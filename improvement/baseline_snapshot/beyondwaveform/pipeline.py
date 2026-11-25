"""Resumable generation, saved-waveform evaluation, and provenance isolation."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import torch
from tqdm import tqdm
from .attacks import feature_attack, waveform_attack
from .config import AttackConfig, SpectralConfig
from .evaluation import QualityScorer, aggregate_errors, error_counts, defend, quality_provenance
from .io import atomic_json, digest, environment, load_audio, read_jsonl, save_audio, seed_everything, write_jsonl
from .models.feature import load_feature_models
from .models.targets import load_target, load_extension
from .models.whisper import load_whisper


def config_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def generate(config, manifest, output, device="cuda", limit=None):
    output = Path(output)
    rows = read_jsonl(manifest)
    if limit:
        rows = rows[:limit]
    if len({r["language"] for r in rows}) != 1 or rows[0]["language"] != config["surrogate"].get("language", "en"):
        raise ValueError("Manifest and surrogate languages must match")
    identity = {"config": config, "manifest_sha256": digest(manifest), "ids": [r["id"] for r in rows]}
    fingerprint = config_fingerprint(identity)
    metadata_file = output / "run.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to a different config/manifest; choose another run directory")
    print("Loading surrogate Whisper checkpoint...", flush=True)
    surrogate, surrogate_meta = load_whisper(config["surrogate"], device)
    method = config.get("method", "feature")
    feature_meta = None
    if method in {"feature", "reconstruction"}:
        print("Loading WavLM and feature-conditioned HiFi-GAN...", flush=True)
        encoder, vocoder, feature_meta = load_feature_models(config["features"], device)
    if method == "extension":
        extension = load_extension(config["extension"])
    metadata = {**identity, "fingerprint": fingerprint, "manifest": str(Path(manifest).resolve()),
                "roles": sorted({r["role"] for r in rows}), "surrogate": surrogate_meta,
                "features": feature_meta, "environment": environment(), "status": "running"}
    if metadata_file.exists():
        prior = json.loads(metadata_file.read_text())
        if prior["surrogate"] != surrogate_meta or prior["features"] != feature_meta:
            raise ValueError("Model artifacts changed; refusing to mix generations")
    atomic_json(metadata_file, metadata)
    records = []
    for row in tqdm(rows, desc="Generating"):
        uid = row["id"]
        record_file = output / "records" / f"{uid}.json"
        if record_file.exists():
            record = json.loads(record_file.read_text())
            if record["input_audio_sha256"] != digest(row["audio"]):
                raise ValueError("Input audio changed since generation")
            for asset in record["audio"].values():
                if digest(asset["path"]) != asset["sha256"]:
                    raise ValueError("Saved waveform corrupted")
            records.append(record)
            continue
        actual_hash = digest(row["audio"])
        if row.get("audio_sha256", actual_hash) != actual_hash:
            raise ValueError(f"Corpus hash mismatch: {uid}")
        seed = int(config.get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
        seed_everything(seed % 2**32)
        x = load_audio(row["audio"]).to(device)
        started = time.perf_counter()
        if method in {"feature", "reconstruction"}:
            settings = dict(config.get("attack", {}))
            if method == "reconstruction":
                settings.update(steps=0, initialization="zero")
            result = feature_attack(x, row["text"], encoder, vocoder, surrogate, AttackConfig(**settings),
                                    SpectralConfig(**config.get("spectral", {})))
            adversarial, reconstruction, trace = result.audio, result.reconstruction, result.trace
            torch.save(result.delta.cpu(), output / f"{uid}.delta.pt") if config.get("save_delta", False) else None
        elif method in {"pgd", "mi-fgsm", "vmi-fgsm"}:
            adversarial, trace = waveform_attack(x, row["text"], surrogate, method=method, **config.get("waveform", {}))
            reconstruction = None
        elif method == "clean":
            adversarial, reconstruction, trace = x, None, []
        elif method == "extension":
            adversarial, trace = extension.generate(x, row["text"], surrogate)
            reconstruction = None
        else:
            raise ValueError(f"Unsupported attack {method!r}; no invented Muting/Sloth implementation")
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        seconds = time.perf_counter() - started
        assets = {"clean": save_audio(output / "audio" / "clean" / f"{uid}.wav", x),
                  "adversarial": save_audio(output / "audio" / "adversarial" / f"{uid}.wav", adversarial)}
        if reconstruction is not None:
            assets["reconstruction"] = save_audio(output / "audio" / "reconstruction" / f"{uid}.wav", reconstruction)
        record = {"id": uid, "text": row["text"], "language": row["language"], "role": row["role"],
                  "input_audio_sha256": actual_hash, "audio": assets, "seed": seed % 2**32,
                  "generation_seconds": seconds, "trace": trace}
        atomic_json(record_file, record)
        records.append(record)
    write_jsonl(output / "generated.jsonl", records)
    metadata.update(status="complete", count=len(records), generated_sha256=digest(output / "generated.jsonl"),
                    generation_seconds=sum(r["generation_seconds"] for r in records))
    atomic_json(metadata_file, metadata)
    return metadata


def evaluate(run, target_config, output, device="cuda", variants=None, quality=False, artifact_root="artifacts"):
    run, output = Path(run), Path(output)
    metadata = json.loads((run / "run.json").read_text())
    if metadata["status"] != "complete" or digest(run / "generated.jsonl") != metadata["generated_sha256"]:
        raise ValueError("Incomplete or altered generation run")
    records = read_jsonl(run / "generated.jsonl")
    print("Loading evaluation target...", flush=True)
    target, target_meta = load_target(target_config, device)
    variants = variants or sorted(set.intersection(*(set(r["audio"]) for r in records)))
    scorer = QualityScorer(artifact_root) if quality else None
    defense = target_config.get("defense", {"kind": "none"})
    rows = []
    for record in tqdm(records, desc="Evaluating saved audio"):
        for variant in variants:
            asset = record["audio"][variant]
            if digest(asset["path"]) != asset["sha256"]:
                raise ValueError("Evaluation audio hash differs from generated asset")
            audio = load_audio(asset["path"]).to(device)
            transcript = target.transcribe(defend(audio, defense))
            row = {"id": record["id"], "variant": variant, "audio_sha256": asset["sha256"],
                   "reference": record["text"], "hypothesis": transcript,
                   **error_counts(record["text"], transcript, record["language"])}
            if scorer:
                row["quality"] = scorer(audio)  # score exported audio BEFORE any defense
            rows.append(row)
    summary = {}
    for variant in variants:
        pool = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"count": len(pool), **aggregate_errors(pool)}
        if scorer:
            summary[variant]["quality"] = {metric: sum(r["quality"][metric] for r in pool) / len(pool)
                                             for metric in scorer.metrics}
    is_surrogate = target_config["kind"] == "whisper" and target_meta == metadata["surrogate"] and defense.get("kind", "none") == "none"
    result = {"generation_run": str(run.resolve()), "generation_fingerprint": metadata["fingerprint"],
              "manifest_sha256": metadata["manifest_sha256"], "roles": metadata["roles"],
              "evaluation_ids": [r["id"] for r in records],
              "evaluation_scope": "surrogate" if is_surrogate else "target", "target": target_config,
              "target_artifacts": target_meta, "metric": "CER" if records[0]["language"] == "zh" else "WER",
              "normalization": "NFKC/lowercase/punctuation-to-space; Chinese whitespace removed",
              "summary": summary, "quality_provenance": quality_provenance(artifact_root) if scorer else None}
    write_jsonl(output / "utterances.jsonl", rows)
    result["utterances_sha256"] = digest(output / "utterances.jsonl")
    atomic_json(output / "summary.json", result)
    return result
