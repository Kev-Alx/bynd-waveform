"""Corpus indexing, deterministic independently sampled subsets and split audits."""
from __future__ import annotations
import json
import random
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from .io import digest, atomic_json, read_jsonl, write_jsonl


def prepare_manifest(corpus, root, split, role, output, count=None, seed=2026,
                     max_seconds=30.0, ids_file=None):
    if role not in {"train", "finetune", "validation", "test", "pilot"}:
        raise ValueError("Unknown manifest role")
    if split.startswith("test") and role != "test":
        raise ValueError("Official test splits must retain role=test; do not relabel them for tuning")
    root = Path(root).resolve()
    rows, excluded = [], []
    if corpus == "librispeech":
        pool = root / "LibriSpeech" / split
        candidates = []
        for text_file in tqdm(sorted(pool.rglob("*.trans.txt")), desc=f"Indexing {split} transcripts", unit="files"):
            for line in text_file.read_text().splitlines():
                uid, text = line.split(maxsplit=1)
                candidates.append((uid, text, text_file.parent / f"{uid}.flac", uid.split("-")[0]))
        language = "en"
    elif corpus == "aishell":
        pool = root / "data_aishell"
        texts = dict(line.split(maxsplit=1) for line in (pool / "transcript" / "aishell_transcript_v0.8.txt").read_text().splitlines() if line.strip())
        candidates = [(p.stem, texts[p.stem], p, p.parent.name)
                      for p in tqdm((pool / "wav" / split).rglob("*.wav"),
                                    desc=f"Indexing AISHELL {split}", unit="files") if p.stem in texts]
        language = "zh"
    else:
        raise ValueError(corpus)
    selected_ids = set(Path(ids_file).read_text().split()) if ids_file else None
    for uid, text, audio, speaker in tqdm(candidates, desc=f"Hashing {split} audio", unit="files"):
        if selected_ids is not None and uid not in selected_ids:
            continue
        info = sf.info(audio)
        duration = info.frames / info.samplerate
        if duration > max_seconds or duration < 0.045:
            excluded.append({"id": uid, "duration": duration, "reason": "duration_policy"})
            continue
        rows.append({"id": uid, "text": text.strip(), "audio": str(audio.resolve()), "audio_sha256": digest(audio),
                     "corpus": corpus, "split": split, "role": role, "language": language, "speaker": speaker,
                     "sample_rate": info.samplerate, "samples": info.frames, "duration": duration})
    if selected_ids is not None and selected_ids != {r["id"] for r in rows}:
        raise ValueError("Requested IDs missing or excluded by duration policy; exact manifest must not silently change")
    if count is not None:
        if count <= 0 or count > len(rows):
            raise ValueError(f"Requested {count} utterances; only {len(rows)} eligible")
        rows = random.Random(seed).sample(rows, count)
    rows.sort(key=lambda r: r["id"])
    if not rows:
        raise ValueError("No corpus recordings found; check root and split")
    write_jsonl(output, rows)
    atomic_json(str(output) + ".metadata.json", {"corpus": corpus, "split": split, "role": role,
        "count": len(rows), "seed": seed, "manifest_sha256": digest(output), "excluded": excluded,
        "selection": "provided IDs" if ids_file else "independent deterministic subset; NOT recovered author manifest",
        "max_seconds": max_seconds})
    return rows


def audit_disjoint(manifests):
    seen_ids, seen_audio = {}, {}
    report = {}
    for manifest in manifests:
        rows = read_jsonl(manifest)
        for row in rows:
            uid = (row.get("corpus", "custom"), row["id"])
            audio_hash = row.get("audio_sha256") or digest(row["audio"])
            for key, mapping in [(uid, seen_ids), (audio_hash, seen_audio)]:
                if key in mapping and mapping[key] != str(manifest):
                    raise ValueError(f"Data leakage between {mapping[key]} and {manifest}: {row['id']}")
                mapping[key] = str(manifest)
        report[str(manifest)] = {"count": len(rows), "sha256": digest(manifest), "roles": sorted({r["role"] for r in rows})}
    return report


def partition_manifest(manifest, output, validation_fraction=0.1, seed=2026, remainder_role="finetune"):
    """Speaker-disjoint independent partition, useful for AISHELL dev fine-tuning."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Fraction must lie strictly between zero and one")
    rows = read_jsonl(manifest)
    if any(r["split"].startswith("test") for r in rows):
        raise ValueError("Cannot partition final test data into tuning data")
    speakers = sorted({r["speaker"] for r in rows})
    if len(speakers) < 2:
        raise ValueError("Speaker-disjoint partition requires at least two speakers")
    random.Random(seed).shuffle(speakers)
    validation_speakers = set(speakers[:max(1, min(len(speakers)-1, round(len(speakers)*validation_fraction)))])
    validation, remainder = [], []
    for row in tqdm(rows, desc="Partitioning manifest", unit="utterances"):
        row = dict(row)
        row["role"] = "validation" if row["speaker"] in validation_speakers else remainder_role
        (validation if row["speaker"] in validation_speakers else remainder).append(row)
    output = Path(output)
    paths = [output / f"{remainder_role}.jsonl", output / "validation.jsonl"]
    write_jsonl(paths[0], remainder)
    write_jsonl(paths[1], validation)
    audit = audit_disjoint(paths)
    atomic_json(output / "partition.json", {"input_sha256": digest(manifest), "seed": seed,
                "validation_speakers": sorted(validation_speakers), "audit": audit,
                "status": "independent partition; original author split unavailable"})
    return paths
