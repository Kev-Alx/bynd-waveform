"""Audit frozen test generation and saved-audio evaluations without rerunning ASR."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.pipeline import config_fingerprint

METRICS = ("dnsmos", "nisqa", "utmos")
VARIANTS = ("clean", "reconstruction", "adversarial")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    return {"min": float(min(values)), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "max": float(max(values))}


def verify_ctc_target(target):
    require(re.fullmatch(r"[0-9a-f]{40}", target["revision"]), "Unpinned CTC target")
    require(target["language"] == "en" and target["sample_rate"] == 16000 and
            target["decoding"] == {"kind": "greedy", "external_language_model": False}, "Changed CTC decoding protocol")
    snapshot = Path(target["model_id"]).resolve()
    require(snapshot.is_dir() and snapshot.name == target["revision"] and target["artifact_hashes"], "Wrong local CTC snapshot")
    require(any(n == "pytorch_model.bin" or n.endswith(".safetensors") for n in target["artifact_hashes"]), "Missing CTC weight hash")
    for filename, expected in target["artifact_hashes"].items():
        path = snapshot / filename
        require(path.is_file() and digest(path) == expected, f"CTC artifact changed: {filename}")


def verify(run, evaluation, freeze_path, hash_audio=True, ablation_freeze_path=None):
    run, evaluation, freeze_path = Path(run).resolve(), Path(evaluation).resolve(), Path(freeze_path).resolve()
    frozen = json.loads(freeze_path.read_text())
    ablation = None
    if ablation_freeze_path:
        from ablation_protocol import load_ablation_freeze
        frozen, ablation = load_ablation_freeze(freeze_path, ablation_freeze_path)
    metadata = json.loads((run / "run.json").read_text())
    result_path = evaluation / "summary.json"
    result = json.loads(result_path.read_text())
    require(digest(frozen["config_path"]) == frozen["config_sha256"] and
            load_config(frozen["config_path"]) == metadata["config"] == frozen["config"], "Frozen config changed")
    require(metadata["surrogate"] == frozen["surrogate_artifacts"] and
            metadata["features"] == frozen["feature_artifacts"], "Generation artifacts differ from freeze")
    require(digest(frozen["test_manifest"]) == metadata["manifest_sha256"] ==
            result["manifest_sha256"] == frozen["test_manifest_sha256"], "Test manifest changed")
    source = read_jsonl(frozen["test_manifest"])
    ids = [r["id"] for r in source]
    require(len(ids) == frozen["expected_count"] == metadata["count"] == 999, "Wrong test count")
    require(ids == frozen["test_ids"] == metadata["ids"] == result["evaluation_ids"], "Mismatched test IDs")
    require(metadata["status"] == "complete" and metadata["roles"] == result["roles"] == ["test"], "Incomplete/mixed-role test")
    require(all(r["role"] == "test" and r["language"] == "en" and r["split"] == "test-clean" for r in source),
            "Invalid test manifest role/language/split")
    identity = {key: metadata[key] for key in ("config", "manifest_sha256", "ids")}
    require(config_fingerprint(identity) == metadata["fingerprint"] == result["generation_fingerprint"], "Generation fingerprint mismatch")
    require(Path(result["generation_run"]).resolve() == run and result["metric"] == "WER", "Wrong generation/metric")
    require(result["normalization"] == "NFKC/lowercase/punctuation-to-space; Chinese whitespace removed", "Changed text normalization")
    require(result["target"]["kind"] in {"whisper", "ctc"} and
            result["target"].get("defense", {}).get("kind", "none") == "none", "This audit requires raw Whisper/CTC evaluation")
    target = result["target_artifacts"]
    if result["target"]["kind"] == "whisper":
        require(digest(target["checkpoint"]) == target["sha256"], "Target checkpoint changed")
        require(target["language"] == "en" and target["decoding"] == frozen["config"]["surrogate"]["decoding"], "Changed decoding protocol")
        expected_scope = "surrogate" if target == metadata["surrogate"] else "target"
    else:
        require(target == result["target"], "Mismatched CTC target")
        verify_ctc_target(target)
        expected_scope = "target"
    require(result["evaluation_scope"] == expected_scope, "Misclassified evaluation scope")
    if expected_scope == "surrogate":
        require(result["quality_provenance"] == frozen["quality_provenance"], "Changed quality scorer provenance")
    has_quality = result["quality_provenance"] is not None
    require(digest(run / "generated.jsonl") == metadata["generated_sha256"], "Generation JSONL changed")
    require(digest(evaluation / "utterances.jsonl") == result["utterances_sha256"], "Evaluation JSONL changed")
    generated = read_jsonl(run / "generated.jsonl")
    require([r["id"] for r in generated] == ids, "Incomplete generation index")
    # Evaluation IDs legitimately repeat across three variants. Validate pair uniqueness.
    scored = [json.loads(line) for line in (evaluation / "utterances.jsonl").read_text().splitlines() if line.strip()]
    pairs = {(r["id"], r["variant"]): r for r in scored}
    require(len(pairs) == len(scored) == 2997 and set(pairs) == {(uid, v) for uid in ids for v in VARIANTS},
            "Duplicate or missing (ID, variant) evaluation rows")
    require(set(result["summary"]) == set(VARIANTS), "Incomplete summary controls")
    require({p.stem for p in (run / "records").glob("*.json")} == set(ids), "Missing/extra generation records")
    for variant in VARIANTS:
        require({p.stem for p in (run / "audio" / variant).glob("*.wav")} == set(ids), "Missing/extra saved WAVs")
    radii, first_nll, last_nll, seconds, length_differences = [], [], [], [], []
    waveform_hashes = 0
    steps = frozen["config"]["attack"]["steps"]
    rho = frozen["config"]["attack"]["rho"]
    for row, record in tqdm(zip(source, generated), total=len(ids), desc="Verifying test controls and traces", unit="clips"):
        uid = row["id"]
        require(record == json.loads((run / "records" / f"{uid}.json").read_text()), f"Record/index differ: {uid}")
        require(record["text"] == row["text"] and record["role"] == "test" and record["language"] == "en", "Changed reference/role/language")
        require(record["input_audio_sha256"] == row["audio_sha256"], "Input hash correspondence mismatch")
        expected_seed = (int(frozen["config"].get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % 2**32
        require(record["seed"] == expected_seed, "Changed per-utterance seed")
        if hash_audio:
            require(digest(row["audio"]) == row["audio_sha256"], f"Input FLAC changed: {uid}")
        require(set(record["audio"]) == set(VARIANTS), "Missing generation controls")
        for variant in VARIANTS:
            asset, score = record["audio"][variant], pairs[(uid, variant)]
            require(Path(asset["path"]).resolve() == run / "audio" / variant / f"{uid}.wav", "Unexpected saved asset path")
            require(asset["sha256"] == score["audio_sha256"] and score["reference"] == row["text"], "Wrong evaluated audio/reference")
            require(asset["format"] == "WAV/FLOAT" and asset["sample_rate"] == 16000 and math.isfinite(asset["peak"]), "Invalid audio policy")
            if hash_audio:
                require(digest(asset["path"]) == asset["sha256"], f"Saved WAV changed: {uid}/{variant}")
                info = sf.info(asset["path"])
                require(info.subtype == "FLOAT" and info.samplerate == 16000 and info.channels == 1 and
                        info.frames == asset["samples"], "Exported WAV format/length differs")
                waveform_hashes += 1
            for key, value in error_counts(score["reference"], score["hypothesis"], "en").items():
                require(score[key] == value, f"Incorrect edit counts: {uid}/{variant}")
            if has_quality:
                require(set(score["quality"]) == set(METRICS) and all(math.isfinite(v) for v in score["quality"].values()),
                        "Missing/nonfinite utterance quality")
            else:
                require("quality" not in score, "Unprovenanced quality values")
        trace = record["trace"]
        require([t["step"] for t in trace] == list(range(1, steps + 1)), "Incomplete attack steps")
        for step in trace:
            require(all(math.isfinite(v) for v in step.values()), "Nonfinite attack trace")
            require(0 <= step["radius_after_update"] <= rho + 1e-6, "Feature projection violation")
            if ablation:
                settings = frozen["config"]["attack"]
                if not settings["attack"]:
                    require(step["nll_before_update"] == 0, "Removed attack term was evaluated")
                if not settings["temporal"] or settings["lambda_perc"] == 0:
                    require(step["tv_ratio"] == 0, "Removed TV term was evaluated")
                if not settings["high_frequency"] or settings["lambda_perc"] == 0:
                    require(step["hf_ratio"] == 0, "Removed HF term was evaluated")
                expected_loss = -step["nll_before_update"] + settings["lambda_perc"] * (
                    step["tv_ratio"] + settings["alpha_hf"] * step["hf_ratio"])
                require(math.isclose(step["loss_before_update"], expected_loss, rel_tol=1e-5, abs_tol=1e-5),
                        "Ablation objective does not match retained terms")
        radii.append(trace[-1]["radius_after_update"])
        first_nll.append(trace[0]["nll_before_update"])
        last_nll.append(trace[-1]["nll_before_update"])
        require(math.isfinite(record["generation_seconds"]) and record["generation_seconds"] > 0, "Invalid timing")
        seconds.append(record["generation_seconds"])
        require(record["audio"]["adversarial"]["samples"] == record["audio"]["reconstruction"]["samples"], "Different vocoder control lengths")
        length_differences.append(record["audio"]["clean"]["samples"] - record["audio"]["adversarial"]["samples"])
    quality_diagnostics = {}
    for variant in VARIANTS:
        pool = [pairs[(uid, variant)] for uid in ids]
        summary = result["summary"][variant]
        require(summary["count"] == 999, "Incorrect variant count")
        for key, value in aggregate_errors(pool).items():
            require(math.isfinite(summary[key]) and math.isclose(summary[key], value, rel_tol=1e-12), "Incorrect corpus aggregate")
        if has_quality:
            for metric in METRICS:
                mean = sum(r["quality"][metric] for r in pool) / len(pool)
                require(math.isfinite(summary["quality"][metric]) and math.isclose(summary["quality"][metric], mean, rel_tol=1e-12),
                        "Incorrect quality aggregate")
            quality_diagnostics[variant] = {
                "mean_gate_passed": all(v > 2.5 for v in summary["quality"].values()),
                "utterances_at_or_below_2_5": {m: sum(r["quality"][m] <= 2.5 for r in pool) for m in METRICS},
                "utterances_with_all_three_above_2_5": sum(all(v > 2.5 for v in r["quality"].values()) for r in pool)}
    require(math.isclose(sum(seconds), metadata["generation_seconds"], rel_tol=1e-12), "Incorrect generation timing sum")
    return {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(), "generation_run": str(run),
            "evaluation": str(evaluation), "summary_sha256": digest(result_path), "generation_fingerprint": metadata["fingerprint"],
            "generated_sha256": metadata["generated_sha256"], "utterances_sha256": result["utterances_sha256"],
            "freeze_sha256": digest(freeze_path), "config_sha256": frozen["config_sha256"],
            "ablation": {"name": ablation["name"], "freeze_sha256": digest(ablation_freeze_path),
                         "removed_terms_verified": True} if ablation else None,
            "test_manifest_sha256": metadata["manifest_sha256"], "count": 999, "evaluation_rows": len(scored),
            "evaluation_scope": expected_scope, "summary": result["summary"], "quality_diagnostics": quality_diagnostics,
            "integrity": {"source_flacs_rehashed": 999 if hash_audio else 0, "saved_wavs_rehashed": waveform_hashes,
                          "all_49950_update_traces_finite_and_projected": True},
            "attack_diagnostics": {"rho_limit": rho, "final_radius": distribution(radii),
                "final_radius_at_limit_count": sum(math.isclose(v, rho, abs_tol=1e-6) for v in radii),
                "first_preupdate_nll": distribution(first_nll), "last_preupdate_nll": distribution(last_nll),
                "preupdate_nll_increase_count": sum(b > a for a, b in zip(first_nll, last_nll)),
                "nll_note": "First and last logged pre-update NLL; not a separately measured post-update final NLL. "
                            "When attack loss is removed, logged NLL is a zero placeholder and is not measured.",
                "generation_seconds": distribution(seconds), "clean_minus_vocoder_samples": distribution(length_differences)},
            "scope": "Frozen protocol, complete test IDs/control pairs, references, edit counts, corpus WER/MOS aggregates, "
                     "generation records, finite projected traces; source/WAV contents and WAV headers when hash_audio=True. "
                     "ASR and quality models were not rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--ablation-freeze")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-only", action="store_true", help="Skip rehashing corpus/saved WAVs (evaluation already hashes WAVs)")
    args = parser.parse_args()
    audit = verify(args.run, args.evaluation, args.freeze, hash_audio=not args.metadata_only,
                   ablation_freeze_path=args.ablation_freeze)
    atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
