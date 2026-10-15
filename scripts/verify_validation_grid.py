"""Audit a completed feature grid against its manifest and saved utterance scores."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.data import audit_disjoint
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.pipeline import config_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grid = Path(args.grid).resolve()
    job = json.loads((grid / "job.json").read_text())
    require(job["status"] == "complete" and job["exit_code"] == 0, "Grid job did not complete")
    plan = json.loads((grid / "plan.json").read_text())
    require(plan["suite"] == "feature-grid" and len(plan["experiments"]) == 16, "Unexpected grid plan")
    manifests = audit_disjoint([args.pilot_manifest, args.manifest, args.test_manifest])
    rows = read_jsonl(args.manifest)
    ids = [r["id"] for r in rows]
    require(len(rows) == 100 and all(r["role"] == "validation" and r["split"] == "dev-clean"
                                   and r["language"] == "en" for r in rows), "Invalid validation pool")
    test = read_jsonl(args.test_manifest)
    require(len(test) == 999 and all(r["role"] == "test" and r["split"] == "test-clean"
                                   and r["language"] == "en" for r in test), "Invalid final test pool")
    reference = {r["id"]: r for r in rows}
    manifest_hash = digest(args.manifest)
    protocol = None
    candidates = []
    settings = set()
    for entry in tqdm(plan["experiments"], desc="Auditing validation grid", unit="settings"):
        run = grid / entry["name"]
        metadata = json.loads((run / "run.json").read_text())
        path = run / "evaluation" / "summary.json"
        result = json.loads(path.read_text())
        require(metadata["status"] == "complete" and metadata["count"] == 100, f"Incomplete run: {run}")
        require(metadata["roles"] == result["roles"] == ["validation"], f"Mixed roles: {run}")
        require(metadata["manifest_sha256"] == result["manifest_sha256"] == manifest_hash, f"Manifest mismatch: {run}")
        require(metadata["ids"] == result["evaluation_ids"] == ids, f"ID mismatch: {run}")
        identity = {k: metadata[k] for k in ("config", "manifest_sha256", "ids")}
        require(config_fingerprint(identity) == metadata["fingerprint"] == result["generation_fingerprint"],
                f"Fingerprint mismatch: {run}")
        require(Path(result["generation_run"]).resolve() == run, f"Wrong generation path: {path}")
        require(metadata["config"] == load_config(entry["config"]), f"Plan config differs: {run}")
        config = deepcopy(metadata["config"])
        attack = config["attack"]
        settings.add((attack.pop("rho"), attack.pop("lambda_perc")))
        require(attack["steps"] == 50 and attack["learning_rate"] == 0.05, f"Unexpected attack protocol: {run}")
        require(result["evaluation_scope"] == "surrogate" and result["metric"] == "WER", f"Non-surrogate evaluation: {run}")
        require(result["target_artifacts"] == metadata["surrogate"] and result["target"] ==
                {"kind": "whisper", **metadata["config"]["surrogate"]}, f"Different or defended target: {run}")
        current = {k: result[k] for k in ("target_artifacts", "quality_provenance", "normalization", "metric")}
        current.update(config=config, features=metadata["features"], environment=metadata["environment"])
        require(current["quality_provenance"] is not None, f"Missing quality provenance: {run}")
        if protocol is None:
            protocol = current
        require(current == protocol, f"Mismatched fixed protocol: {run}")
        generated_path = run / "generated.jsonl"
        utterances_path = run / "evaluation" / "utterances.jsonl"
        require(digest(generated_path) == metadata["generated_sha256"], f"Altered generation records: {run}")
        require(digest(utterances_path) == result["utterances_sha256"], f"Altered evaluation rows: {run}")
        generated = read_jsonl(generated_path)
        scored = read_jsonl(utterances_path)
        require([r["id"] for r in generated] == [r["id"] for r in scored] == ids, f"Incomplete saved rows: {run}")
        radius_max = 0.0
        for record, score in zip(generated, scored):
            source = reference[record["id"]]
            require(record["role"] == "validation" and record["language"] == "en", "Invalid generation role/language")
            require(record["text"] == source["text"] == score["reference"], "Reference mismatch")
            require(record["input_audio_sha256"] == source["audio_sha256"], "Input hash mismatch")
            require(set(record["audio"]) == {"clean", "reconstruction", "adversarial"}, "Missing saved controls")
            require(score["variant"] == "adversarial" and score["audio_sha256"] ==
                    record["audio"]["adversarial"]["sha256"], "Evaluation waveform identity mismatch")
            for key, value in error_counts(score["reference"], score["hypothesis"], "en").items():
                require(score[key] == value, f"Incorrect saved error counts: {record['id']}")
            require(set(score["quality"]) == {"dnsmos", "nisqa", "utmos"} and
                    all(math.isfinite(v) for v in score["quality"].values()), "Invalid utterance quality")
            trace = record["trace"]
            require([t["step"] for t in trace] == list(range(1, 51)), "Incomplete attack trace")
            for step in trace:
                require(all(math.isfinite(v) for v in step.values()), "Nonfinite attack trace")
                radius = step["radius_after_update"]
                require(0 <= radius <= metadata["config"]["attack"]["rho"] + 1e-6, "Projection violation")
                radius_max = max(radius_max, radius)
        require(set(result["summary"]) == {"adversarial"}, "Unexpected grid variants")
        summary = result["summary"]["adversarial"]
        require(summary["count"] == 100, "Incorrect summary count")
        for key, value in aggregate_errors(scored).items():
            require(math.isfinite(summary[key]) and math.isclose(summary[key], value, rel_tol=1e-12),
                    f"Incorrect corpus aggregate: {key}")
        for metric in ("dnsmos", "nisqa", "utmos"):
            mean = sum(r["quality"][metric] for r in scored) / len(scored)
            require(math.isfinite(summary["quality"][metric]) and
                    math.isclose(summary["quality"][metric], mean, rel_tol=1e-12), "Incorrect quality mean")
        candidates.append({"name": entry["name"], "summary": str(path), "summary_sha256": digest(path),
                           "run_sha256": digest(run / "run.json"), "count": 100,
                           "error_percent": summary["error_percent"], "quality": summary["quality"],
                           "max_recorded_radius": radius_max})
    require(settings == {(r, w) for r in (0.05, 0.1, 0.5, 1.0) for w in (0.1, 0.5, 1.0, 2.0)},
            "Missing/duplicate rho/lambda settings")
    audit = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
             "grid_job_sha256": digest(grid / "job.json"), "manifests": manifests,
             "protocol": protocol, "candidates": candidates,
             "scope": "Saved JSONL hashes, complete IDs, references, recorded waveform hash correspondence, "
                      "recomputed lexical edit counts/corpus WER/MOS means, fixed protocol and 50-step finite projected traces. "
                      "This audit does not rerun models or rehash every WAV; final evaluation rehashes each saved WAV."}
    atomic_json(args.output, audit)
    print(f"Verified all {len(candidates)} settings, 1,600 validation utterance results; split audit passed.", flush=True)


if __name__ == "__main__":
    main()
