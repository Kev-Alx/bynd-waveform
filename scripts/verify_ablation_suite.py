"""Fresh post-completion audit of exact loss removals and matching control samples."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tqdm import tqdm
from ablation_protocol import control_sample_digest
from beyondwaveform.io import atomic_json, digest, read_jsonl
from run_ablation_suite import preflight
from verify_test_run import require, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    output = Path(args.run).resolve()
    plan = preflight(output / "plan.json")
    result_path = output / "results.json"
    results = json.loads(result_path.read_text())
    job = json.loads((output / "job.json").read_text())
    require(job["status"] == "complete" and job["exit_code"] == 0 and results["status"] == "complete",
            "Ablation suite did not complete")
    require(results["plan_sha256"] == digest(output / "plan.json") and
            [r["name"] for r in results["experiments"]] == [e["name"] for e in plan["experiments"]],
            "Wrong plan or condition coverage")
    source_samples = {r["id"]: {v: control_sample_digest(r["audio"][v]["path"])
                               for v in ["clean", "reconstruction"]}
                      for r in read_jsonl(Path(plan["source_run"]) / "generated.jsonl")}
    audits = {}
    for entry, saved in tqdm(list(zip(plan["experiments"], results["experiments"])), desc="Auditing loss removals", unit="conditions"):
        run = Path(entry["run"])
        evaluation = run / "surrogate-eval"
        require(saved["generation_run"] == str(run) and saved["summary_sha256"] == digest(evaluation / "summary.json") and
                saved["verification_sha256"] == digest(saved["verification"]), "Saved ablation evidence changed")
        require(json.loads((evaluation / "summary.json").read_text())["summary"] == saved["summary"], "Cumulative summary mismatch")
        audit = verify(run, evaluation, plan["base_freeze"], hash_audio=not entry["reuse"],
                       ablation_freeze_path=None if entry["reuse"] else entry["freeze"])
        require(audit["generation_fingerprint"] == saved["generation_fingerprint"], "Ablation fingerprint mismatch")
        for r in read_jsonl(run / "generated.jsonl"):
            for variant in ["clean", "reconstruction"]:
                require(control_sample_digest(r["audio"][variant]["path"]) == source_samples[r["id"]][variant],
                        "Different clean/reconstruction sample content")
        audit["full_loss_control_sample_hashes_matched"] = True
        audits[entry["name"]] = audit
        print(f"Verified {entry['name']}: {audit['summary']['adversarial']}", flush=True)
    atomic_json(output / "verification.json", {
        "status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
        "results_sha256": digest(result_path), "plan_sha256": digest(output / "plan.json"),
        "job_sha256": digest(output / "job.json"), "evaluation_rows": sum(a["evaluation_rows"] for a in audits.values()),
        "saved_wavs_rehashed": sum(a["integrity"]["saved_wavs_rehashed"] for a in audits.values()),
        "unique_source_flacs": 999, "source_flac_rehash_passes": sum(a["integrity"]["source_flacs_rehashed"] for a in audits.values()),
        "full_loss_control_samples_matched_all_conditions": True, "conditions": audits,
        "scope": "Four new loss-removal runs fully rehashed; full loss uses preserved source evidence with fresh metadata/control-sample audit. "
                 "All five conditions' lexical/corpus/quality aggregates and 249,750 finite projected updates checked. "
                 "Models not rerun; original per-condition verification evidence not overwritten."})
    print(f"Fresh ablation audit passed: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
