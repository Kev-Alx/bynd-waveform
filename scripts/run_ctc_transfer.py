"""Pinned raw Wav2Vec2/HuBERT transfer on the existing frozen English test WAVs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from beyondwaveform.config import load_config
from beyondwaveform.experiments import report
from beyondwaveform.io import atomic_json, digest
from verify_test_run import require, verify, verify_ctc_target


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan_sha = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    output, run = Path(plan["output"]), Path(plan["generation_run"])
    state = {"status": "running", "phase": "preflight", "started_at": now(), "plan": str(plan_path),
             "plan_sha256": plan_sha, "expected_count": 999, "generation_run": str(run), "completed_targets": []}

    def progress(**updates):
        state.update(updates, updated_at=now())
        if not args.check_only:
            atomic_json(output / "progress.json", state)

    def phase(name, command):
        progress(phase=name, command=command, phase_started_at=now(), phase_elapsed_seconds=0)
        print(f"\nStarting {state['target']} / {name}: {shlex.join(command)}", flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        progress(phase_pid=child.pid)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress(phase_elapsed_seconds=round(time.monotonic() - started, 1))
                    print(f"Heartbeat: {state['target']} / {name}, {len(state['completed_targets'])}/2 targets complete, "
                          f"phase elapsed {state['phase_elapsed_seconds']}s", flush=True)
            require(code == 0, f"{state['target']} / {name} exited with code {code}; stopping")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        progress(last_completed_phase=name, phase_elapsed_seconds=round(time.monotonic() - started, 1))

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    progress()
    try:
        require(plan["status"] == "frozen" and plan["count"] == 999, "Unfrozen/wrong-count CTC plan")
        require([r["key"] for r in plan["targets"]] == ["wav2vec2", "hubert"], "Unexpected CTC targets")
        require(plan["variants"] == ["clean", "reconstruction", "adversarial"] and
                plan["decoding"] == {"kind": "greedy", "external_language_model": False}, "Changed controls/decoder")
        for path, sha in [(plan["freeze"], plan["freeze_sha256"]), (str(run / "run.json"), plan["generation_metadata_sha256"]),
                          (str(run / "generated.jsonl"), plan["generated_sha256"]),
                          (plan["source_verification"], plan["source_verification_sha256"])]:
            require(digest(path) == sha, f"Frozen source changed: {path}")
        source_audit = json.loads(Path(plan["source_verification"]).read_text())
        require(source_audit["status"] == "passed" and source_audit["generation_fingerprint"] == plan["generation_fingerprint"] and
                source_audit["integrity"]["saved_wavs_rehashed"] == 2997, "Source has not passed full WAV verification")
        for entry in plan["targets"]:
            require(re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) and entry["upstream_files"], "Missing immutable HF pin")
            require("pytorch_model.bin" in entry["upstream_files"] and
                    re.fullmatch(r"[0-9a-f]{64}", entry["upstream_files"]["pytorch_model.bin"]["lfs_sha256"]), "Missing published weight digest")
        print("CTC preflight passed: two immutable targets, greedy decoding, same 999 frozen clips and three controls.", flush=True)
        if args.check_only:
            return
        summaries, measured = [], []
        for index, entry in enumerate(plan["targets"], start=1):
            key = entry["key"]
            progress(target=key, target_index=index)
            require(digest(plan_path) == plan_sha, "CTC plan changed")
            phase("download", [sys.executable, "-u", str(ROOT / "scripts/download_ctc.py"), "--model-id", entry["repo"],
                               "--revision", entry["revision"], "--cache-dir", plan["cache_dir"], "--output", entry["config"]])
            config = load_config(entry["config"])
            receipt_path = Path(entry["config"] + ".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            require(receipt["repo"] == entry["repo"] and receipt["revision"] == entry["revision"] and
                    receipt["upstream_files"] == entry["upstream_files"], "Downloaded source differs from frozen metadata")
            expected = {"name": entry["repo"], "kind": "ctc", "model_id": entry["snapshot"], "revision": entry["revision"],
                        "repo_id": entry["repo"], "language": "en", "sample_rate": 16000,
                        "decoding": plan["decoding"], "artifact_hashes": receipt["files"]}
            require(config == expected and set(config["artifact_hashes"]) == set(entry["upstream_files"]), "Changed CTC target recipe/files")
            verify_ctc_target(config)
            resolved = {"target": config, "config_sha256": digest(entry["config"]), "receipt_sha256": digest(receipt_path),
                        "plan_sha256": plan_sha, "snapshot": entry["snapshot"]}
            evaluation = output / key
            resolved_path = evaluation / "target-freeze.json"
            if resolved_path.exists():
                require(json.loads(resolved_path.read_text()) == resolved, "Previously frozen CTC target changed")
            else:
                atomic_json(resolved_path, resolved)
                resolved_path.chmod(0o444)
            Path(entry["config"]).chmod(0o444)
            receipt_path.chmod(0o444)
            progress(resolved_target=str(resolved_path))
            require(digest(run / "generated.jsonl") == plan["generated_sha256"], "Source generation index changed")
            summary_path = evaluation / "summary.json"
            if not summary_path.exists():
                phase("evaluation", [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"), "--run", str(run),
                                     "--target", entry["config"], "--output", str(evaluation), "--device", args.device,
                                     "--variants", "clean", "reconstruction", "adversarial"])
            else:
                print(f"Auditing completed {key} evaluation for reuse; no rescore.", flush=True)
            progress(phase="verification")
            require(digest(entry["config"]) == resolved["config_sha256"], "Frozen target config changed during evaluation")
            result = json.loads(summary_path.read_text())
            require(result["target"] == result["target_artifacts"] == config and result["evaluation_scope"] == "target" and
                    result["quality_provenance"] is None, "Wrong raw CTC result/provenance")
            audit = verify(run, evaluation, plan["freeze"], hash_audio=False)
            atomic_json(evaluation / "verification.json", audit)
            observed = result["summary"]["adversarial"]["error_percent"]
            measured.append({"model": key, "repo": entry["repo"], "revision": entry["revision"],
                "summary": result["summary"], "summary_sha256": digest(summary_path),
                "verification_sha256": digest(evaluation / "verification.json"), "target_freeze_sha256": digest(resolved_path),
                "paper_raw_comparison": {"observed_wer": observed, "paper_wer": entry["paper_raw_wer"],
                    "gap_percentage_points": observed - entry["paper_raw_wer"],
                    "caveat": "Independent target checkpoint and greedy CTC decoder; paper target revisions/decoders/subsets unspecified."}})
            summaries.append(str(summary_path))
            state["completed_targets"].append(key)
            report(summaries, output / "report.md")
            atomic_json(output / "results.json", {"status": "running", "generation_fingerprint": plan["generation_fingerprint"],
                "generated_sha256": plan["generated_sha256"], "targets": measured})
            progress()
            print(f"Completed and verified {key}: {result['summary']}", flush=True)
        atomic_json(output / "results.json", {"status": "complete", "plan_sha256": plan_sha,
            "generation_fingerprint": plan["generation_fingerprint"], "generated_sha256": plan["generated_sha256"], "targets": measured})
        with (output / "report.md").open("a") as stream:
            stream.write("\nBoth independently pinned raw CTC targets use the same 999 saved clean/reconstruction/adversarial WAVs from "
                "the frozen Whisper-small attack. Decoding is native argmax/CTC collapse with no external language model, through each "
                "checkpoint's AutoProcessor at 16 kHz. This preserves the existing independent CTC implementation. "
                "All file checksums, model/processor configs, upstream Git/LFS metadata and immutable revisions are recorded in target receipts. "
                "Per-target evaluation rehashes each saved WAV; the auditor recomputes all 2,997 lexical/corpus score rows. "
                "No audio regeneration, target-guided attack tuning or quality rescoring occurred. Original target checkpoint IDs and CTC decoder "
                "recipes are absent from the paper, so raw paper gaps are descriptive. Defenses, baselines and exact Figure 4 arrays remain unavailable.\n")
        progress(status="complete", phase="complete", finished_at=now(), results=str(output / "results.json"))
        print(f"Raw Wav2Vec2/HuBERT transfer complete: {output / 'results.json'}", flush=True)
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", finished_at=now(), error=str(error))
        raise


if __name__ == "__main__":
    main()
