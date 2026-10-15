"""Download pinned raw Whisper targets and evaluate the same frozen 999 saved clips.

Use detached tmux and run_logged.py. The plan is fixed before target evaluation;
this script never generates audio, tunes attacks, or reruns quality scorers.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import whisper
from beyondwaveform.config import load_config
from beyondwaveform.experiments import report
from beyondwaveform.io import atomic_json, digest
from beyondwaveform.references import OTHER_PROSE_ANCHORS
from verify_test_run import require, verify


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan_hash = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    run = Path(plan["generation_run"])
    state = {"status": "running", "phase": "preflight", "started_at": now(),
             "plan": str(plan_path), "plan_sha256": plan_hash,
             "generation_run": str(run), "expected_count": 999, "completed_targets": []}

    def progress(**updates):
        state.update(updates, updated_at=now())
        if not args.check_only:
            atomic_json(output / "progress.json", state)

    def phase(name, command):
        progress(phase=name, command=command, phase_started_at=now(), phase_elapsed_seconds=0)
        print(f"\nStarting {name}: {shlex.join(command)}", flush=True)
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
                    print(f"Heartbeat: {state['target']} / {name}, "
                          f"{len(state['completed_targets'])}/{len(plan['targets'])} targets complete, "
                          f"phase elapsed {state['phase_elapsed_seconds']}s", flush=True)
            require(code == 0, f"{name} exited with code {code}; stopping transfer")
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
        require(plan["status"] == "frozen" and plan["count"] == 999, "Unfrozen or wrong cohort plan")
        require([r["model"] for r in plan["targets"]] == ["tiny", "base", "medium", "large-v2"], "Unexpected transfer targets")
        require(digest(plan["freeze"]) == plan["freeze_sha256"], "Attack freeze changed")
        require(digest(run / "run.json") == plan["generation_metadata_sha256"] and
                digest(run / "generated.jsonl") == plan["generated_sha256"], "Generation metadata/index changed")
        require(digest(plan["source_verification"]) == plan["source_verification_sha256"], "Source audit changed")
        source_audit = json.loads(Path(plan["source_verification"]).read_text())
        require(source_audit["status"] == "passed" and source_audit["integrity"]["saved_wavs_rehashed"] == 2997,
                "Source saved audio has not passed complete integrity verification")
        require(source_audit["generation_fingerprint"] == plan["generation_fingerprint"] and
                source_audit["generated_sha256"] == plan["generated_sha256"], "Audit belongs to a different run")
        for entry in plan["targets"]:
            require(digest(entry["config"]) == entry["config_sha256"] and load_config(entry["config"]) == entry["target_config"],
                    "Target config changed")
            require(whisper._MODELS[entry["model"]] == entry["url"] and
                    entry["url"].split("/")[-2] == entry["checkpoint_sha256"], "Published model URL/hash changed")
            require(entry["target_config"]["kind"] == "whisper" and entry["target_config"]["language"] == "en" and
                    entry["target_config"]["decoding"] == {"beam_size": 5, "temperature": 0.0} and
                    entry["target_config"].get("defense", {}).get("kind", "none") == "none", "Unexpected raw decoding protocol")
        print("Transfer preflight passed: four pinned raw models, fixed config/manifest, 999 previously generated clips.", flush=True)
        if args.check_only:
            return
        summaries, results = [], []
        for index, entry in enumerate(plan["targets"], start=1):
            model = entry["model"]
            progress(target=model, target_index=index)
            require(digest(plan_path) == plan_hash, "Transfer plan changed during execution")
            phase("download", [sys.executable, "-u", str(ROOT / "scripts/download_models.py"),
                               "--root", str(ROOT / "artifacts"), "--components", "whisper", "--whisper", model])
            checkpoint = Path(entry["checkpoint"])
            require(digest(checkpoint) == entry["checkpoint_sha256"], "Downloaded checkpoint differs from pinned hash")
            require(digest(entry["config"]) == entry["config_sha256"], "Target config changed before evaluation")
            require(digest(run / "generated.jsonl") == plan["generated_sha256"], "Saved generation index changed")
            evaluation = output / model
            summary_path = evaluation / "summary.json"
            if not summary_path.exists():
                phase("evaluation", [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"), "--run", str(run),
                                     "--target", entry["config"], "--output", str(evaluation), "--device", args.device,
                                     "--variants", "clean", "reconstruction", "adversarial"])
            else:
                print(f"Checking completed {model} evaluation for reuse; no rescore.", flush=True)
            progress(phase="verification")
            summary = json.loads(summary_path.read_text())
            require(summary["target"] == entry["target_config"] and
                    summary["target_artifacts"]["sha256"] == entry["checkpoint_sha256"] and
                    summary["evaluation_scope"] == "target" and summary["quality_provenance"] is None,
                    "Existing evaluation does not match the pinned raw transfer protocol")
            audit = verify(run, evaluation, plan["freeze"], hash_audio=False)
            atomic_json(evaluation / "verification.json", audit)
            summaries.append(str(summary_path))
            results.append({"model": model, "checkpoint_sha256": entry["checkpoint_sha256"],
                            "summary": summary["summary"], "summary_sha256": digest(summary_path),
                            "verification_sha256": digest(evaluation / "verification.json")})
            state["completed_targets"].append(model)
            progress()
            report(summaries, output / "report.md")
            atomic_json(output / "results.json", {"status": "running", "generation_fingerprint": plan["generation_fingerprint"],
                                                   "generated_sha256": plan["generated_sha256"], "targets": results})
            print(f"Completed and verified {model}: {summary['summary']}", flush=True)
        large = results[-1]["summary"]["adversarial"]["error_percent"]
        anchor = OTHER_PROSE_ANCHORS["whisper_large_raw_WER"]
        atomic_json(output / "results.json", {"status": "complete", "plan_sha256": plan_hash,
                   "generation_fingerprint": plan["generation_fingerprint"], "generated_sha256": plan["generated_sha256"],
                   "targets": results, "paper_large_raw_comparison": {"observed_large_v2_wer": large, "paper_large_wer": anchor,
                   "gap_percentage_points": large - anchor, "caveat": "Paper large revision, subset, config and decoder unspecified; large-v2 is an independent choice."}})
        with (output / "report.md").open("a") as stream:
            stream.write("\nAll four targets use the identical 999 saved clean/reconstruction/adversarial WAVs from the frozen feature run. "
                         "Per-target evaluation verifies each saved waveform hash; verification recomputes all 2,997 edit-count rows. "
                         "Audio was not regenerated or attack settings retuned. Quality scores are target-independent and remain in the "
                         "[verified source report](../english-feature-test-999/report.md); quality models were not rerun. "
                         "Targets are pinned multilingual Whisper tiny/base/medium/large-v2 with English forced, beam size 5 and temperature 0. "
                         f"Paper large raw WER is {anchor:.2f}%; independently chosen large-v2 measures {large:.4f}% (gap {large-anchor:+.4f} pp). "
                         "Paper checkpoint revisions/subsets/decoders are unidentified; no exact Figure 3 curve or defense claim is made.\n")
        progress(status="complete", phase="complete", finished_at=now(), results=str(output / "results.json"))
        print(f"All four raw Whisper transfer targets complete: {output / 'results.json'}", flush=True)
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", finished_at=now(), error=str(error))
        raise


if __name__ == "__main__":
    main()
