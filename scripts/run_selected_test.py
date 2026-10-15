"""Generate and evaluate the frozen English test config, with stage/heartbeat progress.

Run this script through run_logged.py inside detached tmux. It never selects or tunes.
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
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.io import atomic_json, digest, read_jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--ablation-freeze", help="Predefined loss-removal receipt; preserves the original selection freeze")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    freeze_path = Path(args.freeze).resolve()
    freeze_hash = digest(freeze_path)
    frozen = json.loads(freeze_path.read_text())
    ablation_hash = None
    if args.ablation_freeze:
        from ablation_protocol import load_ablation_freeze
        frozen, _ = load_ablation_freeze(freeze_path, args.ablation_freeze)
        ablation_hash = digest(args.ablation_freeze)
    output = Path(args.output).resolve()
    progress_path = output / "progress.json"
    state = {"status": "running", "phase": "preflight", "started_at": now(),
             "freeze": str(freeze_path), "freeze_sha256": freeze_hash,
             "expected_count": frozen["expected_count"], "output": str(output)}
    if args.ablation_freeze:
        state.update(ablation_freeze=args.ablation_freeze, ablation_freeze_sha256=ablation_hash)

    def progress(**updates):
        state.update(updates, updated_at=now())
        state["generated_records"] = len(list((output / "records").glob("*.json")))
        if not args.check_only:
            atomic_json(progress_path, state)

    def phase(name, command):
        progress(phase=name, phase_started_at=now(), command=command)
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
                    print(f"Heartbeat: {name}, {state['generated_records']}/{frozen['expected_count']} "
                          f"clips generated, phase elapsed {state['phase_elapsed_seconds']}s", flush=True)
            require(code == 0, f"{name} exited with code {code}; stopping the pipeline")
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
        require(frozen["status"] == "frozen", "Config is not frozen")
        proofs = {frozen["config_path"]: frozen["config_sha256"],
                  frozen["selection_path"]: frozen["selection_sha256"],
                  frozen["selected"]["summary"]: frozen["selected_summary_sha256"],
                  frozen["source_run_path"]: frozen["source_run_sha256"],
                  frozen["validation_audit_path"]: frozen["validation_audit_sha256"],
                  frozen["validation_manifest"]: frozen["validation_manifest_sha256"],
                  frozen["test_manifest"]: frozen["test_manifest_sha256"],
                  **frozen["artifact_hashes"]}
        for path, expected in tqdm(proofs.items(), desc="Checking frozen inputs", unit="files"):
            require(digest(path) == expected, f"Frozen input changed: {path}")
        require(load_config(frozen["config_path"]) == frozen["config"], "Frozen config fields differ")
        rows = read_jsonl(frozen["test_manifest"])
        ids = [r["id"] for r in rows]
        require(len(rows) == frozen["expected_count"] == 999 and ids == frozen["test_ids"], "Wrong test IDs/count")
        require(all(r["role"] == "test" and r["language"] == "en" and r["split"] == "test-clean" for r in rows),
                "Invalid test role/language/split")
        require(frozen["evaluation"] == {"target": "surrogate", "variants": ["clean", "reconstruction", "adversarial"],
                                         "quality": True}, "Unexpected frozen evaluation protocol")
        print("Preflight passed: frozen config, manifests, selection evidence and model hashes; 999 test clips.", flush=True)
        if args.ablation_freeze:
            print("Config is a predefined loss removal of the selected config; no new selection or tuning.", flush=True)
        if args.check_only:
            return
        phase("generation", [sys.executable, "-u", str(ROOT / "scripts/generate.py"),
                             "--config", frozen["config_path"], "--manifest", frozen["test_manifest"],
                             "--output", str(output), "--device", args.device])
        metadata = json.loads((output / "run.json").read_text())
        require(metadata["status"] == "complete" and metadata["count"] == 999 and metadata["ids"] == ids,
                "Generation did not cover the complete test manifest")
        require(metadata["config"] == frozen["config"] and metadata["surrogate"] == frozen["surrogate_artifacts"] and
                metadata["features"] == frozen["feature_artifacts"], "Generation artifacts/config differ from selection")
        require(digest(freeze_path) == freeze_hash and digest(frozen["config_path"]) == frozen["config_sha256"] and
                metadata["manifest_sha256"] == frozen["test_manifest_sha256"], "Frozen protocol changed")
        if args.ablation_freeze:
            require(digest(args.ablation_freeze) == ablation_hash, "Ablation freeze changed")
        evaluation = output / "surrogate-eval"
        phase("evaluation", [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"), "--run", str(output),
                             "--target", "surrogate", "--output", str(evaluation), "--device", args.device,
                             "--variants", "clean", "reconstruction", "adversarial", "--quality",
                             "--artifact-root", frozen["config"]["features"]["artifact_root"]])
        summary_path = evaluation / "summary.json"
        result = json.loads(summary_path.read_text())
        require(result["evaluation_scope"] == "surrogate" and result["roles"] == ["test"] and
                result["evaluation_ids"] == ids and result["manifest_sha256"] == frozen["test_manifest_sha256"],
                "Evaluation does not match the frozen test manifest/surrogate")
        require(result["quality_provenance"] == frozen["quality_provenance"], "Quality scorer provenance changed")
        require(set(result["summary"]) == {"clean", "reconstruction", "adversarial"} and
                all(v["count"] == 999 for v in result["summary"].values()), "Incomplete evaluation controls")
        phase("report", [sys.executable, str(ROOT / "scripts/report.py"), str(summary_path),
                         "--output", str(output / "report.md")])
        progress(status="complete", phase="complete", finished_at=now(), summary=str(summary_path))
        print(f"Complete: 999 test clips and all three evaluated controls. Results: {summary_path}", flush=True)
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                 finished_at=now(), error=str(error))
        raise


if __name__ == "__main__":
    main()
