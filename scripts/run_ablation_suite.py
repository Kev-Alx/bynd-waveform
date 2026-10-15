"""Run four predefined loss removals on the fixed 999 English test clips."""
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
from ablation_protocol import control_sample_digest, load_ablation_freeze
from beyondwaveform.experiments import experiment_configs
from beyondwaveform.io import atomic_json, digest, read_jsonl
from verify_test_run import require, verify


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    base = json.loads(Path(plan["base_freeze"]).read_text())
    require(plan["status"] == "frozen" and plan["suite"] == "predefined_loss_removals" and plan["count"] == 999,
            "Wrong or mutable ablation plan")
    require(digest(plan["base_freeze"]) == plan["base_freeze_sha256"], "Base freeze changed")
    presets = dict(experiment_configs(base["config"], "ablations"))
    require([e["name"] for e in plan["experiments"]] == list(presets), "Changed ablation names/order")
    require(plan["evaluation"] == base["evaluation"], "Changed ablation evaluation protocol")
    require(plan["experiments"][0]["run"] == plan["source_run"] and plan["experiments"][0]["reuse"] is True,
            "Full loss must reuse the fixed source run")
    for path, expected in plan["source_proofs"].items():
        require(digest(path) == expected, f"Source evidence changed: {path}")
    source = json.loads((Path(plan["source_run"]) / "run.json").read_text())
    require(source["status"] == "complete" and source["config"] == base["config"] and source["count"] == 999,
            "Full-loss source differs from selected config")
    for entry in plan["experiments"][1:]:
        require(entry["reuse"] is False and Path(entry["run"]).resolve() == Path(plan["output"]) / entry["name"],
                "Unexpected ablation output path")
        require(digest(entry["freeze"]) == entry["freeze_sha256"], "Ablation receipt changed")
        protocol, receipt = load_ablation_freeze(plan["base_freeze"], entry["freeze"])
        require(receipt["name"] == entry["name"] and protocol["config"] == presets[entry["name"]], "Wrong loss removal")
    return plan


def write_results(output, plan, measured, complete=False):
    atomic_json(output / "results.json", {"status": "complete" if complete else "running",
        "plan_sha256": digest(output / "plan.json"), "base_freeze_sha256": plan["base_freeze_sha256"],
        "count": 999, "experiments": measured})
    lines = ["# Fixed-setting English loss-removal measurements", "",
        "Only the predefined loss component changes. The selected rho=0.5, lambda=0.1 (except its removal), Adam LR=0.05, 50 updates, artifacts, decoder, seeds and 999-clip test manifest remain fixed. Full loss reuses the verified core run. All three controls and quality metrics are evaluated for each removal; these results do not select a new configuration.", "",
        "| Loss configuration | Clean WER % | Reconstruction WER % | Perturbed WER % | DNSMOS | NISQA | UTMOS |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in measured:
        s = row["summary"]
        values = [s[v]["error_percent"] for v in ["clean", "reconstruction", "adversarial"]]
        q = s["adversarial"]["quality"]
        lines.append("| " + " | ".join([row["name"], *[f"{v:.4f}" for v in values],
                     *[f"{q[m]:.4f}" for m in ["dnsmos", "nisqa", "utmos"]]]) + " |")
    pending = [e["name"] for e in plan["experiments"] if e["name"] not in {m["name"] for m in measured}]
    lines += ["", f"Status: {'complete' if complete else 'running'}. Pending: {', '.join(pending) or 'none'}.", "",
        "Without attack loss means 50 perceptual-only updates, not zero-delta reconstruction. Logged NLL is a zero placeholder for this removal, because NLL is not evaluated. Disabled TV/HF terms likewise log zero placeholders. The auditor checks removed terms and reconstructs the retained scalar objective.", "",
        "These are independent ablations of the validation-selected config. Paper defaults rho=0.1/lambda=1 differ; author Figure 8 configuration, exact arrays and scorer recipes are unavailable, and its full-loss DNSMOS conflicts with Table 4. Hold-fixed learning rate can change the effective optimization balance when a loss is removed. No results are chosen or tuned on test measurements.", "",
        "Evidence: frozen plan and per-removal config/receipt; each completed run's generated WAVs, summary/scored rows, and verification. Exact decoded float32 sample hashes of clean/reconstruction must match the original full-loss controls. WAV file hashes may differ because libsndfile writes an export timestamp in the PEAK chunk; each run's own file hashes remain fully checked."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = preflight(plan_path)
    plan_sha = digest(plan_path)
    print("Ablation preflight passed: four exact loss removals, same 999 clips and settings; full-loss result reused.", flush=True)
    if args.check_only:
        return
    output = Path(plan["output"])
    state = {"status": "running", "phase": "preflight", "started_at": now(), "plan": str(plan_path),
             "plan_sha256": plan_sha, "expected_count": 999, "completed_experiments": []}

    def progress(**updates):
        state.update(updates, updated_at=now())
        atomic_json(output / "progress.json", state)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    progress()
    measured = []
    source_records = {r["id"]: r for r in read_jsonl(Path(plan["source_run"]) / "generated.jsonl")}
    source_samples = {uid: {v: control_sample_digest(r["audio"][v]["path"])
                            for v in ["clean", "reconstruction"]} for uid, r in source_records.items()}
    try:
        for entry in plan["experiments"]:
            name, run = entry["name"], Path(entry["run"])
            evaluation = run / "surrogate-eval"
            progress(experiment=name, phase="verification" if entry["reuse"] else "pipeline", generated_records=0)
            require(digest(plan_path) == plan_sha, "Ablation plan changed")
            if not entry["reuse"] and not (evaluation / "summary.json").exists():
                command = [sys.executable, "-u", str(ROOT / "scripts/run_selected_test.py"),
                    "--freeze", plan["base_freeze"], "--ablation-freeze", entry["freeze"],
                    "--output", str(run), "--device", args.device]
                print(f"\nStarting {name}: {shlex.join(command)}", flush=True)
                child = subprocess.Popen(command, cwd=ROOT)
                progress(phase_pid=child.pid, command=command, phase_started_at=now())
                started = time.monotonic()
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            inner_path = run / "progress.json"
                            inner = json.loads(inner_path.read_text()) if inner_path.exists() else {}
                            progress(phase=inner.get("phase", "pipeline"),
                                     generated_records=len(list((run / "records").glob("*.json"))),
                                     phase_elapsed_seconds=round(time.monotonic() - started, 1))
                            print(f"Heartbeat: {name} / {state['phase']}, {state['generated_records']}/999 generated, "
                                  f"{len(measured)-1}/4 removals complete, elapsed {state['phase_elapsed_seconds']}s", flush=True)
                    require(code == 0, f"{name} failed with exit code {code}; stopping")
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
            progress(phase="verification")
            receipt = None if entry["reuse"] else entry["freeze"]
            audit = verify(run, evaluation, plan["base_freeze"], hash_audio=not entry["reuse"],
                           ablation_freeze_path=receipt)
            file_matches = {"clean": 0, "reconstruction": 0}
            for record in read_jsonl(run / "generated.jsonl"):
                original = source_records[record["id"]]
                for variant in ["clean", "reconstruction"]:
                    require(control_sample_digest(record["audio"][variant]["path"]) == source_samples[record["id"]][variant],
                            f"Control changed across losses: {record['id']}/{variant}")
                    file_matches[variant] += record["audio"][variant]["sha256"] == original["audio"][variant]["sha256"]
            audit["full_loss_control_sample_hashes_matched"] = True
            audit["full_loss_control_file_hash_match_counts"] = file_matches
            audit["control_hash_note"] = ("Exact decoded mono float32 samples at 16 kHz, SHA256. Independently exported "
                "WAV file hashes can differ because PEAK contains an export timestamp; own-run file hashes remain verified.")
            audit_path = output / "full-verification.json" if entry["reuse"] else run / "verification.json"
            atomic_json(audit_path, audit)
            result = json.loads((evaluation / "summary.json").read_text())
            measured.append({"name": name, "generation_run": str(run), "reused": entry["reuse"],
                "generation_fingerprint": result["generation_fingerprint"], "summary": result["summary"],
                "summary_sha256": digest(evaluation / "summary.json"),
                "verification": str(audit_path), "verification_sha256": digest(audit_path)})
            state["completed_experiments"].append(name)
            write_results(output, plan, measured)
            progress()
            print(f"Completed and verified {name}: {result['summary']['adversarial']}", flush=True)
        write_results(output, plan, measured, complete=True)
        progress(status="complete", phase="complete", finished_at=now(), results=str(output / "results.json"))
        print(f"Ablation suite complete: {output / 'results.json'}", flush=True)
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", finished_at=now(), error=str(error))
        raise


if __name__ == "__main__":
    main()
