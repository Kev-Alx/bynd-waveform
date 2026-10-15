"""Execute the existing waveform grids and select each method on validation only."""
import argparse
from collections import Counter
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
import yaml
from tqdm import tqdm
from beyondwaveform.config import load_config
from beyondwaveform.data import audit_disjoint
from beyondwaveform.experiments import experiment_configs
from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.selection import select_configuration
from verify_waveform_run import require, verify_waveform_run


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    require(plan["status"] == "frozen" and plan["suite"] == "waveform-grid" and plan["count"] == 100, "Wrong waveform plan")
    for path, sha in tqdm(plan["proofs"].items(), desc="Checking frozen waveform inputs", unit="files"):
        require(digest(path) == sha, f"Frozen input changed: {path}")
    base = json.loads(Path(plan["base_freeze"]).read_text())
    require(digest(plan["base_freeze"]) == plan["base_freeze_sha256"] and
            digest(plan["validation_manifest"]) == plan["validation_manifest_sha256"] == base["validation_manifest_sha256"] and
            digest(plan["test_manifest"]) == plan["test_manifest_sha256"] == base["test_manifest_sha256"],
            "Changed parent freeze/validation/test identities")
    expected = experiment_configs(base["config"], "waveform-grid")
    require(len(plan["experiments"]) == len(expected) == 135, "Wrong waveform grid size")
    for entry, (name, config) in zip(plan["experiments"], expected):
        require(entry["name"] == name and entry["method"] == config["method"] and
                digest(entry["config"]) == entry["config_sha256"] and load_config(entry["config"]) == config,
                "Changed waveform setting/order/config")
    require(Counter(e["method"] for e in plan["experiments"]) == {"pgd": 15, "mi-fgsm": 30, "vmi-fgsm": 90},
            "Missing waveform method/settings")
    require(plan["selection"] == {"threshold": 2.5, "strict": True, "separate_by_method": True} and
            plan["evaluation_variants"] == ["adversarial"], "Changed evaluation/selection protocol")
    require(plan["policy"] == {"objective": "maximize lexical-mean teacher-forced Whisper-small NLL", "norm": "L-infinity",
            "random_start": False, "waveform_range": [-1, 1], "updates": 50, "validation_only": True, "no_test_scoring": True},
            "Changed documented independent waveform policy")
    rows = read_jsonl(plan["validation_manifest"])
    require(len(rows) == 100 and [r["id"] for r in rows] == plan["validation_ids"] and
            all(r["role"] == "validation" and r["split"] == "dev-clean" and r["language"] == "en" for r in rows),
            "Invalid validation pool")
    require(plan["surrogate_artifacts"] == base["surrogate_artifacts"] and plan["quality_provenance"] == base["quality_provenance"],
            "Changed model/quality artifact protocol")
    audit_disjoint([plan["pilot_manifest"], plan["validation_manifest"], plan["test_manifest"]])
    return plan


def immutable_json(path, value):
    if path.exists():
        require(json.loads(path.read_text()) == value, "Previously frozen baseline receipt changed")
    else:
        atomic_json(path, value)
        path.chmod(0o444)


def select_method(output, plan, method, rows):
    folder = output / "selected"
    folder.mkdir(exist_ok=True)
    selection_path = folder / f"{method}.selection.json"
    try:
        selected = select_configuration([r["summary_path"] for r in rows], selection_path)["selected"]
    except ValueError as error:
        if (str(error) != "No configuration passed all three quality gates; audit saved" or
                not selection_path.exists() or json.loads(selection_path.read_text())["selected"] is not None):
            raise
        print(f"No {method} setting passed all three strict MOS gates; no fallback selected.", flush=True)
        return {"status": "no_eligible_configuration", "selection": str(selection_path), "selected": None}
    run = Path(selected["generation_run"])
    config = json.loads((run / "run.json").read_text())["config"]
    path = folder / f"{method}.yaml"
    if path.exists():
        require(load_config(path) == config, "Previously frozen selected baseline changed")
    else:
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        path.chmod(0o444)
    receipt = {"status": "frozen", "scope": "validation_only_waveform_selection", "method": method,
        "plan_sha256": digest(output / "plan.json"), "config_path": str(path), "config_sha256": digest(path), "config": config,
        "selection_path": str(selection_path), "selection_sha256": digest(selection_path), "selected": selected,
        "selected_summary_sha256": digest(selected["summary"]), "generation_metadata_sha256": digest(run / "run.json"),
        "validation_manifest": plan["validation_manifest"], "validation_manifest_sha256": plan["validation_manifest_sha256"],
        "validation_ids": plan["validation_ids"], "test_manifest": plan["test_manifest"],
        "test_manifest_sha256": plan["test_manifest_sha256"], "surrogate_artifacts": plan["surrogate_artifacts"],
        "quality_provenance": plan["quality_provenance"], "base_freeze": plan["base_freeze"],
        "base_freeze_sha256": digest(plan["base_freeze"]),
        "fidelity": "Independent L-infinity lexical-NLL adaptation with existing grid/update rules. Author loss/norm/start/implementation recipes unspecified. No test/target feedback used for selection."}
    frozen_path = folder / f"{method}.freeze.json"
    immutable_json(frozen_path, receipt)
    print(f"Selected {method} on validation only: {selected}", flush=True)
    return {"status": "selected", "selection": str(selection_path), "freeze": str(frozen_path),
            "freeze_sha256": digest(frozen_path), "selected": selected}


def save_results(output, plan, measured, selections, complete=False):
    atomic_json(output / "results.json", {"status": "complete" if complete else "running", "plan_sha256": digest(output / "plan.json"),
        "expected_settings": 135, "completed_settings": len(measured), "validation_count": 100,
        "candidates": measured, "selections": selections})
    lines = ["# Independent waveform baseline validation", "",
        f"Status: {'complete' if complete else 'running'}; {len(measured)}/135 settings audited. All candidates use the same 100 validation clips and raw Whisper-small, with 50 updates and all three quality metrics. Separate existing strict >2.5 gates select PGD, MI-FGSM and VMI-FGSM by validation corpus WER. No test scoring or target feedback occurs.", "",
        "| Setting | Method | Validation WER % | DNSMOS | NISQA | UTMOS | Eligible |",
        "|---|---|---:|---:|---:|---:|---|"]
    for r in measured:
        s = r["summary"]
        q = s["quality"]
        lines.append("| " + " | ".join([r["name"], r["method"], f"{s['error_percent']:.4f}",
            *[f"{q[m]:.6f}" for m in ["dnsmos", "nisqa", "utmos"]], str(all(v > 2.5 for v in q.values()))]) + " |")
    lines += ["", "Selection status:"]
    for method in ["pgd", "mi-fgsm", "vmi-fgsm"]:
        lines.append(f"- {method}: {selections[method]['status'] if method in selections else 'pending'}.")
    lines += ["", "These preserve the documented independent waveform implementation: maximize lexical-mean teacher-forced NLL; L-infinity projection around original audio, [-1,1] clipping, zero start, sign updates, existing momentum/variance-neighbor rules. Feature/vocoder models are not used. Paper norms, objectives and full adaptation recipes are absent. VMI uses extra gradient evaluations per update, so 50 updates do not imply equal compute. Validation evaluates saved perturbed audio; each run also saves its clean control. A selected baseline's final test evaluation will include clean controls.", "",
        "Evidence: immutable plan/configs, per-setting generation/audio/evaluation/verification, and each method's selection audit plus read-only config/freeze. Missing eligible settings are reported without replacing the gate. Muting Whisper, Sloth, defenses, human studies and OTA are not reconstructed here."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = preflight(plan_path)
    print("Waveform preflight passed: 135 fixed settings, 100 validation clips, independent strict selection for each method.", flush=True)
    if args.check_only:
        return
    output, plan_sha = Path(plan["output"]), digest(plan_path)
    state = {"status": "running", "phase": "preflight", "started_at": now(), "plan": str(plan_path),
             "plan_sha256": plan_sha, "expected_settings": 135, "expected_count": 100, "completed_settings": 0}
    def progress(**updates):
        state.update(updates, updated_at=now())
        if state.get("setting"):
            state["generated_records"] = len(list((output / state["setting"] / "records").glob("*.json")))
        atomic_json(output / "progress.json", state)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")
    for sig in [signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, interrupt)
    def phase(name, command):
        progress(phase=name, phase_started_at=now(), command=command)
        print(f"\nStarting {state['setting_index']}/135 {state['method']} / {name}: {shlex.join(command)}", flush=True)
        child = subprocess.Popen(command, cwd=ROOT)
        progress(phase_pid=child.pid)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress(phase_elapsed_seconds=round(time.monotonic()-started, 1))
                    print(f"Heartbeat: {state['completed_settings']}/135 settings, {state['setting']} / {name}, "
                          f"{state['generated_records']}/100 generated, phase elapsed {state['phase_elapsed_seconds']}s", flush=True)
            require(code == 0, f"{state['setting']} / {name} failed with exit code {code}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    measured, selections = [], {}
    progress()
    save_results(output, plan, measured, selections)
    try:
        for index, entry in enumerate(plan["experiments"], start=1):
            run = output / entry["name"]
            progress(setting=entry["name"], method=entry["method"], setting_index=index)
            require(digest(plan_path) == plan_sha and digest(entry["config"]) == entry["config_sha256"], "Frozen plan/config changed")
            if not (run / "evaluation/summary.json").exists():
                phase("generation", [sys.executable, "-u", str(ROOT / "scripts/generate.py"), "--config", entry["config"],
                    "--manifest", plan["validation_manifest"], "--output", str(run), "--device", args.device])
                phase("evaluation", [sys.executable, "-u", str(ROOT / "scripts/evaluate.py"), "--run", str(run),
                    "--target", "surrogate", "--output", str(run / "evaluation"), "--device", args.device,
                    "--variants", "adversarial", "--quality", "--artifact-root", plan["artifact_root"]])
            progress(phase="verification")
            audit = verify_waveform_run(run, entry, plan)
            atomic_json(run / "verification.json", audit)
            measured.append({"name": entry["name"], "method": entry["method"], "summary": audit["summary"],
                "summary_path": str(run / "evaluation/summary.json"), "summary_sha256": audit["summary_sha256"],
                "verification_sha256": digest(run / "verification.json"), "generation_fingerprint": audit["generation_fingerprint"]})
            progress(completed_settings=len(measured))
            if index == 135 or plan["experiments"][index]["method"] != entry["method"]:
                progress(phase="selection")
                selections[entry["method"]] = select_method(output, plan, entry["method"],
                    [r for r in measured if r["method"] == entry["method"]])
            save_results(output, plan, measured, selections)
            print(f"Completed and audited {index}/135: {entry['name']} {audit['summary']}", flush=True)
        save_results(output, plan, measured, selections, complete=True)
        progress(status="complete", phase="complete", finished_at=now(), results=str(output / "results.json"))
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", finished_at=now(), error=str(error))
        raise


if __name__ == "__main__":
    main()
