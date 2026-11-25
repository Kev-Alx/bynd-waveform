"""Validation-driven refinement of paper-unspecified NLL/update choices.

Discord is disabled. Reuse audited controls, run fourteen pilot variants, then
validate three candidates. Only a candidate meeting both WER and quality goals
on validation and a confirmation run may advance to the paired 999-clip test.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.models.whisper import nll_transcript
from parallel_jobs import Supervisor, now
import run_quality_research as protocol

DEFAULT = ROOT / "reproduction-runs/paper-quality-refinement"
WER_FLOOR = 75.43 - 2.0
METRICS = protocol.METRICS
TARGETS = protocol.QUALITY_TARGET


def candidate_grid(previous, old_root):
    lookup = {c["name"]: c for c in previous["candidates"]}
    rows = []
    for name in ("mean-lr-0.05", "sum-lr-0.1", "sum-lr-0.02"):
        rows.append({"name": name, "config": copy.deepcopy(lookup[name]["config"]),
                     "cached": {"pilot": str(old_root / "pilot" / name)}})
    rows[1]["cached"]["validation"] = str(old_root / "validation/sum-lr-0.1")
    rows[2]["cached"]["validation"] = str(old_root / "validation/sum-lr-0.02")
    for policy in ("lower", "sentence"):
        for reduction in ("mean", "sum"):
            for lr in (0.05, 0.1):
                config = copy.deepcopy(lookup[f"{reduction}-lr-{lr:g}"]["config"])
                config["surrogate"]["nll_transcript_policy"] = policy
                rows.append({"name": f"{policy}-{reduction}-lr-{lr:g}", "config": config})
    for lr in (1.0, 5.0, 20.0):
        config = copy.deepcopy(lookup["sum-lr-0.1"]["config"])
        config["attack"].update(optimizer="sgd", learning_rate=lr)
        rows.append({"name": f"sgd-sum-lr-{lr:g}", "config": config})
    for alpha in (0.1, 10.0):
        config = copy.deepcopy(lookup["sum-lr-0.1"]["config"])
        config["attack"]["alpha_hf"] = alpha
        rows.append({"name": f"hf-{alpha:g}-sum-lr-0.1", "config": config})
    config = copy.deepcopy(lookup["sum-lr-0.1"]["config"])
    config["attack"].update(initialization="normal", init_fraction=0.1)
    rows.append({"name": "normal-start-sum-lr-0.1", "config": config})
    return rows


def broad_gate(row):
    return math.isfinite(row["error_percent"]) and all(
        math.isfinite(row["quality"][m]) and row["quality"][m] > 2.5 for m in METRICS)


def qualifies(row):
    return broad_gate(row) and row["error_percent"] >= WER_FLOOR and all(
        row["quality"][m] >= TARGETS[m] for m in METRICS)


def select_validation(rows):
    first_two = protocol.promotions(rows)["names"]
    remaining = [r for r in rows if broad_gate(r) and r["name"] not in first_two]
    if remaining:
        balanced = max(remaining, key=lambda r: (
            min(r["error_percent"] / WER_FLOOR, *(r["quality"][m] / TARGETS[m] for m in METRICS)),
            r["error_percent"], r["name"]))
        first_two.append(balanced["name"])
    return first_two


def decision_evidence(rows):
    """Use immutable score identity; a repeated audit has a new timestamp."""
    return [{key: row[key] for key in ("name", "pool", "run", "summary_sha256", "error_percent", "quality")}
            for row in rows]


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Refinement plan already exists")
    old_root = ROOT / "reproduction-runs/paper-quality-research"
    old = protocol.read(old_root / "plan.json")
    history = protocol.read(output / "history/verification.json")
    protocol.require(digest(old_root / "plan.json") == history["previous_plan_sha256"], "Previous plan changed")
    for role, pool in old["pools"].items():
        protocol.require(digest(pool["manifest"]) == pool["sha256"], f"Changed {role} manifest")
        protocol.require([r["id"] for r in read_jsonl(pool["manifest"])] == pool["ids"], "Changed IDs")
    entries = candidate_grid(old, old_root)
    files = [Path(__file__).resolve(), Path(protocol.__file__).resolve(), old_root / "plan.json",
             old_root / "results.json", output / "history/verification.json",
             *[Path(p["manifest"]) for p in old["pools"].values()],
             *[p for p in (output / "history").rglob("*") if p.is_file()]]
    for entry in entries:
        path = output / "configs" / (entry["name"] + ".json")
        atomic_json(path, entry["config"])
        path.chmod(0o444)
        entry["config_path"] = str(path)
        files.append(path)
        confirmation = {"name": entry["name"] + "-confirmation", "config": copy.deepcopy(entry["config"])}
        confirmation["config"]["seed"] = 2027
        cpath = output / "configs" / (confirmation["name"] + ".json")
        atomic_json(cpath, confirmation["config"])
        cpath.chmod(0o444)
        confirmation["config_path"] = str(cpath)
        entry["confirmation"] = confirmation
        files.append(cpath)
        for cached in entry.get("cached", {}).values():
            folder = Path(cached)
            audit = protocol.read(folder / "verification.json")
            protocol.require(digest(folder / "evaluation/summary.json") == audit["summary_sha256"], "Changed cached audit")
            files.extend(folder / n for n in ("run.json", "generated.jsonl", "verification.json",
                                               "evaluation/summary.json", "evaluation/utterances.jsonl"))
    files.extend((ROOT / "beyondwaveform").rglob("*.py"))
    files.extend(ROOT / "scripts" / n for n in ("generate.py", "run_logged.py", "parallel_jobs.py"))
    hashes = {str(p.resolve()): digest(p) for p in files}
    for path, expected in old["frozen_files"].items():
        if path.startswith(str(ROOT / "artifacts")):
            protocol.require(digest(path) == expected, f"Changed artifact: {path}")
            hashes[path] = expected
    plan = {"created_at": now(), "pools": old["pools"], "quality": old["quality"], "runtime": protocol.runtime(),
            "feature_artifacts": old["feature_artifacts"], "base_surrogate_artifacts": old["base_surrogate_artifacts"],
            "candidates": entries, "frozen_files": hashes, "wer_floor": WER_FLOOR, "quality_targets": TARGETS,
            "previous_results": str(old_root / "results.json"), "discord_enabled": False,
            "selection": "Promote quality-first, attack-first, and balanced candidates from pilot only. Confirm validation qualifiers with seed 2027 before one test evaluation.",
            "test_rule": "At least 73.43% validation WER and all Table 4 feature-quality anchors on both validation runs. Highest-WER stable qualifier advances once; no test feedback selects it.",
            "uncertainty": "2 percentage points WER tolerance is a research target, not an equivalence margin established statistically. Repeated old test IDs provide a paired benchmark, not a fresh blind holdout."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.check(plan)
    protocol.validate_candidate_files([c["confirmation"] for c in plan["candidates"]])
    for entry in plan["candidates"]:
        nll_transcript("validate", entry["config"]["surrogate"].get("nll_transcript_policy", "original"))


def report(output, rows, status, outcome=None):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows,
                "outcome": outcome, "quality_targets": TARGETS, "wer_floor": WER_FLOOR})
    lines = ["# WER and sound-quality refinement", "", f"Status: **{status}**.", "",
             "Paper-budget radius 0.1, lambda 1, 50 updates. Pilot/validation evidence selects experiments; the paired test is conditional on stable validation quality and WER.", "",
             "| Pool | Candidate | WER % | DNSMOS | NISQA | UTMOS | Meets joint target | Cached |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        q = row["quality"]
        lines.append(f"| {row['pool']} | {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {qualifies(row)} | {row['cached']} |")
    lines += ["", "Joint target: WER >=73.43%, DNSMOS >=2.954, NISQA >=3.526, UTMOS >=3.590. The WER floor is two percentage points below the published 75.43%, defined before this batch's results. These descriptive targets do not establish author parity.", "",
              "Transcript policy affects lexical teacher forcing only. Recognition decoding and scoring references remain fixed. Cached controls are earlier audited measurements, not fresh replications. All saved deltas are independently checked.", ""]
    if outcome:
        lines += [f"Outcome: {outcome}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def phase(supervisor, name, command):
    folder = supervisor.output / "phases" / name
    # Retain stage logs independently of the now-disabled Discord hooks.
    wrapped = [sys.executable, "-u", "scripts/run_logged.py", "--log", str(folder / "stage.log"),
               "--status", str(folder / "job.json"), "--", *command]
    supervisor.phase(name, wrapped, threads=4)


def run_case(output, plan, entry, role, rows, supervisor, pool=None):
    pool = pool or role
    cached = entry.get("cached", {}).get(role) if pool == role else None
    folder = Path(cached) if cached else output / pool / entry["name"]
    check(plan)
    if not (folder / "evaluation/summary.json").exists():
        phase(supervisor, f"{pool}-{entry['name']}-generate", [sys.executable, "-u", "scripts/generate.py",
            "--config", entry["config_path"], "--manifest", plan["pools"][role]["manifest"], "--output", str(folder)])
        phase(supervisor, f"{pool}-{entry['name']}-evaluate", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--output", str(output), "--stage", "evaluate", "--run", str(folder)])
    audit_name = "refinement-verification.json" if cached else "verification.json"
    phase(supervisor, f"{pool}-{entry['name']}-verify", [sys.executable, "-u", str(Path(__file__).resolve()),
        "--output", str(output), "--stage", "verify", "--run", str(folder), "--config", entry["config_path"],
        "--role", role, "--audit-name", audit_name])
    audit = protocol.read(folder / audit_name)
    row = {"name": entry["name"], "pool": pool, "role": role, "cached": bool(cached), "run": str(folder),
           **audit["summary"]["adversarial"], "audit_sha256": digest(folder / audit_name),
           "summary_sha256": audit["summary_sha256"]}
    rows.append(row)
    report(output, rows, "running")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stage", choices=("evaluate", "verify"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--role", choices=("pilot", "validation", "test"))
    parser.add_argument("--audit-name", default="verification.json")
    args = parser.parse_args()
    os.environ["BEYONDWAVEFORM_NOTIFY_DISCORD"] = "0"
    output = args.output.resolve()
    plan = prepare(output) if args.prepare else protocol.read(output / "plan.json")
    check(plan)
    if args.prepare or args.check_only:
        print(f"Refinement preflight passed: {len(plan['candidates'])} pilot entries; notifications disabled.", flush=True)
        return
    if args.stage:
        protocol.require(args.run is not None, "Stage needs --run")
        if args.stage == "evaluate":
            protocol.evaluate_stage(plan, args.run)
        else:
            protocol.require(args.config is not None and args.role is not None, "Verification needs config and role")
            protocol.verify_stage(plan, args.run, protocol.load_config(args.config), args.role, audit_name=args.audit_name)
        return
    supervisor, rows = Supervisor(output), []
    try:
        report(output, rows, "running")
        pilots = [run_case(output, plan, entry, "pilot", rows, supervisor) for entry in plan["candidates"]]
        names = select_validation(pilots)
        selection = {"names": names, "pilot_evidence": decision_evidence(pilots)}
        decision = output / "validation-promotion.json"
        if decision.exists():
            protocol.require(protocol.read(decision) == selection, "Pilot selection changed on resume")
        else:
            atomic_json(decision, selection)
            decision.chmod(0o444)
        validated = []
        for name in names:
            entry = next(c for c in plan["candidates"] if c["name"] == name)
            validated.append((entry, run_case(output, plan, entry, "validation", rows, supervisor)))
        candidates = sorted([(e, r) for e, r in validated if qualifies(r)],
                            key=lambda pair: (pair[1]["error_percent"], pair[0]["name"]), reverse=True)
        winner = None
        for entry, row in candidates:
            confirmation = run_case(output, plan, entry["confirmation"], "validation", rows, supervisor, pool="confirmation")
            if qualifies(confirmation):
                winner = entry
                break
        if winner:
            frozen = {"name": winner["name"], "config_path": winner["config_path"],
                      "config_sha256": digest(winner["config_path"]),
                      "validation_evidence": decision_evidence([r for r in rows if r["pool"] in {"validation", "confirmation"}])}
            path = output / "selected-test-freeze.json"
            if path.exists():
                protocol.require(protocol.read(path) == frozen, "Test selection changed on resume")
            else:
                atomic_json(path, frozen)
                path.chmod(0o444)
            result = run_case(output, plan, winner, "test", rows, supervisor)
            outcome = "Selected test meets the joint target." if qualifies(result) else "Selected test misses the joint target; no test-based retuning or replacement."
        else:
            outcome = "No candidate met the joint target on both validation runs; no new 999-clip test was launched."
        check(plan)
        report(output, rows, "complete", outcome)
        supervisor.finish(outcome=outcome)
    except BaseException as error:
        report(output, rows, "failed", str(error))
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
