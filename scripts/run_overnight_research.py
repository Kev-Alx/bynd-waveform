"""Three sequential paper-budget studies, then validation and confirmation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.data import prepare_manifest, audit_disjoint
from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl
from beyondwaveform.config import load_config
from parallel_jobs import Supervisor, now
import run_quality_research as protocol
import run_quality_refinement as selection
from run_quality_contracts import frozen_decision, reconstruction_gate

DEFAULT = ROOT / "reproduction-runs/paper-quality-overnight"
SEED = 20261018
GROUPS = ("conditioning", "schedules", "l2-updates")


def fresh_pilot(index, excluded, count=24, seed=SEED):
    ids = {r["id"] for r in excluded}
    hashes = {r["audio_sha256"] for r in excluded}
    eligible = sorted((r for r in index if r["id"] not in ids and r["audio_sha256"] not in hashes), key=lambda r: r["id"])
    sampled = random.Random(seed).sample(eligible, count)
    return sorted(({**r, "role": "pilot"} for r in sampled), key=lambda r: r["id"])


def candidate_grid(original):
    lookup = {c["name"]: c for c in original["candidates"]}
    entries = []
    for reduction in ("sum", "mean"):
        for lr in (0.02, 0.05, 0.1):
            config = copy.deepcopy(lookup[f"{reduction}-lr-{lr:g}"]["config"])
            config["surrogate"]["nll_timestamp_policy"] = "initial_timestamp"
            entries.append({"name": f"timestamp-{reduction}-lr-{lr:g}", "group": "conditioning", "config": config})
    for lr in (0.02, 0.05):
        config = copy.deepcopy(lookup[f"mean-lr-{lr:g}"]["config"])
        config["surrogate"].update(nll_timestamp_policy="initial_timestamp", nll_transcript_policy="lower")
        entries.append({"name": f"timestamp-lower-mean-lr-{lr:g}", "group": "conditioning", "config": config})
    for schedule in ("cosine", "two_stage"):
        for lr in (0.05, 0.1):
            config = copy.deepcopy(lookup[f"sum-lr-{lr:g}"]["config"])
            config["attack"].update(lr_schedule=schedule, lr_final_ratio=0.1)
            entries.append({"name": f"{schedule}-sum-lr-{lr:g}", "group": "schedules", "config": config})
    for reduction in ("sum", "mean"):
        for fraction in (0.005, 0.01, 0.02):
            config = copy.deepcopy(lookup[f"{reduction}-lr-0.05"]["config"])
            config["attack"].update(optimizer="l2_pgd", learning_rate=fraction)
            entries.append({"name": f"l2-{reduction}-step-{fraction:g}", "group": "l2-updates", "config": config})
    for name in ("sum-lr-0.02", "sum-lr-0.1"):
        entries.append({"name": "control-" + name, "group": "controls", "config": copy.deepcopy(lookup[name]["config"]),
                        "cached": {"validation": str(ROOT / "reproduction-runs/paper-quality-research/validation" / name)}})
    return entries


def effective_plan(plan, config):
    expected = dict(plan["base_surrogate_artifacts"])
    if "nll_timestamp_policy" in config["surrogate"]:
        expected["nll_timestamp_policy"] = config["surrogate"]["nll_timestamp_policy"]
    return {**plan, "base_surrogate_artifacts": expected}


def historical_hashes(previous, receipt):
    hashes = {}
    for path, expected in previous["frozen_files"].items():
        archived = receipt["historical_sources"].get(path)
        actual = archived["snapshot"] if archived else path
        protocol.require(digest(actual) == expected, f"Changed historical evidence: {path}")
        if archived:
            protocol.require(archived["sha256"] == expected, "Historical source mapping mismatch")
        hashes[actual] = expected
    return hashes


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Overnight plan already exists")
    previous = protocol.read(output / "history/plan.json")
    receipt = protocol.read(output / "history/review.json")
    protocol.require(digest(output / "history/plan.json") == receipt["previous_plan_sha256"], "Changed reviewed plan")
    hashes = historical_hashes(previous, receipt)
    index_path = output / "data/dev-clean-index.jsonl"
    if not index_path.exists():
        prepare_manifest("librispeech", ROOT / "data", "dev-clean", "pilot", index_path)
    protocol.require(digest(index_path) == protocol.read(str(index_path) + ".metadata.json")["manifest_sha256"], "Changed corpus index")
    exclusions = [ROOT / "data/manifests" / n for n in ("reconstruction-pilot.jsonl", "pilot.jsonl", "english-validation.jsonl", "english-test-999.jsonl")]
    excluded = [r for path in exclusions for r in read_jsonl(path)]
    sampled = fresh_pilot(read_jsonl(index_path), excluded)
    pilot = output / "data/pilot-24.jsonl"
    if pilot.exists():
        protocol.require(read_jsonl(pilot) == sampled, "Changed fresh pilot selection")
    else:
        write_jsonl(pilot, sampled)
    pilot.chmod(0o444)
    audit_disjoint([pilot, *exclusions[:1], *exclusions[2:]])
    original = protocol.read(ROOT / "reproduction-runs/paper-quality-research/plan.json")
    entries = candidate_grid(original)
    pools = {**previous["pools"], "pilot": {"manifest": str(pilot), "sha256": digest(pilot), "ids": [r["id"] for r in sampled]}}
    for pool in pools.values():
        for row in read_jsonl(pool["manifest"]):
            protocol.require(digest(row["audio"]) == row["audio_sha256"], "Changed source audio")
    files = [Path(__file__).resolve(), index_path, Path(str(index_path) + ".metadata.json"), pilot, *exclusions]
    files.extend(p for p in (output / "history").rglob("*") if p.is_file())
    files.extend(Path(p) for p in receipt["historical_sources"])
    for entry in entries:
        confirmation = {"name": entry["name"] + "-confirmation", "group": entry["group"], "config": copy.deepcopy(entry["config"])}
        confirmation["config"]["seed"] = 2027
        entry["confirmation"] = confirmation
        for item in (entry, confirmation):
            path = output / "configs" / (item["name"] + ".json")
            atomic_json(path, item["config"])
            path.chmod(0o444)
            item["config_path"] = str(path)
            files.append(path)
    import whisper
    files.extend(Path(whisper.__file__).parent / n for n in ("decoding.py", "transcribe.py", "tokenizer.py"))
    hashes.update({str(p.resolve()): digest(p) for p in files})
    plan = {"created_at": now(), "pools": pools, "quality": previous["quality"], "runtime": protocol.runtime(),
            "feature_artifacts": previous["feature_artifacts"], "base_surrogate_artifacts": previous["base_surrogate_artifacts"],
            "candidates": entries, "frozen_files": hashes, "historical_sources": receipt["historical_sources"],
            "groups": GROUPS, "discord_enabled": False, "pilot_seed": SEED, "exclusion_manifests": [str(p) for p in exclusions],
            "selection": "Run all three independent groups sequentially. Select up to three new candidates plus the strongest joint-target control for validation, using the fresh pilot only. Reconstruction quality/intelligibility must pass too. Qualifiers repeat with seed 2027 before one paired 999-clip test.",
            "wer_floor": selection.WER_FLOOR, "quality_targets": protocol.QUALITY_TARGET,
            "paper_budget": {"rho": 0.1, "lambda_perc": 1, "steps": 50, "cutoff_hz": 6000},
            "scope": "Independent conditioning/update assumptions; original decoder, references, scorer channels/weights and paired 999 IDs fixed. A 0.00-second attack timestamp is not an oracle alignment or recovery of author settings."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.check(plan)
    protocol.validate_candidate_files([c["confirmation"] for c in plan["candidates"]])
    for entry in plan["candidates"]:
        config = entry["config"]
        attack = config["attack"]
        protocol.require(attack["rho"] == 0.1 and attack["lambda_perc"] == 1 and attack["steps"] == 50 and
                         config["spectral"]["cutoff_hz"] == 6000 and config["spectral"]["sample_rate"] == 16000,
                         "Changed paper budget/audio contract")
        protocol.require(config["surrogate"].get("nll_timestamp_policy", "no_timestamps") in {"no_timestamps", "initial_timestamp"}, "Unknown timestamp policy")


def qualifies(row):
    return selection.qualifies(row) and reconstruction_gate(row)


def promotions(rows):
    new = [r for r in rows if r["group"] != "controls" and reconstruction_gate(r) and selection.broad_gate(r)]
    controls = [r for r in rows if r["group"] == "controls" and reconstruction_gate(r) and selection.broad_gate(r)]
    names = selection.select_validation(new) if new else []
    if controls:
        names += selection.select_validation(controls)[:1]
    protocol.require(bool(names), "No pilot candidate passes the quality/reconstruction gates")
    return names


def report(output, plan, rows, failures, status, outcome=None):
    batches = {}
    for group in GROUPS:
        expected = sum(c["group"] == group for c in plan["candidates"])
        done = sum(r["group"] == group and r["pool"] == "pilot" for r in rows)
        failed = sum(r["group"] == group and r["pool"] == "pilot" for r in failures)
        batches[group] = {"completed": done, "failed": failed, "total": expected}
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows,
        "batches": batches, "failures": failures, "outcome": outcome})
    lines = ["# Overnight WER/quality queue", "", f"Status: **{status}**.", "",
        "Three sequential experiments share a fresh, randomly selected 24-clip development pilot. All attacks retain rho=0.1, lambda=1, 50 updates and a 6000 Hz cutoff. Decoder/scorers and the 100/999 pools stay fixed.", "",
        "| Experiment | Audited pilot cases | Failed cases | Planned cases |", "|---|---:|---:|---:|"]
    lines += [f"| {g} | {v['completed']} | {v['failed']} | {v['total']} |" for g, v in batches.items()]
    lines += ["", "| Pool | Candidate | WER % | DNSMOS | NISQA | UTMOS | Reconstruction WER % | Joint target | Cached |",
        "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        q = row["quality"]
        lines.append(f"| {row['pool']} | {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {row['reconstruction']['error_percent']:.4f} | {qualifies(row)} | {row['cached']} |")
    lines += ["", "Joint exploratory target: WER >=73.43%, DNSMOS/NISQA/UTMOS >=2.954/3.526/3.590, with reconstruction WER <=clean+5 percentage points and reconstruction quality also meeting those MOS anchors. These are research targets, not author parity or statistical equivalence.", "",
        "Timestamp conditioning adds a fixed, unscored 0.00-second token before lexical attack targets. Recognition decoding stays unchanged. Schedules decay Adam LR to 10% by cosine or at the midpoint. L2 PGD normalizes the complete objective gradient; its LR is a fraction of the utterance's clean feature norm per step. All 50 updates are projected into the same ball.", "",
        "Original hypotheses/references and scorer channels are fixed. The fresh pilot excludes all prior pilot/validation/test IDs and audio hashes. Old cached 100-clip controls are labeled; fresh pilot controls are new measurements. Validation qualifiers repeat before any one paired 999-clip test. Test feedback cannot select a replacement. Discord is disabled.", ""]
    if failures:
        lines += ["Failed independent cases (other cases still run):", ""] + [f"- {r['pool']}/{r['name']}: {r['error']}" for r in failures] + [""]
    if outcome:
        lines += [f"Outcome: {outcome}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_case(output, plan, entry, role, rows, failures, supervisor, pool=None):
    pool = pool or role
    cached = entry.get("cached", {}).get(role) if pool == role else None
    folder = Path(cached) if cached else output / pool / entry["name"]
    check(plan)
    supervisor.progress(batch=entry["group"], completed_cases=len(rows), failed_cases=len(failures))
    if not (folder / "evaluation/summary.json").exists():
        selection.phase(supervisor, f"{pool}-{entry['name']}-generate", [sys.executable, "-u", "scripts/generate.py",
            "--config", entry["config_path"], "--manifest", plan["pools"][role]["manifest"], "--output", str(folder)])
        selection.phase(supervisor, f"{pool}-{entry['name']}-evaluate", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--output", str(output), "--stage", "evaluate", "--run", str(folder)])
    audit_name = "overnight-verification.json" if cached else "verification.json"
    selection.phase(supervisor, f"{pool}-{entry['name']}-verify", [sys.executable, "-u", str(Path(__file__).resolve()),
        "--output", str(output), "--stage", "verify", "--run", str(folder), "--config", entry["config_path"],
        "--role", role, "--audit-name", audit_name])
    audit = protocol.read(folder / audit_name)
    row = {"name": entry["name"], "group": entry["group"], "pool": pool, "role": role, "run": str(folder),
        "cached": bool(cached), **audit["summary"]["adversarial"], "clean": audit["summary"]["clean"],
        "reconstruction": audit["summary"]["reconstruction"], "audit_sha256": digest(folder / audit_name),
        "summary_sha256": audit["summary_sha256"]}
    rows.append(row)
    report(output, plan, rows, failures, "running")
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
        print(f"Overnight preflight passed: three batches, {len(plan['candidates'])} fresh-pilot cases; notifications disabled.", flush=True)
        return
    if args.stage:
        protocol.require(args.run is not None, "Stage needs --run")
        if args.stage == "evaluate":
            protocol.evaluate_stage(plan, args.run)
        else:
            protocol.require(args.config is not None and args.role is not None, "Audit needs config and role")
            config = load_config(args.config)
            protocol.verify_stage(effective_plan(plan, config), args.run, config, args.role, audit_name=args.audit_name)
            if config["attack"].get("lr_schedule", "constant") != "constant" or config["attack"]["optimizer"] == "l2_pgd":
                from beyondwaveform.attacks import learning_rate_at_step
                from beyondwaveform.config import AttackConfig
                settings = AttackConfig(**config["attack"])
                for record in read_jsonl(args.run / "generated.jsonl"):
                    protocol.require(all(abs(step["learning_rate"] - learning_rate_at_step(settings, i)) < 1e-12
                        for i, step in enumerate(record["trace"])), "Incorrect update schedule in trace")
        return
    supervisor, rows, failures = Supervisor(output), [], []
    try:
        report(output, plan, rows, failures, "running")
        for entry in plan["candidates"]:
            try:
                run_case(output, plan, entry, "pilot", rows, failures, supervisor)
            except RuntimeError as error:
                failures.append({"name": entry["name"], "group": entry["group"], "pool": "pilot", "error": str(error)})
                supervisor.progress(phase="continuing-after-independent-case-failure", error=str(error), failed_cases=len(failures))
                report(output, plan, rows, failures, "running")
                print(f"Case failed; continuing independent cases: {entry['name']}: {error}", flush=True)
        pilots = [r for r in rows if r["pool"] == "pilot"]
        names = promotions(pilots)
        frozen_decision(output / "validation-promotion.json", {"names": names,
            "pilot_evidence": selection.decision_evidence(pilots), "failed_pilot_cases": failures.copy()})
        validated = []
        for name in names:
            entry = next(c for c in plan["candidates"] if c["name"] == name)
            try:
                validated.append((entry, run_case(output, plan, entry, "validation", rows, failures, supervisor)))
            except RuntimeError as error:
                failures.append({"name": name, "group": entry["group"], "pool": "validation", "error": str(error)})
                report(output, plan, rows, failures, "running")
        winner = None
        for entry, _ in sorted([(e, r) for e, r in validated if qualifies(r)],
                               key=lambda pair: (pair[1]["error_percent"], pair[0]["name"]), reverse=True):
            try:
                confirmation = run_case(output, plan, entry["confirmation"], "validation", rows, failures, supervisor, pool="confirmation")
            except RuntimeError as error:
                failures.append({"name": entry["name"], "group": entry["group"], "pool": "confirmation", "error": str(error)})
                continue
            if qualifies(confirmation):
                winner = entry
                break
        if winner and not failures:
            frozen_decision(output / "selected-test-freeze.json", {"name": winner["name"], "config_path": winner["config_path"],
                "config_sha256": digest(winner["config_path"]), "validation_evidence": selection.decision_evidence(
                    [r for r in rows if r["pool"] in {"validation", "confirmation"}])})
            tested = run_case(output, plan, winner, "test", rows, failures, supervisor)
            outcome = "Selected test meets joint targets." if qualifies(tested) else "Selected test misses joint targets; no test-based replacement."
        elif failures:
            outcome = "Independent cases processed; 999-clip test withheld because some cases failed."
        else:
            outcome = "No stable joint WER/quality qualifier; no new 999-clip test launched."
        check(plan)
        if failures:
            raise RuntimeError(f"Queue processed remaining independent cases but {len(failures)} case(s) failed. {outcome}")
        report(output, plan, rows, failures, "complete", outcome)
        supervisor.finish(outcome=outcome, completed_cases=len(rows), failed_cases=0, error=None)
    except BaseException as error:
        report(output, plan, rows, failures, "failed", str(error))
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
