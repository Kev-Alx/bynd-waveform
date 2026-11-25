"""Study feature preprocessing and compatible vocoders at the paper budget.

Reconstruction controls screen each new path before expanding its optimizer
grid. Pilot-only selection, validation, and confirmation precede any test.
No kNN matching, altered scoring, or Discord notification is introduced.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.io import atomic_json, digest, read_jsonl
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.models.feature import load_feature_models
from parallel_jobs import Supervisor, now
import run_quality_research as protocol
import run_quality_refinement as selection

DEFAULT = ROOT / "reproduction-runs/paper-quality-contracts"
CONTRACTS = (("prematched-raw", "prematched", False),
             ("regular-normalized", "regular", True),
             ("prematched-normalized", "prematched", True))


def contract_key(features):
    return f"{features['vocoder_variant']}-{'normalized' if features['normalize'] else 'raw'}"


def candidate_grid(previous, old_root):
    controls = {c["name"]: c for c in previous["candidates"]}
    entries = []
    for source in previous["candidates"]:
        entry = {"name": source["name"], "config": copy.deepcopy(source["config"]),
                 "contract": "regular-raw", "cached": {"pilot": str(old_root / "pilot" / source["name"])}}
        if source["name"] in {"sum-lr-0.02", "sum-lr-0.1"}:
            entry["cached"]["validation"] = str(old_root / "validation" / source["name"])
        entries.append(entry)
    order = ("sum-lr-0.02", "sum-lr-0.05", "sum-lr-0.1",
             "mean-lr-0.02", "mean-lr-0.05", "mean-lr-0.1")
    for key, variant, normalize in CONTRACTS:
        for i, source in enumerate(order):
            config = copy.deepcopy(controls[source]["config"])
            config["features"].update(vocoder_variant=variant, normalize=normalize)
            entries.append({"name": key + "-" + source, "contract": key, "config": config,
                            "contract_probe": i == 0})
    return entries


def reconstruction_gate(summary):
    clean, reconstruction = summary["clean"], summary["reconstruction"]
    return math.isfinite(reconstruction["error_percent"]) and (
        reconstruction["error_percent"] <= clean["error_percent"] + 5.0 and
        all(math.isfinite(reconstruction["quality"][m]) and
            reconstruction["quality"][m] >= protocol.QUALITY_TARGET[m] for m in protocol.METRICS))


def effective_plan(plan, config):
    key = contract_key(config["features"])
    metadata = plan["feature_contracts"][key]
    protocol.require(metadata["normalize"] == config["features"]["normalize"] and
                     metadata["vocoder_variant"] == config["features"]["vocoder_variant"] and
                     metadata["layer"] == config["features"]["layer"], "Wrong feature contract")
    return {**plan, "feature_artifacts": metadata}


def read_scores(path):
    """Evaluation has three rows per ID; uniqueness is (ID, variant)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol.require(len({(r["id"], r["variant"]) for r in rows}) == len(rows), "Duplicate evaluation ID/variant pairs")
    return rows


def review_previous(output):
    """Recompute completed score evidence without changing the old batch."""
    previous = ROOT / "reproduction-runs/paper-quality-refinement"
    plan = protocol.read(previous / "plan.json")
    selection.check(plan)
    protocol.require(all(protocol.read(previous / n)["status"] == "complete"
                         for n in ("job.json", "progress.json", "results.json")), "Previous batch incomplete")
    reviewed = []
    for row in protocol.read(previous / "results.json")["results"]:
        run = Path(row["run"])
        audit_path = run / ("refinement-verification.json" if row["cached"] else "verification.json")
        audit, result = protocol.read(audit_path), protocol.read(run / "evaluation/summary.json")
        protocol.require(digest(audit_path) == row["audit_sha256"], "Previous audit changed")
        protocol.require(digest(run / "evaluation/summary.json") == audit["summary_sha256"] == row["summary_sha256"], "Previous summary changed")
        protocol.require(digest(run / "evaluation/utterances.jsonl") == result["utterances_sha256"], "Previous score rows changed")
        records, scores = read_jsonl(run / "generated.jsonl"), read_scores(run / "evaluation/utterances.jsonl")
        pool = plan["pools"][row["role"]]
        original = read_jsonl(pool["manifest"])
        metadata = protocol.read(run / "run.json")
        identity = {"config": metadata["config"], "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
        protocol.require(metadata["status"] == "complete" and metadata["config"] == audit["config"] and
                         metadata["fingerprint"] == protocol.config_fingerprint(identity) == result["generation_fingerprint"] and
                         digest(run / "generated.jsonl") == metadata["generated_sha256"], "Previous generation identity changed")
        protocol.require([r["id"] for r in records] == pool["ids"], "Previous coverage changed")
        pairs = {(r["id"], r["variant"]): r for r in scores}
        protocol.require(len(pairs) == len(scores) == len(original)*3, "Previous score coverage changed")
        for source, record in zip(original, records):
            protocol.require(len(record["trace"]) == 50 and all(math.isfinite(v) for step in record["trace"] for v in step.values()), "Previous trace changed")
            for variant, asset in record["audio"].items():
                score = pairs[(record["id"], variant)]
                protocol.require(digest(asset["path"]) == asset["sha256"] == score["audio_sha256"], "Previous WAV changed")
                protocol.require(source["text"] == record["text"] == score["reference"], "Previous reference changed")
                for key, value in error_counts(source["text"], score["hypothesis"], "en").items():
                    protocol.require(score[key] == value, "Previous lexical counts changed")
        for variant, measured in result["summary"].items():
            subset = [r for r in scores if r["variant"] == variant]
            protocol.require(measured["count"] == len(original), "Previous count changed")
            for key, value in aggregate_errors(subset).items():
                protocol.require(measured[key] == value, "Previous aggregate error changed")
            for metric in protocol.METRICS:
                values = [r["quality"][metric] for r in subset]
                protocol.require(all(math.isfinite(v) for v in values) and
                    abs(sum(values)/len(values)-measured["quality"][metric]) < 1e-10, "Previous quality aggregate changed")
        protocol.require(result["summary"] == audit["summary"] and audit["max_independent_delta_radius"] <= 0.100001,
                         "Previous verified budget/results changed")
        protocol.require(all(row[key] == value for key, value in result["summary"]["adversarial"].items()),
                         "Previous cumulative report differs from underlying results")
        reviewed.append({"run": str(run), "audit_sha256": digest(audit_path), "summary_sha256": audit["summary_sha256"]})
    history = output / "history"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "results.json", "report.md", "validation-promotion.json"):
        target = history / name
        if target.exists():
            protocol.require(digest(target) == digest(previous / name), "Historical snapshot changed")
        else:
            shutil.copy2(previous / name, target)
    receipt = {"reviewed_at": now(), "previous_plan_sha256": digest(previous / "plan.json"),
               "frozen_inputs_checked": len(plan["frozen_files"]), "score_sets_checked": reviewed,
               "note": "Previous audits remain unchanged; WAV hashes, coverage, WER, MOS and trace finiteness recomputed. Delta-radius evidence retained from hash-verified audits."}
    atomic_json(output / "prior-review.json", receipt)
    return plan


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Contract plan already exists")
    output.mkdir(parents=True, exist_ok=True)
    previous = review_previous(output)
    old_root = ROOT / "reproduction-runs/paper-quality-research"
    original = protocol.read(old_root / "plan.json")
    entries = candidate_grid(original, old_root)
    # Strict state-dict loading checks each experimental path before its freeze.
    feature_contracts = {"regular-raw": previous["feature_artifacts"]}
    for key, _, _ in CONTRACTS:
        config = next(c["config"]["features"] for c in entries if c["contract"] == key)
        print(f"Validating feature/vocoder path: {key}", flush=True)
        encoder, vocoder, metadata = load_feature_models(config, "cpu")
        feature_contracts[key] = metadata
        del encoder, vocoder
    files = [Path(__file__).resolve(), Path(protocol.__file__).resolve(), Path(selection.__file__).resolve(),
             output / "prior-review.json", *[p for p in (output / "history").rglob("*") if p.is_file()]]
    for entry in entries:
        for item in (entry, {"name": entry["name"] + "-confirmation", "config": copy.deepcopy(entry["config"])}):
            if item is not entry:
                item["config"]["seed"] = 2027
                entry["confirmation"] = item
            path = output / "configs" / (item["name"] + ".json")
            atomic_json(path, item["config"])
            path.chmod(0o444)
            item["config_path"] = str(path)
            files.append(path)
        for cached in entry.get("cached", {}).values():
            folder = Path(cached)
            audit = protocol.read(folder / "verification.json")
            protocol.require(digest(folder / "evaluation/summary.json") == audit["summary_sha256"], "Cached summary changed")
            files.extend(folder / name for name in ("run.json", "generated.jsonl", "verification.json",
                         "evaluation/summary.json", "evaluation/utterances.jsonl"))
    hashes = dict(previous["frozen_files"])
    files += [ROOT / "artifacts/checkpoints/prematch_g_02500000.pt",
              ROOT / "artifacts/checkpoints/prematch_g_02500000.pt.receipt.json"]
    files += [ROOT / "scripts/update_job_status.py", ROOT / "artifacts/sources" /
              "knn-vc-c616845c4e309e24d5927f15adbdf277a3d65358/README.md"]
    hashes.update({str(p.resolve()): digest(p) for p in files})
    plan = {"created_at": now(), "pools": previous["pools"], "quality": previous["quality"],
            "runtime": protocol.runtime(), "base_surrogate_artifacts": previous["base_surrogate_artifacts"],
            "feature_artifacts": previous["feature_artifacts"], "feature_contracts": feature_contracts,
            "candidates": entries, "frozen_files": hashes, "discord_enabled": False,
            "paper_budget": {"rho": 0.1, "lambda_perc": 1, "steps": 50, "cutoff_hz": 6000},
            "reconstruction_gate": "Pilot reconstruction WER <= clean WER +5 percentage points, and all three reconstruction MOS reach Table 4 feature anchors. A failing path stops after its sum/LR=0.02 probe.",
            "selection": "Same quality-first, attack-first and balanced pilot selection; validation joint target and seed-2027 confirmation before one paired test.",
            "wer_floor": selection.WER_FLOOR, "quality_targets": protocol.QUALITY_TARGET,
            "scope": "Independent upstream-compatible layer-6 generators; normalization is experimental preprocessing. No kNN replacement. Author artifacts and subset remain unknown; historical test is not a fresh blind holdout."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.check(plan)
    protocol.validate_candidate_files([c["confirmation"] for c in plan["candidates"]])
    for entry in plan["candidates"]:
        config = entry["config"]
        protocol.require(contract_key(config["features"]) == entry["contract"], "Changed candidate path")
        effective_plan(plan, config)
        attack = config["attack"]
        protocol.require(attack["rho"] == 0.1 and attack["lambda_perc"] == 1 and attack["steps"] == 50 and
                         config["spectral"]["cutoff_hz"] == 6000, "Changed explicit paper setting")


def report(output, rows, skipped, status, outcome=None):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows,
                "skipped": skipped, "outcome": outcome, "quality_targets": protocol.QUALITY_TARGET,
                "wer_floor": selection.WER_FLOOR})
    lines = ["# Feature/vocoder quality study", "", f"Status: **{status}**.", "",
             "All attacks retain rho=0.1, lambda=1, 50 updates, and the stated 6000 Hz cutoff. Six cached controls and up to 18 new pilots compare generator/preprocessing choices.", "",
             "| Pool | Candidate | WER % | DNSMOS | NISQA | UTMOS | Reconstruction WER % | Reconstruction UTMOS | Joint target | Cached |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        q, reconstruction = row["quality"], row["reconstruction"]
        lines.append(f"| {row['pool']} | {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {reconstruction['error_percent']:.4f} | {reconstruction['quality']['utmos']:.4f} | {selection.qualifies(row)} | {row['cached']} |")
    lines += ["", "Joint exploratory target: WER >=73.43%, DNSMOS >=2.954, NISQA >=3.526, UTMOS >=3.590. This does not establish author parity or statistical equivalence. Validation qualifiers must repeat before one paired test; test feedback cannot choose a replacement.", "",
              "Prematched refers to the generator's upstream training, not to nearest-neighbor conversion here. All paths directly vocode the current utterance's perturbed features. Normalization is tested with reconstruction controls rather than assumed compatible with training. Cached controls are not fresh replications.", ""]
    if skipped:
        lines += ["Paths rejected by reconstruction gate:", ""] + [f"- {key}: {value['reason']}" for key, value in skipped.items()] + [""]
    if outcome:
        lines += [f"Outcome: {outcome}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def frozen_decision(path, evidence):
    if path.exists():
        protocol.require(protocol.read(path) == evidence, f"Decision changed on resume: {path.name}")
    else:
        atomic_json(path, evidence)
        path.chmod(0o444)


def run_case(output, plan, entry, role, rows, skipped, supervisor, pool=None):
    pool = pool or role
    cached = entry.get("cached", {}).get(role) if pool == role else None
    folder = Path(cached) if cached else output / pool / entry["name"]
    check(plan)
    if not (folder / "evaluation/summary.json").exists():
        selection.phase(supervisor, f"{pool}-{entry['name']}-generate", [sys.executable, "-u", "scripts/generate.py",
            "--config", entry["config_path"], "--manifest", plan["pools"][role]["manifest"], "--output", str(folder)])
        selection.phase(supervisor, f"{pool}-{entry['name']}-evaluate", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--output", str(output), "--stage", "evaluate", "--run", str(folder)])
    audit_name = "contracts-verification.json" if cached else "verification.json"
    selection.phase(supervisor, f"{pool}-{entry['name']}-verify", [sys.executable, "-u", str(Path(__file__).resolve()),
        "--output", str(output), "--stage", "verify", "--run", str(folder), "--config", entry["config_path"],
        "--role", role, "--audit-name", audit_name])
    audit = protocol.read(folder / audit_name)
    row = {"name": entry["name"], "pool": pool, "role": role, "cached": bool(cached), "run": str(folder),
           "contract": contract_key(entry["config"]["features"]), **audit["summary"]["adversarial"],
           "reconstruction": audit["summary"]["reconstruction"], "clean": audit["summary"]["clean"],
           "audit_sha256": digest(folder / audit_name), "summary_sha256": audit["summary_sha256"]}
    rows.append(row)
    report(output, rows, skipped, "running")
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
        print(f"Contract preflight passed: {len(plan['candidates'])} pilot entries; notifications disabled.", flush=True)
        return
    if args.stage:
        protocol.require(args.run is not None, "Stage needs --run")
        if args.stage == "evaluate":
            protocol.evaluate_stage(plan, args.run)
        else:
            protocol.require(args.config is not None and args.role is not None, "Audit needs config and role")
            config = protocol.load_config(args.config)
            protocol.verify_stage(effective_plan(plan, config), args.run, config, args.role, audit_name=args.audit_name)
        return
    supervisor, rows, skipped = Supervisor(output), [], {}
    try:
        report(output, rows, skipped, "running")
        pilots = []
        for entry in plan["candidates"]:
            key = entry["contract"]
            if key in skipped:
                continue
            row = run_case(output, plan, entry, "pilot", rows, skipped, supervisor)
            if entry.get("contract_probe"):
                passed = reconstruction_gate({"clean": row["clean"], "reconstruction": row["reconstruction"]})
                evidence = {"contract": key, "passed": passed, "probe": selection.decision_evidence([row])[0]}
                frozen_decision(output / "reconstruction-gates" / (key + ".json"), evidence)
                if not passed:
                    skipped[key] = {"reason": "Pilot reconstruction fails fixed WER/quality screen; remaining five settings skipped.",
                                    "probe_summary_sha256": row["summary_sha256"]}
                    report(output, rows, skipped, "running")
                    continue
            pilots.append(row)
        names = selection.select_validation(pilots)
        frozen_decision(output / "validation-promotion.json", {"names": names, "pilot_evidence": selection.decision_evidence(pilots), "skipped": skipped})
        validated = []
        for name in names:
            entry = next(c for c in plan["candidates"] if c["name"] == name)
            validated.append((entry, run_case(output, plan, entry, "validation", rows, skipped, supervisor)))
        winner = None
        for entry, _ in sorted([(e, r) for e, r in validated if selection.qualifies(r)],
                               key=lambda pair: (pair[1]["error_percent"], pair[0]["name"]), reverse=True):
            confirmation = run_case(output, plan, entry["confirmation"], "validation", rows, skipped, supervisor, pool="confirmation")
            if selection.qualifies(confirmation):
                winner = entry
                break
        if winner:
            frozen_decision(output / "selected-test-freeze.json", {"name": winner["name"], "config_path": winner["config_path"],
                "config_sha256": digest(winner["config_path"]), "validation_evidence": selection.decision_evidence(
                    [r for r in rows if r["pool"] in {"validation", "confirmation"}])})
            tested = run_case(output, plan, winner, "test", rows, skipped, supervisor)
            outcome = "Selected test meets the joint target." if selection.qualifies(tested) else "Selected test misses the joint target; no test-based replacement."
        else:
            outcome = "No candidate met joint WER/quality targets on validation and confirmation; no new 999-clip test launched."
        check(plan)
        report(output, rows, skipped, "complete", outcome)
        supervisor.finish(outcome=outcome)
    except BaseException as error:
        report(output, rows, skipped, "failed", str(error))
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
