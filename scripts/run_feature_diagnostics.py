"""Controlled saved-feature-delta diagnostics, not new optimized attacks.

Hold features, generator, ASR and scorers fixed. Contrast temporal smoothing
with norm-matched controls. No 999-clip test is automatically selected here.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl, load_audio, save_audio, seed_everything
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.pipeline import config_fingerprint, evaluate
from beyondwaveform.evaluation import error_counts, aggregate_errors
from beyondwaveform.attacks import relative_radius, project_relative_ball_
from beyondwaveform.feature_transforms import transform_delta
from parallel_jobs import Supervisor, now
import run_quality_research as protocol
import run_quality_refinement as selection
from run_quality_contracts import read_scores, frozen_decision

DEFAULT = ROOT / "reproduction-runs/feature-delta-diagnostics"


def cases():
    rows = [{"name": "identity", "mode": "identity", "window": 1}]
    for window in (3, 5, 9):
        for mode in ("attenuation_matched", "smooth", "smooth_norm_matched"):
            rows.append({"name": f"{mode}-{window}", "mode": mode, "window": window})
    return rows


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Diagnostic plan already exists")
    old = ROOT / "reproduction-runs/paper-quality-overnight"
    previous = protocol.read(old / "plan.json")
    protocol.check(previous)
    protocol.require(protocol.read(old / "job.json")["status"] == "complete", "Overnight job not complete")
    output.mkdir(parents=True, exist_ok=True)
    history = output / "history"
    history.mkdir(exist_ok=True)
    files = [Path(__file__).resolve(), ROOT / "beyondwaveform/feature_transforms.py",
             ROOT / "scripts/probe_feature_gradients.py"]
    for name in ("plan.json", "results.json", "report.md", "job.json", "validation-promotion.json"):
        target = history / name
        if target.exists():
            protocol.require(digest(target) == digest(old / name), "Changed historical snapshot")
        else:
            shutil.copy2(old / name, target)
        files.append(target)
    review = ROOT / "reproduction-runs/overnight-review-20261019"
    files += [review / "frozen-integrity.json", review / "integrity-and-analysis.json", review / "report.md"]
    hashes = dict(previous["frozen_files"])
    parents = {}
    for role, run in (("pilot", old / "pilot/control-sum-lr-0.1"),
                      ("validation", ROOT / "reproduction-runs/paper-quality-research/validation/sum-lr-0.1")):
        metadata = protocol.read(run / "run.json")
        audit = protocol.read(run / "verification.json")
        protocol.require(digest(run / "evaluation/summary.json") == audit["summary_sha256"], "Changed parent summary")
        config = metadata["config"]
        settings = AttackConfig(**config["attack"])
        spectral = SpectralConfig(**config["spectral"])
        protocol.require(settings.rho == 0.1 and settings.lambda_perc == 1 and settings.steps == 50 and spectral.cutoff_hz == 6000,
                         "Parent does not use paper settings")
        protocol.require(metadata["ids"] == previous["pools"][role]["ids"], "Parent pool mismatch")
        delta_hashes = {uid: digest(run / (uid + ".delta.pt")) for uid in metadata["ids"]}
        parents[role] = {"run": str(run), "config": config, "delta_hashes": delta_hashes,
                         "summary_sha256": audit["summary_sha256"]}
        files.extend(run / n for n in ("run.json", "generated.jsonl", "verification.json",
                                      "evaluation/summary.json", "evaluation/utterances.jsonl"))
        hashes.update({str(run / (uid + ".delta.pt")): sha for uid, sha in delta_hashes.items()})
    hashes.update({str(p.resolve()): digest(p) for p in files})
    plan = {"created_at": now(), "pools": {k: previous["pools"][k] for k in ("pilot", "validation")},
        "parents": parents, "cases": cases(), "frozen_files": hashes, "runtime": protocol.runtime(),
        "quality": previous["quality"], "feature_artifacts": previous["feature_artifacts"],
        "base_surrogate_artifacts": previous["base_surrogate_artifacts"], "discord_enabled": False,
        "waveform_reproduction_tolerance": 2e-5, "gaussian_sigma": "(window-1)/4", "boundary": "replicate",
        "validation_rule": "For smooth, compare to attenuation_matched at the same window; for smooth_norm_matched, compare to identity. Promote at most two with >=0.05 UTMOS gain, <=5 WER-point loss and all MOS >2.5, ranking weakest target ratio. Also evaluate their matched controls and identity on validation.",
        "scope": "Saved-delta structural diagnostic: zero new optimization steps, source delta from 50 paper-budget updates. No automatic 999-clip evaluation or claim of a recovered author optimizer."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.require(protocol.runtime() == plan["runtime"], "Changed runtime")
    for path, expected in plan["frozen_files"].items():
        protocol.require(digest(path) == expected, f"Changed frozen diagnostic input: {path}")
    for role, parent in plan["parents"].items():
        pool = plan["pools"][role]
        protocol.require(digest(pool["manifest"]) == pool["sha256"], "Changed manifest")
        protocol.require([r["id"] for r in read_jsonl(pool["manifest"])] == pool["ids"], "Changed IDs")
        settings = AttackConfig(**parent["config"]["attack"])
        protocol.require(settings.rho == 0.1 and settings.lambda_perc == 1 and settings.steps == 50, "Changed source budget")


def parent_check(output, plan, role):
    import torch
    from tqdm import tqdm
    from beyondwaveform.models.feature import load_feature_models
    parent, pool = plan["parents"][role], plan["pools"][role]
    run = Path(parent["run"])
    metadata, result = protocol.read(run / "run.json"), protocol.read(run / "evaluation/summary.json")
    records, source, scores = read_jsonl(run / "generated.jsonl"), read_jsonl(pool["manifest"]), read_scores(run / "evaluation/utterances.jsonl")
    identity = {"config": parent["config"], "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    protocol.require(metadata["status"] == "complete" and metadata["config"] == parent["config"] and
        metadata["features"] == plan["feature_artifacts"] and digest(run / "generated.jsonl") == metadata["generated_sha256"] and
        digest(run / "evaluation/utterances.jsonl") == result["utterances_sha256"] and
        metadata["fingerprint"] == config_fingerprint(identity) == result["generation_fingerprint"], "Changed parent generation/scores")
    protocol.require(result["evaluation_scope"] == "surrogate" and result["research_scorer_manifest"] == plan["quality"] and
        result["evaluation_environment"] == plan["runtime"] and result["target_artifacts"] == metadata["surrogate"], "Changed parent evaluation protocol")
    pairs = {(r["id"], r["variant"]): r for r in scores}
    protocol.require(len(pairs) == len(source)*3 and [r["id"] for r in records] == pool["ids"], "Parent coverage mismatch")
    for original, record in zip(source, records):
        protocol.require(digest(original["audio"]) == original["audio_sha256"], "Changed original source audio")
        protocol.require(original["text"] == record["text"] and original["audio_sha256"] == record["input_audio_sha256"], "Changed parent source")
        protocol.require(len(record["trace"]) == 50 and all(math.isfinite(v) for step in record["trace"] for v in step.values()), "Invalid parent trace")
        for variant, asset in record["audio"].items():
            row = pairs[(record["id"], variant)]
            protocol.require(digest(asset["path"]) == asset["sha256"] == row["audio_sha256"] and row["reference"] == original["text"], "Changed parent WAV/reference")
            protocol.require(all(row[k] == v for k, v in error_counts(original["text"], row["hypothesis"], "en").items()), "Parent edit counts mismatch")
    for variant, summary in result["summary"].items():
        subset = [r for r in scores if r["variant"] == variant]
        protocol.require(summary["count"] == len(source) and all(summary[k] == v for k, v in aggregate_errors(subset).items()), "Parent aggregate errors mismatch")
        for metric in protocol.METRICS:
            values = [r["quality"][metric] for r in subset]
            protocol.require(all(math.isfinite(v) for v in values) and abs(sum(values)/len(values)-summary["quality"][metric]) < 1e-10, "Parent MOS mismatch")
    audit_path = output / "parent-audits" / (role + ".json")
    if audit_path.exists():
        prior = protocol.read(audit_path)
        protocol.require(prior["parent_summary_sha256"] == parent["summary_sha256"] and
            prior["role"] == role and prior["features"] == plan["feature_artifacts"] and
            [r["id"] for r in prior["items"]] == pool["ids"] and
            all(r["parent_delta_sha256"] == parent["delta_hashes"][r["id"]] for r in prior["items"]) and
            all(digest(r["q_path"]) == r["q_sha256"] for r in prior["items"]), "Changed cached clean features")
        print(f"Reused hash-verified {role} clean-feature cache.", flush=True)
        return
    encoder, vocoder, features = load_feature_models(parent["config"]["features"], "cuda")
    protocol.require(features == plan["feature_artifacts"], "Changed feature models")
    items = []
    for original, record in tqdm(list(zip(source, records)), desc="Checking parent deltas and resynthesis"):
        seed_everything(record["seed"])
        with torch.no_grad():
            q = encoder(load_audio(original["audio"]).cuda()).detach()
            delta = torch.load(run / (original["id"] + ".delta.pt"), map_location="cuda", weights_only=True)
            protocol.require(delta.shape == q.shape and torch.isfinite(delta).all(), "Invalid source delta")
            radius = relative_radius(delta, q).item()
            protocol.require(radius <= 0.100001 and abs(radius-record["trace"][-1]["radius_after_update"]) < 1e-6, "Source delta exceeds/mismatches radius")
            regenerated = vocoder(q + delta)
            saved = load_audio(record["audio"]["adversarial"]["path"]).cuda()
            protocol.require(regenerated.shape == saved.shape, "Source waveform length mismatch")
            difference = (regenerated-saved).abs().max().item()
            protocol.require(difference <= plan["waveform_reproduction_tolerance"], "Saved delta does not reproduce parent waveform")
        path = output / "feature-cache" / role / (original["id"] + ".q.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(q.cpu(), path)
        items.append({"id": original["id"], "q_path": str(path), "q_sha256": digest(path),
                      "parent_radius": radius, "resynthesis_max_abs_error": difference,
                      "parent_delta_sha256": parent["delta_hashes"][original["id"]]})
    atomic_json(audit_path, {"verified_at": now(), "role": role, "parent_summary_sha256": parent["summary_sha256"],
        "features": features, "items": items, "score_rows_checked": len(scores), "parent_optimization_steps": 50})


def diagnostic_config(plan, entry, role):
    parent = plan["parents"][role]
    return {"method": "saved_feature_delta_diagnostic", "features": parent["config"]["features"],
        "surrogate": parent["config"]["surrogate"], "transform": entry, "parent_run": parent["run"],
        "new_optimization_steps": 0, "parent_optimization_steps": 50}


def generate_case(output, plan, entry, role):
    import torch
    from tqdm import tqdm
    from beyondwaveform.models.feature import load_feature_models
    parent, pool = plan["parents"][role], plan["pools"][role]
    folder = output / role / entry["name"]
    config = diagnostic_config(plan, entry, role)
    identity = {"config": config, "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    fingerprint = config_fingerprint(identity)
    if (folder / "run.json").exists():
        protocol.require(protocol.read(folder / "run.json")["fingerprint"] == fingerprint, "Changed diagnostic identity")
    metadata = protocol.read(Path(parent["run"]) / "run.json")
    generated = {**identity, "fingerprint": fingerprint, "roles": [role], "features": metadata["features"],
        "surrogate": metadata["surrogate"], "status": "running", "environment": protocol.runtime(),
        "scope": "saved-delta structural diagnostic; zero new optimization updates"}
    atomic_json(folder / "run.json", generated)
    source = read_jsonl(pool["manifest"])
    records = read_jsonl(Path(parent["run"]) / "generated.jsonl")
    cache = protocol.read(output / "parent-audits" / (role + ".json"))
    cache_items = {r["id"]: r for r in cache["items"]}
    encoder, vocoder, features = load_feature_models(parent["config"]["features"], "cuda")
    del encoder
    protocol.require(features == plan["feature_artifacts"], "Changed generator")
    produced = []
    for original, record in tqdm(list(zip(source, records)), desc="Transforming saved feature deltas"):
        path = folder / "records" / (original["id"] + ".json")
        if path.exists():
            item = protocol.read(path)
            protocol.require(digest(item["delta"]["path"]) == item["delta"]["sha256"] and
                digest(item["audio"]["adversarial"]["path"]) == item["audio"]["adversarial"]["sha256"], "Changed saved diagnostic output")
            produced.append(item)
            continue
        info = cache_items[original["id"]]
        protocol.require(digest(info["q_path"]) == info["q_sha256"], "Changed q cache")
        seed_everything(record["seed"])
        q = torch.load(info["q_path"], map_location="cuda", weights_only=True)
        source_delta = torch.load(Path(parent["run"]) / (original["id"] + ".delta.pt"), map_location="cuda", weights_only=True)
        with torch.no_grad():
            delta = transform_delta(source_delta, entry["window"], entry["mode"])
            project_relative_ball_(delta, q, 0.1)
            radius = relative_radius(delta, q).item()
            protocol.require(torch.isfinite(delta).all() and radius <= 0.100001, "Invalid transformed radius")
            audio = vocoder(q + delta)
        delta_path = folder / "deltas" / (original["id"] + ".delta.pt")
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta.cpu(), delta_path)
        asset = save_audio(folder / "audio/adversarial" / (original["id"] + ".wav"), audio)
        item = {"id": original["id"], "text": original["text"], "language": "en", "role": role,
            "input_audio_sha256": original["audio_sha256"], "audio": {**record["audio"], "adversarial": asset},
            "delta": {"path": str(delta_path), "sha256": digest(delta_path)}, "transform": entry,
            "diagnostic": {"source_radius": info["parent_radius"], "radius": radius,
                "delta_norm_ratio": (delta.norm()/source_delta.norm()).item()},
            "parent_run": parent["run"], "parent_optimization_steps": 50, "new_optimization_steps": 0}
        atomic_json(path, item)
        produced.append(item)
    write_jsonl(folder / "generated.jsonl", produced)
    generated.update(status="complete", count=len(produced), generated_sha256=digest(folder / "generated.jsonl"))
    atomic_json(folder / "run.json", generated)


def evaluate_case(output, plan, entry, role):
    folder = output / role / entry["name"]
    protocol.configure_scorers(plan["quality"])
    target = {"kind": "whisper", **plan["parents"][role]["config"]["surrogate"]}
    result = evaluate(folder, target, folder / "evaluation", variants=["adversarial"], quality=True,
                      artifact_root=str(ROOT / "artifacts"))
    result["research_scorer_manifest"] = plan["quality"]
    result["evaluation_environment"] = protocol.runtime()
    result["scope"] = "post-optimization saved-delta diagnostic"
    atomic_json(folder / "evaluation/summary.json", result)


def verify_case(output, plan, entry, role):
    import torch
    from tqdm import tqdm
    from beyondwaveform.models.feature import load_feature_models
    folder = output / role / entry["name"]
    pool, parent = plan["pools"][role], plan["parents"][role]
    source, records = read_jsonl(pool["manifest"]), read_jsonl(folder / "generated.jsonl")
    result, metadata = protocol.read(folder / "evaluation/summary.json"), protocol.read(folder / "run.json")
    config = diagnostic_config(plan, entry, role)
    expected = {"config": config, "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    protocol.require(metadata["status"] == "complete" and metadata["config"] == config and
        metadata["fingerprint"] == config_fingerprint(expected) == result["generation_fingerprint"] and
        digest(folder / "generated.jsonl") == metadata["generated_sha256"], "Changed derived identity")
    protocol.require(result["target"] == {"kind": "whisper", **parent["config"]["surrogate"]} and
        result["target_artifacts"] == metadata["surrogate"] and result["evaluation_scope"] == "surrogate" and
        result["roles"] == [role] and result["research_scorer_manifest"] == plan["quality"] and
        result["evaluation_environment"] == plan["runtime"], "Changed diagnostic scoring protocol")
    protocol.require(digest(folder / "evaluation/utterances.jsonl") == result["utterances_sha256"], "Changed score rows")
    scores = read_scores(folder / "evaluation/utterances.jsonl")
    protocol.require([r["id"] for r in records] == result["evaluation_ids"] == pool["ids"] and
        len(scores) == len(source) and {(r["id"], r["variant"]) for r in scores} == {(r["id"], "adversarial") for r in source}, "Diagnostic coverage mismatch")
    by_id = {r["id"]: r for r in scores}
    parent_records = {r["id"]: r for r in read_jsonl(Path(parent["run"]) / "generated.jsonl")}
    cache = {r["id"]: r for r in protocol.read(output / "parent-audits" / (role + ".json"))["items"]}
    encoder, vocoder, features = load_feature_models(parent["config"]["features"], "cuda")
    del encoder
    protocol.require(features == metadata["features"] == plan["feature_artifacts"], "Changed diagnostic generator")
    radii, errors = [], []
    for original, record in tqdm(list(zip(source, records)), desc="Auditing transformed deltas and WAVs"):
        row = by_id[original["id"]]
        protocol.require(row["reference"] == record["text"] == original["text"] and
            record["new_optimization_steps"] == 0 and record["parent_optimization_steps"] == 50,
            "Changed references or derivation steps")
        protocol.require(all(row[k] == v for k, v in error_counts(original["text"], row["hypothesis"], "en").items()), "Incorrect diagnostic edit counts")
        asset, info = record["audio"]["adversarial"], cache[original["id"]]
        protocol.require(digest(asset["path"]) == asset["sha256"] == row["audio_sha256"] and
            digest(record["delta"]["path"]) == record["delta"]["sha256"] and digest(info["q_path"]) == info["q_sha256"], "Changed diagnostic/cache artifacts")
        seed_everything(parent_records[original["id"]]["seed"])
        q = torch.load(info["q_path"], map_location="cuda", weights_only=True)
        delta = torch.load(record["delta"]["path"], map_location="cuda", weights_only=True)
        before = torch.load(Path(parent["run"]) / (original["id"] + ".delta.pt"), map_location="cuda", weights_only=True)
        expected_delta = transform_delta(before, entry["window"], entry["mode"])
        project_relative_ball_(expected_delta, q, 0.1)
        protocol.require(delta.shape == q.shape and torch.isfinite(delta).all() and torch.allclose(delta, expected_delta, atol=1e-7, rtol=1e-5), "Transformed delta differs from frozen rule")
        radius = relative_radius(delta, q).item()
        protocol.require(radius <= 0.100001 and abs(radius-record["diagnostic"]["radius"]) < 1e-6, "Incorrect transformed radius")
        with torch.no_grad(): regenerated = vocoder(q + delta)
        saved = load_audio(asset["path"]).cuda()
        protocol.require(regenerated.shape == saved.shape, "Derived waveform duration mismatch")
        difference = (regenerated-saved).abs().max().item()
        protocol.require(difference <= plan["waveform_reproduction_tolerance"], "Delta does not reproduce derived WAV")
        radii.append(radius); errors.append(difference)
    summary = result["summary"]["adversarial"]
    protocol.require(summary["count"] == len(source) and all(summary[k] == v for k, v in aggregate_errors(scores).items()), "Incorrect diagnostic aggregate edits")
    for metric in protocol.METRICS:
        values = [r["quality"][metric] for r in scores]
        protocol.require(all(math.isfinite(v) for v in values) and abs(sum(values)/len(values)-summary["quality"][metric]) < 1e-10, "Incorrect diagnostic MOS")
    atomic_json(folder / "verification.json", {"verified_at": now(), "count": len(source),
        "summary_sha256": digest(folder / "evaluation/summary.json"), "summary": result["summary"],
        "max_independent_delta_radius": max(radii), "max_resynthesis_abs_error": max(errors),
        "new_optimization_steps": 0, "parent_optimization_steps": 50})


def promotion(rows):
    by_name = {r["name"]: r for r in rows}
    eligible = []
    for row in rows:
        if row["mode"] not in {"smooth", "smooth_norm_matched"}:
            continue
        control = by_name["identity" if row["mode"] == "smooth_norm_matched" else f"attenuation_matched-{row['window']}"]
        if (selection.broad_gate(row) and row["quality"]["utmos"] - control["quality"]["utmos"] >= 0.05 and
            row["error_percent"] >= control["error_percent"] - 5):
            eligible.append(row)
    ranked = sorted(eligible, key=lambda r: (min(r["error_percent"]/selection.WER_FLOOR,
        *(r["quality"][m]/protocol.QUALITY_TARGET[m] for m in protocol.METRICS)), r["error_percent"], r["name"]), reverse=True)
    names = [r["name"] for r in ranked[:2]]
    controls = ["identity"] if names else []
    controls += [f"attenuation_matched-{by_name[n]['window']}" for n in names if by_name[n]["mode"] == "smooth"]
    return {"names": list(dict.fromkeys([*controls, *names])), "promising_directions": names}


def report(output, rows, status, outcome=None):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows,
        "outcome": outcome, "scope": "saved-delta diagnostic; no new 50-step attack or automatic test"})
    lines = ["# Feature perturbation structure diagnostics", "", f"Status: **{status}**.", "",
        "Saved source deltas came from 50 updates at rho=0.1/lambda=1/6000 Hz. Transformations use zero new optimization steps. All derived deltas are projected/independently checked against the same radius and re-vocoded. Decoder and MOS models stay fixed.", "",
        "| Pool | Transform | WER % | DNSMOS | NISQA | UTMOS | Cached source control |",
        "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        q = row["quality"]
        lines.append(f"| {row['role']} | {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {row['cached']} |")
    lines += ["", "For each 3/5/9-frame Gaussian: smooth is compared with an unsmoothed attenuation having the same norm; smooth_norm_matched is compared with original identity at the same norm. Replicate boundaries and sigma=(window-1)/4 are fixed. Every result is reported, including matched controls.", "",
        "Only a directional improvement of at least 0.05 UTMOS with at most five WER-point loss and all MOS >2.5 can advance to validation (up to two directions plus their controls). This exploratory screen is not statistical equivalence. No 999-clip run or paper-parity claim follows automatically. The gradient probe measures the existing objective without changing it.", ""]
    if outcome:
        lines += [f"Outcome: {outcome}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_case(output, plan, entry, role, rows, supervisor):
    check(plan)
    if entry["mode"] == "identity":
        run = Path(plan["parents"][role]["run"])
        result = protocol.read(run / "evaluation/summary.json")
        row = {**entry, "role": role, "pool": role, "run": str(run), "cached": True,
            **result["summary"]["adversarial"], "summary_sha256": digest(run / "evaluation/summary.json")}
    else:
        run = output / role / entry["name"]
        base = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output),
                "--role", role, "--case", entry["name"], "--stage"]
        if not (run / "evaluation/summary.json").exists():
            selection.phase(supervisor, f"{role}-{entry['name']}-generate", [*base, "generate"])
            selection.phase(supervisor, f"{role}-{entry['name']}-evaluate", [*base, "evaluate"])
        selection.phase(supervisor, f"{role}-{entry['name']}-verify", [*base, "verify"])
        audit = protocol.read(run / "verification.json")
        row = {**entry, "role": role, "pool": role, "run": str(run), "cached": False,
            **audit["summary"]["adversarial"], "summary_sha256": audit["summary_sha256"],
            "audit_sha256": digest(run / "verification.json")}
    rows.append(row)
    report(output, rows, "running")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--stage", choices=("parent", "generate", "evaluate", "verify"))
    parser.add_argument("--role", choices=("pilot", "validation"))
    parser.add_argument("--case", choices=tuple(r["name"] for r in cases()))
    args = parser.parse_args()
    os.environ["BEYONDWAVEFORM_NOTIFY_DISCORD"] = "0"
    output = args.output.resolve()
    plan = prepare(output) if args.prepare else protocol.read(output / "plan.json")
    check(plan)
    if args.prepare or args.check_only:
        print(f"Diagnostic preflight passed: {len(plan['cases'])} controlled cases; zero new optimization updates; Discord disabled.", flush=True)
        return
    if args.stage:
        protocol.require(args.role is not None, "Stage needs --role")
        if args.stage == "parent":
            parent_check(output, plan, args.role)
        else:
            protocol.require(args.case is not None and args.case != "identity", "Transform stage needs nonidentity --case")
            entry = next(r for r in plan["cases"] if r["name"] == args.case)
            {"generate": generate_case, "evaluate": evaluate_case, "verify": verify_case}[args.stage](output, plan, entry, args.role)
        return
    supervisor, rows = Supervisor(output), []
    try:
        report(output, rows, "running")
        selection.phase(supervisor, "pilot-parent-check", [sys.executable, "-u", str(Path(__file__).resolve()),
            "--output", str(output), "--stage", "parent", "--role", "pilot"])
        gradient_path = output / "gradient-probe.json"
        if gradient_path.exists():
            gradient = protocol.read(gradient_path)
            protocol.require(gradient["status"] == "complete" and gradient["count"] == 3 and
                gradient["plan_sha256"] == digest(output / "plan.json") and
                gradient["parent_cache_audit_sha256"] == digest(output / "parent-audits/pilot.json") and
                all(digest(path) == expected for path, expected in gradient["source_hashes"].items()), "Changed completed gradient probe")
        else:
            selection.phase(supervisor, "pilot-component-gradients", [sys.executable, "-u", "scripts/probe_feature_gradients.py",
                "--plan", str(output / "plan.json"), "--output", str(gradient_path), "--count", "3"])
        pilots = [run_case(output, plan, entry, "pilot", rows, supervisor) for entry in plan["cases"]]
        decision = promotion(pilots)
        frozen_decision(output / "validation-promotion.json", {**decision, "pilot_evidence": selection.decision_evidence(pilots)})
        if decision["names"]:
            selection.phase(supervisor, "validation-parent-check", [sys.executable, "-u", str(Path(__file__).resolve()),
                "--output", str(output), "--stage", "parent", "--role", "validation"])
            for name in decision["names"]:
                run_case(output, plan, next(r for r in plan["cases"] if r["name"] == name), "validation", rows, supervisor)
            outcome = "Pilot-selected directions and matched controls evaluated on validation. These are post-optimization diagnostics; no automatic 999-clip evaluation."
        else:
            outcome = "No smoothing direction cleared the predeclared pilot gate versus its norm-matched control; no validation/test expansion."
        check(plan)
        report(output, rows, "complete", outcome)
        supervisor.finish(outcome=outcome)
    except BaseException as error:
        report(output, rows, "failed", str(error))
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
