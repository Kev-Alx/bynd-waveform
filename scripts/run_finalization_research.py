"""Frozen two-trajectory study of returned deltas within fifty updates.

Each family shares a fresh terminal trajectory, its last-ten-iterate mean and
a terminal direction matched to that mean's norm. Only a predefined paired
pilot improvement can expand to a fresh confirmation pool. No test is run.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.attacks import relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.evaluation import error_counts, aggregate_errors
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, save_audio, seed_everything, write_jsonl
from beyondwaveform.pipeline import config_fingerprint, evaluate
from beyondwaveform.trajectory_diagnostics import SNAPSHOT_STEPS, collect_trajectory, finalize_deltas
from parallel_jobs import Supervisor, now
import run_quality_research as protocol
import run_quality_refinement as stages
from run_quality_contracts import frozen_decision, read_scores, reconstruction_gate

DEFAULT = ROOT / "reproduction-runs/paper-quality-finalization"
DECODER = ROOT / "reproduction-runs/decoder-alignment-diagnostics"
OVERNIGHT = ROOT / "reproduction-runs/paper-quality-overnight"
FAMILIES = ("control-sum-lr-0.1", "timestamp-sum-lr-0.05")
MODES = ("terminal", "average_41_50", "terminal_norm_matched")
GAIN = .05
WER_LOSS = 5.0


def validate_config(config):
    attack, spectral = AttackConfig(**config["attack"]), SpectralConfig(**config["spectral"])
    protocol.require(attack.steps == 50 and attack.rho == .1 and attack.lambda_perc == 1 and
        attack.optimizer == "adam" and attack.initialization == "zero" and attack.lr_schedule == "constant" and
        spectral.sample_rate == 16000 and spectral.cutoff_hz == 6000, "Changed fifty-update paper-budget protocol")


def validate_family(family):
    config = family["config"]
    validate_config(config)
    expected = {FAMILIES[0]: (.1, "no_timestamps"), FAMILIES[1]: (.05, "initial_timestamp")}
    protocol.require(family["name"] in expected, "Unknown fixed family")
    rate, prefix = expected[family["name"]]
    protocol.require(config["attack"]["learning_rate"] == rate and
        config["surrogate"]["nll_reduction"] == "sum" and
        config["surrogate"].get("nll_timestamp_policy", "no_timestamps") == prefix and
        config["surrogate"].get("nll_transcript_policy", "original") == "original",
        "Changed fixed family learning rate, reduction or conditioning")


def prepare(output):
    from finalization_pool import prepare_pool
    import run_decoder_diagnostics as decoder
    protocol.require(not (output / "plan.json").exists(), "Finalization plan already exists")
    previous = protocol.read(DECODER / "plan.json")
    decoder.check(previous)
    decoded = protocol.read(DECODER / "report.json")
    decoder.verify_report(previous, decoded)
    audit = protocol.read(DECODER / "verification.json")
    protocol.require(protocol.read(DECODER / "job.json")["status"] == "complete" and
        protocol.read(DECODER / "results.json")["status"] == "complete" and
        audit["report_sha256"] == digest(DECODER / "report.json") and
        audit["plan_sha256"] == digest(DECODER / "plan.json") and not decoded["mechanism_gate"]["eligible"],
        "Decoder diagnostic must complete with a negative static-suppression screen")
    old = protocol.read(OVERNIGHT / "plan.json")
    protocol.require(old["pools"]["pilot"] == previous["pools"]["pilot"], "Changed inherited pilot identity")
    exclusions = [ROOT / "data/manifests" / name for name in
        ("reconstruction-pilot.jsonl", "pilot.jsonl", "english-validation.jsonl", "english-test-999.jsonl")]
    exclusions.append(Path(old["pools"]["pilot"]["manifest"]))
    index = OVERNIGHT / "data/dev-clean-index.jsonl"
    confirmation = prepare_pool(output / "data/confirmation", index, exclusions, seed=20261019, count=48)
    families = []
    files = [Path(__file__).resolve(), ROOT / "tests/test_finalization_research.py",
        ROOT / "beyondwaveform/trajectory_diagnostics.py", ROOT / "tests/test_trajectory_diagnostics.py",
        ROOT / "scripts/finalization_pool.py", ROOT / "tests/test_finalization_pool.py",
        ROOT / "docs/QUALITY_FINALIZATION.md", index, *exclusions]
    files += [DECODER / name for name in ("plan.json", "report.json", "results.json", "verification.json", "job.json")]
    for name in FAMILIES:
        source = next(row for row in old["candidates"] if row["name"] == name)
        config = copy.deepcopy(source["config"])
        validate_config(config)
        path = output / "configs" / (name + ".json")
        atomic_json(path, config)
        path.chmod(0o444)
        metadata_path = OVERNIGHT / "pilot" / name / "run.json"
        metadata = protocol.read(metadata_path)
        protocol.require(metadata["config"] == config and metadata["features"] == old["feature_artifacts"], "Changed source model contract")
        families.append({"name": name, "config": config, "config_path": str(path),
                         "surrogate_artifacts": metadata["surrogate"]})
        validate_family(families[-1])
        files.extend((path, metadata_path))
    pools = {"pilot": old["pools"]["pilot"], "confirmation": confirmation}
    files += [p for p in (output / "data/confirmation").rglob("*") if p.is_file()]
    hashes = dict(previous["frozen_files"])
    for pool in pools.values():
        files.append(Path(pool["manifest"]))
        for row in read_jsonl(pool["manifest"]):
            protocol.require(digest(row["audio"]) == row["audio_sha256"], "Changed pool audio")
            hashes[str(Path(row["audio"]).resolve())] = row["audio_sha256"]
    hashes.update({str(path.resolve()): digest(path) for path in files})
    plan = {"created_at": now(), "frozen_files": hashes, "runtime": protocol.runtime(), "families": families,
        "pools": pools, "quality": old["quality"], "feature_artifacts": old["feature_artifacts"],
        "modes": list(MODES), "snapshot_steps": list(SNAPSHOT_STEPS), "waveform_tolerance": 2e-5,
        "additional_dependencies": previous.get("additional_dependencies", {}),
        "discord_enabled": False, "decoder_plan": str(DECODER / "plan.json"),
        "directional_gate": {"minimum_utmos_gain": GAIN, "maximum_wer_point_loss": WER_LOSS,
            "minimum_mean_mos_exclusive": 2.5, "terminal_reconstruction_gate": "WER <= clean+5pp and all paper quality anchors"},
        "scope": "Two fixed fresh fifty-update trajectories; last-ten mean versus norm-matched fresh terminal direction. Conditional fresh48 confirmation only; no historical 100-clip validation or 999-clip test. No extra optimization updates per derived output."}
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.require(protocol.runtime() == plan["runtime"], "Changed finalization runtime")
    protocol.require(all(importlib.metadata.version(name) == value for name, value in
        plan.get("additional_dependencies", {}).items()), "Changed tokenizer runtime")
    protocol.require([f["name"] for f in plan["families"]] == list(FAMILIES) and
        plan["modes"] == list(MODES) and plan["snapshot_steps"] == list(SNAPSHOT_STEPS), "Changed fixed research scope")
    protocol.require(plan["directional_gate"]["minimum_utmos_gain"] == GAIN and
        plan["directional_gate"]["maximum_wer_point_loss"] == WER_LOSS and
        plan["directional_gate"]["minimum_mean_mos_exclusive"] == 2.5, "Changed directional screen")
    for path, expected in plan["frozen_files"].items():
        protocol.require(digest(path) == expected, f"Changed frozen finalization input: {path}")
    for family in plan["families"]:
        validate_family(family)
        protocol.require(protocol.read(family["config_path"]) == family["config"], "Changed serialized family config")
    for role, pool in plan["pools"].items():
        rows = read_jsonl(pool["manifest"])
        protocol.require(digest(pool["manifest"]) == pool["sha256"] and [r["id"] for r in rows] == pool["ids"] and
            len(rows) == (24 if role == "pilot" else 48), "Changed pilot/confirmation pool")


def inputs(plan, role, smoke=False):
    protocol.require(not smoke or role == "pilot", "Smoke uses only the first pilot clip")
    pool = plan["pools"][role]
    rows = read_jsonl(pool["manifest"])
    return pool, rows[:1] if smoke else rows


def family_root(output, role, family, smoke=False):
    return output / ("smoke" if smoke else role) / family["name"]


def derived_config(family, mode, smoke=False):
    protocol.require(mode in MODES, "Unknown finalization mode")
    return {"method": "feature_trajectory_finalization", "parent_config": family["config"],
        "mode": mode, "snapshot_steps": list(SNAPSHOT_STEPS), "shared_optimization_updates": 50,
        "additional_optimization_updates": 0, "trace_scope": "terminal optimizer trajectory, independent of returned delta",
        "smoke": smoke}


def identity(plan, family, role, mode, smoke=False):
    pool, rows = inputs(plan, role, smoke)
    return {"config": derived_config(family, mode, smoke), "manifest_sha256": pool["sha256"], "ids": [r["id"] for r in rows]}


def load_models(plan, family, device="cuda", surrogate=True):
    from beyondwaveform.models.feature import load_feature_models
    from beyondwaveform.models.whisper import load_whisper
    encoder, vocoder, features = load_feature_models(family["config"]["features"], device)
    protocol.require(features == plan["feature_artifacts"], "Changed feature/vocoder artifacts")
    adapter = None
    if surrogate:
        adapter, metadata = load_whisper(family["config"]["surrogate"], device)
        protocol.require(metadata == family["surrogate_artifacts"], "Changed Whisper artifacts")
    return encoder, vocoder, adapter


def receipt_check(receipt, source, family):
    expected_seed = (int(family["config"].get("seed", 2026)) + int(hashlib.sha256(source["id"].encode()).hexdigest()[:8], 16)) % 2**32
    protocol.require(receipt["status"] == "complete" and receipt["id"] == source["id"] and
        receipt["text"] == source["text"] and receipt["input_audio_sha256"] == source["audio_sha256"] and
        receipt["config"] == family["config"] and receipt["seed"] == expected_seed and
        list(map(int, receipt["snapshots"])) == list(SNAPSHOT_STEPS), "Changed shared trajectory identity")
    protocol.require(digest(source["audio"]) == source["audio_sha256"], "Changed source audio")
    assets = [*receipt["snapshots"].values(), *receipt["deltas"].values(), *receipt["audio"].values()]
    protocol.require(set(receipt["deltas"]) == set(MODES) and
        all(digest(asset["path"]) == asset["sha256"] for asset in assets), "Changed shared trajectory artifact")


def generate_family(output, plan, family, role, smoke=False, device="cuda"):
    pool, source = inputs(plan, role, smoke)
    root = family_root(output, role, family, smoke)
    encoder, vocoder, surrogate = load_models(plan, family, device)
    attack, spectral = AttackConfig(**family["config"]["attack"]), SpectralConfig(**family["config"]["spectral"])
    produced = {mode: [] for mode in MODES}
    for mode in MODES:
        expected = identity(plan, family, role, mode, smoke)
        path = root / mode / "run.json"
        if path.exists():
            protocol.require(protocol.read(path)["fingerprint"] == config_fingerprint(expected), "Changed derived run identity")
        atomic_json(path, {**expected, "fingerprint": config_fingerprint(expected), "status": "running",
            "roles": [role], "features": plan["feature_artifacts"], "surrogate": family["surrogate_artifacts"],
            "environment": protocol.runtime(), "scope": plan["scope"]})
    for original in source:
        uid = original["id"]
        shared = root / "shared" / "records" / (uid + ".json")
        marker = root / "shared" / (uid + ".incomplete")
        if shared.exists():
            receipt = protocol.read(shared)
            receipt_check(receipt, original, family)
        else:
            protocol.require(not marker.exists(), f"Partial trajectory {uid}; archive it before rerunning")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("Uncommitted fifty-update trajectory; do not mix partial outputs.\n", encoding="utf-8")
            protocol.require(digest(original["audio"]) == original["audio_sha256"], "Changed generation input")
            seed = (int(family["config"].get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % 2**32
            seed_everything(seed)
            x = load_audio(original["audio"]).to(device)
            collection = collect_trajectory(x, original["text"], encoder, vocoder, surrogate, attack, spectral)
            outputs = finalize_deltas(collection.snapshots)
            snapshots, deltas = {}, {}
            for step, tensor in collection.snapshots.items():
                path = root / "shared/snapshots" / uid / f"{step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, path)
                snapshots[str(step)] = {"path": str(path), "sha256": digest(path)}
            audio = {"clean": save_audio(root / "shared/audio/clean" / (uid+".wav"), x),
                     "reconstruction": save_audio(root / "shared/audio/reconstruction" / (uid+".wav"), collection.result.reconstruction)}
            radii = {}
            for mode, tensor in outputs.items():
                path = root / mode / "deltas" / (uid+".pt")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, path)
                deltas[mode] = {"path": str(path), "sha256": digest(path)}
                saved_delta = torch.load(path, map_location=device, weights_only=True)
                radii[mode] = relative_radius(saved_delta, collection.q, attack.eps).item()
                protocol.require(radii[mode] <= .100001 and torch.isfinite(saved_delta).all(), "Returned delta exceeds ball")
                with torch.no_grad(): wave = vocoder(collection.q + saved_delta)
                audio[mode] = save_audio(root / mode / "audio/adversarial" / (uid+".wav"), wave)
            receipt = {"status": "complete", "id": uid, "text": original["text"], "config": family["config"],
                "input_audio_sha256": original["audio_sha256"], "seed": seed, "snapshots": snapshots, "deltas": deltas,
                "audio": audio, "trace": collection.result.trace, "generation_seconds": collection.result.seconds,
                "returned_radii": radii, "shared_optimization_updates": 50, "additional_optimization_updates": 0,
                "trace_scope": "terminal optimizer trajectory"}
            atomic_json(shared, receipt)
            marker.unlink()
        for mode in MODES:
            produced[mode].append({"id": uid, "text": original["text"], "language": original["language"], "role": role,
                "input_audio_sha256": original["audio_sha256"], "seed": receipt["seed"], "trace": receipt["trace"],
                "trace_scope": "terminal optimizer trajectory", "returned_delta_mode": mode,
                "shared_trajectory": {"path": str(shared), "sha256": digest(shared)}, "delta": receipt["deltas"][mode],
                "returned_delta_radius": receipt["returned_radii"][mode],
                "terminal_trace_radius": receipt["trace"][-1]["radius_after_update"],
                "shared_optimization_updates": 50, "additional_optimization_updates": 0,
                "audio": {"clean": receipt["audio"]["clean"], "reconstruction": receipt["audio"]["reconstruction"],
                          "adversarial": receipt["audio"][mode]}})
        print(f"Collected {role} {family['name']} {uid}: one 50-update trajectory, three returned deltas.", flush=True)
    for mode in MODES:
        folder = root / mode
        write_jsonl(folder / "generated.jsonl", produced[mode])
        metadata = protocol.read(folder / "run.json")
        metadata.update(status="complete", count=len(source), generated_sha256=digest(folder / "generated.jsonl"))
        atomic_json(folder / "run.json", metadata)


def evaluate_mode(output, plan, family, role, mode, smoke=False):
    folder = family_root(output, role, family, smoke) / mode
    protocol.configure_scorers(plan["quality"])
    result = evaluate(folder, {"kind": "whisper", **family["config"]["surrogate"]}, folder / "evaluation",
        variants=["clean", "reconstruction", "adversarial"] if mode == "terminal" else ["adversarial"],
        quality=True, artifact_root=str(ROOT / "artifacts"))
    result.update(research_scorer_manifest=plan["quality"], evaluation_environment=protocol.runtime())
    atomic_json(folder / "evaluation/summary.json", result)


def verify_family(output, plan, family, role, smoke=False, device="cuda"):
    pool, source = inputs(plan, role, smoke)
    root = family_root(output, role, family, smoke)
    encoder, vocoder, _ = load_models(plan, family, device, surrogate=False)
    results, records, audits = {}, {}, {mode: [] for mode in MODES}
    for mode in MODES:
        folder = root / mode
        metadata, result = protocol.read(folder / "run.json"), protocol.read(folder / "evaluation/summary.json")
        expected = identity(plan, family, role, mode, smoke)
        rows = read_scores(folder / "evaluation/utterances.jsonl")
        data = read_jsonl(folder / "generated.jsonl")
        variants = ["clean", "reconstruction", "adversarial"] if mode == "terminal" else ["adversarial"]
        protocol.require(metadata["status"] == "complete" and metadata["config"] == expected["config"] and
            metadata["ids"] == expected["ids"] and metadata["count"] == len(source) and metadata["environment"] == plan["runtime"] and
            metadata["fingerprint"] == config_fingerprint(expected) == result["generation_fingerprint"] and
            metadata["features"] == plan["feature_artifacts"] and metadata["surrogate"] == family["surrogate_artifacts"] and
            metadata["manifest_sha256"] == pool["sha256"] and digest(folder / "generated.jsonl") == metadata["generated_sha256"] and
            result["manifest_sha256"] == pool["sha256"] and result["roles"] == [role] and result["evaluation_scope"] == "surrogate" and
            result["target"] == {"kind": "whisper", **family["config"]["surrogate"]} and
            result["target_artifacts"] == family["surrogate_artifacts"] and result["research_scorer_manifest"] == plan["quality"] and
            result["evaluation_environment"] == plan["runtime"] and digest(folder / "evaluation/utterances.jsonl") == result["utterances_sha256"] and
            [r["id"] for r in data] == result["evaluation_ids"] == expected["ids"] and
            {(r["id"], r["variant"]) for r in rows} == {(r["id"], v) for r in source for v in variants},
            "Changed finalization generation/evaluation identity or coverage")
        pairs = {(row["id"], row["variant"]): row for row in rows}
        for original, record in zip(source, data):
            protocol.require(record["text"] == original["text"] and record["language"] == original["language"] and
                record["input_audio_sha256"] == original["audio_sha256"] and
                record["returned_delta_mode"] == mode and record["shared_optimization_updates"] == 50 and
                record["additional_optimization_updates"] == 0 and record["trace_scope"] == "terminal optimizer trajectory",
                "Changed finalization record/reference")
            for variant in variants:
                asset, row = record["audio"][variant], pairs[(original["id"], variant)]
                protocol.require(digest(asset["path"]) == asset["sha256"] == row["audio_sha256"] and row["reference"] == original["text"] and
                    all(row[k] == v for k, v in error_counts(original["text"], row["hypothesis"], original["language"]).items()),
                    "Incorrect WAV/reference/lexical edits")
        protocol.require(set(result["summary"]) == set(variants), "Wrong scored variants")
        for variant in variants:
            subset = [r for r in rows if r["variant"] == variant]
            summary = result["summary"][variant]
            protocol.require(summary["count"] == len(source) and all(summary[k] == v for k, v in aggregate_errors(subset).items()), "Incorrect corpus edits")
            for metric in protocol.METRICS:
                values = [r["quality"][metric] for r in subset]
                protocol.require(all(math.isfinite(v) for v in values) and abs(sum(values)/len(values)-summary["quality"][metric]) < 1e-10, "Incorrect MOS mean")
        results[mode], records[mode] = result, data
    for index, original in enumerate(source):
        terminal = records["terminal"][index]
        asset = terminal["shared_trajectory"]
        protocol.require(digest(asset["path"]) == asset["sha256"], "Changed shared snapshot receipt")
        shared = protocol.read(asset["path"])
        receipt_check(shared, original, family)
        trace = shared["trace"]
        protocol.require(len(trace) == 50 and [r["step"] for r in trace] == list(range(1, 51)) and
            all(math.isfinite(v) for r in trace for v in r.values()) and
            all(r["radius_after_update"] <= .100001 for r in trace), "Invalid terminal trajectory trace")
        snapshots = {int(step): torch.load(v["path"], map_location="cpu", weights_only=True) for step, v in shared["snapshots"].items()}
        expected_deltas = finalize_deltas(snapshots)
        seed_everything(shared["seed"])
        original_audio = load_audio(original["audio"]).to(device)
        with torch.no_grad():
            q = encoder(original_audio).detach()
            reconstruction = vocoder(q)
        control_errors = {}
        for variant, expected_wave in (("clean", original_audio), ("reconstruction", reconstruction)):
            saved_control = load_audio(shared["audio"][variant]["path"]).to(device)
            protocol.require(saved_control.shape == expected_wave.shape, "Control waveform duration mismatch")
            control_errors[variant] = (saved_control-expected_wave).abs().max().item()
            protocol.require(control_errors[variant] <= plan["waveform_tolerance"], "Clean/reconstruction control does not reproduce")
        for step, delta in snapshots.items():
            radius = relative_radius(delta.to(device), q).item()
            protocol.require(delta.shape == q.shape and torch.isfinite(delta).all() and radius <= .100001 and
                abs(radius-trace[step-1]["radius_after_update"]) < 1e-6, "Saved projected iterate violates/mismatches trace")
        for mode in MODES:
            record = records[mode][index]
            protocol.require(record["shared_trajectory"] == asset and record["trace"] == trace and
                record["delta"] == shared["deltas"][mode] and
                record["seed"] == shared["seed"] and
                record["audio"] == {"clean": shared["audio"]["clean"], "reconstruction": shared["audio"]["reconstruction"], "adversarial": shared["audio"][mode]} and
                record["returned_delta_radius"] == shared["returned_radii"][mode] and
                record["terminal_trace_radius"] == trace[-1]["radius_after_update"], "Changed returned/terminal association")
            actual = torch.load(record["delta"]["path"], map_location="cpu", weights_only=True)
            protocol.require(torch.equal(actual, expected_deltas[mode]), "Actual returned delta differs from fixed CPU finalization arithmetic")
            radius = relative_radius(actual.to(device), q).item()
            protocol.require(radius <= .100001 and abs(radius-record["returned_delta_radius"]) < 1e-6, "Incorrect returned delta radius")
            with torch.no_grad(): wave = vocoder(q + actual.to(device))
            saved = load_audio(record["audio"]["adversarial"]["path"]).to(device)
            protocol.require(saved.shape == wave.shape, "Derived waveform duration mismatch")
            difference = (saved-wave).abs().max().item()
            protocol.require(difference <= plan["waveform_tolerance"], "Returned delta does not reproduce actual WAV")
            audits[mode].append({"id": original["id"], "returned_radius": radius,
                "terminal_trace_radius": trace[-1]["radius_after_update"], "max_resynthesis_abs_error": difference,
                "control_resynthesis_abs_errors": control_errors,
                "shared_trajectory_sha256": asset["sha256"], "actual_delta_sha256": record["delta"]["sha256"]})
        average = torch.load(shared["deltas"]["average_41_50"]["path"], map_location="cpu", weights_only=True)
        control = torch.load(shared["deltas"]["terminal_norm_matched"]["path"], map_location="cpu", weights_only=True)
        protocol.require(torch.equal(average, torch.stack([snapshots[s] for s in SNAPSHOT_STEPS]).mean(0)) and
            torch.isclose(average.norm(), control.norm(), rtol=1e-5, atol=1e-7), "Independent mean/norm-match check failed")
        print(f"Independently audited {role} {family['name']} {original['id']}: snapshots, three deltas and WAVs.", flush=True)
    for mode in MODES:
        folder = root / mode
        atomic_json(folder / "verification.json", {"verified_at": now(), "count": len(source), "mode": mode,
            "summary_sha256": digest(folder / "evaluation/summary.json"), "summary": results[mode]["summary"],
            "items": audits[mode], "shared_optimization_updates": 50, "additional_optimization_updates": 0,
            "trace_scope": "terminal optimizer trajectory; returned radius independently recorded"})


def directional_gate(terminal_summary, average, matched):
    gain = average["quality"]["utmos"] - matched["quality"]["utmos"]
    loss = matched["error_percent"] - average["error_percent"]
    checks = {"terminal_reconstruction": reconstruction_gate(terminal_summary),
        "finite_quality": stages.broad_gate(average) and stages.broad_gate(matched),
        "utmos_gain": math.isfinite(gain) and gain >= GAIN,
        "wer_loss": math.isfinite(loss) and loss <= WER_LOSS}
    return {"eligible": all(checks.values()), "checks": checks, "utmos_gain": gain, "wer_point_loss": loss,
            "thresholds": {"minimum_utmos_gain": GAIN, "maximum_wer_point_loss": WER_LOSS, "all_mean_mos_exclusive": 2.5}}


def rows_for_family(output, plan, family, role, smoke=False):
    root = family_root(output, role, family, smoke)
    rows = []
    for mode in MODES:
        folder = root / mode
        audit = protocol.read(folder / "verification.json")
        protocol.require(audit["summary_sha256"] == digest(folder / "evaluation/summary.json"), "Changed audited summary")
        rows.append({"name": family["name"]+"/"+mode, "family": family["name"], "mode": mode,
            "pool": role, "role": role, "smoke": smoke, "cached": False, "run": str(folder),
            **audit["summary"]["adversarial"], "summary_sha256": audit["summary_sha256"],
            "audit_sha256": digest(folder / "verification.json")})
    return rows


def report(output, rows, status, gates, outcome=None):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows, "directional_gates": gates, "outcome": outcome})
    lines = ["# Last-ten-iterate finalization study", "", f"Status: **{status}**.", "",
        "Two fixed configurations each share one fresh 50-update trajectory per clip. Average steps 41–50 is compared with the fresh terminal direction at the same norm. All attacks retain rho=0.1/lambda=1/6000 Hz; no automatic 999-clip test.", "",
        "| Pool | Family | Returned delta | WER % | DNSMOS | NISQA | UTMOS |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        q = row["quality"]
        lines.append(f"| {row['pool']} | {row['family']} | {row['mode']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} |")
    lines += ["", "Confirmation requires >=0.05 UTMOS improvement and <=5 WER-point loss versus the matched terminal control, all mean MOS>2.5, plus acceptable fresh terminal reconstruction. Every family runs on the 24-clip pilot; only passing families expand to the predeclared fresh 48-clip confirmation. No historical 100-clip validation or 999-clip test follows.", ""]
    for name, gate in gates.items():
        lines.append(f"{name}: directional screen **{gate['eligible']}**, UTMOS gain {gate['utmos_gain']:+.4f}, WER-point loss {gate['wer_point_loss']:+.4f}.")
    if outcome:
        lines += ["", f"Outcome: {outcome}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_family(output, plan, family, role, supervisor, smoke=False):
    check(plan)
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output), "--family", family["name"], "--role", role]
    if smoke:
        base.append("--smoke")
    root = family_root(output, role, family, smoke)
    if not all((root / mode / "evaluation/summary.json").exists() for mode in MODES):
        stages.phase(supervisor, f"{role}-{family['name']}-generate", [*base, "--stage", "generate"])
        for mode in MODES:
            if not (root / mode / "evaluation/summary.json").exists():
                stages.phase(supervisor, f"{role}-{family['name']}-{mode}-evaluate", [*base, "--mode", mode, "--stage", "evaluate"])
    stages.phase(supervisor, f"{role}-{family['name']}-verify", [*base, "--stage", "verify"])
    rows = rows_for_family(output, plan, family, role, smoke)
    terminal = protocol.read(root / "terminal/evaluation/summary.json")["summary"]
    return rows, directional_gate(terminal, rows[1], rows[2])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("generate", "evaluate", "verify"))
    parser.add_argument("--role", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args(argv)
    os.environ["BEYONDWAVEFORM_NOTIFY_DISCORD"] = "0"
    output = args.output.resolve()
    plan = prepare(output) if args.prepare else protocol.read(output / "plan.json")
    check(plan)
    if args.prepare or args.check_only:
        print(f"Finalization preflight passed: two families, fixed 50 updates, pilot 24, fresh confirmation 48; {len(plan['frozen_files'])} frozen inputs.", flush=True)
        return
    if args.stage:
        protocol.require(args.family is not None, "Stage needs --family")
        family = next(f for f in plan["families"] if f["name"] == args.family)
        if args.stage == "evaluate":
            protocol.require(args.mode is not None, "Evaluation needs --mode")
            evaluate_mode(output, plan, family, args.role, args.mode, args.smoke)
        else:
            {"generate": generate_family, "verify": verify_family}[args.stage](output, plan, family, args.role, args.smoke)
        return
    if not args.smoke and (output / "results.json").exists():
        protocol.require(protocol.read(output / "results.json")["status"] != "complete", "Study already complete; use --stage verify to re-audit")
    lock_path = output / ("smoke.lock" if args.smoke else "research.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another finalization supervisor is active") from error
        supervisor = Supervisor(output / "smoke" if args.smoke else output)
        rows, gates = [], {}
        try:
            if not args.smoke:
                report(output, rows, "running", gates)
            for family in plan["families"]:
                new, gate = run_family(output, plan, family, "pilot", supervisor, args.smoke)
                rows.extend(new)
                gates[family["name"]] = gate
                if not args.smoke:
                    report(output, rows, "running", gates)
            if args.smoke:
                atomic_json(output / "smoke/results.json", {"status": "complete", "results": rows,
                    "scope": "One first pilot clip per family, integrity integration only; excluded from selection"})
                supervisor.finish(scope="smoke only")
                return
            decision = {"families": [f["name"] for f in plan["families"] if gates[f["name"]]["eligible"]],
                "pilot_evidence": stages.decision_evidence(rows), "gates": gates}
            frozen_decision(output / "confirmation-promotion.json", decision)
            for family in plan["families"]:
                if family["name"] in decision["families"]:
                    new, gate = run_family(output, plan, family, "confirmation", supervisor)
                    rows.extend(new)
                    gates["confirmation/"+family["name"]] = gate
                    report(output, rows, "running", gates)
            check(plan)
            outcome = ("Pilot-selected families evaluated on the fresh 48-clip confirmation pool; no automatic historical validation or 999-clip test."
                       if decision["families"] else "Neither averaging direction passed the paired pilot gate; this family of experiments stopped without confirmation/test expansion.")
            report(output, rows, "complete", gates, outcome)
            supervisor.finish(outcome=outcome)
        except BaseException as error:
            if not args.smoke:
                report(output, rows, "failed", gates, str(error))
            supervisor.fail(error)
            raise


if __name__ == "__main__":
    main()
