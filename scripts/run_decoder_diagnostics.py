"""Freeze and supervise a saved-audio diagnostic of Whisper loss/decoder alignment.

This stage does not optimize an attack, change recognition, score MOS again,
or automatically expand to validation/test data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.io import atomic_json, digest, read_jsonl
from parallel_jobs import Supervisor, now
import run_feature_diagnostics as parent_protocol
import run_quality_research as protocol
import run_quality_refinement as stages

DEFAULT = ROOT / "reproduction-runs/decoder-alignment-diagnostics"
PARENT = ROOT / "reproduction-runs/feature-delta-diagnostics"


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Plan already exists")
    previous = protocol.read(PARENT / "plan.json")
    parent_protocol.check(previous)
    protocol.require(protocol.read(PARENT / "job.json")["status"] == "complete", "Parent not complete")
    protocol.require(protocol.read(PARENT / "validation-promotion.json")["names"] == [],
                     "Review the positive smoothing decision before changing direction")
    frozen = dict(previous["frozen_files"])
    files = [Path(__file__).resolve(), ROOT / "scripts/probe_decoder_alignment.py",
             ROOT / "tests/test_decoder_alignment_probe.py", ROOT / "tests/test_decoder_diagnostics.py",
             ROOT / "docs/QUALITY_DECODER_ALIGNMENT.md",
             PARENT / "plan.json", PARENT / "job.json", PARENT / "results.json",
             PARENT / "report.md", PARENT / "validation-promotion.json", PARENT / "gradient-probe.json",
             PARENT / "parent-audits/pilot.json"]
    for module_name in ("whisper.decoding", "whisper.tokenizer", "whisper.transcribe", "whisper.model", "whisper.audio"):
        files.append(Path(inspect.getfile(importlib.import_module(module_name))))
    tokenizer_module = importlib.import_module("whisper.tokenizer")
    files.append(Path(tokenizer_module.__file__).parent / "assets/multilingual.tiktoken")
    from probe_decoder_alignment import source_paths
    files.extend(source_paths())
    parent_run = Path(previous["parents"]["pilot"]["run"])
    for row in read_jsonl(parent_run / "generated.jsonl"):
        files.extend(Path(asset["path"]) for asset in row["audio"].values())
    files.extend(Path(item["q_path"]) for item in protocol.read(PARENT / "parent-audits/pilot.json")["items"])
    frozen.update({str(path.resolve()): digest(path) for path in files})
    from whisper.decoding import DecodingOptions
    surrogate = previous["parents"]["pilot"]["config"]["surrogate"]
    decoding_options = asdict(DecodingOptions(language=surrogate["language"], task="transcribe", fp16=False,
                                             **surrogate["decoding"]))
    plan = {"created_at": now(), "parent_plan": str(PARENT / "plan.json"),
        "parent_plan_sha256": digest(PARENT / "plan.json"), "frozen_files": frozen,
        "runtime": protocol.runtime(), "count": 24, "pools": {"pilot": previous["pools"]["pilot"]},
        "variants": ["clean", "reconstruction", "adversarial"],
        "prefixes": ["no_timestamps", "initial_timestamp"], "discord_enabled": False,
        "native_decoding_options": decoding_options,
        "additional_dependencies": {"tiktoken": importlib.metadata.version("tiktoken")},
        "mechanism_screen": {"minimum_static_gap_share_of_nll_increase": 0.10,
            "minimum_adversarial_median_suppressed_mass": 0.10,
            "minimum_statically_allowed_reference_coverage": 0.95,
            "positive_nll_increase_guard": 1e-8,
            "scope": "Actual parent attack prefix, matched statically valid lexical positions. A positive screen motivates a separately frozen study; it is not evidence of better WER or quality."},
        "scope": "Saved pilot audio only; no new optimization, recognition or MOS inference, no validation/test expansion. Native ordered dynamic filters are diagnostic and do not define a replacement objective."}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    protocol.require(protocol.runtime() == plan["runtime"], "Runtime changed")
    protocol.require(all(importlib.metadata.version(name) == value for name, value in
                         plan.get("additional_dependencies", {}).items()), "Tokenizer runtime changed")
    protocol.require(plan["count"] == 24 and plan["variants"] == ["clean", "reconstruction", "adversarial"] and
        plan["prefixes"] == ["no_timestamps", "initial_timestamp"], "Changed diagnostic scope")
    for path, expected in plan["frozen_files"].items():
        protocol.require(digest(path) == expected, f"Changed frozen diagnostic input: {path}")
    parent_protocol.check(protocol.read(plan["parent_plan"]))


def report(output, result):
    lines = ["# Whisper decoder alignment diagnostics", "", "Status: **complete**.", "",
        "Measured frozen clean/reconstruction/adversarial pilot audio under both lexical teacher-forcing prefixes. No new attack, recognition, MOS scoring or validation/test run was performed.", "",
        "| Teacher prefix | Audio | Mean lexical NLL per token | Mean static gap (allowed targets) | Median suppressed mass (allowed targets) |",
        "|---|---|---:|---:|---:|"]
    for view, entry in result["summary"]["views"].items():
        for variant, state in entry["states"].items():
            row = state["token_micro"]
            lines.append(f"| {view} | {variant} | {row['raw_nll_mean']:.6f} | {row['static_gap_mean_valid']:.6f} | {row['static_suppressed_mass_median_valid']:.6f} |")
    evidence = result["mechanism_gate"]["evidence"]
    share = evidence["static_gap_share_of_raw_nll_increase"]
    share_text = f"{100*share:.4f}%" if share is not None else "undefined (no positive NLL increase)"
    lines += ["", f"Predeclared mechanism screen: **{result['mechanism_gate']['eligible']}**. On matched statically valid attack-prefix targets, static suppression accounts for **{share_text}** of the adversarial-minus-reconstruction NLL increase. Static target coverage is **{100*evidence['static_valid_coverage']:.2f}%**.", "",
        "[Numerical evidence](" + str(output / "report.json") + ") contains token-micro and utterance-macro summaries, exact native suppression options, forbidden targets, and separate dynamic-filter reachability diagnostics.", "",
        "The predeclared mechanism screen requires static suppression to explain at least 10% of the adversarial-minus-reconstruction NLL increase, adversarial median suppressed mass at least 0.10, and at least 95% reference-token coverage. It is measured on matched, statically valid tokens under the actual attack prefix. Passing this screen would justify a separately frozen experiment, without establishing WER or quality improvement.", "",
        "Dynamic timestamp/blank filters are reported separately; reference paths can become unreachable. Their locally masked losses cannot be substituted for the static matched-position decomposition or for free-decoding WER.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_report(plan, result):
    from probe_decoder_alignment import aggregate, mechanism_gate, source_paths
    parent = protocol.read(plan["parent_plan"])
    parent_run = Path(parent["parents"]["pilot"]["run"])
    parent_metadata = protocol.read(parent_run / "run.json")
    protocol.require(result["status"] == "complete" and result["count"] == plan["count"] and
        result["plan_sha256"] == plan["parent_plan_sha256"] and result["plan"] == plan["parent_plan"] and
        result["parent_run"] == parent["parents"]["pilot"]["run"] and
        result["surrogate"] == parent_metadata["surrogate"] and
        result["runtime"] == plan["runtime"] and
        result["effective_native_recognition_options"] == plan["native_decoding_options"] and
        result["variant_order"] == plan["variants"] and result["teacher_views"] == ["attack", "native_timestamp0"] and
        [item["id"] for item in result["items"]] == plan["pools"]["pilot"]["ids"],
        "Changed completed diagnostic identity or coverage")
    protocol.require(set(result["source_hashes"]) == {str(path) for path in source_paths()} and
        all(digest(path) == expected == plan["frozen_files"].get(path)
        for path, expected in result["source_hashes"].items()), "Changed diagnostic source receipt")
    expected_assets = {}
    records = read_jsonl(parent_run / "generated.jsonl")
    originals = read_jsonl(plan["pools"]["pilot"]["manifest"])
    cache = protocol.read(Path(plan["parent_plan"]).parent / "parent-audits/pilot.json")
    for original, record, cached in zip(originals, records, cache["items"]):
        expected_assets[original["audio"]] = original["audio_sha256"]
        expected_assets[cached["q_path"]] = cached["q_sha256"]
        expected_assets[str(parent_run / (original["id"] + ".delta.pt"))] = parent["parents"]["pilot"]["delta_hashes"][original["id"]]
        expected_assets.update({asset["path"]: asset["sha256"] for asset in record["audio"].values()})
    protocol.require(result["checked_asset_hashes"] == expected_assets and
        all(digest(path) == expected for path, expected in expected_assets.items()),
        "Changed measured artifact")
    protocol.require(result["parent_cache_audit_sha256"] == digest(Path(plan["parent_plan"]).parent / "parent-audits/pilot.json"),
        "Changed cached parent evidence")
    rebuilt = aggregate(result["items"])
    gate = mechanism_gate(rebuilt)
    protocol.require(rebuilt == result["summary"] and gate == result["mechanism_gate"], "Incorrect diagnostic aggregation or gate")
    screen = plan["mechanism_screen"]
    expected = {"minimum_static_gap_share": screen["minimum_static_gap_share_of_nll_increase"],
        "minimum_adversarial_token_median_suppressed_mass": screen["minimum_adversarial_median_suppressed_mass"],
        "minimum_static_valid_coverage": screen["minimum_statically_allowed_reference_coverage"],
        "minimum_positive_nll_increase": screen["positive_nll_increase_guard"]}
    protocol.require(gate["thresholds"] == expected, "Mechanism screen differs from frozen thresholds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.environ["BEYONDWAVEFORM_NOTIFY_DISCORD"] = "0"
    output = args.output.resolve()
    plan = prepare(output) if args.prepare else protocol.read(output / "plan.json")
    check(plan)
    if args.prepare or args.check_only:
        print(f"Decoder diagnostic preflight passed; {len(plan['frozen_files'])} frozen inputs.", flush=True)
        return
    supervisor = Supervisor(output)
    try:
        target = output / "report.json"
        protocol.require(not target.exists(), "Completed report exists; verify it instead of overwriting")
        stages.phase(supervisor, "pilot-decoder-alignment", [sys.executable, "-u",
            str(ROOT / "scripts/probe_decoder_alignment.py"), "--plan", plan["parent_plan"],
            "--output", str(target), "--count", str(plan["count"]),
            "--progress", str(output / "measurement-progress.json")])
        check(plan)
        result = protocol.read(target)
        verify_report(plan, result)
        atomic_json(output / "verification.json", {"verified_at": now(), "count": result["count"],
            "report_sha256": digest(target), "plan_sha256": digest(output / "plan.json"),
            "frozen_input_count": len(plan["frozen_files"]),
            "checked_asset_count": len(result["checked_asset_hashes"]),
            "mechanism_gate": result["mechanism_gate"]})
        report(output, result)
        atomic_json(output / "results.json", {"status": "complete", "updated_at": now(), "results": [],
            "diagnostic_report": str(target), "diagnostic_report_sha256": digest(target),
            "mechanism_gate": result["mechanism_gate"], "scope": plan["scope"]})
        supervisor.finish(report_sha256=digest(target), scope=plan["scope"])
    except BaseException as error:
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
