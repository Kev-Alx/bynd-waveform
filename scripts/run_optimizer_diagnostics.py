"""Supervise fixed numerical-repeatability and Adam-contribution diagnostics."""
from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.io import atomic_json, digest
from parallel_jobs import Supervisor, now
import run_finalization_research as previous_protocol
import run_quality_research as protocol
import run_quality_refinement as stages

DEFAULT = ROOT / "reproduction-runs/optimizer-mechanism-diagnostics"
PREVIOUS = ROOT / "reproduction-runs/paper-quality-finalization"
PARENT = ROOT / "reproduction-runs/feature-delta-diagnostics/plan.json"
MODES = ("auto", "math", "strict_math")


def prepare(output):
    protocol.require(not (output / "plan.json").exists(), "Plan already exists")
    previous = protocol.read(PREVIOUS / "plan.json")
    previous_protocol.check(previous)
    protocol.require(protocol.read(PREVIOUS / "job.json")["status"] == "complete" and
        protocol.read(PREVIOUS / "confirmation-promotion.json")["families"] == [],
        "Review the previous study before changing research direction")
    parent = protocol.read(PARENT)
    frozen = dict(previous["frozen_files"])
    files = [Path(__file__).resolve(), ROOT / "scripts/probe_numerical_repeatability.py",
        ROOT / "scripts/probe_adam_contributions.py", ROOT / "beyondwaveform/adam_diagnostics.py",
        ROOT / "tests/test_numerical_repeatability.py", ROOT / "tests/test_adam_diagnostics.py",
        ROOT / "tests/test_adam_contributions.py", ROOT / "tests/test_optimizer_diagnostics.py",
        ROOT / "docs/QUALITY_OPTIMIZER_DIAGNOSTICS.md", PARENT]
    files.extend(PREVIOUS / name for name in ("plan.json", "results.json", "report.md", "job.json", "confirmation-promotion.json"))
    for module in ("torch.optim.adam", "torch.optim.optimizer", "torch.nn.functional", "whisper.model", "whisper.audio"):
        files.append(Path(inspect.getfile(importlib.import_module(module))))
    from probe_numerical_repeatability import source_paths
    files.extend(source_paths())
    frozen.update({str(path.resolve()): digest(path) for path in files})
    plan = {"created_at": now(), "frozen_files": frozen, "runtime": protocol.runtime(),
        "parent_plan": str(PARENT), "parent_plan_sha256": digest(PARENT),
        "ids": parent["pools"]["pilot"]["ids"][:3], "count": 3, "repeats": 3,
        "modes": list(MODES), "adam_capture_steps": [1, 10, 25, 50],
        "additional_dependencies": previous.get("additional_dependencies", {}), "discord_enabled": False,
        "scope": "Fixed development diagnostics, no ASR/MOS inference or validation/test expansion. Auto/math gradients repeated at fixed zero/saved deltas; strict failures labeled unsupported. Three fresh full-objective 50-update trajectories with moment-matched counterfactual branches at four fixed steps; no replacement loss or optimizer automatically selected."}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    return plan


def check(plan):
    import importlib.metadata
    protocol.require(plan["count"] == 3 and plan["repeats"] == 3 and plan["modes"] == list(MODES) and
        plan["adam_capture_steps"] == [1, 10, 25, 50], "Changed diagnostic scope")
    protocol.require(protocol.runtime() == plan["runtime"] and all(importlib.metadata.version(k) == v
        for k, v in plan["additional_dependencies"].items()), "Changed runtime")
    for path, expected in plan["frozen_files"].items():
        protocol.require(digest(path) == expected, f"Changed frozen optimizer-diagnostic input: {path}")
    protocol.require(digest(plan["parent_plan"]) == plan["parent_plan_sha256"] and
        protocol.read(plan["parent_plan"])["pools"]["pilot"]["ids"][:3] == plan["ids"], "Changed parent/selection")


def report(output, rows, status):
    atomic_json(output / "results.json", {"status": status, "updated_at": now(), "results": rows,
        "scope": "Numerical and local optimizer diagnostics; no new WER, MOS or candidate promotion."})
    lines = ["# Numerical and optimizer diagnostics", "", f"Status: **{status}**.", "",
        "Three fixed development clips; original objective, radius and model artifacts. No new WER/MOS scoring or validation/test use.", "",
        "| Diagnostic | Status | Bitwise fixed-point gradient agreement | Evidence |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['status']} | {row.get('numerical_bitwise_agreement', 'N/A')} | [JSON]({row['report']}) |")
    lines += ["", "Strict deterministic mode may report unsupported rather than silently substituting a frontend. Repeated gradients at fixed inputs do not establish complete trajectory repeatability. Adam counterfactuals hold moment history fixed and do not constitute full loss ablations or quality estimates.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_result(plan, result, mode):
    import torch
    parent = protocol.read(plan["parent_plan"])
    cache_path = Path(plan["parent_plan"]).parent / "parent-audits/pilot.json"
    protocol.require(result["parent_plan"] == plan["parent_plan"] and
        result["parent_plan_sha256"] == plan["parent_plan_sha256"] and
        result["parent_cache_audit_sha256"] == digest(cache_path) and result["runtime"] == plan["runtime"],
        "Changed diagnostic parent/runtime binding")
    for key in ("source_hashes", "checked_asset_hashes"):
        protocol.require(result[key] and all(digest(path) == wanted == plan["frozen_files"].get(path)
            for path, wanted in result[key].items()), "Changed measured input/source receipt")
    expected_surrogate = protocol.read(Path(parent["parents"]["pilot"]["run"]) / "run.json")["surrogate"]
    protocol.require(result["surrogate"] == expected_surrogate, "Changed diagnostic surrogate")
    if mode != "adam":
        from probe_numerical_repeatability import compare_gradients, tensor_value_sha256, known_determinism_failure
        protocol.require(result["mode"] == mode and result["requested_count"] == 3 and result["repeats_per_state"] == 3 and
            [item["id"] for item in result["items"]] == plan["ids"][:len(result["items"])], "Changed repeatability selection")
        flags = result["backend"]
        if mode != "auto":
            protocol.require(flags["math_sdp_enabled"] and not flags["flash_sdp_enabled"] and
                not flags["efficient_sdp_enabled"] and not flags["cudnn_sdp_enabled"], "Math backend was not enforced")
        if result["status"] == "unsupported":
            protocol.require(mode == "strict_math" and flags["deterministic_algorithms"] and
                known_determinism_failure(RuntimeError(result["unsupported"]["message"])), "Invalid unsupported diagnostic")
        else:
            protocol.require(result["completed_clip_count"] == 3 and len(result["items"]) == 3 and
                all(set(item["states"]) == {"zero_delta", "saved_parent_delta"} for item in result["items"]),
                "Incomplete repeated-gradient measurements")
        agreements = []
        for item in result["items"]:
            for state in item["states"].values():
                protocol.require(len(state["repeats"]) == 3 and len(state["pairs"]) == 3, "Wrong repeat coverage")
                tensors = []
                for repeat in state["repeats"]:
                    asset = repeat["gradient_artifact"]
                    protocol.require(digest(asset["path"]) == asset["file_sha256"], "Changed saved gradient")
                    tensor = torch.load(asset["path"], map_location="cpu", weights_only=True)
                    protocol.require(torch.isfinite(tensor).all().item() and tensor_value_sha256(tensor) ==
                        asset["value_sha256"] == repeat["gradient_value_sha256"], "Incorrect gradient value hash")
                    tensors.append(tensor)
                for pair in state["pairs"]:
                    left, right = pair["left_repeat"], pair["right_repeat"]
                    rebuilt = compare_gradients(tensors[left], tensors[right])
                    loss_equal = state["repeats"][left]["loss_value_sha256"] == state["repeats"][right]["loss_value_sha256"]
                    protocol.require(rebuilt == pair["gradient"] and loss_equal == pair["loss_values_bitwise_equal"] and
                        pair["numerical_bitwise_agreement"] == (rebuilt["raw_value_sha256_equal"] and loss_equal), "Incorrect repeat comparison")
                agreement = all(pair["numerical_bitwise_agreement"] for pair in state["pairs"])
                protocol.require(state["numerical_bitwise_agreement"] == agreement, "Incorrect state repeatability")
                agreements.append(agreement)
        if result["status"] == "complete":
            protocol.require(result["numerical_bitwise_agreement"] == all(agreements), "Incorrect overall repeatability")
    else:
        protocol.require(result["count"] == 3 and [item["id"] for item in result["items"]] == plan["ids"] and
            result["capture_steps"] == plan["adam_capture_steps"], "Changed Adam selection")
        for item in result["items"]:
            protocol.require(item["actual_optimization_updates"] == 50 and item["counterfactual_trajectory_continuations"] == 0 and
                len(item["trace"]) == 50 and [r["step"] for r in item["captures"]] == plan["adam_capture_steps"], "Changed actual-update count")
            protocol.require(all(math.isfinite(value) for row in item["trace"] for value in row.values()) and
                all(row["radius_after_update"] <= .100001 for row in item["trace"]), "Invalid full trajectory")
            for capture in item["captures"]:
                protocol.require(digest(capture["evidence_path"]) == capture["evidence_sha256"], "Changed Adam evidence")
                evidence = torch.load(capture["evidence_path"], map_location="cpu", weights_only=True)
                protocol.require(torch.equal(evidence["actual_delta_after"], evidence["branches"]["full"]["delta_projected"]) and
                    capture["full_clone_max_abs_error"] == 0, "Counterfactual full branch did not match actual step")


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
        print(f"Optimizer diagnostic preflight passed: {len(plan['frozen_files'])} frozen inputs.", flush=True)
        return
    supervisor, rows = Supervisor(output), []
    try:
        report(output, rows, "running")
        for mode in MODES:
            path = output / ("repeatability-" + mode + ".json")
            protocol.require(not path.exists(), "Existing diagnostic needs review, not overwrite")
            command = [sys.executable, "-u", str(ROOT / "scripts/probe_numerical_repeatability.py"),
                "--parent-plan", plan["parent_plan"], "--output", str(path), "--mode", mode, "--count", "3"]
            if mode == "strict_math":
                command = ["/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command]
            stages.phase(supervisor, "fixed-gradient-" + mode, command)
            check(plan)
            result = protocol.read(path)
            protocol.require(result["status"] in {"complete", "unsupported"}, "Incomplete numerical diagnostic")
            verify_result(plan, result, mode)
            rows.append({"name": mode, "status": result["status"], "report": str(path),
                "report_sha256": digest(path), "numerical_bitwise_agreement": result.get("numerical_bitwise_agreement")})
            report(output, rows, "running")
        path = output / "adam-contributions.json"
        protocol.require(not path.exists(), "Existing Adam report needs review")
        stages.phase(supervisor, "actual-adam-contributions", [sys.executable, "-u",
            str(ROOT / "scripts/probe_adam_contributions.py"), "--parent-plan", plan["parent_plan"],
            "--output", str(path), "--count", "3"])
        check(plan)
        result = protocol.read(path)
        protocol.require(result["status"] == "complete", "Incomplete Adam diagnostic")
        verify_result(plan, result, "adam")
        rows.append({"name": "adam-contributions", "status": "complete", "report": str(path), "report_sha256": digest(path)})
        report(output, rows, "complete")
        supervisor.finish(scope=plan["scope"])
    except BaseException as error:
        report(output, rows, "failed")
        supervisor.fail(error)
        raise


if __name__ == "__main__":
    main()
