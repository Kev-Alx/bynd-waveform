"""The single separately registered R1-native-plus-deletion repair.

Stages are deliberately separate so the shared single GPU can be scheduled by
the parent task. This file never opens confirmation/final audio or loads a
held-out evaluation model. Any failure of this single repair is recorded and
stops the condition without another repair.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "scripts")]

import torch

import execution_runtime as runtime
from r1_sequence import DegenerateRiskError, calibrate_scale, gradient_probe
from r1_deletion_repair import DeletionRepairObjective, RULE


REGISTRATION = runtime.RUN / "r1-native-plus-deletion-registration.json"
CALIBRATION = runtime.RUN / "r1-deletion-calibration"
STAGES = {"pilot": ("r1-deletion-pilot3", True, False),
          "development": ("r1-deletion-development24", False, False),
          "c1-pilot": ("r1-deletion-c1-pilot3", True, True)}


def code_hashes():
    return {str(path): runtime.digest(path) for path in
            (Path(__file__), HERE / "r1_deletion_repair.py", HERE / "r1_sequence.py",
             HERE / "execution_runtime.py")}


def register(plan):
    runtime.check(plan)
    if REGISTRATION.exists():
        raise FileExistsError("Preserve the registered single-deletion repair")
    canonical = runtime.read(runtime.RUN / "r1-calibrated-development24/result.json")
    if canonical["status"] != "negative" or canonical["actual_count"] != 0:
        raise ValueError("Register this one repair only after the documented canonical all-correct stop")
    for path, expected in canonical["bindings"]["code_hashes"].items():
        if runtime.digest(path) != expected:
            raise ValueError("Frozen canonical implementation changed")
    value = {"name": "R1-native-plus-deletion", "created_at_utc": runtime.stamp(),
             "status": "registered_before_repair_outputs", "maximum_repairs": 1,
             "rule": RULE, "same_k4_reference_eos_refresh_optimizer_constraints": True,
             "calibration_clips": [row["id"] for row in runtime.read_jsonl(plan["calibration_manifest"])],
             "scale": "recalibrate once on same three clips; same finite/nonzero/median/bounds guards",
             "canonical_stop_sha256": runtime.digest(runtime.RUN / "r1-calibrated-development24/result.json"),
             "plan_sha256": runtime.digest(runtime.RUN / "plan.json"), "code_hashes": code_hashes(),
             "test_source_sha256": runtime.digest(HERE / "test_r1_deletion_repair.py"),
             "caveat": "The deletion is a synthetic counterfactual, not a posterior beam hypothesis; no novelty or preserved human audio-content claim follows.",
             "target_models_permitted": [], "confirmation_or_final_inference": False,
             "stop": "One repair only; any degeneracy stops it without another candidate or condition"}
    runtime.atomic_json(REGISTRATION, value)
    return value


def prerequisites(plan):
    runtime.check(plan)
    registration = runtime.read(REGISTRATION)
    if registration["code_hashes"] != code_hashes() or registration["rule"] != RULE:
        raise ValueError("Changed preregistered deletion repair source/rule")
    if registration["plan_sha256"] != runtime.digest(runtime.RUN / "plan.json"):
        raise ValueError("Changed registered plan")
    baseline = runtime.RUN / "baseline-preflight/result.json"
    if not baseline.exists() or runtime.read(baseline).get("status") != "passed":
        raise RuntimeError("R1 requires the passed exact 150-update baseline preflight")
    expected = {"k": 4, "refresh_every": 5, "tau": 1, "include_eos": True,
                "learning_rate": .1, "steps": 50}
    if any(plan["r1"].get(key) != value for key, value in expected.items()):
        raise ValueError("Changed registered R1 condition")
    config = plan["base_config"]
    attack = runtime.AttackConfig(**config["attack"])
    if not (attack.rho == .1 and attack.lambda_perc == 1 and attack.alpha_hf == 1
            and attack.initialization == "zero" and attack.temporal and attack.high_frequency):
        raise ValueError("R1 must retain the registered feature/perceptual contract")
    if config["surrogate"]["nll_reduction"] != "sum":
        raise ValueError("R1 calibration requires the historical lexical-sum-NLL units")
    return {"plan_sha256": runtime.digest(runtime.RUN / "plan.json"),
            "baseline_preflight_sha256": runtime.digest(baseline),
            "code_hashes": code_hashes(), "repair_registration_sha256": runtime.digest(REGISTRATION)}


def objective(surrogate):
    return DeletionRepairObjective(surrogate, k=4, refresh_every=5, tau=1., include_eos=True)


def calibration(plan):
    bindings = prerequisites(plan)
    if CALIBRATION.exists():
        raise FileExistsError("Preserve existing or partial R1 calibration evidence")
    CALIBRATION.mkdir(parents=True)
    result = {"status": "running", "stage": "r1-native-plus-deletion-calibration", "source_only": True,
              "target_models_evaluated": [], "created_at_utc": runtime.stamp(),
              "candidate_fallback": RULE, "bindings": bindings, "items": []}
    runtime.atomic_json(CALIBRATION / "result.json", result)
    started = time.perf_counter()
    try:
        with runtime.backend_mode("strict_math"):
            vocoder, surrogate, _, metadata = runtime.models(plan)
            result["source_metadata"] = metadata
            result["backend"] = runtime.backend_manifest()
            torch.cuda.reset_peak_memory_stats()
            for row, q, x in runtime.clips(plan, calibration=True):
                item = {"id": row["id"], "duration": row["duration"],
                        "duration_stratum": plan["calibration_strata"][row["id"]],
                        "status": "running"}
                risk_objective = objective(surrogate)
                clip_started = time.perf_counter()
                delta = torch.zeros_like(q, requires_grad=True)
                risk_result = audio = nll = None
                try:
                    audio = vocoder(q + delta)
                    risk_result = risk_objective.evaluate(audio, row["text"], step=0)
                    item["diagnostics"] = risk_result.diagnostics()
                    nll = surrogate.nll(audio, row["text"])
                    probe = gradient_probe(risk_result.risk, nll, delta, row["id"],
                                           item["duration_stratum"])
                    item["probe"] = probe
                    item["status"] = "valid" if probe["valid"] else "degenerate"
                    if not probe["valid"]:
                        item["reason"] = "Zero/nonfinite source physical-feature risk gradient"
                except DegenerateRiskError as error:
                    item.update(status="degenerate", reason=str(error))
                finally:
                    if risk_objective.candidate_set is not None:
                        item["candidate_set"] = risk_objective.candidate_set.receipt()
                    torch.cuda.synchronize()
                    item["seconds"] = time.perf_counter() - clip_started
                    item["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    item["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    runtime.atomic_json(CALIBRATION / f'{row["id"]}.json', item)
                    result["items"].append(item)
                    runtime.atomic_json(CALIBRATION / "result.json", result)
                    print(f'R1 CALIBRATION {row["id"]}: {item["status"]}', flush=True)
                    del risk_result, nll, audio, delta, risk_objective, q, x
                    gc.collect()
            if all(item["status"] == "valid" for item in result["items"]):
                try:
                    result["calibration"] = calibrate_scale([item["probe"] for item in result["items"]])
                    result["status"] = "passed"
                except DegenerateRiskError as error:
                    result.update(status="negative", reason=str(error))
            else:
                result.update(status="negative", reason="Single registered deletion repair fails a declared candidate/gradient gate; no further repair")
            if len(result["items"]) != 3:
                raise RuntimeError("Calibration did not inspect all three declared duration clips")
            result["seconds"] = time.perf_counter() - started
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            result["completed_at_utc"] = runtime.stamp()
            runtime.atomic_json(CALIBRATION / "result.json", result)
    except BaseException as error:
        result.update(status="failed_infrastructure", reason=str(error), traceback=traceback.format_exc(),
                      seconds=time.perf_counter() - started)
        runtime.atomic_json(CALIBRATION / "result.json", result)
        raise
    return result


def load_calibration(bindings):
    saved = runtime.read(CALIBRATION / "result.json")
    if saved["status"] != "passed":
        raise DegenerateRiskError("Single deletion-repair calibration did not pass; no optimization or further repair")
    if saved["bindings"] != bindings:
        raise ValueError("R1 implementation/registered inputs changed after calibration")
    return saved


def collect_clip(plan, folder, row, q, x, vocoder, surrogate, scale):
    risk_objective = objective(surrogate)
    spectral = runtime.SpectralConfig(**plan["base_config"]["spectral"])
    reference = runtime.ExplicitReflectionReference(q, x, spectral, 1e-8)
    delta = torch.zeros_like(q, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.1)
    trace, refreshes = [], []
    started = time.perf_counter()

    def fail(reason, step):
        failure_delta = folder / "failures" / f'{row["id"]}-delta.pt'
        failure_delta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(delta.detach().cpu(), failure_delta)
        runtime.atomic_json(folder / "failures" / f'{row["id"]}.json',
                            {"id": row["id"], "status": "negative", "reason": reason,
                             "step_zero_based": step, "rule": RULE,
                             "candidate_set": risk_objective.candidate_set.receipt()
                                              if risk_objective.candidate_set is not None else None,
                             "delta": {"path": str(failure_delta), "sha256": runtime.digest(failure_delta),
                                       "value_sha256": runtime.tensor_value_sha256(delta)}})
        raise DegenerateRiskError(reason)

    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        audio = vocoder(q + delta)
        try:
            risk_result = risk_objective.evaluate(audio, row["text"], step)
        except DegenerateRiskError as error:
            fail(str(error), step)
        if risk_result.candidate_set.refreshed_at == step:
            candidate_path = folder / "candidates" / row["id"] / f"refresh-{step:03d}.json"
            runtime.atomic_json(candidate_path, risk_result.candidate_set.receipt())
            refreshes.append({"step": step, "path": str(candidate_path), "sha256": runtime.digest(candidate_path)})
        tv, hf = reference.terms(q + delta, audio)
        loss = -scale * risk_result.risk + tv + hf
        if not torch.isfinite(loss):
            fail("Nonfinite R1 full objective", step)
        risk_gradient, = torch.autograd.grad(risk_result.risk, delta, retain_graph=True)
        risk_norm = torch.linalg.vector_norm(risk_gradient.double()).item()
        if not math.isfinite(risk_norm) or risk_norm == 0:
            fail("Flat/nonfinite physical-feature risk gradient during optimization", step)
        gradient, = torch.autograd.grad(loss, delta)
        if not torch.isfinite(gradient).all():
            fail("Nonfinite R1 full-objective gradient", step)
        diagnostic_nll = None
        if step % 5 == 0 or step == 49:
            with torch.no_grad():
                diagnostic_nll = surrogate.nll(audio.detach(), row["text"]).item()
        delta.grad = gradient
        optimizer.step()
        runtime.project_relative_ball_(delta, q, .1)
        radius = runtime.relative_radius(delta, q).item()
        if not torch.isfinite(delta).all() or radius > .100001:
            fail("Invalid projected R1 iterate", step)
        trace.append({"step": step + 1, "candidate_step_zero_based": step,
                      "loss": loss.detach().item(), "nll": diagnostic_nll,
                      "tv": tv.detach().item(), "hf": hf.detach().item(),
                      "risk_gradient_norm": risk_norm,
                      "total_gradient_norm": torch.linalg.vector_norm(gradient.double()).item(),
                      "gradient_value_sha256": runtime.tensor_value_sha256(gradient),
                      "delta_value_sha256": runtime.tensor_value_sha256(delta),
                      "radius": radius, **risk_result.diagnostics()})
        runtime.atomic_json(folder / "partial-traces" / f'{row["id"]}.json',
                            {"id": row["id"], "trace": trace, "refreshes": refreshes})
        if step in (0, 9, 24, 49):
            print(f'R1 {folder.name} {row["id"]}: update {step + 1}/50 risk={trace[-1]["risk"]:.6f}', flush=True)
        del risk_result, audio, loss, tv, hf, risk_gradient, gradient
    with torch.no_grad():
        final_audio = vocoder(q + delta)
        source_hypothesis = surrogate.transcribe(final_audio)
    counts = runtime.error_counts(row["text"], source_hypothesis)
    optimizer_path = folder / "optimizer" / f'{row["id"]}.pt'
    optimizer_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.cuda.synchronize()
    result = runtime.save_feature_result(folder.name, row, q, x, delta, trace, vocoder,
                {"actual_optimization_updates": 50, "method": "R1-native-plus-deletion (synthetic counterfactual candidate)",
                 "scale": scale, "candidate_refreshes": refreshes,
                 "source_hypothesis": source_hypothesis, "source_error_counts": counts,
                 "source_empty": not counts["hypothesis_normalized"],
                 "optimizer": {"path": str(optimizer_path), "sha256": runtime.digest(optimizer_path),
                               "value_receipt": runtime.tensor_tree(optimizer.state_dict())},
                 "seconds": time.perf_counter() - started})
    return result


def generate(plan, stage):
    bindings = prerequisites(plan)
    calibrated = load_calibration(bindings)
    name, use_three, c1 = STAGES[stage]
    if stage in ("development", "c1-pilot"):
        pilot = runtime.read(runtime.RUN / STAGES["pilot"][0] / "result.json")
        if pilot["status"] != "completed":
            raise DegenerateRiskError("Calibrated three-clip mechanism pilot must pass first")
    folder = runtime.RUN / name
    if folder.exists():
        raise FileExistsError("Preserve existing or partial R1 optimization evidence")
    folder.mkdir(parents=True)
    scale = 1. if c1 else calibrated["calibration"]["scale"]
    result = {"status": "running", "stage": stage, "method": name, "scale": scale,
              "source_only": True, "target_models_evaluated": [], "candidate_fallback": RULE,
              "created_at_utc": runtime.stamp(), "bindings": bindings,
              "calibration_sha256": runtime.digest(CALIBRATION / "result.json"),
              "items": [], "empty_rate_limit": .05}
    runtime.atomic_json(folder / "result.json", result)
    records, started = [], time.perf_counter()
    planned_count = 3 if use_three else 24
    try:
        with runtime.backend_mode("strict_math"):
            vocoder, surrogate, _, metadata = runtime.models(plan)
            result["source_metadata"], result["backend"] = metadata, runtime.backend_manifest()
            torch.cuda.reset_peak_memory_stats()
            for row, q, x in runtime.clips(plan, calibration=use_three):
                try:
                    record = collect_clip(plan, folder, row, q, x, vocoder, surrogate, scale)
                except DegenerateRiskError as error:
                    result.update(status="negative", reason=str(error), failed_clip=row["id"])
                    break
                records.append(record)
                runtime.write_jsonl(folder / "generated.jsonl", records)
                result["items"].append({"id": row["id"], "seconds": record["seconds"],
                                        "source_empty": record["source_empty"],
                                        "source_error_counts": record["source_error_counts"]})
                result["empty_count"] = sum(record["source_empty"] for record in records)
                # Use the complete declared cohort denominator; once this lower
                # bound exceeds 5%, additional clips cannot rescue the gate.
                if result["empty_count"] / planned_count > .05:
                    result.update(status="negative", reason="Source empty-output rate exceeds 5% of declared cohort")
                    break
                runtime.atomic_json(folder / "result.json", result)
                del q, x
                gc.collect()
            if result["status"] == "running":
                if len(records) != planned_count:
                    raise RuntimeError("Incomplete registered R1 cohort")
                result["status"] = "completed"
            result.update(seconds=time.perf_counter() - started, completed_at_utc=runtime.stamp(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          actual_count=len(records), planned_count=planned_count)
            if records:
                result["source_empty_rate_observed"] = result["empty_count"] / len(records)
                result["source_error_counts"] = runtime.aggregate_errors([record["source_error_counts"] for record in records])
            runtime.atomic_json(folder / "result.json", result)
    except BaseException as error:
        result.update(status="failed_infrastructure", reason=str(error), traceback=traceback.format_exc(),
                      seconds=time.perf_counter() - started)
        runtime.atomic_json(folder / "result.json", result)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "calibration", *STAGES))
    args = parser.parse_args()
    plan = runtime.read(runtime.RUN / "plan.json")
    if args.stage == "register":
        result = register(plan)
    else:
        result = calibration(plan) if args.stage == "calibration" else generate(plan, args.stage)
    print(f'R1 {args.stage}: {result["status"]}', flush=True)


if __name__ == "__main__":
    main()
