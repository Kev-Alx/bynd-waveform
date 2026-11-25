"""Registered source-only R2 execution, isolated under /workspace/improvement.

Register before generation; generation and saved-waveform evaluation are
separate processes. GPU work is only started by an explicit generate/evaluate
invocation. A smoke3 run uses the registered duration-stratified development
clips. It cannot replace the complete 24-clip canonical condition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import torch

try:
    from improvement import execution_runtime as rt
    from improvement.r2_geometry import optimize_local_neighbors
except ModuleNotFoundError:
    import execution_runtime as rt
    from r2_geometry import optimize_local_neighbors


CANONICAL = "r2-canonical-zero-uniform"
REPAIR = "r2-gradient-logit-bootstrap-repair"
STALL_STATUSES = {"uniform_start_stalled_with_feasible_descent"}


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", name):
        raise ValueError("Run name must be a single safe directory name")
    return name


def _rows(plan, smoke3=False):
    return rt.read_jsonl(plan["calibration_manifest"] if smoke3 else plan["development_manifest"])


def _baseline_records():
    return {row["id"]: row for row in rt.read_jsonl(rt.BASE / "generated.jsonl")}


def _sha_binding(path, expected=None):
    path = Path(path)
    actual = rt.digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Changed registered input: {path}")
    return {"path": str(path), "sha256": actual}


def register(plan, name, smoke3=False, repair_ids=(), canonical_name=None):
    """Freeze condition, exact cohort, source hashes and physical controls."""
    rt.check(plan)
    name = safe_name(name)
    rows = _rows(plan, smoke3)
    if len(rows) != (3 if smoke3 else 24):
        raise ValueError("Expected the complete registered development24 or calibration3 cohort")
    repair_ids = tuple(repair_ids)
    canonical_bindings = {}
    if repair_ids:
        if len(set(repair_ids)) != len(repair_ids) or not canonical_name:
            raise ValueError("Repair needs explicit unique IDs and a completed canonical run")
        canonical_name = safe_name(canonical_name)
        canonical_folder = rt.RUN / canonical_name
        summary = rt.read(canonical_folder / "generation-summary.json")
        if summary.get("status") != "completed_all_registered" or summary.get("condition") != CANONICAL:
            raise ValueError("Repair may follow only a completed canonical condition")
        canonical_registration = rt.read(canonical_folder / "registration.json")
        if canonical_registration["smoke3"] != smoke3:
            raise ValueError("Repair must name a canonical run of the same registered cohort type")
        canonical_records = {r["id"]: r for r in rt.read_jsonl(canonical_folder / "generated.jsonl")}
        for uid in repair_ids:
            record = canonical_records.get(uid)
            if not record or record.get("r2_status") not in STALL_STATUSES:
                raise ValueError(f"Repair ID has no recorded actual canonical clean-start stall: {uid}")
        rows = [row for row in rows if row["id"] in set(repair_ids)]
        if len(rows) != len(repair_ids):
            raise ValueError("Repair IDs must all belong to the registered development cohort")
        canonical_bindings = {
            "canonical_name": canonical_name,
            "registration": _sha_binding(canonical_folder / "registration.json"),
            "generation_summary": _sha_binding(canonical_folder / "generation-summary.json"),
            "generated": _sha_binding(canonical_folder / "generated.jsonl"),
            "selection": "Human-explicit IDs with already recorded canonical stalls; exploratory subset, no replacement or winner selection.",
        }
    elif canonical_name:
        raise ValueError("canonical-name is only meaningful with explicit repair IDs")
    cache = {r["id"]: r for r in rt.read(plan["cache_audit"])["items"]}
    baseline = _baseline_records()
    bindings = []
    for row in rows:
        old, cached = baseline[row["id"]], cache[row["id"]]
        if old["input_audio_sha256"] != row["audio_sha256"]:
            raise ValueError("R2 and frozen baseline do not bind the same original audio")
        delta = old["delta"]
        bindings.append({"id": row["id"], "text": row["text"],
                         "original_audio": _sha_binding(row["audio"], row["audio_sha256"]),
                         "clean_features": _sha_binding(cached["q_path"], cached["q_sha256"]),
                         "baseline_delta": _sha_binding(delta["path"], delta.get("file_sha256", delta.get("sha256"))),
                         "baseline_delta_value_sha256": delta["value_sha256"],
                         "baseline_adversarial": _sha_binding(old["audio"]["adversarial"]["path"], old["audio"]["adversarial"]["sha256"])})
    source_paths = [Path(__file__), Path(__file__).with_name("r2_geometry.py"),
                    Path(__file__).with_name("execution_runtime.py")]
    specification = {
        "condition": REPAIR if repair_ids else CANONICAL,
        "name": name, "smoke3": bool(smoke3), "cohort_ids": [r["id"] for r in rows],
        "registered_count": len(rows), "repair_ids": list(repair_ids),
        "canonical_bindings": canonical_bindings,
        "parent_registration": _sha_binding(rt.RUN / "plan.json"),
        "source_files": [_sha_binding(p) for p in source_paths],
        "input_bindings": bindings,
        "method": {"steps": 50, "learning_rate": .1, "rho": .1, "cosine_floor": .95,
                   "initialization": "exact clean a=0; uniform valid-neighbor logits",
                   "gradient_bootstrap": bool(repair_ids), "optimizer": "Adam(a, logits)",
                   "adam_betas": [.9, .999], "adam_eps": 1e-8,
                   "bank": "current clean utterance, temporal +/-2, <=4 nonself plus exact self, clean cosine>=.95",
                   "objective": "same frozen public Whisper-small sum NLL + full original TV/HF: -NLL+TV+HF",
                   "stopping": "Named no-bank, flat-gradient or clean-start stall, retained in complete cohort outputs.",
                   "budget_view": "50-update native-method; early mathematical stops reported, no padding or clip replacement"},
        "attenuation_diagnostic": {
            "source": "Frozen baseline final physical delta on the SAME clean q",
            "factor": "min(1, ||R2_delta||F / ||baseline_delta||F)",
            "selection": "Deterministic norm matching only; no source/quality/target-based selection",
            "claim_limit": "An attenuated unconstrained baseline direction is a causal diagnostic, not an optimized equal-rank geometry control."},
        "variants": ["clean", "reconstruction", "adversarial", "baseline_attenuated", "baseline_frozen"],
        "source_only": True, "target_models_evaluated": [],
        "cohort_interpretation": "Exploratory explicit stalled-clip repair subset" if repair_ids else "All registered clips, including degenerate or negative outcomes",
    }
    folder = rt.RUN / name
    path = folder / "registration.json"
    if path.exists():
        existing = rt.read(path)
        if {k: v for k, v in existing.items() if k != "created_at_utc"} != specification:
            raise ValueError("Run registration changed; use a separately named condition before any new outputs")
        return existing
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Registration must precede every output in the condition directory")
    folder.mkdir(parents=True, exist_ok=True)
    rt.atomic_json(path, dict(specification, created_at_utc=rt.stamp()))
    path.chmod(0o444)
    return rt.read(path)


def check_registration(registration):
    for binding in registration["source_files"] + [registration["parent_registration"]]:
        _sha_binding(binding["path"], binding["sha256"])
    for bindings in registration["input_bindings"]:
        for key in ("original_audio", "clean_features", "baseline_delta", "baseline_adversarial"):
            _sha_binding(bindings[key]["path"], bindings[key]["sha256"])
    for value in registration["canonical_bindings"].values():
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            _sha_binding(value["path"], value["sha256"])


def _physical_summary(q, delta):
    q64, d64 = q.detach().double(), delta.detach().double()
    nonzero = q64.norm(dim=-1) > 0
    z64 = q64 + d64
    cosines = (z64 * q64).sum(-1) / (q64.norm(dim=-1) * z64.norm(dim=-1)).clamp_min(1e-300)
    return {"absolute_delta_norm": float(d64.norm().item()),
            "relative_radius": float((d64.norm() / (q64.norm() + 1e-8)).item()),
            "minimum_nonzero_frame_cosine": float(cosines[nonzero].min().item()) if nonzero.any() else None}


def _cpu_state(state):
    return {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in state.items()}


def _verify_completed_record(record):
    if not record.get("record_complete"):
        raise ValueError("Incomplete per-clip output cannot count as a completed registered outcome")
    for asset in record["audio"].values():
        _sha_binding(asset["path"], asset["sha256"])
    for key in ("delta", "geometry_parameters", "attenuated_delta"):
        binding = record[key]
        _sha_binding(binding["path"], binding["sha256"])
    return record


def generate(plan, registration):
    rt.check(plan)
    check_registration(registration)
    gate = rt.read(rt.RUN / "baseline-preflight/result.json")
    if gate.get("status") != "passed" or gate.get("checked_updates") != 150:
        raise ValueError("Exact frozen-control preflight must pass before R2 generation")
    folder, name = rt.RUN / registration["name"], registration["name"]
    canonical = registration["condition"] == CANONICAL
    rows = {row["id"]: row for row in _rows(plan, registration["smoke3"])}
    bindings = {binding["id"]: binding for binding in registration["input_bindings"]}
    baseline = _baseline_records()
    selected_ids = registration["cohort_ids"]
    records = []
    attempt = {"status": "running", "started_at_utc": rt.stamp(), "registered_count": len(selected_ids)}
    attempt_path = folder / "attempts" / (attempt["started_at_utc"].replace(":", "-") + ".json")
    rt.atomic_json(attempt_path, attempt)
    rt.atomic_json(folder / "generation-status.json", attempt)
    started = time.perf_counter()
    try:
        with rt.backend_mode("strict_math"):
            model_start = time.perf_counter()
            vocoder, surrogate, native, metadata = rt.models(plan)
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - model_start
            if any(p.requires_grad for model in (vocoder, surrogate, native) for p in model.parameters()):
                raise ValueError("R2 execution requires frozen model weights")
            for row, q, x in rt.clips(plan, registration["smoke3"]):
                uid = row["id"]
                if uid not in selected_ids:
                    continue
                record_path = folder / "records" / f"{uid}.json"
                if record_path.exists() and rt.read(record_path).get("record_complete"):
                    record = _verify_completed_record(rt.read(record_path))
                    if record.get("registration_sha256") != rt.digest(folder / "registration.json"):
                        raise ValueError("Existing output belongs to a different registration")
                    records.append(record)
                    continue
                torch.cuda.reset_peak_memory_stats()
                clip_started = time.perf_counter()
                apply = rt.objective(q, x, row["text"], vocoder, surrogate)
                result = optimize_local_neighbors(q, apply, steps=50, learning_rate=.1, rho=.1,
                                                   cosine_floor=.95, gradient_bootstrap=not canonical)
                torch.cuda.synchronize()
                optimization_seconds = time.perf_counter() - clip_started
                parameters_path = folder / "geometry-parameters" / f"{uid}.pt"
                parameters_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_cpu_state(result.state), parameters_path)
                base_binding = bindings[uid]["baseline_delta"]
                base_delta = torch.load(base_binding["path"], map_location=q.device, weights_only=True).detach()
                if base_delta.shape != q.shape or not torch.isfinite(base_delta).all():
                    raise ValueError("Frozen baseline physical delta does not match q")
                if rt.tensor_value_sha256(base_delta) != bindings[uid]["baseline_delta_value_sha256"]:
                    raise ValueError("Frozen baseline delta value binding changed")
                baseline_norm = float(base_delta.double().norm().item())
                r2_norm = float(result.delta.double().norm().item())
                attenuation = min(1., r2_norm / baseline_norm) if baseline_norm > 0 else 0.
                attenuated = base_delta * attenuation
                attenuation_path = folder / "attenuated-deltas" / f"{uid}.pt"
                attenuation_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(attenuated.cpu(), attenuation_path)
                record = rt.save_feature_result(name, row, q, x, result.delta, result.trace, vocoder,
                    extra={"r2_status": result.status, "condition": registration["condition"],
                           "registered_updates": 50, "actual_optimization_updates": len(result.trace),
                           "source_objective_queries": result.state["objective_queries"],
                           "registration_sha256": rt.digest(folder / "registration.json"),
                           "input_binding": bindings[uid], "r2_diagnostics": result.diagnostics,
                           "geometry_parameters": _sha_binding(parameters_path),
                           "attenuated_delta": _sha_binding(attenuation_path)})
                with torch.no_grad():
                    attenuated_audio = vocoder(q + attenuated)
                record["audio"]["baseline_attenuated"] = rt.save_audio(folder / "audio/baseline_attenuated" / f"{uid}.wav", attenuated_audio)
                record["audio"]["baseline_frozen"] = baseline[uid]["audio"]["adversarial"]
                torch.cuda.synchronize()
                record["resource_use"] = {
                    "optimization_seconds": optimization_seconds,
                    "complete_generation_seconds": time.perf_counter() - clip_started,
                    "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
                    "waveform_evaluation_included": False,
                    "bank_retrieval_and_geometry_diagnostics_included": True,
                }
                record["physical_comparison"] = {
                    "same_clean_q": True, "baseline": _physical_summary(q, base_delta),
                    "r2": _physical_summary(q, result.delta),
                    "baseline_attenuated": _physical_summary(q, attenuated),
                    "baseline_attenuation_factor": attenuation,
                    "norm_matching_residual": float(attenuated.double().norm().item() - r2_norm),
                }
                record["record_complete"] = True
                rt.atomic_json(record_path, record)
                records.append(record)
                rt.write_jsonl(folder / "generated.jsonl", records)
                print(f'R2 {uid}: {result.status}; updates={len(result.trace)} radius={result.diagnostics["radius"]:.6f} coverage={result.diagnostics["neighbor_coverage"]:.3f}', flush=True)
                del apply, result, q, x, base_delta, attenuated, attenuated_audio
            actual_ids = [record["id"] for record in records]
            if set(actual_ids) != set(selected_ids) or len(actual_ids) != len(selected_ids):
                raise ValueError("Incomplete registered R2 cohort; no clips may be removed")
            records.sort(key=lambda record: selected_ids.index(record["id"]))
            rt.write_jsonl(folder / "generated.jsonl", records)
            summary = {
                "status": "completed_all_registered", "condition": registration["condition"],
                "completed_at_utc": rt.stamp(), "registered_count": len(selected_ids), "completed_count": len(records),
                "outcome_counts": dict(Counter(record["r2_status"] for record in records)),
                "actual_total_optimization_updates": sum(record["actual_optimization_updates"] for record in records),
                "source_objective_queries": sum(record["source_objective_queries"] for record in records),
                "generated_manifest": _sha_binding(folder / "generated.jsonl"),
                "registration": _sha_binding(folder / "registration.json"),
                "model_load_seconds_this_invocation": model_load_seconds,
                "wall_seconds_this_invocation": time.perf_counter() - started,
                "source_metadata": metadata, "backend": rt.backend_manifest(),
                "runtime": rt.environment(), "source_only": True, "target_models_evaluated": [],
                "interpretation": registration["cohort_interpretation"],
            }
            rt.atomic_json(folder / "generation-summary.json", summary)
            attempt.update(status=summary["status"], completed_at_utc=rt.stamp(), wall_seconds=time.perf_counter() - started, completed_ids=[r["id"] for r in records])
            rt.atomic_json(attempt_path, attempt)
            rt.atomic_json(folder / "generation-status.json", attempt)
            return summary
    except Exception as error:
        attempt.update({
            "status": "failed_infrastructure", "failed_at_utc": rt.stamp(), "wall_seconds": time.perf_counter() - started,
            "exception": type(error).__name__, "message": str(error),
            "completed_ids": [record["id"] for record in records],
            "uncompleted_registered_ids": [uid for uid in selected_ids if uid not in {r["id"] for r in records}],
        })
        rt.atomic_json(attempt_path, attempt)
        rt.atomic_json(folder / "generation-status.json", attempt)
        raise


def evaluate(plan, registration):
    rt.check(plan)
    check_registration(registration)
    folder = rt.RUN / registration["name"]
    summary = rt.read(folder / "generation-summary.json")
    if summary["status"] != "completed_all_registered":
        raise ValueError("Score only the complete registered cohort, including all failures")
    _sha_binding(summary["generated_manifest"]["path"], summary["generated_manifest"]["sha256"])
    records = [_verify_completed_record(record) for record in rt.read_jsonl(folder / "generated.jsonl")]
    if [record["id"] for record in records] != registration["cohort_ids"]:
        raise ValueError("Evaluation cohort differs from registration")
    # The shared scorer uses its historical native source/quality backend.
    # Do not wrap this in generation's strict_math context.
    return rt.evaluate_records(plan, registration["name"], records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["register", "generate", "evaluate"])
    parser.add_argument("--smoke3", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--repair-ids", nargs="+", default=[])
    parser.add_argument("--canonical-name")
    args = parser.parse_args()
    default = REPAIR if args.repair_ids else CANONICAL
    name = args.name or default + ("-smoke3" if args.smoke3 else "-dev24")
    plan = rt.prepare()
    registration = register(plan, name, args.smoke3, args.repair_ids, args.canonical_name)
    if args.stage == "generate":
        result = generate(plan, registration)
    elif args.stage == "evaluate":
        result = evaluate(plan, registration)
    else:
        result = {"status": "registered", "registration": str(rt.RUN / name / "registration.json"),
                  "condition": registration["condition"], "count": registration["registered_count"]}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
