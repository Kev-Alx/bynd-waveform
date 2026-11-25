"""Logging-only wrapper for R1 development after the frozen v1 pilot.

The scientific implementation, model calls, candidates, objective, optimizer and
calibrated scale remain those of r1_real_runner.py. This separately registered
wrapper adds the candidate set to failure receipts, including failures raised
before the ordinary successful-refresh archive is written. It does not retry,
repair, skip, replace or rescore a failed clip.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_real_runner as frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "c1-pilot"))
    args = parser.parse_args()
    runtime = frozen.runtime
    plan = runtime.read(runtime.RUN / "plan.json")
    bindings = frozen.prerequisites(plan)
    calibration = frozen.load_calibration(bindings)
    original = HERE / "r1_real_runner.py"
    preserved = HERE / "r1_real_runner_v1.py"
    if runtime.digest(original) != runtime.digest(preserved):
        raise RuntimeError("Preserved v1 runner differs from the calibrated original")
    stage_name = frozen.STAGES[args.stage][0]
    registration_path = runtime.RUN / f"{stage_name}-v2-registration.json"
    if registration_path.exists():
        raise FileExistsError("Preserve prior v2 logging registration")
    registration = {
        "created_at_utc": runtime.stamp(),
        "version": "v2 logging only; scientific execution remains frozen v1",
        "stage": args.stage,
        "wrapper": {"path": str(Path(__file__)), "sha256": runtime.digest(Path(__file__))},
        "frozen_runner": {"path": str(original), "sha256": runtime.digest(original)},
        "preserved_v1_copy": {"path": str(preserved), "sha256": runtime.digest(preserved)},
        "calibration_bindings": bindings,
        "frozen_scale": calibration["calibration"]["scale"],
        "pilot_result_sha256": runtime.digest(runtime.RUN / frozen.STAGES["pilot"][0] / "result.json"),
        "changes": ["Archive latest detached candidate set when canonical optimization raises DegenerateRiskError"],
        "scientific_changes": [],
        "user_data_scope": "same registered development cohort; no confirmation or final inference",
    }
    runtime.atomic_json(registration_path, registration)

    original_objective = frozen.objective
    original_collect = frozen.collect_clip
    active = {}

    def logging_objective(surrogate):
        instance = original_objective(surrogate)
        active["objective"] = instance
        return instance

    def logging_collect(plan, folder, row, q, x, vocoder, surrogate, scale):
        active.clear()
        try:
            return original_collect(plan, folder, row, q, x, vocoder, surrogate, scale)
        except frozen.DegenerateRiskError as error:
            instance = active.get("objective")
            receipt = {"id": row["id"], "status": "negative", "reason": str(error),
                       "logging_registration": str(registration_path),
                       "logging_registration_sha256": runtime.digest(registration_path),
                       "candidate_set": instance.candidate_set.receipt()
                                        if instance is not None and instance.candidate_set is not None else None}
            # Failure introspection records the existing state only; it neither
            # mutates the live tensors nor calls another model/optimizer pass.
            current = error.__traceback__
            while current is not None:
                frame = current.tb_frame
                if frame.f_code is original_collect.__code__:
                    receipt["step_zero_based"] = frame.f_locals.get("step")
                    delta = frame.f_locals.get("delta")
                    if delta is not None:
                        path = folder / "failures" / f'{row["id"]}-delta.pt'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        frozen.torch.save(delta.detach().cpu(), path)
                        receipt["delta"] = {"path": str(path), "sha256": runtime.digest(path),
                                            "value_sha256": runtime.tensor_value_sha256(delta)}
                    break
                current = current.tb_next
            runtime.atomic_json(folder / "failures" / f'{row["id"]}.json', receipt)
            raise

    frozen.objective = logging_objective
    frozen.collect_clip = logging_collect
    try:
        result = frozen.generate(plan, args.stage)
        print(f'R1 {args.stage}: {result["status"]}', flush=True)
    finally:
        result_path = runtime.RUN / stage_name / "result.json"
        if result_path.exists():
            result = runtime.read(result_path)
            result["execution_wrapper"] = {"registration": str(registration_path),
                                           "registration_sha256": runtime.digest(registration_path),
                                           "scientific_execution_version": "frozen v1",
                                           "failure_logging_version": "v2"}
            runtime.atomic_json(result_path, result)


if __name__ == "__main__":
    main()
