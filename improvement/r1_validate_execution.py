"""CPU-only audit of saved R1 evidence, without new ASR/quality inference."""
from __future__ import annotations

import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import execution_runtime as runtime
from r1_sequence import calibrate_scale, word_cost


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def candidate_check(candidate_set):
    rows = candidate_set["candidates"]
    native = [row for row in rows if row["origin"] == "native_beam"]
    require(len(native) == len({tuple(row["tokens"]) for row in native}) == 4,
            "Candidate set does not contain four distinct real beam paths")
    require(any(row["cost"] == 0 for row in rows), "Missing zero-cost reference path")
    for row in rows:
        cost, counts = word_cost(candidate_set["reference"], row["text"])
        require(cost == row["cost"], "Candidate word cost changed")
        require([counts[k] for k in ("substitutions", "deletions", "insertions", "reference_units")]
                == [row[k] for k in ("substitutions", "deletions", "insertions", "reference_words")],
                "Candidate edit-count receipt changed")
        require(all(0 <= token < 50257 for token in row["tokens"]), "Nonlexical path token")
    return len(rows)


def main():
    plan = runtime.read(runtime.RUN / "plan.json")
    runtime.check(plan)
    calibration = runtime.read(runtime.RUN / "r1-calibration/result.json")
    pilot = runtime.read(runtime.RUN / "r1-calibrated-pilot3/result.json")
    stopped = runtime.read(runtime.RUN / "r1-calibrated-development24/result.json")
    require(calibration["status"] == "passed" and pilot["status"] == "completed", "Missing valid calibration/pilot")
    require(stopped["status"] == "negative" and stopped["actual_count"] == 0, "Canonical stop status changed")
    recalculated = calibrate_scale([row["probe"] for row in calibration["items"]])
    require(recalculated == calibration["calibration"], "Frozen calibration does not recompute")
    for path, expected in calibration["bindings"]["code_hashes"].items():
        require(runtime.digest(path) == expected, "Changed frozen R1 source")
    require(runtime.digest(HERE / "r1_real_runner.py") == runtime.digest(HERE / "r1_real_runner_v1.py"), "v1 copy differs")
    candidate_count = sum(candidate_check(row["candidate_set"]) for row in calibration["items"])
    rows = runtime.read_jsonl(runtime.RUN / "r1-calibrated-pilot3/generated.jsonl")
    declared = [row["id"] for row in runtime.read_jsonl(plan["calibration_manifest"])]
    require([row["id"] for row in rows] == declared, "Pilot used undeclared clips")
    cache = {row["id"]: row for row in runtime.read(plan["cache_audit"])["items"]}
    refresh_count, update_count, wave_count = 0, 0, 0
    for row in rows:
        require(len(row["trace"]) == row["actual_optimization_updates"] == 50, "Incomplete R1 trajectory")
        for index, update in enumerate(row["trace"]):
            require(update["step"] == index + 1 and update["candidate_step_zero_based"] == index, "Wrong update index")
            require(update["risk_gradient_norm"] > 0 and math.isfinite(update["risk_gradient_norm"]), "Flat/nonfinite risk gradient")
            require(0 <= update["risk"] <= 1 and update["radius"] <= .100001, "Invalid bounded risk/radius")
            require(abs(sum(update["posterior"]) - 1) < 1e-6, "Posterior is not normalized")
            require(abs(sum(p * c for p, c in zip(update["posterior"], update["costs"])) - update["risk"]) < 1e-6,
                    "Risk differs from archived posterior/cost expectation")
            update_count += 1
        require([item["step"] for item in row["candidate_refreshes"]] == list(range(0, 50, 5)), "Changed refresh cadence")
        for receipt in row["candidate_refreshes"]:
            require(runtime.digest(receipt["path"]) == receipt["sha256"], "Changed refresh candidate artifact")
            candidates = runtime.read(receipt["path"])
            require(candidates["refreshed_at"] == receipt["step"], "Refresh index mismatch")
            candidate_count += candidate_check(candidates)
            refresh_count += 1
        delta = torch.load(row["delta"]["path"], map_location="cpu", weights_only=True)
        require(runtime.digest(row["delta"]["path"]) == row["delta"]["sha256"], "Changed final perturbation file")
        require(runtime.tensor_value_sha256(delta) == row["delta"]["value_sha256"] == row["trace"][-1]["delta_value_sha256"],
                "Final perturbation differs from update 50")
        clean_features = torch.load(cache[row["id"]]["q_path"], map_location="cpu", weights_only=True)
        radius = runtime.relative_radius(delta, clean_features).item()
        require(radius <= .100001 and abs(radius - row["actual_relative_radius"]) < 1e-6, "Saved physical-feature radius invalid")
        require(runtime.digest(row["optimizer"]["path"]) == row["optimizer"]["sha256"], "Changed optimizer file")
        optimizer = torch.load(row["optimizer"]["path"], map_location="cpu", weights_only=True)
        require(runtime.tensor_tree(optimizer) == row["optimizer"]["value_receipt"], "Changed optimizer tensor values")
        for asset in row["audio"].values():
            require(runtime.digest(asset["path"]) == asset["sha256"], "Changed saved waveform")
            audio = runtime.load_audio(asset["path"])
            require(torch.isfinite(audio).all().item() and audio.numel() > 0, "Invalid saved waveform")
            wave_count += 1
        require(runtime.error_counts(row["text"], row["source_hypothesis"]) == row["source_error_counts"], "Source edit counts do not recompute")
    failure_path = runtime.RUN / "r1-calibrated-development24/failures" / f'{stopped["failed_clip"]}.json'
    failure = runtime.read(failure_path)
    require(failure["step_zero_based"] == 0, "Stop happened after unexpected updates")
    candidate_count += candidate_check(failure["candidate_set"])
    require(all(row["cost"] == 0 for row in failure["candidate_set"]["candidates"]), "Canonical failure candidates are not all zero-cost")
    delta = torch.load(failure["delta"]["path"], map_location="cpu", weights_only=True)
    require(torch.count_nonzero(delta).item() == 0, "Failed initial candidate probe changed physical features")
    registration_path = stopped["execution_wrapper"]["registration"]
    require(runtime.digest(registration_path) == stopped["execution_wrapper"]["registration_sha256"], "Changed v2 logging registration")
    registration = runtime.read(registration_path)
    require(registration["scientific_changes"] == [], "Undeclared scientific change")
    require(runtime.digest(registration["wrapper"]["path"]) == registration["wrapper"]["sha256"], "Changed v2 logging wrapper")
    result = {"status": "passed", "created_at_utc": runtime.stamp(), "cpu_only": True,
              "new_model_inference": False, "calibration_probes": 3, "pilot_updates_verified": update_count,
              "refresh_candidate_sets_verified": refresh_count, "candidate_rows_verified": candidate_count,
              "saved_waveforms_verified": wave_count, "all_zero_cost_initial_stop_verified": True,
              "source_files_immutable": True, "validation_source_sha256": runtime.digest(Path(__file__))}
    runtime.atomic_json(runtime.RUN / "r1-validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
