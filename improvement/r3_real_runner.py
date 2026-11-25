"""Bounded real-data R3 feasibility runner; root schedules the single GPU.

Launch only after the shared runtime's frozen baseline preflight has passed.
All new files remain under improvement/execution/<name>. This script never
loads the four sealed targets or evaluates confirmation/final audio.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT), "/workspace", "/workspace/scripts"]
import torch

import execution_runtime as rt
from beyondwaveform.assets import KNN_REVISION
from beyondwaveform.models.feature import WavLMEncoder
from r3_waveform import (DifferentiableWavLM6, PenaltyWeights, SourceOnlyTrainer,
                         TrainingConfig, calibrate_penalty_weights, make_generator,
                         multiresolution_stft_loss, relative_feature_loss)


class GateFailure(RuntimeError):
    pass


class TrainingDeadline(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateFailure(message)


def cuda_measurement(start):
    torch.cuda.synchronize()
    return {"wall_seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def status(folder, stage, state, **extra):
    value = {"stage": stage, "status": state, "updated_at_utc": rt.stamp(), **extra}
    rt.atomic_json(folder / "status.json", value)
    print(json.dumps(value, allow_nan=False), flush=True)
    return value


def load_encoder_only(plan):
    """Same pinned WavLM loader contract, without constructing a vocoder."""
    settings, expected = plan["base_config"]["features"], plan["feature_artifacts"]
    artifact_root = rt.ROOT / settings.get("artifact_root", "artifacts")
    source = artifact_root / "sources" / f"knn-vc-{KNN_REVISION}"
    checkpoint_path = artifact_root / "checkpoints" / "WavLM-Large.pt"
    require(rt.digest(checkpoint_path) == expected["encoder_sha256"], "Changed WavLM checkpoint")
    require(KNN_REVISION == expected["source_revision"], "Changed WavLM implementation revision")
    require(settings.get("layer", 6) == expected["layer"] == 6, "Content layer must remain six")
    require(settings.get("normalize", False) == expected["normalize"], "Changed WavLM normalization")
    sys.path.insert(0, str(source))
    try:
        from wavlm.WavLM import WavLM, WavLMConfig
    finally:
        sys.path.pop(0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["cfg"] == expected["encoder_config"], "Changed WavLM checkpoint configuration")
    model = WavLM(WavLMConfig(checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    native = WavLMEncoder(model, layer=6, normalize=settings.get("normalize", False)).cuda().eval()
    adapter = DifferentiableWavLM6.from_encoder(native)
    return native, adapter, {"checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": expected["encoder_sha256"], "source_revision": KNN_REVISION,
        "layer": 6, "normalize": native.normalize, "vocoder_loaded": False,
        "adapter": "same frozen inference path; bypass no_grad/GradMultiply only",
        "content_execution": "strict_math", "native_feature_grad_mult": model.feature_grad_mult}


def registration(plan, folder, allow_new_reference):
    rt.check(plan)
    baseline = rt.RUN / "baseline-preflight" / "result.json"
    require(baseline.exists() and rt.read(baseline)["status"] == "passed",
            "Shared frozen-control preflight must pass before R3 GPU work")
    train_manifest = rt.RUN / "data" / "train-1h.jsonl"
    require(train_manifest.exists(), "Frozen disjoint one-hour training manifest is not ready")
    receipt = train_manifest.with_suffix(".receipt.json")
    require(receipt.exists(), "Frozen training manifest needs its acquisition/selection receipt")
    require(rt.read(receipt)["manifest_sha256"] == rt.digest(train_manifest), "Changed training manifest")
    require(plan["r3"]["seed"] == 11 and plan["r3"]["accumulation"] == 4, "Changed R3 seed/accumulation")
    require(plan["r3"]["maximum_training_seconds"] <= 3600, "R3 training allowance exceeds one hour")
    files = [Path(__file__), OUT / "r3_waveform.py", OUT / "execution_runtime.py",
             OUT / "source_evaluate.py",
             rt.RUN / "plan.json", train_manifest, receipt,
             Path(plan["development_manifest"]), Path(plan["calibration_manifest"])]
    result = {"stage": "registered", "seed": 11, "source_only": True,
        "target_models_evaluated": [], "confirmation_or_final_inference": False,
        "train_manifest": str(train_manifest), "train_manifest_sha256": rt.digest(train_manifest),
        "implementation_hashes": {str(path): rt.digest(path) for path in files},
        "generator": make_generator().registration(), "maximum_training_seconds": 3600,
        "maximum_updates": 12000, "batch_size": 1, "accumulation": 4,
        "training_crops": None, "reference_text_policy": "unchanged original transcript",
        "content_policy": "strict_math, frozen WavLM6; all new clean features use this identical transform",
        "allow_new_content_reference": bool(allow_new_reference),
        "execution_repair": ("Declared strict WavLM6 reference regeneration; preserve initial cached-reference failure"
                             if allow_new_reference else None),
        "cached_feature_tolerances": {"relative_frobenius": 1e-5, "maximum_absolute": 1e-4},
        "stft": {"fft": [256, 512, 1024], "hop": "fft/4", "window": "periodic Hann",
                 "center": True, "padding": "operation-equivalent explicit reflected concatenation",
                 "epsilon": 1e-5},
        "linear_upsample_execution": "native CUDA forward; native CPU derivative plus synchronous copies under strict algorithms",
        "calibration": "three frozen duration-stratified dev clips, seeded Rademacher1e-4, seed11",
        "selection": "source-only checkpoints every500 updates; terminal feasibility checkpoint if none qualify",
        "identity_control": "zero-residual identity equals original clean waveform exactly",
        "initial_control": "untrained seeded generator with identical bounds and architecture"}
    result["source_evaluation_policy"] = {
        "execution": "fresh source_evaluate.py subprocess; only native Whisper-small and MOS",
        "cublas_workspace_config": "removed from child environment",
        "backend": "inherited historical saved-audio native policy, established by helper",
        "checkpoint_cost": "charged to one-hour training wall-time ceiling",
        "checkpoint_gpu_memory": "training models paused and resident; allocator cache emptied before child launch; combined process memory must fit"}
    path = folder / "registration.json"
    if path.exists():
        require(rt.read(path) == result, "R3 registration changed; use a named new execution variant")
    else:
        rt.atomic_json(path, result)
    return result


def cached_clean_reference(folder, row, audio, encoder):
    path = folder / "clean-features" / f'{row["id"]}.pt'
    if path.exists():
        return torch.load(path, map_location="cuda", weights_only=True)
    with torch.no_grad():
        value = encoder(audio).detach()
    require(bool(torch.isfinite(value).all()), "Nonfinite clean content features")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.cpu(), path)
    return value


def load_training_audio(row):
    require(rt.digest(row["audio"]) == row["audio_sha256"], f'Changed training audio {row["id"]}')
    require(row.get("role") == "train" and row.get("split") == "train-clean-100", "Wrong R3 data role/split")
    require(row.get("sample_rate") == 16000, "Only native16k training is registered")
    return rt.load_audio(row["audio"]).cuda()


def preflight(plan, folder, registered, source, native_encoder, encoder):
    result_path = folder / "preflight.json"
    if result_path.exists() and rt.read(result_path)["status"] == "passed":
        return rt.read(result_path)
    status(folder, "preflight", "running")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    conformance = []
    clips = []
    for row, cached_q, audio in rt.clips(plan, calibration=True):
        with torch.no_grad():
            native_q = native_encoder(audio)
            current = encoder(audio)
            repeated = encoder(audio)
        require(current.shape == cached_q.shape, "Native content frame alignment differs from cached q")
        diff = current - cached_q
        relative = float(diff.norm() / cached_q.norm().clamp_min(1e-12))
        maximum = float(diff.abs().max())
        cached_matches = relative <= 1e-5 and maximum <= 1e-4
        item = {"id": row["id"], "duration": row["duration"],
            "native_adapter_bitwise_equal": rt.tensor_value_sha256(current) == rt.tensor_value_sha256(native_q),
            "strict_repeat_bitwise_equal": rt.tensor_value_sha256(current) == rt.tensor_value_sha256(repeated),
            "cached_q_bitwise_equal": rt.tensor_value_sha256(current) == rt.tensor_value_sha256(cached_q),
            "cached_q_within_declared_tolerance": cached_matches,
            "cached_q_relative_frobenius_error": relative, "cached_q_maximum_absolute_error": maximum,
            "strict_value_sha256": rt.tensor_value_sha256(current),
            "cached_value_sha256": rt.tensor_value_sha256(cached_q)}
        conformance.append(item)
        rt.atomic_json(folder / "encoder-conformance.json", {"items": conformance,
            "reference_policy": registered["content_policy"], "cached_parity_claimed": False})
        require(item["native_adapter_bitwise_equal"], "Audio-gradient adapter changes native strict WavLM values")
        require(item["strict_repeat_bitwise_equal"], "Strict WavLM reference did not repeat exactly")
        clips.append({**row, "audio": audio})
    if not all(item["cached_q_within_declared_tolerance"] for item in conformance):
        if not registered["allow_new_content_reference"]:
            failed = {"status": "failed_infrastructure", "mechanism": "cached content representation mismatch",
                "calibration_executed": False, "training_executed": False, "items": conformance,
                "resource": cuda_measurement(started)}
            rt.atomic_json(result_path, failed)
            raise GateFailure("Strict WavLM6 differs from frozen cached q; stop before calibration. "
                              "Any regenerated strict clean-reference repair requires a named registered variant.")
        rt.atomic_json(folder / "content-reference-repair.json", {
            "status": "declared_execution_repair", "same_checkpoint_layer_and_audio": True,
            "old_cached_q_parity_claimed": False, "clean_and_adversarial_transform": "same strict_math WavLM6",
            "cached_reference_mismatches": conformance,
            "limitation": "Distinct encoder numerical execution from inherited feature-cache contract"})
    # No generator waveform has been emitted before this coefficient freeze.
    weights, calibration = calibrate_penalty_weights(clips, source.nll, encoder,
        linf=plan["r3"]["linf"], rms_limit=plan["r3"]["rms_limit"])
    calibration["input_manifest_sha256"] = rt.digest(plan["calibration_manifest"])
    calibration["implementation_hashes"] = registered["implementation_hashes"]
    rt.atomic_json(folder / "penalty-calibration.json", calibration)
    print(f'R3 FROZEN WEIGHTS stft={weights.stft:.8g} content={weights.content:.8g}', flush=True)
    # Use the longest eligible training utterance, not just the longest dev clip.
    train_rows = rt.read_jsonl(registered["train_manifest"])
    longest = max(train_rows, key=lambda row: (row["duration"], row["id"]))
    audio = load_training_audio(longest)
    clean = cached_clean_reference(folder, longest, audio, encoder)
    fit_model = make_generator(11, linf=plan["r3"]["linf"]).cuda()
    fit_trainer = SourceOnlyTrainer(fit_model, source.nll, encoder, weights,
        TrainingConfig(seed=11, max_gpu_hours=1.0))
    fit_started = time.perf_counter()
    fit_steps, first_gradient = [], None
    for index in range(4):
        torch.cuda.synchronize(); before = time.perf_counter()
        log = fit_trainer.microbatch(audio, longest["text"], clean)
        torch.cuda.synchronize()
        log["wall_seconds"] = time.perf_counter() - before
        if index == 0:
            gradients = [parameter.grad for parameter in fit_model.parameters() if parameter.grad is not None]
            first_gradient = {"all_finite": all(bool(torch.isfinite(g).all()) for g in gradients),
                "norm": math.sqrt(sum(float(g.double().square().sum()) for g in gradients)),
                "gradient_tensors": len(gradients)}
            require(first_gradient["all_finite"] and first_gradient["norm"] > 0,
                    "R3 parameter gradient fit failed")
        fit_steps.append(log)
        print(f'R3 LONGEST TRAIN FIT {longest["id"]} microbatch{index + 1}/4', flush=True)
    with torch.no_grad():
        adversarial = fit_model(audio)
        residual = adversarial - audio
        bounds = {"original_samples": audio.numel(), "output_samples": adversarial.numel(),
                  "linf": float(residual.abs().max()), "rms": float(residual.square().mean().sqrt()),
                  "peak": float(adversarial.abs().max()), "finite": bool(torch.isfinite(adversarial).all())}
    require(bounds["original_samples"] == bounds["output_samples"] and bounds["finite"]
            and bounds["linf"] <= plan["r3"]["linf"] + 1e-7
            and bounds["rms"] <= 0.003 + 1e-7 and bounds["peak"] <= 1,
            "Real GPU generator duration/bounds failed")
    fit = {"id": longest["id"], "duration": longest["duration"], "parameter_count": fit_model.parameter_count,
           "optimizer_updates": fit_trainer.updates, "microbatches": fit_steps,
           "first_parameter_gradients": first_gradient, "bounds": bounds,
           "wall_seconds": time.perf_counter() - fit_started}
    require(fit_trainer.updates == 1, "Longest-clip fit did not allocate AdamW state through one update")
    del fit_trainer, fit_model, audio, clean
    gc.collect(); torch.cuda.empty_cache()
    result = {"status": "passed", "source_only": True, "encoder_conformance": conformance,
        "cached_q_parity_claimed": all(item["cached_q_bitwise_equal"] for item in conformance),
        "clean_reference": "new strict frozen WavLM6 features under identical clean/adversarial transform",
        "calibration": {"path": str(folder / "penalty-calibration.json"),
                        "sha256": rt.digest(folder / "penalty-calibration.json")},
        "longest_training_clip_fit": fit, "resource": cuda_measurement(started)}
    rt.atomic_json(result_path, result)
    status(folder, "preflight", "passed", resource=result["resource"])
    return result


def source_evaluation_command(name, records_path, output_dir, *, checkpoint_mode=False,
                              parent_environment=None):
    """Build the policy-isolated source evaluator invocation without GPU work."""
    require(Path(name).name == name and name.startswith("r3-"), "Use a simple R3 source-evaluation name")
    require(Path(records_path).resolve().is_relative_to(OUT), "Source records stay inside improvement/")
    require(Path(output_dir).resolve().is_relative_to(OUT), "Source score outputs stay inside improvement/")
    environment = dict(os.environ if parent_environment is None else parent_environment)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(OUT / "source_evaluate.py"), "--name", name,
               "--records", str(records_path), "--output-dir", str(output_dir)]
    return command, environment


def evaluate_source_subprocess(name, records_path, output_dir, *, checkpoint_mode=False,
                                deadline=None):
    """Pause generation and evaluate saved WAVs in a fresh native-policy process."""
    require((OUT / "source_evaluate.py").exists(), "Native-policy source evaluation helper is not ready")
    output_dir = Path(output_dir)
    require(output_dir.resolve().is_relative_to(OUT), "Source evaluation artifacts stay inside improvement/")
    output_dir.mkdir(parents=True, exist_ok=True)
    command, environment = source_evaluation_command(name, records_path, output_dir,
                                                     checkpoint_mode=checkpoint_mode)
    # Save a terminal state with margin after a time-limited child evaluation.
    timeout = None if deadline is None else deadline - time.monotonic() - 5.0
    if timeout is not None and timeout <= 0:
        raise TrainingDeadline("No training time remains for native-policy checkpoint scoring")
    memory = {"parent_cuda_initialized": torch.cuda.is_initialized()}
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        memory.update({"parent_allocated_bytes": torch.cuda.memory_allocated(),
                       "parent_reserved_bytes": torch.cuda.memory_reserved()})
    started = time.monotonic()
    receipt = {"command": command, "cublas_workspace_config_present": "CUBLAS_WORKSPACE_CONFIG" in environment,
               "checkpoint_mode": checkpoint_mode, "timeout_seconds": timeout,
               "parent_models_paused": True, "memory": memory, "target_models_evaluated": [],
               "status": "running"}
    try:
        with (output_dir / "source-evaluation.log").open("a") as stream:
            process = subprocess.run(command, env=environment, cwd=str(rt.ROOT),
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        receipt["returncode"] = process.returncode
        require(process.returncode == 0, f'Native source evaluator failed; inspect {output_dir / "source-evaluation.log"}')
        result = rt.read(output_dir / "source-summary.json")
        require(result.get("source_only") is True and result.get("target_models_evaluated") == [],
                "Source helper receipt must exclude held-out target inference")
        receipt["status"] = "completed"
        return result
    except subprocess.TimeoutExpired as error:
        receipt["status"] = "training_deadline"
        raise TrainingDeadline("One-hour budget interrupted native checkpoint evaluation") from error
    except Exception:
        receipt["status"] = "failed_infrastructure"
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        rt.atomic_json(output_dir / "source-subprocess.json", receipt)


def checkpoint_source_evaluation(plan, model, deadline, folder, update):
    checkpoint_folder = folder / "checkpoints" / f'update-{update:05d}-evaluation'
    records, spectral_losses = [], []
    for row, _, audio in rt.clips(plan):
        if time.monotonic() >= deadline:
            raise TrainingDeadline("Training deadline during source-only checkpoint evaluation")
        with torch.no_grad():
            adversarial = model(audio)
            spectral = float(multiresolution_stft_loss(adversarial, audio))
        asset = rt.save_audio(checkpoint_folder / "audio" / f'{row["id"]}.wav', adversarial)
        records.append({"id": row["id"], "text": row["text"], "role": "pilot", "language": "en",
                        "source_only": True, "audio": {"adversarial": asset}})
        spectral_losses.append(spectral)
    records_path = checkpoint_folder / "generated.jsonl"
    rt.write_jsonl(records_path, records)
    source_folder = checkpoint_folder / "native-source"
    receipt = evaluate_source_subprocess(f'{folder.name}-checkpoint-{update}', records_path,
        source_folder, checkpoint_mode=True, deadline=deadline)
    summary = receipt["summary"]["adversarial"]
    quality = summary["quality"]
    admissible_proxy = quality["utmos"] >= 3.5 and quality["dnsmos"] >= 3.0 and quality["nisqa"] >= 3.75
    return {"source_only": True, **summary, "quality": quality,
            "mean_stft_loss": sum(spectral_losses) / len(spectral_losses),
            "automatic_proxy_admissible": admissible_proxy, "human_admissibility_established": False,
            "native_policy_receipt": str(source_folder / "source-summary.json"),
            "native_policy_receipt_sha256": rt.digest(source_folder / "source-summary.json")}


def train(plan, folder, registered, preflight_result, source, encoder):
    summary_path = folder / "training.json"
    if summary_path.exists() and rt.read(summary_path)["status"] == "completed":
        return rt.read(summary_path)
    calibration = rt.read(folder / "penalty-calibration.json")
    require(rt.digest(folder / "penalty-calibration.json") == preflight_result["calibration"]["sha256"],
            "Frozen R3 calibration receipt changed")
    weights = PenaltyWeights(**calibration["weights"])
    rows = rt.read_jsonl(registered["train_manifest"])
    generator = make_generator(11, linf=plan["r3"]["linf"]).cuda()
    trainer = SourceOnlyTrainer(generator, source.nll, encoder, weights,
                                TrainingConfig(seed=11, max_gpu_hours=1.0))
    # At entry, source/content models and all clean calibration references exist;
    # coefficient calibration, longest-clip fit and later terminal evaluation have
    # separate accounting from this one-hour training wall-time ceiling.
    started = time.monotonic()
    deadline = started + min(3600, plan["r3"]["maximum_training_seconds"])
    torch.cuda.reset_peak_memory_stats()
    random_generator = random.Random(11)
    timings = [row["wall_seconds"] for row in preflight_result["longest_training_clip_fit"]["microbatches"]]
    max_microbatch_seconds = max(timings)
    gpu_elapsed_milliseconds = 0.0
    loss_totals = {"source_sum_nll": 0.0, "stft_loss": 0.0, "content_loss": 0.0}
    epoch = 0
    log_path = folder / "training-trace.jsonl"
    require(not log_path.exists(), "Training trace exists without completed summary; preserve it and start a named resumed variant")
    selected_checkpoint = None
    reason = "update_limit"
    last_progress = time.monotonic()
    status(folder, "training", "running", maximum_seconds=3600)
    while trainer.updates < 12000:
        order = list(range(len(rows)))
        random_generator.shuffle(order)
        epoch += 1
        for row_index in order:
            # Reserve four full longest-fit microbatches plus margin before a
            # new accumulation block; no incomplete update is promoted.
            reserve = max(15.0, max_microbatch_seconds * 1.5 + 5.0)
            remaining = deadline - time.monotonic()
            needed = (4 - trainer.microbatches % 4) * reserve
            if remaining <= needed or trainer.budget_exhausted:
                reason = "one_hour_wall_time_cap_conservative_reserve"
                break
            row = rows[row_index]
            audio = load_training_audio(row)
            clean = cached_clean_reference(folder, row, audio, encoder)
            event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); micro_start = time.perf_counter(); event_start.record()
            log = trainer.microbatch(audio, row["text"], clean)
            event_end.record(); torch.cuda.synchronize()
            seconds = time.perf_counter() - micro_start
            max_microbatch_seconds = max(max_microbatch_seconds, seconds)
            gpu_elapsed_milliseconds += event_start.elapsed_time(event_end)
            for key in loss_totals:
                loss_totals[key] += log[key]
            log.update({"id": row["id"], "epoch": epoch, "duration": row["duration"],
                        "wall_seconds": seconds, "elapsed_training_seconds": time.monotonic() - started})
            with log_path.open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            del audio, clean
            if time.monotonic() - last_progress >= 30 or (log["optimizer_step"] and trainer.updates == 1):
                status(folder, "training", "running", updates=trainer.updates,
                    microbatches=trainer.microbatches, elapsed_seconds=time.monotonic() - started,
                    latest_losses={key: log[key] for key in loss_totals})
                last_progress = time.monotonic()
            if log["optimizer_step"] and trainer.checkpoint_due:
                checkpoint = folder / "checkpoints" / f'update-{trainer.updates:05d}.pt'
                trainer.save_checkpoint(checkpoint, manifest_sha256=registered["train_manifest_sha256"], calibration=calibration)
                try:
                    score = checkpoint_source_evaluation(plan, generator, deadline, folder, trainer.updates)
                    rt.atomic_json(checkpoint.with_suffix(".source.json"), score)
                    selected = trainer.consider_checkpoint(source_wer=score["error_percent"],
                        log_stft=score["mean_stft_loss"], admissible=score["automatic_proxy_admissible"])
                    if selected:
                        selected_checkpoint = checkpoint
                except TrainingDeadline:
                    reason = "one_hour_wall_time_cap_during_checkpoint_evaluation"
                    break
            if trainer.updates >= 12000:
                break
        if reason != "update_limit" or trainer.updates >= 12000:
            break
    # Discard partial gradient accumulation without representing it as an
    # optimizer update. The completed parameter state remains auditable.
    incomplete = trainer.microbatches % 4
    if incomplete:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.microbatches -= incomplete
    terminal = folder / "checkpoints" / "terminal.pt"
    trainer.save_checkpoint(terminal, manifest_sha256=registered["train_manifest_sha256"], calibration=calibration)
    result = {"status": "completed", "interpretation": "one-seed one-hour feasibility; no generalization or transfer claim",
        "seed": 11, "updates": trainer.updates, "completed_microbatches": trainer.microbatches,
        "discarded_partial_microbatches": incomplete, "stop_reason": reason,
        "epochs_started": epoch, "training_wall_seconds": time.monotonic() - started,
        "maximum_training_wall_seconds": 3600, "cuda_event_microbatch_span_seconds": gpu_elapsed_milliseconds / 1000,
        "cuda_timing_note": "Event spans of microbatch work; not hardware-profiler active-kernel occupancy",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "mean_microbatch_losses": {key: value / max(1, trainer.microbatches + incomplete) for key, value in loss_totals.items()},
        "terminal_checkpoint": {"path": str(terminal), "sha256": rt.digest(terminal)},
        "selected_checkpoint": str(selected_checkpoint or terminal),
        "checkpoint_selection": "automatic source-only proxy" if selected_checkpoint else "terminal feasibility only; no admissible500-update checkpoint selected",
        "human_admissibility_established": False, "training_manifest_sha256": registered["train_manifest_sha256"],
        "trace_sha256": rt.digest(log_path) if log_path.exists() else None,
        "cached_q_parity_claimed": preflight_result["cached_q_parity_claimed"],
        "weights": asdict(weights), "generator": generator.registration(), "training_crops": None}
    rt.atomic_json(summary_path, result)
    status(folder, "training", "completed", updates=trainer.updates,
           training_wall_seconds=result["training_wall_seconds"], stop_reason=reason)
    del trainer, generator
    gc.collect(); torch.cuda.empty_cache()
    return result


def generate(plan, folder, training_result):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    generator = make_generator(11, linf=plan["r3"]["linf"]).cuda().eval()
    initial = make_generator(11, linf=plan["r3"]["linf"]).cuda().eval()
    state = torch.load(training_result["selected_checkpoint"], map_location="cpu", weights_only=True)
    generator.load_state_dict(state["generator"], strict=True)
    records = []
    for row, _, audio in rt.clips(plan):
        with torch.no_grad():
            adversarial, untrained = generator(audio), initial(audio)
            identity = audio + torch.zeros_like(audio)
        require(torch.equal(identity, audio), "Identity waveform control must be exact")
        residual = adversarial - audio
        bound = {"linf": float(residual.abs().max()), "rms": float(residual.square().mean().sqrt()),
                 "peak": float(adversarial.abs().max()), "output_samples": adversarial.numel(),
                 "input_samples": audio.numel(), "finite": bool(torch.isfinite(adversarial).all())}
        require(bound["linf"] <= plan["r3"]["linf"] + 1e-7 and bound["rms"] <= 0.003 + 1e-7
                and bound["peak"] <= 1 and bound["finite"] and adversarial.shape == audio.shape,
                "Generated waveform violates registered constraints")
        assets = {variant: rt.save_audio(folder / "audio" / variant / f'{row["id"]}.wav', waveform)
                  for variant, waveform in (("clean", audio), ("initial-generator", untrained),
                                             ("adversarial", adversarial))}
        records.append({"id": row["id"], "text": row["text"], "language": "en", "role": "pilot",
            "input_audio_sha256": row["audio_sha256"], "audio": assets, "bounds": bound,
            "identity_control_equals_clean": True, "source_only": True, "seed": 11})
    rt.write_jsonl(folder / "generated.jsonl", records)
    rt.atomic_json(folder / "generation-resource.json", {
        "count": len(records), "generator_forwards_per_clip": 2,
        "initial_generator_control_included": True, "identity_control_equals_clean": True,
        "resource": cuda_measurement(started), "source_model_calls": 0, "content_model_calls": 0,
        "vocoder_calls": 0, "generated_manifest_sha256": rt.digest(folder / "generated.jsonl")})
    del generator, initial, state
    gc.collect(); torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "generate", "evaluate", "run"))
    parser.add_argument("--name", default="r3-waveform")
    parser.add_argument("--allow-new-content-reference", action="store_true",
                        help="Explicitly named repair using regenerated strict WavLM6 clean references")
    args = parser.parse_args()
    require(Path(args.name).name == args.name and args.name.startswith("r3-"), "Use a simple R3 artifact name")
    folder = rt.RUN / args.name
    folder.mkdir(parents=True, exist_ok=True)
    try:
        plan = rt.prepare()
        registered = registration(plan, folder, args.allow_new_content_reference)
        if args.stage == "evaluate":
            evaluate_source_subprocess(args.name, folder / "generated.jsonl", folder)
        elif args.stage == "generate":
            with rt.backend_mode("strict_math"):
                generate(plan, folder, rt.read(folder / "training.json"))
        else:
            with rt.backend_mode("strict_math"):
                _, source, native_source, source_metadata = rt.models(plan, with_vocoder=False)
                require(source.reduction == "sum", "Source NLL must score lexical SUM loss")
                require(source.nll_transcript_policy == "original", "Original transcript policy required")
                native_encoder, encoder, encoder_metadata = load_encoder_only(plan)
                rt.atomic_json(folder / "model-provenance.json", {"source": source_metadata, "content": encoder_metadata})
                result = preflight(plan, folder, registered, source, native_encoder, encoder)
                if args.stage in ("train", "run"):
                    training_result = train(plan, folder, registered, result, source, encoder)
                del source, native_source, native_encoder, encoder
                gc.collect(); torch.cuda.empty_cache()
                if args.stage in ("train", "run"):
                    generate(plan, folder, training_result)
            if args.stage == "run":
                evaluate_source_subprocess(args.name, folder / "generated.jsonl", folder)
        status(folder, args.stage, "completed", source_only=True, target_models_evaluated=[])
    except Exception as error:
        status(folder, args.stage, "failed_infrastructure", error=str(error), error_type=type(error).__name__)
        rt.atomic_json(folder / f'{args.stage}-failure.json', {"error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc(),
            "source_only": True, "target_models_evaluated": []})
        raise


if __name__ == "__main__":
    main()
