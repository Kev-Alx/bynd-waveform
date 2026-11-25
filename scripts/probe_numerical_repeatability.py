"""Repeat the unchanged full-objective derivative at fixed feature deltas.

This read-only diagnostic performs no optimizer updates, decoding or MOS
inference. Strict deterministic failures are evidence of an unsupported
execution path, not permission to silently weaken deterministic settings.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beyondwaveform.attacks import freeze, relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything
from beyondwaveform.losses import CleanReference
from beyondwaveform.models.whisper import load_whisper
from beyondwaveform.pipeline import config_fingerprint
from probe_feature_gradients import load_frozen_vocoder

MODES = ("auto", "math", "strict_math")
REPEATS = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths():
    paths = [Path(__file__).resolve(), ROOT / "scripts/probe_feature_gradients.py",
        ROOT / "beyondwaveform/attacks.py", ROOT / "beyondwaveform/losses.py",
        ROOT / "beyondwaveform/io.py", ROOT / "beyondwaveform/config.py",
        ROOT / "beyondwaveform/models/feature.py", ROOT / "beyondwaveform/models/whisper.py"]
    paths += [Path(importlib.import_module(name).__file__).resolve() for name in
        ("whisper.model", "whisper.audio", "whisper.tokenizer", "torch", "torch.functional",
         "torch.nn.functional", "torch.nn.attention", "torch.backends.cuda")]
    assets = Path(importlib.import_module("whisper").__file__).resolve().parent / "assets"
    paths += sorted(assets.glob("*.tiktoken")) + [assets / "mel_filters.npz"]
    return list(dict.fromkeys(paths))


def tensor_value_sha256(value):
    """Hash dtype/shape and raw contiguous CPU values, not a save container."""
    cpu = value.detach().cpu().contiguous()
    description = json.dumps({"dtype": str(cpu.dtype), "shape": list(cpu.shape)}, sort_keys=True).encode()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(description + b"\0" + raw).hexdigest()


def compare_gradients(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype, "Gradient identity mismatch")
    a, b = left.detach().cpu().double().flatten(), right.detach().cpu().double().flatten()
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), "Nonfinite repeated gradient")
    an, bn = a.norm().item(), b.norm().item()
    difference = a-b
    return {"raw_value_sha256_equal": tensor_value_sha256(left) == tensor_value_sha256(right),
        "numeric_values_equal": torch.equal(left, right), "max_abs_difference": difference.abs().max().item(),
        "l2_difference": difference.norm().item(), "left_l2_norm": an, "right_l2_norm": bn,
        "relative_l2_difference_to_left": difference.norm().item()/an if an else (0.0 if not bn else None),
        "cosine": max(-1.0, min(1.0, a.dot(b).item()/(an*bn))) if an and bn else None}


def backend_manifest():
    from whisper.model import MultiHeadAttention, SDPA_AVAILABLE
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "whisper_use_sdpa": MultiHeadAttention.use_sdpa, "whisper_sdpa_available": SDPA_AVAILABLE,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}


@contextmanager
def backend_mode(mode):
    require(mode in MODES, "Unknown numerical backend mode")
    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if mode == "strict_math":
            torch.use_deterministic_algorithms(True, warn_only=False)
        if mode == "auto":
            yield
        else:
            # cuDNN is disabled too: leaving its default enabled would not
            # actually force the mathematical SDPA implementation.
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                    enable_mem_efficient=False, enable_cudnn=False):
                yield
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)


def known_determinism_failure(error):
    message = str(error).lower()
    known = ("reflection_pad1d_backward_cuda", "reflection_pad2d_backward_cuda", "reflection_pad3d_backward_cuda",
        "reflection_pad1d_backward_out_cuda", "reflection_pad2d_backward_out_cuda", "reflection_pad3d_backward_out_cuda",
        "replication_pad1d_backward_cuda", "replication_pad3d_backward_cuda", "nll_loss_forward_cuda",
        "nll_loss2d_forward_cuda", "_scaled_dot_product_efficient_attention_backward",
        "memory_efficient_attention_backward")
    return isinstance(error, RuntimeError) and any(name in message for name in known) and (
        "deterministic implementation" in message or "not deterministic" in message or "nondeterministic" in message)


def full_derivative(q, x, delta, text, vocoder, surrogate, attack, spectral):
    """One total derivative of the unchanged parent expression, no term split."""
    q, x = q.detach(), x.detach()
    live = delta.detach().clone().requires_grad_()
    require(q.ndim == 2 and q.shape == live.shape and q.shape[0] >= 2 and
        torch.isfinite(q).all().item() and torch.isfinite(live).all().item(), "Invalid fixed features/delta")
    reference = CleanReference(q, x, spectral, attack.eps)
    audio = vocoder(q + live)
    nll = surrogate.nll(audio, text) if attack.attack else live.new_zeros(())
    tv, hf = reference.terms(q + live, audio, temporal=attack.temporal and attack.lambda_perc > 0,
        high_frequency=attack.high_frequency and attack.lambda_perc > 0 and attack.alpha_hf > 0)
    loss = -nll + attack.lambda_perc * (tv + attack.alpha_hf * hf)
    require(torch.isfinite(loss).item(), "Nonfinite full objective")
    gradient, = torch.autograd.grad(loss, live)
    require(torch.isfinite(gradient).all().item(), "Nonfinite full-objective gradient")
    scalars = torch.stack([loss.detach(), nll.detach(), tv.detach(), hf.detach()]).cpu()
    cpu_gradient = gradient.detach().cpu().contiguous()
    require(not any(p.grad is not None for module in (vocoder, surrogate) for p in module.parameters()),
            "Diagnostic accumulated model gradients")
    return {"loss_total": scalars[0].item(), "nll": scalars[1].item(), "tv_ratio": scalars[2].item(),
        "hf_ratio": scalars[3].item(), "original_clean_tv": reference.tv.item(), "original_clean_hf": reference.hf.item(),
        "loss_value_sha256": tensor_value_sha256(scalars), "gradient_value_sha256": tensor_value_sha256(cpu_gradient),
        "gradient_l2_norm": cpu_gradient.double().norm().item(), "relative_delta_radius": relative_radius(delta, q, attack.eps).item(),
        "model_parameters_frozen": not any(p.requires_grad for module in (vocoder, surrogate) for p in module.parameters()),
        "parameter_gradients_present": False}, cpu_gradient


def repeat_state(q, x, delta, text, vocoder, surrogate, attack, spectral, seed, directory):
    measurements, gradients = [], []
    for index in range(REPEATS):
        seed_everything(seed)
        measurement, gradient = full_derivative(q, x, delta, text, vocoder, surrogate, attack, spectral)
        path = directory / f"repeat-{index}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gradient, path)
        measurements.append({"repeat": index, **measurement,
            "gradient_artifact": {"path": str(path), "file_sha256": digest(path), "value_sha256": tensor_value_sha256(gradient)}})
        gradients.append(gradient)
    comparisons = []
    for left, right in itertools.combinations(range(REPEATS), 2):
        values = compare_gradients(gradients[left], gradients[right])
        loss_equal = measurements[left]["loss_value_sha256"] == measurements[right]["loss_value_sha256"]
        comparisons.append({"left_repeat": left, "right_repeat": right, "gradient": values,
            "loss_values_bitwise_equal": loss_equal, "numerical_bitwise_agreement": values["raw_value_sha256_equal"] and loss_equal})
    return {"repeats": measurements, "pairs": comparisons,
        "numerical_bitwise_agreement": all(row["numerical_bitwise_agreement"] for row in comparisons)}


def profile_derivative(arguments, seed):
    seed_everything(seed)
    # CPU dispatch events identify the actual aten SDPA implementation and
    # backward operators without requiring CUPTI access to GPU kernel traces.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        full_derivative(*arguments)
    operations = [{"name": entry.key, "count": entry.count} for entry in profile.key_averages()]
    relevant = [entry for entry in operations if any(part in entry["name"].lower()
        for part in ("scaled_dot", "efficient_attention", "flash_attention", "cudnn_attention", "reflection_pad", "pad_backward"))]
    return {"activities": ["CPU dispatcher events"], "placement": "separate extra derivative after all equality repeats",
        "relevant_operations": relevant, "all_operation_names": sorted(entry["name"] for entry in operations),
        "limitation": "These are executed aten operator names, not a CUPTI trace or proof of CUDA kernel determinism."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)
    import run_feature_diagnostics as parent_protocol
    import run_quality_research as protocol
    parent_path, output = args.parent_plan.resolve(), args.output.resolve()
    require(not output.exists(), "Refusing to overwrite a numerical diagnostic")
    require(args.count > 0, "count must be positive")
    if args.mode == "strict_math" and args.device == "cuda":
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
            "Strict mode requires CUBLAS_WORKSPACE_CONFIG in the fresh process environment before CUDA initialization")
    plan = protocol.read(parent_path)
    parent_protocol.check(plan)
    plan_sha = digest(parent_path)
    sources = {str(path): digest(path) for path in source_paths()}
    pool, parent = plan["pools"]["pilot"], plan["parents"]["pilot"]
    originals = read_jsonl(pool["manifest"])
    require(args.count <= len(originals), "count exceeds frozen pilot pool")
    parent_run = Path(parent["run"])
    metadata = protocol.read(parent_run / "run.json")
    result = protocol.read(parent_run / "evaluation/summary.json")
    records = read_jsonl(parent_run / "generated.jsonl")
    expected = {"config": parent["config"], "manifest_sha256": pool["sha256"], "ids": pool["ids"]}
    require(metadata["status"] == "complete" and metadata["config"] == parent["config"] and
        metadata["features"] == plan["feature_artifacts"] and
        metadata["fingerprint"] == config_fingerprint(expected) == result["generation_fingerprint"] and
        digest(parent_run / "generated.jsonl") == metadata["generated_sha256"] and
        digest(parent_run / "evaluation/summary.json") == parent["summary_sha256"] and
        [row["id"] for row in records] == pool["ids"], "Changed parent generation identity")
    cache_path = parent_path.parent / "parent-audits/pilot.json"
    cache = protocol.read(cache_path)
    cache_sha = digest(cache_path)
    require(cache["role"] == "pilot" and cache["features"] == metadata["features"] and
        cache["parent_summary_sha256"] == parent["summary_sha256"] and
        [row["id"] for row in cache["items"]] == pool["ids"], "Changed cached feature identity")
    gradient_root = output.parent / (output.stem + "-gradients")
    require(not gradient_root.exists(), "Partial gradient artifacts exist; archive them before retrying")
    items, assets = [], {}
    unsupported = None
    profile = None
    profile_input = None
    attack, spectral = AttackConfig(**parent["config"]["attack"]), SpectralConfig(**parent["config"]["spectral"])
    with backend_mode(args.mode):
        seed_everything(parent["config"]["seed"])
        flags = backend_manifest()
        vocoder, vocoder_provenance = load_frozen_vocoder(parent["config"]["features"], metadata["features"], args.device)
        surrogate, surrogate_provenance = load_whisper(parent["config"]["surrogate"], args.device)
        freeze(surrogate)
        require(surrogate_provenance == metadata["surrogate"], "Changed parent Whisper checkpoint/conditioning")
        try:
            for original, record, cached in zip(originals[:args.count], records[:args.count], cache["items"][:args.count]):
                uid = original["id"]
                delta_path = str(parent_run / (uid + ".delta.pt"))
                expected_assets = {original["audio"]: original["audio_sha256"], cached["q_path"]: cached["q_sha256"],
                    delta_path: parent["delta_hashes"][uid], **{asset["path"]: asset["sha256"] for asset in record["audio"].values()}}
                expected_seed = (int(parent["config"].get("seed", 2026)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % 2**32
                require(record["text"] == original["text"] and record["input_audio_sha256"] == original["audio_sha256"] and
                    record["seed"] == expected_seed and len(record["trace"]) == 50 and
                    cached["parent_delta_sha256"] == parent["delta_hashes"][uid] and
                    cached["parent_radius"] <= .100001 and
                    cached["resynthesis_max_abs_error"] <= plan["waveform_reproduction_tolerance"] and
                    all(digest(path) == expected for path, expected in expected_assets.items()), "Changed fixed source/q/delta/WAV input")
                assets.update(expected_assets)
                q = torch.load(cached["q_path"], map_location=args.device, weights_only=True).detach()
                delta = torch.load(delta_path, map_location=args.device, weights_only=True).detach()
                x = load_audio(original["audio"]).to(args.device).detach()
                radius = relative_radius(delta, q, attack.eps).item()
                require(delta.shape == q.shape and radius <= .100001 and
                    abs(radius-cached["parent_radius"]) < 1e-6 and
                    abs(radius-record["trace"][-1]["radius_after_update"]) < 1e-6, "Saved parent radius mismatch")
                states = {}
                current = {"id": uid, "reference": original["text"], "seed": record["seed"],
                    "q_value_sha256": tensor_value_sha256(q), "q_file_sha256": cached["q_sha256"],
                    "delta_value_sha256": tensor_value_sha256(delta), "parent_delta_file_sha256": parent["delta_hashes"][uid],
                    "source_audio_sha256": original["audio_sha256"], "states": states}
                items.append(current)
                for state, value in (("zero_delta", torch.zeros_like(q)), ("saved_parent_delta", delta)):
                    states[state] = repeat_state(q, x, value, original["text"], vocoder, surrogate, attack, spectral,
                        record["seed"], gradient_root / uid / state)
                    print(f"Numerical repeats {args.mode}: {uid} {state}; exact agreement={states[state]['numerical_bitwise_agreement']}.", flush=True)
                if profile_input is None:
                    profile_input = ((q, x, torch.zeros_like(q), original["text"], vocoder, surrogate, attack, spectral), record["seed"])
            if not args.no_profile:
                profile = profile_derivative(*profile_input)
        except RuntimeError as error:
            if args.mode != "strict_math" or not known_determinism_failure(error):
                raise
            unsupported = {"kind": "known_nondeterministic_operation", "exception_type": type(error).__name__,
                "message": str(error), "interpretation": "Strict deterministic execution rejected a known operation. The algorithm was not relaxed or substituted."}
            print(f"Strict mathematical backend unsupported: {unsupported['message']}", flush=True)
    parent_protocol.check(plan)
    require(digest(parent_path) == plan_sha and digest(cache_path) == cache_sha and
        all(digest(path) == expected for path, expected in {**sources, **assets}.items()), "Frozen input/source changed during repeatability probe")
    agreement = None if unsupported else all(state["numerical_bitwise_agreement"] for item in items for state in item["states"].values())
    report = {"status": "unsupported" if unsupported else "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_plan": str(parent_path), "parent_plan_sha256": plan_sha, "parent_run": str(parent_run),
        "parent_cache_audit_sha256": cache_sha, "source_hashes": sources, "checked_asset_hashes": assets,
        "mode": args.mode, "device": args.device, "requested_count": args.count, "completed_clip_count": sum(len(item["states"]) == 2 for item in items),
        "repeats_per_state": REPEATS, "backend": flags, "runtime": protocol.runtime(),
        "vocoder": vocoder_provenance, "surrogate": surrogate_provenance, "items": items,
        "numerical_bitwise_agreement": agreement, "unsupported": unsupported, "profile": profile,
        "scope": "Repeated fixed-point full-objective derivatives only; zero optimizer updates, decoding, WER or MOS inference.",
        "objective": "-NLL + lambda_perc * (TV_ratio + alpha_hf * HF_ratio), preserving parent operation ordering, flags and original-clean denominators",
        "limitations": ["Exact agreement is tested within one process and identical hardware/software; it does not prove cross-process or cross-platform determinism.",
            "Mathematical SDPA is an independently declared numerical backend, not identification of the authors' backend.",
            "cuDNN deterministic flags alone do not enforce all PyTorch operations; strict mode can reject reflection-padding backward.",
            "The profiler sample is a separate extra derivative after all repeated measurements and does not enter equality comparisons.",
            "No encoder inference occurs: cached q, source audio, parent delta and prior resynthesis evidence remain bound to their frozen parent.",
            "Raw-gradient reproducibility does not establish full-trajectory or decoded-transcript reproducibility."]}
    atomic_json(output, report)
    print(f"Numerical diagnostic {args.mode}: status={report['status']}, exact agreement={agreement}.", flush=True)
    return report


if __name__ == "__main__":
    main()
