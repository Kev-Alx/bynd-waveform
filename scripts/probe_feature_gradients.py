"""Inspect frozen objective gradients at clean and saved feature deltas.

This diagnostic performs no decoding, optimization, or candidate selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.attacks import freeze, relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig, load_config
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything
from beyondwaveform.losses import CleanReference, temporal_variation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def component_gradients(components, delta, total_loss=None):
    """Differentiate each original loss term and their total at one point."""
    require(delta.is_leaf and delta.requires_grad, "delta must be a live leaf")

    def gradient(value, retain_graph):
        require(value.ndim == 0 and torch.isfinite(value).item(), "Invalid scalar loss term")
        if not value.requires_grad:
            return torch.zeros_like(delta)
        result, = torch.autograd.grad(value, delta, retain_graph=retain_graph, allow_unused=True)
        return torch.zeros_like(delta) if result is None else result.detach()

    gradients = {name: gradient(value, True) for name, value in components.items()}
    total = gradient(sum(components.values()) if total_loss is None else total_loss, False)
    require(all(torch.isfinite(g).all().item() for g in [*gradients.values(), total]), "Nonfinite gradient")
    return gradients, total


def gradient_cosine(left, right):
    a, b = left.detach().double().flatten(), right.detach().double().flatten()
    an, bn = a.norm(), b.norm()
    if an.item() == 0 or bn.item() == 0:
        return None
    return (a.dot(b) / (an * bn)).clamp(-1, 1).item()


def delta_variation(delta):
    value = delta.detach().double()
    energy = value.square().sum().item()
    variation = temporal_variation(value).item()
    return {
        "energy_sum": energy,
        "temporal_variation": variation,
        "temporal_variation_over_energy": variation / energy if energy > 0 else None,
        "difference_energy_over_energy": variation * (value.shape[0] - 1) / energy if energy > 0 else None,
        "normalization": "TV(delta)=mean_t sum_d(diff_t delta)^2; energy=sum_t,d delta^2; zero-energy ratios are undefined",
    }


def probe_state(q, x, delta, text, vocoder, surrogate, attack, spectral):
    """Report the unchanged objective; only the copied delta has gradients."""
    freeze(vocoder)
    freeze(surrogate)
    q, x = q.detach(), x.detach()
    live = delta.detach().clone().requires_grad_()
    require(q.shape == live.shape and q.ndim == 2 and q.shape[0] >= 2, "Invalid feature/delta shape")
    require(torch.isfinite(q).all().item() and torch.isfinite(live).all().item(), "Nonfinite features")
    reference = CleanReference(q, x, spectral, attack.eps)
    audio = vocoder(q + live)
    nll = surrogate.nll(audio, text) if attack.attack else live.new_zeros(())
    tv, hf = reference.terms(q + live, audio,
        temporal=attack.temporal and attack.lambda_perc > 0,
        high_frequency=attack.high_frequency and attack.lambda_perc > 0 and attack.alpha_hf > 0)
    components = {"negative_nll": -nll, "weighted_tv": attack.lambda_perc * tv,
                  "weighted_hf": attack.lambda_perc * attack.alpha_hf * hf}
    # Keep the parent implementation's operation ordering for the independent
    # total derivative, rather than differentiating a reassociated scalar sum.
    total_loss = -nll + attack.lambda_perc * (tv + attack.alpha_hf * hf)
    scalars = {name: value.detach().item() for name, value in components.items()}
    gradients, total = component_gradients(components, live, total_loss)
    summed = sum(gradients.values())
    discrepancy = (summed - total).double()
    maximum_difference = discrepancy.abs().max().item()
    total_scale = total.abs().max().item()
    require(maximum_difference <= 1e-7 + 5e-4 * total_scale, "Component gradients do not recover the total")
    names = list(gradients)
    return {"nll": nll.detach().item(), "tv_ratio": tv.detach().item(), "hf_ratio": hf.detach().item(),
        "original_clean_tv": reference.tv.item(), "original_clean_hf": reference.hf.item(),
        "loss_components": scalars, "loss_total": total_loss.detach().item(), "component_scalar_sum": sum(scalars.values()),
        "gradient_norms": {name: value.double().norm().item() for name, value in gradients.items()},
        "pairwise_gradient_cosines": {f"{left}__{right}": gradient_cosine(gradients[left], gradients[right])
            for i, left in enumerate(names) for right in names[i + 1:]},
        "combined_gradient_norm": total.double().norm().item(),
        "component_sum_gradient_norm": summed.double().norm().item(),
        "component_sum_max_abs_difference": maximum_difference,
        "component_sum_difference_norm": discrepancy.norm().item(),
        "relative_delta_radius": relative_radius(live.detach(), q, attack.eps).item(),
        "delta_variation": delta_variation(live), "q_requires_grad": q.requires_grad,
        "model_parameter_gradients_present": any(p.grad is not None for module in (vocoder, surrogate) for p in module.parameters())}


def load_frozen_vocoder(config, expected, device):
    """Load the identical upstream generator without an unused WavLM model."""
    from beyondwaveform.assets import KNN_REVISION
    from beyondwaveform.models.feature import FeatureVocoder
    artifact_root = Path(config.get("artifact_root", "artifacts"))
    source = artifact_root / "sources" / f"knn-vc-{KNN_REVISION}"
    variant = config.get("vocoder_variant", "regular")
    require(variant in {"regular", "prematched"}, "Unknown vocoder variant")
    weights = artifact_root / "checkpoints" / ("g_02500000.pt" if variant == "regular" else "prematch_g_02500000.pt")
    generator_config = source / "hifigan/config_v1_wavlm.json"
    require(digest(weights) == expected["vocoder_sha256"], "Changed parent vocoder weights")
    settings = json.loads(generator_config.read_text(encoding="utf-8"))
    require(settings == expected["vocoder_config"] and variant == expected["vocoder_variant"] and
            KNN_REVISION == expected["source_revision"], "Changed parent vocoder contract")
    sys.path.insert(0, str(source.resolve()))
    try:
        from hifigan.models import Generator
        from hifigan.utils import AttrDict
    finally:
        sys.path.pop(0)
    generator = Generator(AttrDict(settings))
    generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["generator"], strict=True)
    generator.remove_weight_norm()
    return freeze(FeatureVocoder(generator, settings["hubert_dim"]).to(device)), {
        "weights_path": str(weights.resolve()), "weights_sha256": digest(weights),
        "config_path": str(generator_config.resolve()), "config_sha256": digest(generator_config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    require(args.count > 0, "count must be positive")
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    require(not output.exists(), "Refusing to overwrite an existing diagnostic report")
    # Import only when executing the CLI, after the driver and plan are frozen.
    from run_feature_diagnostics import check
    from beyondwaveform.models.whisper import load_whisper
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    check(plan)
    plan_sha = digest(plan_path)
    source = read_jsonl(plan["pools"]["pilot"]["manifest"])
    require(args.count <= len(source), "count exceeds the pilot pool")
    parent = plan["parents"]["pilot"]
    config_value = parent["config"]
    config = config_value if isinstance(config_value, dict) else load_config(config_value)
    parent_run = Path(parent["run"])
    metadata = json.loads((parent_run / "run.json").read_text(encoding="utf-8"))
    require(metadata["config"] == config and metadata["status"] == "complete", "Changed parent identity")
    cache_path = plan_path.parent / "parent-audits/pilot.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in cache["items"]}
    require(len(items) == len(cache["items"]), "Duplicate cached feature IDs")
    attack, spectral = AttackConfig(**config["attack"]), SpectralConfig(**config["spectral"])
    seed_everything(config["seed"])
    vocoder, vocoder_provenance = load_frozen_vocoder(config["features"], metadata["features"], "cuda")
    surrogate, surrogate_provenance = load_whisper(config["surrogate"], "cuda")
    freeze(surrogate)
    require(surrogate_provenance == metadata["surrogate"], "Changed parent Whisper contract")
    rows = []
    for original in source[:args.count]:
        uid = original["id"]
        item = items[uid]
        q_path = Path(item["q_path"])
        delta_path = parent_run / f"{uid}.delta.pt"
        require(digest(q_path) == item["q_sha256"], "Changed cached clean features")
        require(digest(delta_path) == parent["delta_hashes"][uid], "Changed saved parent delta")
        require(digest(original["audio"]) == original["audio_sha256"], "Changed original source audio")
        q = torch.load(q_path, map_location="cuda", weights_only=True).detach()
        delta = torch.load(delta_path, map_location="cuda", weights_only=True).detach()
        x = load_audio(original["audio"]).to("cuda").detach()
        states = {"zero_delta": torch.zeros_like(q), "saved_parent_delta": delta}
        measurements = {name: probe_state(q, x, value, original["text"], vocoder, surrogate, attack, spectral)
                        for name, value in states.items()}
        require(all(not state["q_requires_grad"] and not state["model_parameter_gradients_present"]
                    for state in measurements.values()), "Probe accumulated parameter/clean-feature gradients")
        rows.append({"id": uid, "selection": "first entries in frozen manifest order", "q_path": str(q_path),
                     "q_sha256": item["q_sha256"], "delta_path": str(delta_path),
                     "delta_sha256": parent["delta_hashes"][uid], "source_audio_sha256": original["audio_sha256"],
                     "feature_shape": list(q.shape), "states": measurements})
        print(f"Audited objective gradients: {uid}", flush=True)
    check(plan)
    require(digest(plan_path) == plan_sha, "Plan changed during gradient probe")
    atomic_json(output, {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path), "plan_sha256": plan_sha, "parent_run": str(parent_run),
        "parent_cache_audit_sha256": digest(cache_path), "config": config,
        "vocoder": vocoder_provenance, "surrogate": surrogate_provenance,
        "source_hashes": {str(path): digest(path) for path in [Path(__file__).resolve(), ROOT / "beyondwaveform/losses.py",
            ROOT / "beyondwaveform/attacks.py", ROOT / "beyondwaveform/models/feature.py", ROOT / "beyondwaveform/models/whisper.py"]},
        "objective": "-NLL + lambda_perc * (TV_ratio + alpha_hf * HF_ratio), preserving parent flags and original-clean denominators",
        "gradient_space": "unpreconditioned derivatives with respect to delta at fixed q; no optimizer update",
        "count": len(rows), "items": rows,
        "limitations": ["Descriptive local derivatives on the first manifest clips, not an attack-success or quality estimate.",
            "Gradient norms/cosines do not establish a causal explanation or optimal loss weights.",
            "No decoding, MOS inference, optimization, or parameter tuning occurs.",
            "Cached clean q is hash-verified; this probe does not re-extract WavLM features.",
            "Zero-vector gradient cosines and zero-energy delta-normalized TV are undefined and reported as null."]})


if __name__ == "__main__":
    main()
