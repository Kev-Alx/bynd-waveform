"""Measure moment-matched local counterfactuals along three full attacks.

Only the original full-objective branch advances. Counterfactual branches
are local diagnostics, not full loss-ablation attacks or quality estimates.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beyondwaveform.adam_diagnostics import counterfactual_step
from beyondwaveform.attacks import freeze, relative_radius, project_relative_ball_
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything
from beyondwaveform.losses import CleanReference
from beyondwaveform.models.whisper import load_whisper
from probe_feature_gradients import component_gradients, gradient_cosine, load_frozen_vocoder
import run_feature_diagnostics as parent_protocol
import run_quality_research as protocol
from parallel_jobs import now

CAPTURE_STEPS = (1, 10, 25, 50)


def compare_branches(optimizer, delta, gradients, q, rho=.1, eps=1e-8):
    """One local update from each gradient with the same moment history."""
    full = gradients["full"]
    values = {"full": full, "asr_only": gradients["negative_nll"],
        "without_tv": full-gradients["weighted_tv"], "without_hf": full-gradients["weighted_hf"]}
    branches = {name: counterfactual_step(optimizer, delta, gradient, q, rho, eps)
                for name, gradient in values.items()}
    baseline = branches["full"]
    reference_change = baseline["projected_change"]
    reference_norm = reference_change.double().norm().item()
    rows = {}
    for name, branch in branches.items():
        difference = (branch["delta_projected"]-baseline["delta_projected"]).double()
        rows[name] = {"gradient_norm": values[name].double().norm().item(),
            "unprojected_step_norm": branch["unprojected_change"].double().norm().item(),
            "projected_step_norm": branch["projected_change"].double().norm().item(),
            "projected_step_cosine_to_full": gradient_cosine(branch["projected_change"], reference_change),
            "projected_difference_from_full_norm": difference.norm().item(),
            "projected_difference_relative_to_full_step": difference.norm().item()/reference_norm if reference_norm else None,
            "projected_radius": relative_radius(branch["delta_projected"], q, eps).item()}
    return branches, rows


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def measure_trajectory(q, x, text, vocoder, surrogate, attack, spectral, evidence_dir):
    q, x = q.detach(), x.detach()
    delta = torch.zeros_like(q, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=attack.learning_rate)
    reference = CleanReference(q, x, spectral, attack.eps)
    captures, trace = [], []
    for step in range(1, 51):
        optimizer.zero_grad(set_to_none=True)
        audio = vocoder(q+delta)
        nll = surrogate.nll(audio, text)
        tv, hf = reference.terms(q+delta, audio)
        loss = -nll+attack.lambda_perc*(tv+attack.alpha_hf*hf)
        protocol.require(torch.isfinite(loss).item(), "Nonfinite trajectory loss")
        record = {"step": step, "nll_before_update": nll.item(), "loss_before_update": loss.item(),
            "tv_ratio": tv.item(), "hf_ratio": hf.item()}
        branches = None
        if step in CAPTURE_STEPS:
            components = {"negative_nll": -nll, "weighted_tv": attack.lambda_perc*tv,
                "weighted_hf": attack.lambda_perc*attack.alpha_hf*hf}
            component_values, total = component_gradients(components, delta, loss)
            gradients = {**component_values, "full": total}
            discrepancy = (sum(component_values.values())-total).double()
            protocol.require(discrepancy.abs().max().item() <= 1e-7+5e-4*total.abs().max().item(),
                "Component gradients do not recover the actual total")
            before = delta.detach().clone()
            state = cpu_copy(optimizer.state_dict())
            branches, comparisons = compare_branches(optimizer, delta, gradients, q, attack.rho, attack.eps)
            pack = {"step": step, "delta_before": before.cpu(), "optimizer_state_before": state,
                "gradients": cpu_copy(gradients), "branches": cpu_copy(branches)}
            current = {**record, "comparisons": comparisons,
                "gradient_sum_max_abs_discrepancy": discrepancy.abs().max().item(),
                "gradient_sum_difference_norm": discrepancy.norm().item(),
                "moment_history": "All branches use the same full-objective moment history before this update."}
        else:
            total, = torch.autograd.grad(loss, delta)
        protocol.require(torch.isfinite(total).all().item(), "Nonfinite trajectory gradient")
        delta.grad = total
        optimizer.step()
        project_relative_ball_(delta, q, attack.rho, attack.eps)
        record["radius_after_update"] = relative_radius(delta, q, attack.eps).item()
        record["gradient_norm"] = total.double().norm().item()
        protocol.require(record["radius_after_update"] <= .100001, "Trajectory exceeds original ball")
        if branches is not None:
            actual = delta.detach()
            cloned = branches["full"]["delta_projected"]
            error = (actual-cloned).abs().max().item()
            protocol.require(torch.equal(actual, cloned), "Full-gradient cloned Adam update differs from actual update")
            current.update(full_clone_max_abs_error=error, actual_radius=record["radius_after_update"])
            pack["actual_delta_after"] = actual.cpu().clone()
            path = evidence_dir / f"step-{step}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pack, path)
            current.update(evidence_path=str(path.resolve()), evidence_sha256=digest(path))
            captures.append(current)
        trace.append(record)
    protocol.require(not q.requires_grad and not any(p.grad is not None for model in (vocoder, surrogate) for p in model.parameters()),
        "Diagnostic accumulated clean-feature/model gradients")
    return {"trace": trace, "captures": captures, "actual_optimization_updates": 50,
        "counterfactual_trajectory_continuations": 0, "final_radius": relative_radius(delta, q, attack.eps).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    output, parent_path = args.output.resolve(), args.parent_plan.resolve()
    protocol.require(not output.exists() and args.count == 3, "Expected a new three-clip diagnostic")
    evidence_root = output.parent / "adam-evidence"
    protocol.require(not evidence_root.exists(), "Existing partial Adam evidence must be archived before a new attempt")
    parent_sha = digest(parent_path)
    plan = protocol.read(parent_path)
    parent_protocol.check(plan)
    parent, pool = plan["parents"]["pilot"], plan["pools"]["pilot"]
    parent_run = Path(parent["run"])
    metadata = protocol.read(parent_run / "run.json")
    protocol.require(metadata["config"] == parent["config"] and metadata["features"] == plan["feature_artifacts"], "Changed parent")
    config = parent["config"]
    attack, spectral = AttackConfig(**config["attack"]), SpectralConfig(**config["spectral"])
    protocol.require(attack.steps == 50 and attack.rho == .1 and attack.lambda_perc == 1 and attack.alpha_hf == 1 and
        attack.learning_rate == .1 and attack.optimizer == "adam" and attack.initialization == "zero" and
        attack.lr_schedule == "constant" and attack.attack and attack.temporal and attack.high_frequency and
        spectral.cutoff_hz == 6000 and spectral.sample_rate == 16000, "Changed full-objective protocol")
    cache_path = parent_path.parent / "parent-audits/pilot.json"
    cache_sha = digest(cache_path)
    cache = protocol.read(cache_path)
    cached = {row["id"]: row for row in cache["items"]}
    seeds = {row["id"]: row["seed"] for row in read_jsonl(parent_run / "generated.jsonl")}
    source_hashes = {str(path): digest(path) for path in (Path(__file__).resolve(),
        ROOT / "beyondwaveform/adam_diagnostics.py", ROOT / "scripts/probe_feature_gradients.py",
        ROOT / "beyondwaveform/attacks.py", ROOT / "beyondwaveform/losses.py", ROOT / "beyondwaveform/models/whisper.py")}
    seed_everything(config["seed"])
    vocoder, vocoder_meta = load_frozen_vocoder(config["features"], metadata["features"], "cuda")
    surrogate, surrogate_meta = load_whisper(config["surrogate"], "cuda")
    freeze(surrogate)
    protocol.require(surrogate_meta == metadata["surrogate"], "Changed Whisper artifacts")
    items, assets = [], {}
    for original in read_jsonl(pool["manifest"])[:3]:
        uid, item = original["id"], cached[original["id"]]
        protocol.require(item["parent_delta_sha256"] == parent["delta_hashes"][uid] and
            cache["features"] == plan["feature_artifacts"] and cache["parent_summary_sha256"] == parent["summary_sha256"],
            "Changed cached clean-feature binding")
        for path, expected in ((original["audio"], original["audio_sha256"]), (item["q_path"], item["q_sha256"])):
            protocol.require(digest(path) == expected, "Changed input/cache tensor")
            assets[path] = expected
        seed_everything(seeds[uid])
        q = torch.load(item["q_path"], map_location="cuda", weights_only=True).detach()
        x = load_audio(original["audio"]).cuda()
        value = measure_trajectory(q, x, original["text"], vocoder, surrogate, attack, spectral, evidence_root / uid)
        items.append({"id": uid, "selection": "first three frozen pilot entries", "seed": seeds[uid], **value})
        print(f"Adam contributions: {uid}, 50 full updates, four local moment-matched comparisons.", flush=True)
    parent_protocol.check(plan)
    protocol.require(digest(parent_path) == parent_sha and digest(cache_path) == cache_sha, "Parent plan/cache receipt changed")
    protocol.require(all(digest(path) == expected for path, expected in {**source_hashes, **assets}.items()), "Input/source changed")
    atomic_json(output, {"status": "complete", "created_at": now(), "parent_plan": str(parent_path),
        "parent_plan_sha256": parent_sha, "parent_cache_audit_sha256": cache_sha, "count": 3,
        "source_hashes": source_hashes, "checked_asset_hashes": assets, "vocoder": vocoder_meta, "surrogate": surrogate_meta,
        "runtime": protocol.runtime(), "capture_steps": list(CAPTURE_STEPS), "items": items,
        "scope": "Local gradient-removal counterfactuals with identical actual Adam moments; three full-objective 50-update trajectories. No free-decoding/MOS inference, full no-perceptual ablation or candidate selection."})


if __name__ == "__main__":
    main()
