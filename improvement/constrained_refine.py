"""Source-NLL waveform refinement with fixed nonlinear quality constraints.

This extends one already completed full-R3 PGD50 waveform by exactly 50 updates.
The constrained condition maximizes the unchanged source sum-NLL while never
exceeding that waveform's STFT/content penalties. It does not decode speech or
choose among candidate attacks by a recognition/quality score. A matched control
continues the original weighted R3 loss for the same 50 gradient updates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Callable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from r3_waveform import multiresolution_stft_loss, project_waveform_residual, relative_feature_loss

STFT_WEIGHT = 65.08367870148994
CONTENT_WEIGHT = 186.6123439723859
Condition = Literal["constrained", "weighted_control"]
TERMS = ("source", "stft", "content")


class InvalidRefinement(ValueError):
    """An observed input, objective, derivative, or feasibility failure."""


@dataclass(frozen=True)
class RefineConfig:
    epsilon: float = 0.003
    rms_limit: float = 0.003
    initial_updates: int = 50
    additional_updates: int = 50
    step_size: float = 0.00012
    stft_weight: float = STFT_WEIGHT
    content_weight: float = CONTENT_WEIGHT
    inward_fraction: float = 0.01
    backtracking_trials: int = 10

    def __post_init__(self):
        if (self.epsilon != .003 or self.rms_limit != .003 or self.initial_updates != 50
                or self.additional_updates != 50 or self.step_size != .00012
                or self.stft_weight != STFT_WEIGHT or self.content_weight != CONTENT_WEIGHT
                or self.inward_fraction != .01 or self.backtracking_trials != 10):
            raise ValueError("Refinement uses frozen PGD100-total bounds, weights, step and backtracking")


@dataclass
class RefineResult:
    adversarial: Tensor
    residual: Tensor
    trace: list[dict]
    final_metrics: dict
    counts: dict


class RefineFailure(RuntimeError):
    def __init__(self, message: str, receipt: Mapping, *, adversarial: Tensor | None = None,
                 residual: Tensor | None = None):
        super().__init__(message)
        self.receipt = dict(receipt)
        self.adversarial = None if adversarial is None else adversarial.detach().clone()
        self.residual = None if residual is None else residual.detach().clone()


def _scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise InvalidRefinement(f"{name} must be a finite scalar")
    return float(value.detach())


def _norm(value: Tensor) -> float:
    result = float(torch.linalg.vector_norm(value.detach().double()))
    if not math.isfinite(result):
        raise InvalidRefinement("Nonfinite audio-gradient norm")
    return result


def _tolerance(value: Tensor, scale: float = 1.) -> float:
    return 4 * torch.finfo(value.dtype).eps * max(1., abs(scale))


def _cap_tolerance(value: Tensor, cap: float) -> float:
    return max(1e-12, 32 * torch.finfo(value.dtype).eps * abs(cap))


def _source_tolerance(value: Tensor, nll: float) -> float:
    return max(1e-7, 8 * torch.finfo(value.dtype).eps * abs(nll))


def physical_bounds(original: Tensor, waveform: Tensor, config: RefineConfig | None = None) -> dict:
    cfg = config or RefineConfig()
    if (waveform.shape != original.shape or not bool(torch.isfinite(waveform).all())):
        raise InvalidRefinement("Nonfinite or misaligned candidate waveform")
    residual = waveform - original
    linf, rms, peak = float(residual.abs().max()), float(residual.square().mean().sqrt()), float(waveform.abs().max())
    tolerance = _tolerance(original)
    if linf > cfg.epsilon + tolerance or rms > cfg.rms_limit + tolerance or peak > 1.:
        raise InvalidRefinement("Waveform violates frozen Linf/RMS/peak constraints")
    return {"residual_linf": linf, "residual_rms": rms, "audio_peak": peak,
            "constraint_roundoff_tolerance": tolerance}


def sequential_component_gradients(waveform: Tensor, source_nll: Callable[[Tensor], Tensor],
                                   stft_loss: Callable[[Tensor], Tensor],
                                   content_loss: Callable[[Tensor], Tensor], *,
                                   counts: dict | None = None) -> tuple[dict[str, Tensor], dict]:
    """Measure three separate derivatives at identical audio without retaining graphs.

    Legitimately zero derivatives are retained, including a zero source gradient
    at a stationary point. No alternative objective or random direction replaces
    them. Counts increment before attempted calls, including failed calls.
    """
    calls = counts if counts is not None else {}
    gradients, values = {}, {}
    for name, function in zip(TERMS, (source_nll, stft_loss, content_loss)):
        current = waveform.detach().requires_grad_(True)
        for key in (f"{name}_forward_calls", f"{name}_gradient_forward_calls", f"{name}_backward_calls"):
            calls.setdefault(key, 0)
        calls[f"{name}_forward_calls"] += 1
        calls[f"{name}_gradient_forward_calls"] += 1
        loss = function(current)
        values[name] = _scalar(loss, name)
        if not loss.requires_grad:
            raise InvalidRefinement(f"{name} has no audio autograd graph")
        calls[f"{name}_backward_calls"] += 1
        gradient, = torch.autograd.grad(loss, current)
        if gradient.shape != waveform.shape or not bool(torch.isfinite(gradient).all()):
            raise InvalidRefinement(f"{name} gradient is nonfinite or misaligned")
        gradients[name] = gradient.detach()
        del loss, current
    return gradients, _metrics(values, gradients)


def _metrics(values: Mapping[str, float], gradients: Mapping[str, Tensor] | None = None) -> dict:
    result = {"source_sum_nll": values["source"], "stft_loss": values["stft"],
              "content_loss": values["content"],
              "full_r3_objective": -values["source"] + STFT_WEIGHT * values["stft"] + CONTENT_WEIGHT * values["content"]}
    if not math.isfinite(result["full_r3_objective"]):
        raise InvalidRefinement("Nonfinite weighted full objective")
    if gradients is not None:
        result.update({f"{name}_gradient_norm": _norm(gradient) for name, gradient in gradients.items()})
    return result


def measure_components(waveform: Tensor, functions: Sequence[Callable[[Tensor], Tensor]], *,
                       counts: dict, purpose: str) -> dict:
    values = {}
    with torch.no_grad():
        for name, function in zip(TERMS, functions):
            for key in (f"{name}_forward_calls", f"{name}_{purpose}_forward_calls"):
                counts.setdefault(key, 0)
                counts[key] += 1
            values[name] = _scalar(function(waveform), f"{purpose} {name}")
    return _metrics(values)


def project_two_halfspaces(preferred: Tensor, normals: Sequence[Tensor], bounds: Sequence[float], *,
                           free_mask: Tensor | None = None) -> tuple[Tensor | None, dict]:
    """Exact small active-set Euclidean projection, including parallel planes.

    Solve min ||d-u||²/2 subject to a_i*d <= b_i for at most two
    halfspaces and d[~free_mask]=0. All four active subsets are considered,
    with analytic one-plane and two-by-two Gram solves and KKT checks. A
    singular two-plane set needs no inverse: feasible singleton candidates
    already cover its nonempty projection. No sequential projection/order bias.
    """
    if len(normals) != len(bounds) or len(normals) > 2:
        raise ValueError("At most two matching halfspace normals/bounds required")
    if not preferred.is_floating_point() or not bool(torch.isfinite(preferred).all()):
        raise InvalidRefinement("Preferred direction must be finite floating point")
    mask = torch.ones_like(preferred, dtype=torch.bool) if free_mask is None else free_mask
    if mask.shape != preferred.shape or mask.dtype != torch.bool:
        raise ValueError("Boolean free-coordinate mask must match preferred direction")
    u = preferred.detach().double().masked_fill(~mask, 0.)
    rows, limits = [], []
    for normal, bound in zip(normals, bounds):
        if normal.shape != preferred.shape or not bool(torch.isfinite(normal).all()) or not math.isfinite(float(bound)):
            raise InvalidRefinement("Invalid quality halfspace")
        rows.append(normal.detach().double().masked_fill(~mask, 0.))
        limits.append(float(bound))
    tol = 128 * torch.finfo(torch.float64).eps * max(1., _norm(u), *(abs(b) for b in limits))
    candidates = []
    examined = 0
    singular_sets = 0
    for size in range(len(rows) + 1):
        for active in itertools.combinations(range(len(rows)), size):
            examined += 1
            multipliers = []
            if size == 0:
                candidate = u
            elif size == 1:
                i, = active
                gram = float((rows[i] * rows[i]).sum())
                if gram == 0.:
                    singular_sets += 1
                    continue
                multiplier = (float((rows[i] * u).sum()) - limits[i]) / gram
                multipliers = [multiplier]
                candidate = u - multiplier * rows[i]
            else:
                i, j = active
                aa = float((rows[i] * rows[i]).sum())
                ab = float((rows[i] * rows[j]).sum())
                bb = float((rows[j] * rows[j]).sum())
                determinant = aa * bb - ab * ab
                if determinant <= 64 * torch.finfo(torch.float64).eps * max(aa * bb, 1e-300):
                    singular_sets += 1
                    continue
                ra = float((rows[i] * u).sum()) - limits[i]
                rb = float((rows[j] * u).sum()) - limits[j]
                multipliers = [(bb * ra - ab * rb) / determinant,
                               (aa * rb - ab * ra) / determinant]
                candidate = u - multipliers[0] * rows[i] - multipliers[1] * rows[j]
            if any(value < -tol for value in multipliers):
                continue
            residuals = [float((normal * candidate).sum()) - bound for normal, bound in zip(rows, limits)]
            if any(value > tol for value in residuals):
                continue
            distance = float((candidate - u).square().sum())
            candidates.append((distance, candidate, active, multipliers, residuals))
    if not candidates:
        return None, {"feasible": False, "active_subsets_examined": examined,
                      "singular_subsets": singular_sets, "numerical_tolerance": tol}
    distance, candidate, active, multipliers, residuals = min(candidates, key=lambda value: value[0])
    return candidate.to(dtype=preferred.dtype), {
        "feasible": True, "active_subsets_examined": examined, "singular_subsets": singular_sets,
        "active_set": list(active), "multipliers": multipliers,
        "halfspace_residuals_float64": residuals, "distance_squared": distance,
        "numerical_tolerance": tol,
    }


def constrained_direction(original: Tensor, waveform: Tensor, gradients: Mapping[str, Tensor], *,
                          config: RefineConfig | None = None) -> tuple[Tensor, dict]:
    cfg = config or RefineConfig()
    preferred = gradients["source"].sign()
    lower = torch.maximum(original - cfg.epsilon, torch.full_like(original, -1.))
    upper = torch.minimum(original + cfg.epsilon, torch.full_like(original, 1.))
    # Suppress coordinates whose preferred direction already points out of the
    # physical box. Projection may rotate another coordinate toward a bound;
    # the physical projection and nonlinear checks below handle that exactly.
    mask = ~(((preferred > 0) & (waveform >= upper)) | ((preferred < 0) & (waveform <= lower)))
    preferred = preferred.masked_fill(~mask, 0.)
    normals, bounds, names = [], [], []
    for name in ("stft", "content"):
        normal = gradients[name].masked_fill(~mask, 0.)
        norm = _norm(normal)
        if norm == 0.:
            continue
        normal = normal / norm
        normals.append(normal)
        # Fixed inward margin is one percent of the conflicting preferred
        # component. Nonconflicting terms use the nonincrease tangent plane.
        bounds.append(-cfg.inward_fraction * max(float((normal.double() * preferred.double()).sum()), 0.))
        names.append(name)
    direction, qp = project_two_halfspaces(preferred, normals, bounds, free_mask=mask)
    used_zero_margin = direction is None
    if direction is None:
        direction, qp = project_two_halfspaces(preferred, normals, [0.] * len(normals), free_mask=mask)
    if direction is None:
        raise InvalidRefinement("Zero-margin quality projection unexpectedly infeasible")
    maximum = float(direction.abs().max())
    if maximum > 0.:
        direction = direction / maximum
    return direction, {"quality_directions": names, "inward_halfspace_bounds": bounds,
                       "inward_bounds_apply_before_linf_normalization": True,
                       "projected_direction_linf_before_normalization": maximum,
                       "linf_normalization_multiplier": 1. / maximum if maximum > 0. else 1.,
                       "final_quality_directional_derivatives": {
                           name: float((normal.double() * direction.double()).sum())
                           for name, normal in zip(names, normals)},
                       "zero_margin_fallback": used_zero_margin, "qp": qp,
                       "free_coordinate_count": int(mask.sum()), "preferred_norm": _norm(preferred),
                       "direction_norm": _norm(direction),
                       "source_directional_derivative": float((gradients["source"].double() * direction.double()).sum()),
                       "direction_rule": "box-free sign(source), exact quality halfspace projection, Linf normalization without re-signing"}


def nonlinear_backtracking_step(original: Tensor, waveform: Tensor, direction: Tensor,
                                functions: Sequence[Callable[[Tensor], Tensor]], *,
                                caps: Mapping[str, float], current_source: float, counts: dict,
                                config: RefineConfig | None = None) -> tuple[Tensor, dict | None, list[dict]]:
    """Return the first valid trial, otherwise retain exactly the current audio."""
    cfg = config or RefineConfig()
    trials = []
    if not bool(direction.abs().max() > 0):
        return waveform.detach(), None, trials
    for trial in range(cfg.backtracking_trials):
        factor = 2. ** (-trial)
        with torch.no_grad():
            residual = project_waveform_residual(original,
                waveform - original + cfg.step_size * factor * direction,
                cfg.epsilon, cfg.rms_limit)
            candidate = (original + residual).detach()
        row = {"trial": trial + 1, "factor": factor, "accepted": False}
        forward_before = {name: counts.get(f"{name}_forward_calls", 0) for name in TERMS}
        try:
            row.update(physical_bounds(original, candidate, cfg))
            metrics = measure_components(candidate, functions, counts=counts, purpose="trial")
            row.update(metrics)
            row["source_nondecrease"] = metrics["source_sum_nll"] >= current_source - _source_tolerance(waveform, current_source)
            row["stft_cap_satisfied"] = metrics["stft_loss"] <= caps["stft_loss"] + _cap_tolerance(waveform, caps["stft_loss"])
            row["content_cap_satisfied"] = metrics["content_loss"] <= caps["content_loss"] + _cap_tolerance(waveform, caps["content_loss"])
            row["accepted"] = row["source_nondecrease"] and row["stft_cap_satisfied"] and row["content_cap_satisfied"]
        except InvalidRefinement as error:
            row["invalid_reason"] = str(error)
            metrics = None
        row["forward_calls"] = {
            name: counts.get(f"{name}_forward_calls", 0) - forward_before[name]
            for name in TERMS}
        trials.append(row)
        if row["accepted"]:
            return candidate, metrics, trials
    return waveform.detach(), None, trials


class WaveformRefiner:
    def __init__(self, source_nll: Callable[[Tensor, str], Tensor],
                 feature_encoder: Callable[[Tensor], Tensor], *,
                 condition: Condition = "constrained", config: RefineConfig | None = None):
        self.config = config or RefineConfig()
        if condition not in ("constrained", "weighted_control"):
            raise ValueError("Unknown refinement condition")
        if isinstance(feature_encoder, nn.Module) and (feature_encoder.training or any(p.requires_grad for p in feature_encoder.parameters())):
            raise ValueError("Content encoder must already be frozen in evaluation mode")
        self.source_nll, self.feature_encoder, self.condition = source_nll, feature_encoder, condition

    def run(self, original: Tensor, text: str, *, initial_waveform: Tensor,
            clean_features: Tensor | None = None, on_step: Callable[[dict], None] | None = None) -> RefineResult:
        cfg = self.config
        counts = {f"{name}_{operation}_calls": 0 for name in TERMS for operation in ("forward", "backward")}
        counts.update(clean_reference_forward_calls=0, completed_updates=0, accepted_updates=0, noop_updates=0,
                      line_search_trials=0, zero_margin_fallbacks=0)
        trace = []
        waveform = None
        stage, step = "input", 0
        try:
            if (original.ndim != 1 or original.numel() <= 512 or not original.is_floating_point()
                    or not bool(torch.isfinite(original).all()) or bool((original.abs() > 1.).any())):
                raise InvalidRefinement("One finite untruncated mono waveform >512 samples in [-1,1] required")
            if not isinstance(text, str) or not text.strip():
                raise InvalidRefinement("Unchanged nonempty original reference required")
            if initial_waveform.dtype != original.dtype or initial_waveform.device != original.device:
                raise InvalidRefinement("Initial waveform must match original dtype and device")
            original, waveform = original.detach(), initial_waveform.detach().clone()
            physical_bounds(original, waveform, cfg)
            stage = "clean_reference"
            if clean_features is None:
                counts["clean_reference_forward_calls"] += 1
                with torch.no_grad():
                    clean_features = self.feature_encoder(original).detach()
            else:
                clean_features = clean_features.detach()
            if not bool(torch.isfinite(clean_features).all()):
                raise InvalidRefinement("Nonfinite frozen clean content reference")
            functions = (lambda x: self.source_nll(x, text),
                         lambda x: multiresolution_stft_loss(x, original),
                         lambda x: relative_feature_loss(self.feature_encoder(x), clean_features))
            stage = "initial_metrics"
            initial_metrics = measure_components(waveform, functions, counts=counts, purpose="initial")
            caps = {name: initial_metrics[name] for name in ("stft_loss", "content_loss")}
            for step in range(cfg.additional_updates):
                stage = "gradient"
                gradients, row = sequential_component_gradients(waveform, *functions, counts=counts)
                stage = "direction"
                if self.condition == "constrained":
                    direction, direction_receipt = constrained_direction(original, waveform, gradients, config=cfg)
                    row.update(direction_receipt)
                    counts["zero_margin_fallbacks"] += int(direction_receipt["zero_margin_fallback"])
                    stage = "nonlinear_backtracking"
                    candidate, accepted_metrics, trials = nonlinear_backtracking_step(
                        original, waveform, direction, functions, caps=caps,
                        current_source=max(row["source_sum_nll"], initial_metrics["source_sum_nll"]),
                        counts=counts, config=cfg)
                    accepted = accepted_metrics is not None
                    row.update(source_eligibility_floor=max(row["source_sum_nll"], initial_metrics["source_sum_nll"]),
                               line_search=trials, accepted_metrics=accepted_metrics,
                               accepted_backtracking_factor=next((t["factor"] for t in trials if t["accepted"]), None))
                    counts["line_search_trials"] += len(trials)
                else:
                    combined = (-gradients["source"] + cfg.stft_weight * gradients["stft"]
                                + cfg.content_weight * gradients["content"])
                    if not bool(torch.isfinite(combined).all()):
                        raise InvalidRefinement("Nonfinite weighted continuation gradient")
                    with torch.no_grad():
                        residual = project_waveform_residual(original,
                            waveform - original - cfg.step_size * combined.sign(), cfg.epsilon, cfg.rms_limit)
                        candidate = (original + residual).detach()
                    accepted = bool((candidate != waveform).any())
                    row.update(combined_gradient_norm=_norm(combined), line_search=[],
                               direction_rule="subtract frozen full-R3 combined-gradient sign")
                stage = "physical_check"
                bounds = physical_bounds(original, candidate, cfg)
                changed = bool((candidate != waveform).any())
                waveform = candidate
                counts["completed_updates"] += 1
                counts["accepted_updates"] += int(accepted and changed)
                counts["noop_updates"] += int(not (accepted and changed))
                row.update(step=step + 1, total_pgd_step=cfg.initial_updates + step + 1,
                           loss_evaluation_iterate=cfg.initial_updates + step,
                           objective_condition=self.condition, accepted=accepted, changed=changed, **bounds)
                trace.append(row)
                stage = "on_step_callback"
                if on_step is not None:
                    on_step(dict(row))
            stage = "final_metrics"
            final_metrics = measure_components(waveform, functions, counts=counts, purpose="final")
            final_metrics.update(physical_bounds(original, waveform, cfg))
            final_metrics.update(objective_condition=self.condition, initial_metrics=initial_metrics,
                                 quality_caps=caps, selected_iterate=cfg.initial_updates + cfg.additional_updates,
                                 initial_updates=cfg.initial_updates, additional_updates=cfg.additional_updates,
                                 selection="last feasible iterate after 50 registered additional updates; no decoding-based selection",
                                 quality_cap_roundoff_tolerance={name: _cap_tolerance(waveform, cap) for name, cap in caps.items()},
                                 source_nondecrease_from_initial=final_metrics["source_sum_nll"] >= initial_metrics["source_sum_nll"] - _source_tolerance(waveform, initial_metrics["source_sum_nll"]))
            final_metrics["stft_cap_satisfied"] = final_metrics["stft_loss"] <= caps["stft_loss"] + _cap_tolerance(waveform, caps["stft_loss"])
            final_metrics["content_cap_satisfied"] = final_metrics["content_loss"] <= caps["content_loss"] + _cap_tolerance(waveform, caps["content_loss"])
            if self.condition == "constrained" and not (final_metrics["stft_cap_satisfied"]
                    and final_metrics["content_cap_satisfied"] and final_metrics["source_nondecrease_from_initial"]):
                raise InvalidRefinement("Final constrained waveform exceeded a quality cap or fell below its initial NLL floor")
            return RefineResult(waveform.detach(), (waveform - original).detach(), trace, final_metrics, dict(counts))
        except InvalidRefinement as error:
            if stage == "on_step_callback":
                raise
            raise RefineFailure(str(error), {
                "status": "failed", "stage": stage, "step": step,
                "reason": str(error), "condition": self.condition, "config": asdict(cfg),
                "counts": dict(counts), "trace": trace,
                "output_role": "failed attempt, retained last feasible iterate, not completed refinement",
            }, adversarial=waveform, residual=None if waveform is None else waveform - original) from error
