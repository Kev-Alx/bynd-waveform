"""Per-utterance waveform optimization with the frozen R3 loss and bounds.

There is no generator, decoding, quality-based iterate selection, random start,
or model loading in this module. The caller owns registration, frozen models,
audio/reference provenance, timing, and persisted receipts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Literal, Mapping

import torch
from torch import Tensor, nn

from r3_waveform import (
    multiresolution_stft_loss, project_waveform_residual, relative_feature_loss,
)


STFT_WEIGHT = 65.08367870148994
CONTENT_WEIGHT = 186.6123439723859
Condition = Literal["full_r3", "nll_only"]


class _InvalidMathematics(ValueError):
    """An explicitly checked input, objective, derivative, or feasible-set failure."""


@dataclass(frozen=True)
class PGDConfig:
    """The single preregistered update rule; no hidden optimizer search."""
    epsilon: float = 0.003
    rms_limit: float = 0.003
    updates: int = 50
    step_size: float = 0.00012
    stft_weight: float = STFT_WEIGHT
    content_weight: float = CONTENT_WEIGHT

    def __post_init__(self):
        if (self.epsilon != 0.003 or self.rms_limit != 0.003 or self.updates != 50
                or self.step_size != 2 * self.epsilon / self.updates
                or self.stft_weight != STFT_WEIGHT or self.content_weight != CONTENT_WEIGHT):
            raise ValueError("Waveform PGD must use its frozen bounds, 50 updates, and weights")


class PGDFailure(RuntimeError):
    """A measured invalid loss/gradient/update, carrying a partial-run receipt."""
    def __init__(self, message: str, receipt: Mapping, *,
                 adversarial: Tensor | None = None, residual: Tensor | None = None):
        super().__init__(message)
        self.receipt = dict(receipt)
        self.adversarial = None if adversarial is None else adversarial.detach().clone()
        self.residual = None if residual is None else residual.detach().clone()


@dataclass
class PGDResult:
    adversarial: Tensor
    residual: Tensor
    trace: list[dict]
    final_metrics: dict
    counts: dict


def _checked_scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise _InvalidMathematics(f"{name} loss must be a finite scalar")
    return float(value.detach())


def _norm(gradient: Tensor) -> float:
    norm = float(torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float64)))
    if not math.isfinite(norm):
        raise _InvalidMathematics("Audio gradient norm is nonfinite")
    return norm


def sequential_objective_gradients(
    waveform: Tensor,
    source_nll: Callable[[Tensor], Tensor],
    stft_loss: Callable[[Tensor], Tensor],
    content_loss: Callable[[Tensor], Tensor],
    *, condition: Condition, config: PGDConfig | None = None,
    counts: dict | None = None,
) -> tuple[Tensor, dict]:
    """Differentiate each active term at identical audio, freeing each graph.

    Clean STFT/content penalties legitimately have zero derivatives. Source NLL
    and the summed active derivative must be nonzero; otherwise no substitute
    objective or random update is supplied. In the NLL-only control the two
    penalties are measured without gradients, so its trace remains comparable.
    ``counts`` is updated before each attempted forward/backward call.
    """
    cfg = config or PGDConfig()
    if condition not in ("full_r3", "nll_only"):
        raise ValueError("Unknown waveform objective condition")
    calls = counts if counts is not None else {}
    for term in ("source", "stft", "content"):
        calls.setdefault(f"{term}_forward_calls", 0)
        calls.setdefault(f"{term}_backward_calls", 0)
    values, gradients = {}, {}
    for name, function in (("source", source_nll), ("stft", stft_loss),
                           ("content", content_loss)):
        active = name == "source" or condition == "full_r3"
        current = waveform.detach().requires_grad_(active)
        calls[f"{name}_forward_calls"] += 1
        with torch.set_grad_enabled(active):
            loss = function(current)
            values[name] = _checked_scalar(loss, name)
        if active:
            if not loss.requires_grad:
                raise _InvalidMathematics(f"{name} loss has no audio autograd graph")
            calls[f"{name}_backward_calls"] += 1
            gradient, = torch.autograd.grad(loss, current)
            if gradient.shape != waveform.shape or not bool(torch.isfinite(gradient).all()):
                raise _InvalidMathematics(f"{name} audio gradient is nonfinite or has wrong shape")
            gradients[name] = gradient.detach()
        # No graph is retained across terms, including frozen model activations.
        del loss, current
    if _norm(gradients["source"]) == 0:
        raise _InvalidMathematics("Source NLL audio gradient is zero")
    summed = -gradients["source"]
    if condition == "full_r3":
        summed = (summed + cfg.stft_weight * gradients["stft"]
                  + cfg.content_weight * gradients["content"])
    if not bool(torch.isfinite(summed).all()) or _norm(summed) == 0:
        raise _InvalidMathematics("Summed objective audio gradient is zero or nonfinite")
    full_objective = (-values["source"] + cfg.stft_weight * values["stft"]
                      + cfg.content_weight * values["content"])
    if not math.isfinite(full_objective):
        raise _InvalidMathematics("Weighted full objective is nonfinite")
    row = {
        "source_sum_nll": values["source"], "stft_loss": values["stft"],
        "content_loss": values["content"], "full_r3_objective": full_objective,
        "optimized_objective": full_objective if condition == "full_r3" else -values["source"],
        "source_gradient_norm": _norm(gradients["source"]),
        "stft_gradient_norm": _norm(gradients["stft"]) if "stft" in gradients else None,
        "content_gradient_norm": _norm(gradients["content"]) if "content" in gradients else None,
        "weighted_source_gradient_norm": _norm(gradients["source"]),
        "weighted_stft_gradient_norm": cfg.stft_weight * _norm(gradients["stft"])
        if "stft" in gradients else 0.0,
        "weighted_content_gradient_norm": cfg.content_weight * _norm(gradients["content"])
        if "content" in gradients else 0.0,
        "combined_gradient_norm": _norm(summed),
        "stft_active": condition == "full_r3", "content_active": condition == "full_r3",
    }
    return summed, row


def waveform_bounds(original: Tensor, residual: Tensor, config: PGDConfig | None = None) -> dict:
    """Validate physical audio constraints, allowing only dtype roundoff."""
    cfg = config or PGDConfig()
    adversarial = original + residual
    if (residual.shape != original.shape or not bool(torch.isfinite(residual).all())
            or not bool(torch.isfinite(adversarial).all())):
        raise _InvalidMathematics("Nonfinite or misaligned waveform update")
    linf = float(residual.abs().max())
    rms = float(residual.square().mean().sqrt())
    peak = float(adversarial.abs().max())
    tolerance = 4 * torch.finfo(original.dtype).eps
    if linf > cfg.epsilon + tolerance or rms > cfg.rms_limit + tolerance or peak > 1:
        raise _InvalidMathematics("Waveform projection violated its registered constraints")
    return {"residual_linf": linf, "residual_rms": rms, "audio_peak": peak,
            "constraint_roundoff_tolerance": tolerance}


class WaveformPGD:
    """Fifty zero-start projected sign-gradient descent updates per utterance."""
    def __init__(self, source_nll: Callable[[Tensor, str], Tensor],
                 feature_encoder: Callable[[Tensor], Tensor], *,
                 condition: Condition = "full_r3", config: PGDConfig | None = None):
        self.config = config or PGDConfig()
        if condition not in ("full_r3", "nll_only"):
            raise ValueError("Unknown waveform objective condition")
        self.source_nll, self.feature_encoder, self.condition = source_nll, feature_encoder, condition
        if isinstance(feature_encoder, nn.Module):
            if feature_encoder.training or any(p.requires_grad for p in feature_encoder.parameters()):
                raise ValueError("The feature encoder must already be frozen in evaluation mode")

    def run(self, original: Tensor, text: str, *, clean_features: Tensor | None = None,
            on_step: Callable[[dict], None] | None = None) -> PGDResult:
        cfg = self.config
        trace: list[dict] = []
        counts = {f"{term}_{operation}_calls": 0
                  for term in ("source", "stft", "content")
                  for operation in ("forward", "backward")}
        counts.update(clean_reference_forward_calls=0, completed_updates=0,
                      final_source_forward_calls=0, final_stft_forward_calls=0,
                      final_content_forward_calls=0)
        stage = "input"
        step = 0
        residual = None
        try:
            if (original.ndim != 1 or original.numel() <= 512 or not original.is_floating_point()
                    or not bool(torch.isfinite(original).all()) or bool((original.abs() > 1).any())):
                raise _InvalidMathematics("One finite, untruncated mono waveform with >512 samples in [-1,1] required")
            if not isinstance(text, str) or not text.strip():
                raise _InvalidMathematics("The unchanged original reference text is required")
            original = original.detach()
            residual = torch.zeros_like(original)
            stage = "clean_reference"
            if clean_features is None:
                counts["clean_reference_forward_calls"] += 1
                with torch.no_grad():
                    clean_features = self.feature_encoder(original).detach()
            else:
                clean_features = clean_features.detach()
            if not bool(torch.isfinite(clean_features).all()):
                raise _InvalidMathematics("Nonfinite frozen clean feature reference")
            for step in range(cfg.updates):
                stage = "gradient"
                waveform = (original + residual).detach()
                gradient, row = sequential_objective_gradients(
                    waveform,
                    lambda value: self.source_nll(value, text),
                    lambda value: multiresolution_stft_loss(value, original),
                    lambda value: relative_feature_loss(self.feature_encoder(value), clean_features),
                    condition=self.condition, config=cfg, counts=counts,
                )
                stage = "projection"
                with torch.no_grad():
                    candidate = project_waveform_residual(
                        original, residual - cfg.step_size * gradient.sign(),
                        cfg.epsilon, cfg.rms_limit).detach()
                bounds = waveform_bounds(original, candidate, cfg)
                residual = candidate
                row.update(step=step + 1, loss_evaluation_iterate=step,
                           objective_condition=self.condition,
                           update_direction="subtract step_size * sign(combined objective gradient)",
                           **bounds)
                counts["completed_updates"] += 1
                trace.append(row)
                stage = "on_step_callback"
                if on_step is not None:
                    on_step(dict(row))
            stage = "final_metrics"
            adversarial = (original + residual).detach()
            with torch.no_grad():
                counts["final_source_forward_calls"] += 1
                nll = _checked_scalar(self.source_nll(adversarial, text), "final source")
                counts["final_stft_forward_calls"] += 1
                stft = _checked_scalar(multiresolution_stft_loss(adversarial, original), "final stft")
                counts["final_content_forward_calls"] += 1
                content = _checked_scalar(relative_feature_loss(
                    self.feature_encoder(adversarial), clean_features), "final content")
            objective = -nll + cfg.stft_weight * stft + cfg.content_weight * content
            if not math.isfinite(objective):
                raise _InvalidMathematics("Final full objective is nonfinite")
            final_metrics = {
                "source_sum_nll": nll, "stft_loss": stft, "content_loss": content,
                "full_r3_objective": objective,
                "optimized_objective": objective if self.condition == "full_r3" else -nll,
                "objective_condition": self.condition, "selected_iterate": cfg.updates,
                "selection": "final iterate, with no quality-based selection",
                **waveform_bounds(original, residual, cfg),
            }
            return PGDResult(adversarial, residual.detach(), trace, final_metrics, dict(counts))
        except _InvalidMathematics as error:
            # Callback/storage errors and unexpected model/runtime exceptions
            # must stop the job. Only explicit mathematical/contract failures
            # become a per-example failure receipt with the last feasible audio.
            if stage == "on_step_callback":
                raise
            last_bounds = waveform_bounds(original, residual, cfg) if residual is not None else None
            raise PGDFailure(str(error), {
                "status": "failed", "stage": stage, "step": step,
                "failure_type": "mathematical" if stage in ("gradient", "final_metrics") else "contract",
                "reason": str(error), "condition": self.condition,
                "config": asdict(cfg), "counts": dict(counts), "trace": trace,
                "last_feasible_bounds": last_bounds,
                "output_role": "failed attempt, last feasible iterate; never a completed attack",
                "selection": "no replacement waveform or fallback objective",
            }, adversarial=None if residual is None else original + residual,
                residual=residual) from error
