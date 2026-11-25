"""R2: same-utterance local feature mixtures with physical constraints.

Only the current utterance's frozen clean (T,D) features enter the bank.
The canonical initialization is EXACTLY clean: a=0, uniform valid logits.
The optional gradient-logit bootstrap is a separately reported mechanism repair,
not part of the canonical condition. No models, datasets or targets load here.

Integration: pass the unchanged frozen-model FV objective as objective(z). It
must return a scalar loss to MINIMIZE, or {'loss': loss, 'nll': ..., ...}.
For the registered condition this is -NLL_sum + TV + alpha_hf*HF. All model
weights must already be frozen; input derivatives remain live. The returned
physical delta can be passed to the existing vocoder as vocoder(q + delta).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import torch
from torch import nn


@dataclass
class LocalSearchResult:
    features: torch.Tensor
    delta: torch.Tensor
    trace: list[dict]
    state: dict
    status: str
    diagnostics: dict


class LocalNeighborMixture(nn.Module):
    """z_t = q_t + a_t * (softmax(logits_t) @ bank_t - q_t).

    Bank slots correspond to offsets [-2,-1,0,1,2]. Self is always valid;
    other entries require a nonzero clean frame and clean cosine >= .95.
    Direct amplitudes make an exact zero start possible, unlike a sigmoid.
    Projection changes amplitudes only, preserving the convex mixture set.
    """

    def __init__(self, q: torch.Tensor, rho: float = 0.1,
                 cosine_floor: float = 0.95, eps: float = 1e-8):
        super().__init__()
        if q.ndim != 2 or min(q.shape) < 1:
            raise ValueError("q must have shape (T,D), T,D >= 1")
        if q.dtype not in (torch.float32, torch.float64) or not torch.isfinite(q).all():
            raise ValueError("q must be finite float32/float64 clean features")
        if not math.isfinite(rho) or rho < 0 or not 0 < cosine_floor <= 1:
            raise ValueError("Invalid physical radius or cosine floor")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.rho, self.cosine_floor, self.eps = float(rho), float(cosine_floor), float(eps)
        clean = q.detach().clone()
        indices = torch.arange(q.shape[0], device=q.device)[:, None] + torch.arange(-2, 3, device=q.device)[None, :]
        in_bounds = (indices >= 0) & (indices < q.shape[0])
        clipped = indices.clamp(0, q.shape[0] - 1)
        bank = clean[clipped]
        clean64, bank64 = clean.double(), bank.double()
        norms = torch.linalg.vector_norm(clean64, dim=-1)
        neighbor_norms = torch.linalg.vector_norm(bank64, dim=-1)
        nonzero = norms > 0
        denominator = norms[:, None] * neighbor_norms
        similarity = (clean64[:, None, :] * bank64).sum(-1) / denominator.clamp_min(torch.finfo(torch.float64).tiny)
        valid = in_bounds & nonzero[:, None] & (neighbor_norms > 0) & (similarity >= cosine_floor)
        valid[:, 2] = True
        differences = bank64 - clean64[:, None, :]
        distinct = (differences != 0).any(-1) & valid
        active = distinct.any(-1) & nonzero
        self.register_buffer("clean", clean)
        self.register_buffer("bank", bank)
        self.register_buffer("indices", clipped)
        self.register_buffer("valid", valid)
        self.register_buffer("nonzero", nonzero)
        self.register_buffer("active", active)
        self.a = nn.Parameter(torch.zeros(q.shape[0], dtype=q.dtype, device=q.device))
        self.logits = nn.Parameter(torch.zeros(q.shape[0], 5, dtype=q.dtype, device=q.device))
        self._bank_summary = self._summarize_bank(differences, distinct)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.masked_fill(~self.valid, -torch.inf), dim=-1)

    def features(self) -> torch.Tensor:
        mixture = (self.weights()[..., None] * self.bank).sum(dim=1)
        amplitude = self.a * self.active.to(self.a.dtype)
        return self.clean + amplitude[:, None] * (mixture - self.clean)

    forward = features

    def physical_delta(self) -> torch.Tensor:
        return self.features() - self.clean

    def _summarize_bank(self, differences: torch.Tensor, distinct: torch.Tensor) -> dict:
        # SVD entropy from each small (5,5) Gram matrix, computed once.
        offsets = differences * self.valid[..., None]
        gram = offsets @ offsets.transpose(-1, -2)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        cutoff = eigenvalues.amax(-1, keepdim=True) * 1e-12
        singular = torch.where(eigenvalues > cutoff, eigenvalues.sqrt(), torch.zeros_like(eigenvalues))
        probabilities = singular / singular.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-300).log()).sum(-1))
        effective_rank = torch.where(self.active, effective_rank, torch.zeros_like(effective_rank))
        active_count = int(self.active.sum().item())
        return {
            "frames": self.clean.shape[0], "feature_dimension": self.clean.shape[1],
            "temporal_window": 2, "maximum_nonself_neighbors": 4,
            "valid_nonself_neighbor_count": int((self.valid.sum() - self.clean.shape[0]).item()),
            "distinct_nonself_neighbor_count": int(distinct.sum().item()),
            "active_frame_count": active_count,
            "neighbor_coverage": active_count / self.clean.shape[0],
            "effective_rank_mean_active": float(effective_rank.sum().item() / max(active_count, 1)),
            "effective_rank_max": float(effective_rank.max().item()),
        }

    @torch.no_grad()
    def diagnostics(self) -> dict:
        z = self.features()
        delta64 = z.double() - self.clean.double()
        clean_norm = torch.linalg.vector_norm(self.clean.double())
        row_qnorm = torch.linalg.vector_norm(self.clean.double(), dim=-1)
        row_znorm = torch.linalg.vector_norm(z.double(), dim=-1)
        cosine = (z.double() * self.clean.double()).sum(-1) / (row_qnorm * row_znorm).clamp_min(torch.finfo(torch.float64).tiny)
        minimum_cosine = float(cosine[self.nonzero].min().item()) if self.nonzero.any() else None
        w = self.weights().double()
        entropy = -(w * w.clamp_min(1e-300).log()).sum(-1)
        zero_delta = delta64[~self.nonzero]
        return dict(self._bank_summary, radius=float((torch.linalg.vector_norm(delta64) / (clean_norm + self.eps)).item()),
                    minimum_cosine=minimum_cosine,
                    zero_frame_max_abs_delta=float(zero_delta.abs().max().item()) if zero_delta.numel() else 0.0,
                    amplitude_mean=float(self.a.mean().item()), amplitude_max=float(self.a.max().item()),
                    weight_entropy_mean=float(entropy.mean().item()),
                    weight_entropy_mean_active=float(entropy[self.active].mean().item()) if self.active.any() else 0.0,
                    maximum_neighbor_weight=float(w.max().item()))

    @torch.no_grad()
    def project_(self) -> dict:
        """Intersect amplitude bounds, content cones and the physical global ball.

        Neighbor mixtures already lie in the clean cosine cone in exact
        arithmetic (a convex second-order cone). The explicit check and scalar
        backtracking protect finite-precision edge cases. Global interpolation
        toward clean preserves both the cone and all mixture coefficients.
        """
        if not torch.isfinite(self.a).all() or not torch.isfinite(self.logits).all():
            raise FloatingPointError("Nonfinite R2 mixture parameters")
        clipped_rows = int(((self.a < 0) | (self.a > 1) | (~self.active & (self.a != 0))).sum().item())
        self.a.clamp_(0, 1)
        self.a.mul_(self.active)
        if self.rho == 0:
            self.a.zero_()
        mixture64 = (self.weights()[..., None] * self.bank).sum(1).double()
        q64 = self.clean.double()
        direction = mixture64 - q64
        qnorm = torch.linalg.vector_norm(q64, dim=-1)

        def row_cos(amplitudes):
            trial = q64 + amplitudes.double()[:, None] * direction
            return (trial * q64).sum(-1) / (torch.linalg.vector_norm(trial, dim=-1) * qnorm).clamp_min(torch.finfo(torch.float64).tiny)

        bad = self.nonzero & (row_cos(self.a) < self.cosine_floor)
        backtracked = int(bad.sum().item())
        if backtracked:
            lower = torch.zeros_like(self.a)
            upper = self.a.clone()
            for _ in range(48):
                midpoint = (lower + upper) * 0.5
                feasible = row_cos(midpoint) >= self.cosine_floor
                lower = torch.where(feasible, midpoint, lower)
                upper = torch.where(feasible, upper, midpoint)
            self.a.copy_(torch.where(bad, lower, self.a))
        factor = 1.0
        bound = self.rho * (torch.linalg.vector_norm(q64) + self.eps)
        # Recheck actual cast/rounded features, not merely ideal coefficients.
        for _ in range(3):
            actual_norm = torch.linalg.vector_norm(self.features().double() - q64)
            if actual_norm <= bound or actual_norm == 0:
                break
            scale = float((bound / actual_norm).item()) * (1 - 8 * torch.finfo(self.a.dtype).eps)
            self.a.mul_(scale)
            factor *= scale
        receipt = self.diagnostics()
        if receipt["radius"] > self.rho + 1e-6:
            raise RuntimeError("R2 physical global radius violated")
        if receipt["minimum_cosine"] is not None and receipt["minimum_cosine"] < self.cosine_floor - 1e-6:
            raise RuntimeError("R2 content anchor violated")
        if receipt["zero_frame_max_abs_delta"] != 0:
            raise RuntimeError("R2 changed a zero clean frame")
        return dict(receipt, amplitude_clip_rows=clipped_rows, cosine_backtracked_rows=backtracked,
                    global_interpolation_factor=factor)

    @torch.no_grad()
    def directional_costs(self, physical_gradient: torch.Tensor) -> torch.Tensor:
        if physical_gradient.shape != self.clean.shape or not torch.isfinite(physical_gradient).all():
            raise ValueError("Expected finite physical feature gradient with shape q")
        costs = ((self.bank.double() - self.clean.double()[:, None, :]) * physical_gradient.double()[:, None, :]).sum(-1)
        return costs.masked_fill(~self.valid, torch.inf)

    @torch.no_grad()
    def bootstrap_logits_(self, physical_gradient: torch.Tensor) -> int:
        """Optional, explicitly labeled repair for uniform-start cancellation.

        At a=0 every logit choice reconstructs q exactly. Favor the best
        first-order feasible neighbor without an extra objective/model query.
        This changes the canonical parameterization and must be reported.
        """
        if torch.count_nonzero(self.a):
            raise ValueError("Gradient bootstrap is only valid at the exact clean start")
        costs = self.directional_costs(physical_gradient)
        best = costs.min(-1).values
        repairable = self.active & (best < 0)
        if not repairable.any():
            return 0
        maximum = costs.masked_fill(~self.valid, 0).clamp_min(0).max(-1).values
        # Off-best positive contributions sum to at most half the descent
        # margin. Log-space arithmetic avoids ratios that underflow first.
        log_fraction = (torch.log((-best).clamp_min(1e-300))
                        - math.log(10.0)
                        - torch.log(torch.maximum(maximum, best.abs()).clamp_min(1e-300))).clamp_max(0)
        prepared = log_fraction[:, None].expand_as(costs).clone()
        prepared = torch.where(costs == best[:, None], torch.zeros_like(prepared), prepared)
        self.logits.copy_(torch.where(repairable[:, None], prepared.to(self.logits.dtype), self.logits))
        return int(repairable.sum().item())


def _objective_parts(output):
    values = {"loss": output} if isinstance(output, torch.Tensor) else output
    if not isinstance(values, Mapping) or "loss" not in values:
        raise TypeError("objective must return scalar loss or mapping containing loss")
    loss = values["loss"]
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("Objective loss must be a finite scalar tensor")
    metrics = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = float(value.detach().item())
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite objective metric: {name}")
            metrics[str(name) + "_before_update"] = value
    return loss, metrics


def optimize_local_neighbors(q: torch.Tensor, objective: Callable,
                             steps: int = 50, learning_rate: float = 0.05,
                             rho: float = 0.1, cosine_floor: float = 0.95,
                             eps: float = 1e-8, gradient_bootstrap: bool = False) -> LocalSearchResult:
    """Minimize a frozen-model objective using Adam over a AND mixture logits.

    Stops with a named status when the bank has no displacement, the source
    objective is flat, or the canonical clean/uniform start cannot leave q.
    A parameterization stall is distinct from absence of feasible directions.
    Each executed step uses exactly one objective query. Source backwards and
    retrieval costs still belong in the caller's measured resource ledger.
    """
    if not isinstance(steps, int) or steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid step count or learning rate")
    geometry = LocalNeighborMixture(q, rho, cosine_floor, eps)
    optimizer = torch.optim.Adam([geometry.a, geometry.logits], lr=learning_rate)
    trace: list[dict] = []
    condition = "gradient_logit_bootstrap_repair" if gradient_bootstrap else "canonical_zero_uniform"
    status = "completed"
    objective_queries, bootstrap_frames = 0, 0
    if not geometry.active.any():
        status = "degenerate_bank"
    elif rho == 0:
        status = "zero_radius"
    else:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            z = geometry.features()
            loss, metrics = _objective_parts(objective(z))
            objective_queries += 1
            if loss.requires_grad:
                physical_gradient, = torch.autograd.grad(loss, z, allow_unused=True)
                if physical_gradient is None:
                    physical_gradient = torch.zeros_like(z)
            else:
                physical_gradient = torch.zeros_like(z)
            if not torch.isfinite(physical_gradient).all():
                raise FloatingPointError("Nonfinite physical feature gradient")
            costs = geometry.directional_costs(physical_gradient)
            feasible_descent_frames = int((costs.min(-1).values < 0).sum().item())
            if step == 0 and gradient_bootstrap:
                bootstrap_frames = geometry.bootstrap_logits_(physical_gradient)
                z = geometry.features()
            # Chain the same physical objective derivative through current
            # mixture parameters; frozen source weights never enter Adam.
            gradients = torch.autograd.grad(z, (geometry.a, geometry.logits), grad_outputs=physical_gradient)
            if not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("Nonfinite R2 parameter gradients")
            geometry.a.grad, geometry.logits.grad = gradients
            optimizer.step()
            receipt = geometry.project_()
            row = dict(step=step + 1, condition=condition, objective_queries=objective_queries,
                       physical_gradient_norm=float(physical_gradient.norm().item()),
                       amplitude_gradient_norm=float(gradients[0].norm().item()),
                       logits_gradient_norm=float(gradients[1].norm().item()),
                       feasible_descent_frames=feasible_descent_frames,
                       bootstrap_frames=bootstrap_frames if step == 0 else 0,
                       **metrics, **receipt)
            trace.append(row)
            if physical_gradient.count_nonzero() == 0:
                status = "flat_objective"
                break
            if step == 0 and not geometry.physical_delta().detach().count_nonzero():
                status = "uniform_start_stalled_with_feasible_descent" if feasible_descent_frames else "no_first_order_feasible_descent"
                break
    with torch.no_grad():
        features = geometry.features().detach().clone()
    state = {"condition": condition, "a": geometry.a.detach().clone(),
             "logits": geometry.logits.detach().clone(), "valid_neighbors": geometry.valid.detach().clone(),
             "neighbor_indices": geometry.indices.detach().clone(), "objective_queries": objective_queries,
             "bootstrap_frames": bootstrap_frames, "rho": rho, "cosine_floor": cosine_floor,
             "initialization": "exact_clean_a_zero", "optimizer": "adam_a_and_logits"}
    return LocalSearchResult(features, features - geometry.clean, trace, state, status, geometry.diagnostics())
