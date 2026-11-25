"""Read-only, same-state counterfactuals for a single ordinary Adam step.

Changing the current gradient while retaining the actual Adam moments is a
local diagnostic. It is not a trajectory trained without a loss component.
"""
from __future__ import annotations

import copy
import math

import torch

from .attacks import project_relative_ball_, relative_radius


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _validate(optimizer, delta, gradient, q, rho, eps):
    _require(type(optimizer) is torch.optim.Adam, "Expected an actual torch.optim.Adam optimizer")
    _require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 1 and
             optimizer.param_groups[0]["params"][0] is delta,
             "Adam diagnostic requires exactly one group containing only the supplied delta")
    _require(all(parameter is delta for parameter in optimizer.state), "Adam has state for another parameter")
    _require(isinstance(delta, torch.Tensor) and delta.is_leaf and delta.requires_grad and
             delta.is_floating_point() and torch.isfinite(delta).all().item(), "Invalid live Adam parameter")
    for name, value in (("gradient", gradient), ("q", q)):
        _require(isinstance(value, torch.Tensor) and value.shape == delta.shape and value.dtype == delta.dtype and
                 value.device == delta.device and torch.isfinite(value).all().item(), f"Invalid {name} shape, dtype, device or values")
    _require(isinstance(rho, (int, float)) and not isinstance(rho, bool) and math.isfinite(rho) and rho >= 0 and
             isinstance(eps, (int, float)) and not isinstance(eps, bool) and math.isfinite(eps) and eps > 0,
             "Invalid projection radius or epsilon")
    group = optimizer.param_groups[0]
    _require(not group.get("differentiable", False), "Differentiable Adam is outside this read-only diagnostic")
    _require(not getattr(optimizer, "_optimizer_step_pre_hooks", {}) and
             not getattr(optimizer, "_optimizer_step_post_hooks", {}), "Unmodeled Adam step hooks")
    _require(getattr(optimizer, "grad_scale", None) is None and getattr(optimizer, "found_inf", None) is None,
             "AMP-scaled Adam state is outside this diagnostic")
    for key in ("lr", "eps", "weight_decay"):
        value = group[key]
        _require(not isinstance(value, torch.Tensor) or value.numel() == 1, f"Invalid Adam {key}")
        number = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
        _require(math.isfinite(number) and number >= 0 and (key != "eps" or number > 0), f"Invalid Adam {key}")
    _require(len(group["betas"]) == 2 and all(math.isfinite(float(b)) and 0 <= float(b) < 1 for b in group["betas"]),
             "Invalid Adam betas")
    state = optimizer.state.get(delta, {})  # .get does not initialize defaultdict state.
    if state:
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["amsgrad"] else set())
        _require(set(state) == expected, "Unexpected initialized Adam state keys")
        step = state["step"]
        _require(isinstance(step, torch.Tensor) and step.numel() == 1 and torch.isfinite(step).item() and
                 step.item() >= 0 and float(step.item()).is_integer(), "Invalid Adam step counter")
        for key in expected - {"step"}:
            value = state[key]
            _require(isinstance(value, torch.Tensor) and value.shape == delta.shape and value.dtype == delta.dtype and
                     value.device == delta.device and torch.isfinite(value).all().item(), f"Invalid Adam {key}")
            if key != "exp_avg":
                _require((value >= 0).all().item(), f"Negative Adam {key}")


@torch.no_grad()
def counterfactual_step(optimizer, delta, gradient, q, rho, eps):
    """Clone CURRENT Adam state, apply one gradient, then the original projection.

    ``state_before`` and ``state_after`` contain detached tensor receipts for
    the same moments/counter that a real Adam update uses. ``delta_before``,
    ``delta_unprojected``, ``delta_projected`` and both change tensors are
    independent copies. Source parameters, gradients, state and groups are
    untouched, including an optimizer whose moments have not been initialized.

    The clone preserves Adam's group options, including foreach, AMSGrad,
    maximize, weight decay, capturable and fused settings. Differentiable/AMP
    Adam and custom step hooks are explicitly outside this diagnostic.
    """
    _validate(optimizer, delta, gradient, q, rho, eps)
    group = optimizer.param_groups[0]
    state_before = _clone(optimizer.state.get(delta, {}))
    group_receipt = {key: _clone(value) for key, value in group.items() if key != "params"}
    defaults = _clone(optimizer.defaults)
    before = delta.detach().clone()
    trial = before.clone().requires_grad_(True)
    clone = torch.optim.Adam([trial], **_clone(defaults))
    # Construct a one-parameter state_dict without invoking source optimizer
    # serialization hooks or initializing its lazy moments.
    serialized_group = {**_clone(group_receipt), "params": [0]}
    serialized_state = {0: _clone(state_before)} if delta in optimizer.state else {}
    clone.load_state_dict({"state": serialized_state, "param_groups": [serialized_group]})
    trial.grad = gradient.detach().clone()
    clone.step()
    unprojected = trial.detach().clone()
    _require(torch.isfinite(unprojected).all().item(), "Counterfactual Adam step produced a nonfinite parameter")
    state_after = _clone(clone.state[trial])
    projected = unprojected.clone()
    project_relative_ball_(projected, q.detach(), rho, eps)
    return {"optimizer_class": "torch.optim.Adam", "param_group": group_receipt,
            "optimizer_defaults": defaults, "initialized_before": bool(state_before),
            "state_before": state_before, "state_after": state_after,
            "delta_before": before, "delta_unprojected": unprojected, "delta_projected": projected,
            "unprojected_change": unprojected - before, "projected_change": projected - before,
            "relative_radius_before": relative_radius(before, q.detach(), eps).item(),
            "relative_radius_unprojected": relative_radius(unprojected, q.detach(), eps).item(),
            "relative_radius_projected": relative_radius(projected, q.detach(), eps).item(),
            "rho": rho, "eps": eps,
            "scope": "One alternative current gradient at identical actual Adam moments; not a complete loss ablation or another optimized trajectory"}
