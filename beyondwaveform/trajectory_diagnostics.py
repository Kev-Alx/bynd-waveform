"""Collect one unchanged attack trajectory for bounded finalization research.

This module leaves the production attack implementation untouched. Averaging
is an author-unspecified returned-delta policy, not a paper formula correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .attacks import AttackResult, freeze, project_relative_ball_, relative_radius
from .losses import CleanReference

SNAPSHOT_STEPS = tuple(range(41, 51))


@dataclass
class TrajectoryCollection:
    result: AttackResult
    q: torch.Tensor
    snapshots: dict[int, torch.Tensor]


def collect_trajectory(x, text, encoder, vocoder, surrogate, config, spectral):
    """Run exactly the existing constant-Adam, zero-init 50-update attack.

    The result/trace describe the terminal optimizer iterate. Only additional
    detached CPU copies of projected iterates 41 through 50 are retained.
    Returned clean features remain detached on their original device.
    """
    if (config.steps != 50 or config.rho != 0.1 or config.lambda_perc != 1 or
            config.optimizer != "adam" or config.initialization != "zero" or
            config.lr_schedule != "constant" or spectral.cutoff_hz != 6000 or
            spectral.sample_rate != 16000):
        raise ValueError("Trajectory diagnostic requires 50 constant-Adam zero-init updates at rho=0.1, lambda=1, 16 kHz and cutoff=6000 Hz")
    for module in (encoder, vocoder, surrogate):
        freeze(module)
    start = time.perf_counter()
    with torch.no_grad():
        q = encoder(x).detach()
        reconstruction = vocoder(q).detach()
    if q.ndim != 2 or q.shape[0] < 2 or not torch.isfinite(q).all():
        raise ValueError("Encoder must produce finite (T,D), T>=2 features")
    reference = CleanReference(q, x, spectral, config.eps)
    delta = torch.zeros_like(q)
    delta.requires_grad_()
    optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
    trace, snapshots = [], {}
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        z = q + delta
        audio = vocoder(z)
        nll = surrogate.nll(audio, text) if config.attack else z.new_zeros(())
        tv, hf = reference.terms(z, audio,
            temporal=config.temporal and config.lambda_perc > 0,
            high_frequency=config.high_frequency and config.lambda_perc > 0 and config.alpha_hf > 0)
        loss = -nll + config.lambda_perc * (tv + config.alpha_hf * hf)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        if loss.requires_grad:
            gradient, = torch.autograd.grad(loss, delta)
        else:
            gradient = torch.zeros_like(delta)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        delta.grad = gradient
        optimizer.step()
        project_relative_ball_(delta, q, config.rho, config.eps)
        radius = relative_radius(delta, q, config.eps).item()
        if radius > config.rho + 1e-6:
            raise RuntimeError("Feature radius violated")
        trace.append({"step": step + 1, "loss_before_update": loss.item(), "nll_before_update": nll.item(),
            "tv_ratio": tv.item(), "hf_ratio": hf.item(), "gradient_norm": gradient.norm().item(),
            "radius_after_update": radius})
        if step + 1 in SNAPSHOT_STEPS:
            snapshots[step + 1] = delta.detach().cpu().clone()
    with torch.no_grad():
        final = vocoder(q + delta).detach()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    result = AttackResult(final, reconstruction, delta.detach(), trace, time.perf_counter() - start)
    return TrajectoryCollection(result=result, q=q, snapshots=snapshots)


def finalize_deltas(snapshots):
    """Return terminal, mean(41..50), and a terminal direction of equal norm.

    Mean arithmetic uses ``torch.stack(...).mean(0)`` in the snapshot dtype.
    There is no projection or optimization. The caller independently verifies
    the returned tensors and radius. A zero terminal with a nonzero mean has
    no norm-matched terminal direction and is explicitly rejected.
    """
    if set(snapshots) != set(SNAPSHOT_STEPS):
        raise ValueError("Expected exactly the projected snapshots from steps 41 through 50")
    values = [snapshots[step] for step in SNAPSHOT_STEPS]
    first = values[0]
    if (not isinstance(first, torch.Tensor) or first.ndim != 2 or first.shape[0] < 2 or
            first.shape[1] < 1 or not first.is_floating_point()):
        raise ValueError("Snapshots must be real floating-point (T,D) tensors with T>=2, D>=1")
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.shape != first.shape or
                value.dtype != first.dtype or value.device != first.device or
                not torch.isfinite(value).all()):
            raise ValueError("Snapshots must have identical shape/dtype/device and finite values")
    terminal = snapshots[50].detach().clone()
    average = torch.stack([value.detach() for value in values]).mean(dim=0)
    # Work relative to each tensor's maximum for norm ratios. This avoids a
    # norm epsilon altering the meaning of tiny nonzero control deltas.
    maximum = terminal.abs().amax()
    if maximum == 0:
        if average.count_nonzero():
            raise ValueError("A zero terminal delta cannot supply a nonzero norm-matched direction")
        matched = torch.zeros_like(terminal)
    else:
        average_maximum = average.abs().amax()
        if average_maximum == 0:
            matched = torch.zeros_like(terminal)
        else:
            terminal_scaled = terminal / maximum
            average_scaled = average / average_maximum
            matched = terminal_scaled * (average_scaled.norm()/terminal_scaled.norm()) * average_maximum
    return {"terminal": terminal, "average_41_50": average, "terminal_norm_matched": matched}
