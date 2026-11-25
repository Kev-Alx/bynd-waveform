"""Clean-statistic coordinates for isolated optimizer experiments.

All weights are strictly positive: delta = weights * theta is a bijection.
The loss and the physical feature-space ball retain their original meaning.
This module is not used by the production attack.
"""
from __future__ import annotations

import torch

from .attacks import project_relative_ball_

KINDS = ("time_energy", "channel_std")
SPEC = {
    "floor_relative_to_max": 0.1,
    "normalization": "unit RMS in CPU float64, then cast to q dtype",
    "time_window_samples": 400,
    "time_hop_samples": 320,
    "channel_std_correction": 0,
    "zero_statistic": "identity weights",
    "mapping": "delta = weights * theta; optimize theta with Adam",
    "projection": "radial projection of physical delta onto original q-relative L2 ball",
}


def _normalized(statistic, dtype):
    if not torch.isfinite(statistic).all() or (statistic < 0).any():
        raise ValueError("Expected a finite nonnegative clean statistic")
    maximum = statistic.max()
    if maximum == 0:
        return torch.ones_like(statistic, dtype=dtype)
    raw = (statistic / maximum).clamp(min=SPEC["floor_relative_to_max"], max=1)
    return (raw / raw.square().mean().sqrt()).to(dtype)


@torch.no_grad()
def coordinate_weights(q, x, kind):
    """Derive weights only from detached clean inputs, always on CPU float64.

    WavLM's seven unpadded convolutions have receptive field 400 and hop 320.
    A geometry mismatch raises instead of interpolating or truncating frames.
    """
    if q.ndim != 2 or min(q.shape) < 2 or not q.is_floating_point() or not torch.isfinite(q).all():
        raise ValueError("Expected finite floating (T,D) clean features")
    if kind == "uniform":
        return torch.ones((1, 1), dtype=q.dtype)
    clean = q.detach().to(device="cpu", dtype=torch.float64)
    if kind == "channel_std":
        return _normalized(clean.std(dim=0, correction=0), q.dtype).unsqueeze(0)
    if kind == "time_energy":
        wave = x.detach().to(device="cpu", dtype=torch.float64)
        if wave.ndim != 1 or wave.numel() < 400 or not torch.isfinite(wave).all():
            raise ValueError("Expected a finite mono waveform of at least 400 samples")
        frames = wave.unfold(0, 400, 320)
        if frames.shape[0] != q.shape[0]:
            raise ValueError("Clean waveform and feature frame geometry differ")
        return _normalized(frames.square().mean(dim=1).sqrt(), q.dtype).unsqueeze(1)
    raise ValueError(f"Unknown optimization coordinates: {kind}")


def physical_delta(theta, weights, identity=False):
    # Avoid even an identity multiplication in the exact control computation.
    return theta if identity else theta * weights


@torch.no_grad()
def project_coordinates_(theta, weights, q, rho, eps=1e-8, identity=False):
    """Return physical delta after radial projection, leaving theta a leaf."""
    if identity:
        return project_relative_ball_(theta, q, rho, eps)
    delta = theta * weights
    radius = rho * (torch.linalg.vector_norm(q.detach()) + eps)
    norm = torch.linalg.vector_norm(delta)
    if norm > radius:
        theta.mul_(radius / norm)
    return theta * weights
