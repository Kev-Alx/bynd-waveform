"""Feature-delta diagnostics; these transforms do not change the paper loss.

The caller must independently project/check the result against the clean
feature radius. In particular, replicate-padded averaging is not assumed to
be a contraction in the Frobenius norm.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def transform_delta(delta, window=1, mode="identity", eps=1e-8):
    """Transform one unpadded ``(time, feature)`` perturbation.

    ``smooth`` uses a symmetric Gaussian with sigma ``(window-1)/4`` and
    replicate padding. ``smooth_norm_matched`` restores its original
    Frobenius norm. ``attenuation_matched`` retains the original direction
    with the norm of the unscaled smoothed delta. The identity mode and a
    one-frame window return the input exactly, without mutating it.

    Normalization uses a common input scale, rather than flooring a nonzero
    norm by epsilon; even tiny perturbations retain the norm-matching
    interpretation. Epsilon supplies a safe scale for an all-zero input.
    Half/bfloat inputs are computed in float32 and cast back.
    """
    if not isinstance(delta, torch.Tensor) or delta.ndim != 2 or delta.shape[0] < 2 or delta.shape[1] < 1:
        raise ValueError("delta must be an unpadded (T,D) tensor with T >= 2 and D >= 1")
    if not delta.is_floating_point() or not torch.isfinite(delta).all():
        raise ValueError("delta must contain finite real floating-point values")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    if mode not in {"identity", "smooth", "smooth_norm_matched", "attenuation_matched"}:
        raise ValueError(f"Unknown feature-delta transform: {mode}")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if mode == "identity" or window == 1:
        return delta

    work = delta if delta.dtype in {torch.float32, torch.float64} else delta.float()
    maximum = work.abs().amax()
    scale = torch.where(maximum > 0, maximum, maximum.new_tensor(eps))
    normalized = work / scale
    offsets = torch.arange(window, device=delta.device, dtype=work.dtype) - window // 2
    sigma = (window - 1) / 4
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, window)
    # Treat features as independent batch elements; never mix feature channels.
    padded = F.pad(normalized.transpose(0, 1).unsqueeze(1), (window // 2, window // 2), mode="replicate")
    smoothed = F.conv1d(padded, kernel).squeeze(1).transpose(0, 1)
    if mode == "smooth":
        return (smoothed * scale).to(delta.dtype)

    original_norm = torch.linalg.vector_norm(normalized)
    smoothed_norm = torch.linalg.vector_norm(smoothed)
    if mode == "smooth_norm_matched":
        denominator = torch.where(smoothed_norm > 0, smoothed_norm, smoothed_norm.new_ones(()))
        transformed = smoothed * (original_norm / denominator)
    else:
        denominator = torch.where(original_norm > 0, original_norm, original_norm.new_ones(()))
        transformed = normalized * (smoothed_norm / denominator)
    return (transformed * scale).to(delta.dtype)
