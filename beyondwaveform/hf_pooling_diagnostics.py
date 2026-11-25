"""Isolated diagnostics for the unspecified time pooling in paper Eq.13.

The existing global-power fraction sums over both frequency and time.  An
equal-frame mean is a separate interpretation, not an established author
formula.  Both use the exact explicit-reflect STFT convention of the current
frontend.  Importing this module does not change any attack or loss function.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SpectralConfig
from .deterministic_frontend import reflect_stft


@dataclass(frozen=True)
class HFFrameSummary:
    """Differentiable frame statistics for one unpadded waveform.

    ``frame_energy`` and ``frame_high_frequency_energy`` sum the one-sided
    STFT power over frequency. ``positive_energy_frames`` identifies the
    frames where their ratio is defined. ``frame_fractions`` uses zero as an
    explicit sentinel for silent frames; those frames are excluded from
    ``equal_frame_fraction``. No epsilon is added to a positive denominator.

    ``global_fraction`` preserves the existing reduction order exactly.
    ``energy_weighted_fraction`` reconstructs it from the frame fractions;
    floating-point reduction order can cause a small rounding difference.
    All scalar fractions retain gradients through the input waveform.
    """

    frame_energy: torch.Tensor
    frame_high_frequency_energy: torch.Tensor
    positive_energy_frames: torch.Tensor
    frame_fractions: torch.Tensor
    total_energy: torch.Tensor
    global_fraction: torch.Tensor
    energy_weighted_fraction: torch.Tensor
    equal_frame_fraction: torch.Tensor

    @property
    def frame_count(self) -> int:
        return self.frame_energy.numel()

    @property
    def positive_frame_count(self) -> int:
        return int(self.positive_energy_frames.sum().item())

    @property
    def silent_frame_count(self) -> int:
        return self.frame_count - self.positive_frame_count


def frame_power_summary(x: torch.Tensor, config: SpectralConfig) -> HFFrameSummary:
    """Compute the existing HF fraction and two frame-pooling diagnostics.

    Hann construction, reflection, centering, hop, FFT size, one-sided bins,
    and strict ``f > cutoff`` match ``deterministic_frontend``. The existing
    minimum waveform length and silence/nonfinite rejection are retained.
    Equal-frame pooling averages strictly positive-energy frames only, so
    silence does not invent an HF ratio or dilute the mean. The membership
    is piecewise constant; no gradient is defined through that discrete
    choice at a frame whose energy changes from exactly zero to positive.
    """
    if x.ndim != 1 or x.numel() <= config.n_fft // 2:
        raise ValueError("HF expects a nonempty waveform long enough for reflection padding")
    window = torch.hann_window(config.win_length, device=x.device, dtype=x.dtype)
    power = reflect_stft(
        x, n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length, window=window, center=config.center,
    ).abs().square()
    energy = power.sum()
    if not torch.isfinite(energy) or energy.detach() <= 0:
        raise ValueError("HF undefined for silence or nonfinite energy")
    frequencies = torch.fft.rfftfreq(config.n_fft, d=1/config.sample_rate).to(x.device)
    high_power = power[frequencies > config.cutoff_hz]
    # Keep these two reductions in the same order as the existing frontend.
    global_fraction = high_power.sum() / energy

    frame_energy = power.sum(dim=0)
    frame_high = high_power.sum(dim=0)
    positive = frame_energy > 0
    # Index before dividing: a silent frame must never produce 0/0 in the
    # autograd graph. Zero is a sentinel only, never an HF denominator floor.
    fractions = torch.zeros_like(frame_energy)
    fractions[positive] = frame_high[positive] / frame_energy[positive]
    equal_fraction = fractions[positive].mean()
    weighted_fraction = (fractions * frame_energy).sum() / frame_energy.sum()
    return HFFrameSummary(
        frame_energy=frame_energy,
        frame_high_frequency_energy=frame_high,
        positive_energy_frames=positive,
        frame_fractions=fractions,
        total_energy=energy,
        global_fraction=global_fraction,
        energy_weighted_fraction=weighted_fraction,
        equal_frame_fraction=equal_fraction,
    )


def global_high_frequency_fraction(x: torch.Tensor, config: SpectralConfig) -> torch.Tensor:
    """Existing global-power interpretation, with unchanged arithmetic."""
    return frame_power_summary(x, config).global_fraction


def equal_frame_high_frequency_fraction(x: torch.Tensor, config: SpectralConfig) -> torch.Tensor:
    """Independent equal-positive-frame interpretation, without an epsilon."""
    return frame_power_summary(x, config).equal_frame_fraction
