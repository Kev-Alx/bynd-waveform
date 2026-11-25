"""Equations (9)–(14). Every operation is per utterance, without padding."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .config import SpectralConfig


def temporal_variation(z: torch.Tensor) -> torch.Tensor:
    """Eq.12: sum feature dimensions, MEAN adjacent time differences."""
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("TV requires an unpadded (T,D) tensor with T >= 2")
    return z.diff(dim=0).square().sum(dim=-1).mean()


def high_frequency_fraction(x: torch.Tensor, config: SpectralConfig) -> torch.Tensor:
    """Eq.13, selected interpretation: sum power over both frequency and time.

    Strictly f > fc. No extra epsilon is added to Eq.13. Zero-energy signals
    are rejected; epsilon belongs to the clean normalizers in Eq.11.
    """
    if x.ndim != 1 or x.numel() <= config.n_fft // 2:
        raise ValueError("HF expects a nonempty waveform long enough for reflection padding")
    window = torch.hann_window(config.win_length, device=x.device, dtype=x.dtype)
    power = torch.stft(x, n_fft=config.n_fft, hop_length=config.hop_length,
                       win_length=config.win_length, window=window, center=config.center,
                       pad_mode="reflect", return_complex=True).abs().square()
    energy = power.sum()
    if not torch.isfinite(energy) or energy.detach() <= 0:
        raise ValueError("HF undefined for silence or nonfinite energy")
    frequencies = torch.fft.rfftfreq(config.n_fft, d=1 / config.sample_rate).to(x.device)
    return power[frequencies > config.cutoff_hz].sum() / energy


def lexical_nll(logits, labels, lexical_mask, reduction="mean"):
    """Already next-token-aligned logits/labels; special positions score zero."""
    if logits.shape[:-1] != labels.shape or labels.shape != lexical_mask.shape:
        raise ValueError("Token shapes do not align")
    mask = lexical_mask.bool()
    if not mask.any():
        raise ValueError("Transcript has no lexical tokens")
    loss = F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("NLL reduction must be sum or mean")


class CleanReference:
    """Detached ORIGINAL-clean denominators, cached once (Eq.11)."""
    def __init__(self, q, x, spectral, eps):
        self.spectral = spectral
        self.eps = eps
        self.tv = temporal_variation(q.detach()).detach()
        self.hf = high_frequency_fraction(x.detach(), spectral).detach()

    def terms(self, z, audio, temporal=True, high_frequency=True):
        zero = z.new_zeros(())
        tv = temporal_variation(z) / (self.tv + self.eps) if temporal else zero
        hf = high_frequency_fraction(audio, self.spectral) / (self.hf + self.eps) if high_frequency else zero
        return tv, hf

