"""Operation-equivalent spectral frontend prototype using explicit reflection.

Only the padding execution path changes: concatenation/slicing replaces the
native reflection-padding operator. Loss definitions, STFT conventions and
native Whisper calibration stay unchanged. This module does not install a
runtime patch or establish CUDA determinism, WER or quality improvement.
"""
from __future__ import annotations

import torch
from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES, mel_filters, pad_or_trim


def reflect_pad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Native reflect values for one nonempty waveform, without F.pad.

    Reflection excludes the endpoint itself. The zero-padding-width path is
    the exact input object; all other paths preserve dtype/device and input
    gradients. As with native reflection, the width must be smaller than T.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError("Expected one nonempty one-dimensional waveform")
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0 or pad >= x.numel():
        raise ValueError("Reflection pad must be an integer with 0 <= pad < waveform length")
    if pad == 0:
        return x
    return torch.cat([x[1:pad+1].flip(0), x, x[-pad-1:-1].flip(0)])


def reflect_stft(x: torch.Tensor, n_fft: int, hop_length=None, win_length=None,
                 window=None, center=True, normalized=False, onesided=None,
                 return_complex=True) -> torch.Tensor:
    """torch.stft with centered reflect values supplied by explicit concat.

    A false ``center`` passes the unpadded input directly to torch.stft. The
    window, hop, normalization, one-sided convention and complex output are
    otherwise forwarded unchanged; neither resampling nor window warping is
    introduced.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 1:
        raise ValueError("STFT prototype expects one one-dimensional waveform")
    if isinstance(n_fft, bool) or not isinstance(n_fft, int) or n_fft < 1:
        raise ValueError("n_fft must be a positive integer")
    padded = reflect_pad_1d(x, n_fft // 2) if center else x
    return torch.stft(padded, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, normalized=normalized, onesided=onesided,
        return_complex=return_complex)


def high_frequency_fraction(x, config):
    """The existing Eq.13 arithmetic with an equivalent padding operation."""
    if x.ndim != 1 or x.numel() <= config.n_fft // 2:
        raise ValueError("HF expects a nonempty waveform long enough for reflection padding")
    window = torch.hann_window(config.win_length, device=x.device, dtype=x.dtype)
    power = reflect_stft(x, n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length, window=window, center=config.center).abs().square()
    energy = power.sum()
    if not torch.isfinite(energy) or energy.detach() <= 0:
        raise ValueError("HF undefined for silence or nonfinite energy")
    frequencies = torch.fft.rfftfreq(config.n_fft, d=1/config.sample_rate).to(x.device)
    return power[frequencies > config.cutoff_hz].sum() / energy


def whisper_mel(audio, n_mels=80):
    """Equivalent WhisperASR.mel, including the exact native float32 assets.

    Native Whisper creates its Hann window and mel filters in float32; those
    choices are retained rather than silently adapting them to float64 input.
    Nonfinite inputs propagate as in the current wrapper. Inputs over thirty
    seconds are rejected before native pad_or_trim can truncate them.
    """
    if audio.ndim != 1 or audio.numel() > N_SAMPLES:
        raise ValueError("Whisper path accepts one utterance <=30 s; truncation is forbidden")
    audio = pad_or_trim(audio, N_SAMPLES)
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = reflect_stft(audio, N_FFT, HOP_LENGTH, window=window)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max()-8.0)
    log_spec = (log_spec+4.0)/4.0
    return log_spec[None]
