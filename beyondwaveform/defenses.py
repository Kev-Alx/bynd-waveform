"""Source-backed defense adapters with explicit, independently chosen settings.

WaveGuard source: shehzeen/waveguard_defense at
4320cfe12d0452e7f34f2903a4e302780662f5e1, Defender/defender_multiple.py
(down_up_sample_np) and Defender/audioio.py (save_as_wav).
This adapter covers the resampling transformation, not WaveGuard detection.
"""
from __future__ import annotations

import numpy as np
import torch

from .io import digest
from .models.targets import load_target


def waveguard_down_up(audio, downsample_hz, sample_rate=16000):
    """Resample with the original kaiser_best policy, then emulate PCM16 I/O.

    librosa 0.7.2 used kaiser_best by default. Modern librosa needs keyword
    sample rates and an explicit res_type. Keep its ceil-based output length;
    do not peak-normalize or crop. The upstream WAV writer clips, truncates
    toward zero to int16, then its reader divides by 32768.
    """
    import librosa

    if sample_rate != 16000 or isinstance(downsample_hz, bool) or not 0 < downsample_hz < sample_rate:
        raise ValueError("WaveGuard requires 16 kHz input and an explicit lower resampling rate")
    if audio.ndim != 1 or not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError("Expected finite, nonempty mono audio")
    x = audio.detach().cpu().float().numpy()
    down = librosa.resample(x, orig_sr=sample_rate, target_sr=downsample_hz,
                           res_type="kaiser_best", fix=True, scale=False)
    up = librosa.resample(down, orig_sr=downsample_hz, target_sr=sample_rate,
                         res_type="kaiser_best", fix=True, scale=False)
    pcm = np.clip(up * 32768., -32768., 32767.).astype(np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.).to(audio.device)


class WaveGuardDownUpASR:
    """Use the existing extension interface without changing attack/eval code."""
    def __init__(self, target_config, checkpoint_sha256, downsample_hz, device="cuda"):
        if target_config.get("kind") != "whisper" or target_config.get("defense", {}).get("kind", "none") != "none":
            raise ValueError("Supply the frozen raw Whisper target")
        if digest(target_config["checkpoint"]) != checkpoint_sha256:
            raise ValueError("Defense target checkpoint changed")
        if isinstance(downsample_hz, bool) or not 0 < downsample_hz < 16000:
            raise ValueError("Invalid explicit downsampling rate")
        self.target, self.target_meta = load_target(target_config, device)
        self.downsample_hz = downsample_hz

    @torch.no_grad()
    def transcribe(self, audio):
        return self.target.transcribe(waveguard_down_up(audio, self.downsample_hz))
