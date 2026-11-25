"""Compare the adapter against the pinned upstream WaveGuard functions."""
import ast
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pytest
import torch

from beyondwaveform.defenses import waveguard_down_up

pytest.importorskip("resampy", reason="Install the defenses optional dependency for WaveGuard tests")

SOURCE = Path(__file__).resolve().parents[1] / "verification/defense-research/waveguard/Defender"


def upstream_function(path, name, globals_):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), globals_)
    return globals_[name]


@pytest.mark.parametrize("rate", [2000, 8000])
def test_waveguard_matches_upstream_resampling_and_pcm_writer(rate, tmp_path):
    # Original librosa 0.7.2 accepts positional rates and defaults kaiser_best.
    shim = SimpleNamespace(resample=lambda x, sr, target: librosa.resample(
        x, orig_sr=sr, target_sr=target, res_type="kaiser_best"))
    down_up = upstream_function(SOURCE / "defender_multiple.py", "down_up_sample_np", {"librosa": shim})
    from scipy.io.wavfile import read, write
    save = upstream_function(SOURCE / "audioio.py", "save_as_wav", {"np": np, "spwavwrite": write})
    # Odd length exercises the upstream ceil convention; amplified input tests
    # clipping and negative truncation in the actual source WAV writer.
    x = np.random.default_rng(2026).normal(0, 0.8, 16003).astype(np.float32)
    path = tmp_path / "official.wav"
    save(path, 16000, down_up(x, down_sample_rate=rate)[:, None, None])
    sr, official_pcm = read(path)
    assert sr == 16000
    expected = torch.from_numpy(official_pcm.astype(np.float32) / 32768.)
    torch.testing.assert_close(waveguard_down_up(torch.from_numpy(x), rate), expected, atol=0, rtol=0)


def test_waveguard_removes_high_frequency_without_destroying_low_tone():
    t = torch.arange(32000) / 16000
    low, high = [0.2 * torch.sin(2 * torch.pi * hz * t) for hz in (500, 6000)]
    low_out = waveguard_down_up(low, 8000)[1000:-1000]
    high_out = waveguard_down_up(high, 8000)[1000:-1000]
    assert low_out.square().mean() > 0.99 * low[1000:-1000].square().mean()
    assert high_out.square().mean() < 1e-5 * high[1000:-1000].square().mean()


@pytest.mark.parametrize("rate", [0, -1, 16000, True])
def test_waveguard_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        waveguard_down_up(torch.zeros(1000), rate)
