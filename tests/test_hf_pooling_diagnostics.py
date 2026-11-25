import pytest
import torch

from beyondwaveform.config import SpectralConfig
from beyondwaveform.deterministic_frontend import high_frequency_fraction as existing_hf
from beyondwaveform.hf_pooling_diagnostics import (
    equal_frame_high_frequency_fraction,
    frame_power_summary,
    global_high_frequency_fraction,
)
from beyondwaveform.losses import high_frequency_fraction as native_hf


def separated_tone_frames(amplitudes=(1.0, 1.0), bins=(64, 448)):
    """Nonoverlapping Hann-windowed tones have analytic one-sided energy.

    Periodic Hann spreads each integer-bin tone over three bins, all wholly
    below or above 6000 Hz. Their powers sum to amplitude**2 * 3*N**2/32.
    """
    n = torch.arange(1024, dtype=torch.float64)
    frames = [amplitude * torch.sin(2 * torch.pi * frequency * n / 1024)
              for amplitude, frequency in zip(amplitudes, bins)]
    return torch.cat(frames), SpectralConfig(hop_length=1024, center=False)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_global_matches_existing_explicit_and_native_forward_exactly(center, dtype):
    x = torch.randn(4099, generator=torch.Generator().manual_seed(721), dtype=dtype)
    before = x.clone()
    config = SpectralConfig(center=center)
    summary = frame_power_summary(x, config)
    torch.testing.assert_close(summary.global_fraction, existing_hf(x, config), rtol=0, atol=0)
    torch.testing.assert_close(summary.global_fraction, native_hf(x, config), rtol=0, atol=0)
    torch.testing.assert_close(global_high_frequency_fraction(x, config), summary.global_fraction,
                               rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    tolerance = 2e-7 if dtype == torch.float32 else 3e-16
    torch.testing.assert_close(summary.energy_weighted_fraction, summary.global_fraction,
                               rtol=0, atol=tolerance)
    assert torch.all(summary.frame_fractions >= 0)
    assert torch.all(summary.frame_fractions <= 1)


def test_equal_energy_low_and_high_tone_frames_have_equal_pooling():
    x, config = separated_tone_frames()
    summary = frame_power_summary(x, config)
    analytic_energy = torch.full((2,), 3 * 1024**2 / 32, dtype=torch.float64)
    torch.testing.assert_close(summary.frame_energy, analytic_energy, rtol=1e-13, atol=1e-9)
    torch.testing.assert_close(summary.frame_fractions, torch.tensor([0., 1.], dtype=torch.float64),
                               rtol=0, atol=1e-25)
    assert summary.frame_count == summary.positive_frame_count == 2
    assert summary.silent_frame_count == 0
    for fraction in (summary.global_fraction, summary.energy_weighted_fraction,
                     summary.equal_frame_fraction):
        torch.testing.assert_close(fraction, torch.tensor(.5, dtype=torch.float64),
                                   rtol=0, atol=1e-14)


def test_loud_low_and_quiet_high_tones_differ_analytically_between_poolings():
    x, config = separated_tone_frames(amplitudes=(1., .1))
    summary = frame_power_summary(x, config)
    torch.testing.assert_close(summary.global_fraction, torch.tensor(.01/1.01, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(summary.equal_frame_fraction, torch.tensor(.5, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(summary.energy_weighted_fraction, summary.global_fraction,
                               rtol=0, atol=1e-17)
    assert summary.equal_frame_fraction > 50 * summary.global_fraction


@pytest.mark.parametrize("gain", [.03125, -2., 16.])
def test_both_poolings_are_gain_invariant(gain):
    x, config = separated_tone_frames(amplitudes=(.8, .07))
    original, scaled = frame_power_summary(x, config), frame_power_summary(x * gain, config)
    for name in ("global_fraction", "equal_frame_fraction", "energy_weighted_fraction"):
        torch.testing.assert_close(getattr(original, name), getattr(scaled, name), rtol=1e-14, atol=1e-16)
    torch.testing.assert_close(scaled.frame_energy, original.frame_energy * gain**2,
                               rtol=1e-14, atol=1e-9)


def test_silent_frames_are_counted_and_excluded_without_diluting_equal_mean():
    tones, config = separated_tone_frames(amplitudes=(1., .1))
    x = torch.cat([tones[:1024], torch.zeros(1024, dtype=tones.dtype), tones[1024:]])
    summary = frame_power_summary(x, config)
    assert summary.frame_count == 3
    assert summary.positive_frame_count == 2
    assert summary.silent_frame_count == 1
    assert summary.positive_energy_frames.tolist() == [True, False, True]
    assert summary.frame_energy[1].item() == summary.frame_fractions[1].item() == 0
    torch.testing.assert_close(summary.equal_frame_fraction, torch.tensor(.5, dtype=x.dtype),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(summary.global_fraction, existing_hf(x, config), rtol=0, atol=0)
    x.requires_grad_()
    summary = frame_power_summary(x, config)
    gradient, = torch.autograd.grad(summary.equal_frame_fraction, x)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[1024:2048]).item() == 0


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_all_silent_or_nonfinite_energy_is_rejected(value):
    with pytest.raises(ValueError, match="silence or nonfinite energy"):
        frame_power_summary(torch.full((2057,), value), SpectralConfig())


@pytest.mark.parametrize("x", [torch.zeros(0), torch.ones(512), torch.ones(2, 1024)])
def test_invalid_input_shape_and_shortness_follow_existing_contract(x):
    with pytest.raises(ValueError, match="waveform long enough"):
        frame_power_summary(x, SpectralConfig())


@pytest.mark.parametrize("pooling", [global_high_frequency_fraction, equal_frame_high_frequency_fraction])
def test_pooling_retains_finite_float32_waveform_gradients_without_mutation(pooling):
    x = torch.randn(3073, generator=torch.Generator().manual_seed(413), requires_grad=True)
    before = x.detach().clone()
    result = pooling(x, SpectralConfig())
    gradient, = torch.autograd.grad(result, x)
    assert torch.isfinite(result)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    torch.testing.assert_close(x, before, rtol=0, atol=0)


def test_global_float64_gradient_matches_independent_native_padding_path():
    x = torch.randn(2057, generator=torch.Generator().manual_seed(719),
                    dtype=torch.float64, requires_grad=True)
    config = SpectralConfig()
    actual, = torch.autograd.grad(global_high_frequency_fraction(x, config), x)
    expected, = torch.autograd.grad(native_hf(x, config), x)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("pooling", [global_high_frequency_fraction, equal_frame_high_frequency_fraction])
def test_float64_autograd_matches_numerical_waveform_derivatives(pooling):
    x = torch.randn(65, generator=torch.Generator().manual_seed(641),
                    dtype=torch.float64, requires_grad=True)
    config = SpectralConfig(n_fft=32, win_length=32, hop_length=8)
    assert torch.autograd.gradcheck(lambda waveform: pooling(waveform, config), (x,),
                                    eps=1e-6, atol=2e-8, rtol=2e-5)
