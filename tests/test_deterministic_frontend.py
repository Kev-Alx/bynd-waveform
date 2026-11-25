import pytest
import torch
import torch.nn.functional as F
import whisper

from beyondwaveform.config import SpectralConfig
from beyondwaveform.deterministic_frontend import (
    high_frequency_fraction, reflect_pad_1d, reflect_stft, whisper_mel,
)
from beyondwaveform.losses import high_frequency_fraction as native_hf


def native_reflect(x, pad):
    return F.pad(x[None, None], (pad, pad), mode="reflect").reshape(-1)


@pytest.mark.parametrize("length,pad", [(2, 0), (2, 1), (7, 1), (7, 6), (513, 512), (1025, 200)])
def test_explicit_reflection_matches_native_values_and_float64_input_gradient(length, pad):
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(length, dtype=torch.float64, generator=generator, requires_grad=True)
    original = x.detach().clone()
    expected, actual = native_reflect(x, pad), reflect_pad_1d(x, pad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.linspace(.3, 1.4, expected.numel(), dtype=torch.float64)
    native_grad, = torch.autograd.grad((expected.square()*weights).sum(), x, retain_graph=True)
    concat_grad, = torch.autograd.grad((actual.square()*weights).sum(), x)
    torch.testing.assert_close(concat_grad, native_grad, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    if pad == 0:
        assert actual is x


@pytest.mark.parametrize("n_fft,hop,win,length", [(8, 2, 8, 37), (9, 3, 7, 41),
    (400, 160, 400, 1600), (1024, 256, 1024, 2057), (1024, 256, 512, 1537)])
@pytest.mark.parametrize("center", [True, False])
def test_explicit_stft_matches_native_values_and_float64_gradient(n_fft, hop, win, length, center):
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(length, dtype=torch.float64, generator=generator, requires_grad=True)
    window = torch.hann_window(win, dtype=x.dtype)
    expected = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=window,
                         center=center, pad_mode="reflect", return_complex=True)
    actual = reflect_stft(x, n_fft, hop_length=hop, win_length=win, window=window, center=center)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A nonuniform spectral weighting exercises overlapping frame/padding
    # gradients rather than an energy identity that could cancel mistakes.
    weights = torch.linspace(.2, 1.1, actual.numel(), dtype=x.dtype).reshape(actual.shape)
    native_grad, = torch.autograd.grad((expected.abs().square()*weights).sum(), x, retain_graph=True)
    concat_grad, = torch.autograd.grad((actual.abs().square()*weights).sum(), x)
    torch.testing.assert_close(concat_grad, native_grad, rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hf_fraction_and_gradient_preserve_original_loss_arithmetic(center, dtype):
    generator = torch.Generator().manual_seed(27)
    x = torch.randn(2057, generator=generator, dtype=dtype, requires_grad=True)
    config = SpectralConfig(center=center)
    expected, actual = native_hf(x, config), high_frequency_fraction(x, config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    native_grad, = torch.autograd.grad(expected, x, retain_graph=True)
    concat_grad, = torch.autograd.grad(actual, x)
    tolerance = (2e-6, 1e-9) if dtype == torch.float32 else (2e-13, 1e-15)
    torch.testing.assert_close(concat_grad, native_grad, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("n_mels,length", [(80, 720), (80, 16000), (128, 2003)])
def test_full_whisper_mel_matches_native_float32_values_and_gradients(n_mels, length):
    generator = torch.Generator().manual_seed(47)
    x = torch.randn(length, generator=generator, dtype=torch.float32, requires_grad=True)*.1
    padded = whisper.pad_or_trim(x, whisper.audio.N_SAMPLES)
    expected = whisper.log_mel_spectrogram(padded, n_mels=n_mels)[None]
    actual = whisper_mel(x, n_mels)
    assert actual.shape == (1, n_mels, 3000)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.linspace(.3, .9, actual.numel()).reshape(actual.shape)
    native_grad, = torch.autograd.grad((expected*weights).sum(), x, retain_graph=True)
    concat_grad, = torch.autograd.grad((actual*weights).sum(), x)
    # Different addition order of duplicated boundary gradients may round
    # differently in float32; this is equivalence, not bitwise-CUDA proof.
    torch.testing.assert_close(concat_grad, native_grad, rtol=2e-5, atol=2e-5)


def test_whisper_zero_and_empty_audio_keep_native_silence_calibration():
    for x in (torch.zeros(0), torch.zeros(720)):
        expected = whisper.log_mel_spectrogram(whisper.pad_or_trim(x, whisper.audio.N_SAMPLES))[None]
        actual = whisper_mel(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.all(actual == -1.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_policies_match_existing_hf_and_whisper_wrappers(value):
    x = torch.ones(1025)
    x[10] = value
    for function in (native_hf, high_frequency_fraction):
        with pytest.raises(ValueError, match="nonfinite energy"):
            function(x, SpectralConfig())
    # WhisperASR.mel currently propagates nonfinite audio instead of rejecting
    # it; this prototype must not invent an extra validation policy.
    expected = whisper.log_mel_spectrogram(whisper.pad_or_trim(x, whisper.audio.N_SAMPLES))[None]
    actual = whisper_mel(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert not torch.isfinite(actual).all()


def test_whisper_retains_native_float32_filter_dtype_contract():
    audio = torch.ones(720, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        whisper.log_mel_spectrogram(whisper.pad_or_trim(audio, whisper.audio.N_SAMPLES))
    with pytest.raises(RuntimeError):
        whisper_mel(audio)


@pytest.mark.parametrize("x,pad", [(torch.zeros(0), 0), (torch.ones(2, 3), 1),
    (torch.ones(3), -1), (torch.ones(3), 3), (torch.ones(3), 4), (torch.ones(3), 1.5), (torch.ones(3), True)])
def test_invalid_reflection_dimensions_and_widths_are_rejected(x, pad):
    with pytest.raises(ValueError):
        reflect_pad_1d(x, pad)


def test_hf_invalid_shape_shortness_and_silence_match_existing_loss():
    for x in (torch.zeros(2, 1025), torch.ones(512), torch.zeros(1025)):
        for function in (native_hf, high_frequency_fraction):
            with pytest.raises(ValueError):
                function(x, SpectralConfig())


def test_whisper_rejects_batching_and_truncation_before_processing():
    for x in (torch.ones(2, 720), torch.ones(whisper.audio.N_SAMPLES+1)):
        with pytest.raises(ValueError, match="truncation is forbidden"):
            whisper_mel(x)
