import math
import pytest
import torch
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.losses import CleanReference, high_frequency_fraction, lexical_nll, temporal_variation
from beyondwaveform.attacks import project_relative_ball_, relative_radius


def test_tv_equation_feature_sum_time_mean():
    x = torch.tensor([[0.0], [1.0], [3.0]])
    assert temporal_variation(x).item() == 2.5
    assert temporal_variation(x.repeat(1, 3)).item() == 7.5
    assert temporal_variation(torch.ones(4, 6)).item() == 0
    with pytest.raises(ValueError):
        temporal_variation(x[:1])


@pytest.mark.parametrize("frequency,expected", [(1000, 0), (7000, 1)])
def test_hf_frequency_in_hz(frequency, expected):
    time = torch.arange(32000, dtype=torch.float64) / 16000
    x = torch.sin(2 * math.pi * frequency * time)
    value = high_frequency_fraction(x, SpectralConfig()).item()
    assert abs(value - expected) < 0.01


def test_hf_strict_cutoff_with_rectangular_dft_fixture():
    # With no center padding and an integer-bin sine, Hann leaks exactly to
    # adjacent bins: at 6000 Hz only the upper side lobe is above the cutoff.
    t = torch.arange(1024, dtype=torch.float64) / 16000
    cfg = SpectralConfig(center=False, hop_length=1024)
    value = high_frequency_fraction(torch.sin(2*math.pi*6000*t), cfg)
    assert value.item() == pytest.approx(1/6, abs=1e-10)


def test_hf_rejects_silence_and_empty_band():
    with pytest.raises(ValueError, match="silence"):
        high_frequency_fraction(torch.zeros(2000), SpectralConfig())
    with pytest.raises(ValueError):
        SpectralConfig(sample_rate=12000)


def test_clean_reference_detaches_and_uses_original_waveform():
    torch.manual_seed(4)
    q = torch.randn(6, 3, requires_grad=True)
    x = torch.randn(2000, requires_grad=True)
    reference = CleanReference(q, x, SpectralConfig(), 1e-8)
    assert not reference.tv.requires_grad and not reference.hf.requires_grad
    z = q.detach().clone().requires_grad_()
    audio = x.detach().clone().requires_grad_()
    tv, hf = reference.terms(z, audio)
    assert tv.item() == pytest.approx(1, abs=1e-6)
    assert hf.item() == pytest.approx(1, abs=1e-6)
    (tv + hf).backward()
    assert q.grad is None and x.grad is None
    assert z.grad.norm() > 0 and audio.grad.norm() > 0
    tv_off, hf_off = reference.terms(z, audio, temporal=False, high_frequency=False)
    assert tv_off.item() == hf_off.item() == 0
    assert not tv_off.requires_grad and not hf_off.requires_grad


def test_projection_is_global_relative_l2():
    q = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    delta = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    project_relative_ball_(delta, q, 0.1)
    assert torch.allclose(delta, torch.tensor([[0.3, 0.4], [0.0, 0.0]]))
    assert relative_radius(delta, q).item() == pytest.approx(0.1)
    project_relative_ball_(delta, q, 0)
    assert torch.count_nonzero(delta) == 0


def test_projection_zero_q_has_epsilon_radius():
    q = torch.zeros(3, 2, dtype=torch.float64)
    delta = torch.ones_like(q)
    project_relative_ball_(delta, q, 0.1, eps=1e-4)
    assert delta.norm().item() == pytest.approx(1e-5)


def test_lexical_nll_masks_special_positions():
    logits = torch.tensor([[[0., 1., 2.], [2., 1., 0.], [1., 0., 2.]]], requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[False, True, False]])
    loss = lexical_nll(logits, labels, mask)
    expected = -logits.log_softmax(-1)[0, 1, 1]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 0].abs().sum() == logits.grad[0, 2].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > 0


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_nll_reduction_matches_manual_log_probs(reduction):
    torch.manual_seed(2)
    logits = torch.randn(1, 5, 10, dtype=torch.float64)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[False, True, True, True, False]])
    values = -logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)[mask]
    expected = values.mean() if reduction == "mean" else values.sum()
    assert lexical_nll(logits, labels, mask, reduction).item() == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.parametrize("term", ["tv", "hf"])
def test_perceptual_directional_derivative(term):
    torch.manual_seed(17)
    shape = (8, 4) if term == "tv" else (2048,)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    direction = torch.randn_like(x)
    direction /= direction.norm()
    fn = temporal_variation if term == "tv" else lambda v: high_frequency_fraction(v, SpectralConfig())
    grad, = torch.autograd.grad(fn(x), x)
    numeric = (fn(x + 1e-5*direction) - fn(x - 1e-5*direction)) / 2e-5
    analytic = (grad * direction).sum()
    torch.testing.assert_close(numeric, analytic, rtol=1e-3, atol=1e-8)

