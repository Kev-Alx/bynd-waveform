import pytest
import torch
from torch import nn
from beyondwaveform.attacks import feature_attack, waveform_attack
from beyondwaveform.config import AttackConfig, SpectralConfig


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        return x[:1024].reshape(32, 32) * self.scale


class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, q):
        return torch.tanh(q.flatten() * self.scale)


class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def nll(self, audio, text):
        # Smooth, nondegenerate differentiable fixture, not a fake ASR result.
        return ((audio + 0.3) * self.scale).square().mean()


def fixture():
    torch.manual_seed(7)
    return torch.randn(1024)*0.15, Encoder(), Vocoder(), Surrogate()


def test_attack_increases_nll_without_updating_weights():
    x, encoder, vocoder, surrogate = fixture()
    params = [p.detach().clone() for m in [encoder, vocoder, surrogate] for p in m.parameters()]
    result = feature_attack(x, "text", encoder, vocoder, surrogate,
                            AttackConfig(steps=8, lambda_perc=0, learning_rate=0.005))
    assert surrogate.nll(result.audio, "text") > surrogate.nll(result.reconstruction, "text")
    assert result.delta.norm() > 0
    for old, param in zip(params, [p for m in [encoder, vocoder, surrogate] for p in m.parameters()]):
        torch.testing.assert_close(old, param)
        assert param.grad is None and not param.requires_grad
    assert len(result.trace) == 8
    assert all(r["radius_after_update"] <= 0.100001 for r in result.trace)


def test_zero_radius_is_reconstruction_not_clean():
    x, encoder, vocoder, surrogate = fixture()
    result = feature_attack(x, "text", encoder, vocoder, surrogate, AttackConfig(steps=3, rho=0))
    torch.testing.assert_close(result.audio, result.reconstruction)
    assert result.delta.count_nonzero() == 0
    assert not torch.allclose(result.audio, x)


def test_perceptual_only_is_optimized_not_zero_delta():
    x, encoder, vocoder, surrogate = fixture()
    result = feature_attack(x, "text", encoder, vocoder, surrogate,
                            AttackConfig(steps=3, attack=False, high_frequency=False, learning_rate=0.001))
    assert result.delta.norm() > 0
    assert all(r["nll_before_update"] == 0 for r in result.trace)
    assert result.trace[-1]["tv_ratio"] < result.trace[0]["tv_ratio"]


def test_all_losses_disabled_is_valid_noop():
    x, encoder, vocoder, surrogate = fixture()
    result = feature_attack(x, "text", encoder, vocoder, surrogate,
                            AttackConfig(steps=2, attack=False, lambda_perc=0))
    assert result.delta.count_nonzero() == 0


@pytest.mark.parametrize("method", ["pgd", "mi-fgsm", "vmi-fgsm"])
def test_baseline_bounds_and_direction(method):
    x, _, _, surrogate = fixture()
    y, trace = waveform_attack(x, "text", surrogate, method=method, steps=3, variance_samples=2)
    assert (y-x).abs().max() <= 0.020001
    assert y.abs().max() <= 1
    assert surrogate.nll(y, "text") > surrogate.nll(x, "text")
    assert len(trace) == 3

