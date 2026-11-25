import math

import pytest
import torch

from beyondwaveform.config import AttackConfig, SpectralConfig
from scripts.probe_feature_gradients import component_gradients, delta_variation, gradient_cosine, probe_state


def test_component_gradients_recover_analytic_total_with_conflicting_terms():
    delta = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    terms = {"negative_nll": -(delta * torch.tensor([2.0, -1.0])).sum(),
             "weighted_tv": delta.square().sum() / 2,
             "weighted_hf": (delta[0] - delta[1]).square()}
    analytic_objective = -(2 * delta[0] - delta[1]) + delta.square().sum() / 2 + (delta[0] - delta[1]).square()
    components, total = component_gradients(terms, delta, analytic_objective)
    assert torch.equal(components["negative_nll"], torch.tensor([-2.0, 1.0], dtype=torch.float64))
    assert torch.equal(components["weighted_tv"], torch.tensor([1.0, -2.0], dtype=torch.float64))
    assert torch.equal(components["weighted_hf"], torch.tensor([6.0, -6.0], dtype=torch.float64))
    assert torch.equal(total, torch.tensor([5.0, -7.0], dtype=torch.float64))
    assert torch.equal(sum(components.values()), total)
    assert total.norm().item() == pytest.approx(math.sqrt(74))
    assert delta.grad is None


def test_disabled_components_and_undefined_cosine_are_explicit():
    delta = torch.ones(2, requires_grad=True)
    components, total = component_gradients({"active": delta.square().sum(), "off": delta.new_zeros(())}, delta)
    assert torch.equal(components["off"], torch.zeros_like(delta))
    assert torch.equal(total, torch.full_like(delta, 2))
    assert gradient_cosine(total, components["off"]) is None
    assert gradient_cosine(total, -total) == pytest.approx(-1)
    assert gradient_cosine(total, total) == pytest.approx(1)


def test_delta_tv_reports_exact_energy_normalizers_and_zero_case():
    delta = torch.tensor([[0.0], [1.0], [3.0]])
    result = delta_variation(delta)
    assert result["energy_sum"] == 10
    assert result["temporal_variation"] == 2.5
    assert result["temporal_variation_over_energy"] == .25
    assert result["difference_energy_over_energy"] == .5
    assert delta_variation(torch.zeros_like(delta))["temporal_variation_over_energy"] is None


def test_full_objective_probe_freezes_models_and_detaches_original_features():
    class TinyVocoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))

        def forward(self, features):
            return features.flatten().repeat_interleave(8) * self.scale

    class TinyNLL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))

        def nll(self, audio, text):
            assert text == "fixed reference"
            return self.scale * audio.square().mean() + audio.mean() / 5

    torch.manual_seed(48)
    q = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    x = torch.randn(64, dtype=torch.float64, requires_grad=True)
    delta = torch.randn_like(q) / 100
    vocoder, surrogate = TinyVocoder(), TinyNLL()
    attack = AttackConfig(lambda_perc=1.7, alpha_hf=.4)
    spectral = SpectralConfig(sample_rate=16, cutoff_hz=4, n_fft=8, hop_length=2, win_length=8)
    result = probe_state(q, x, delta, "fixed reference", vocoder, surrogate, attack, spectral)
    assert result["loss_total"] == pytest.approx(-result["nll"] + 1.7 * (result["tv_ratio"] + .4 * result["hf_ratio"]))
    assert result["combined_gradient_norm"] > 0
    assert result["component_sum_max_abs_difference"] < 1e-12
    assert not result["q_requires_grad"]
    assert not result["model_parameter_gradients_present"]
    assert q.grad is None and x.grad is None
    assert all(not parameter.requires_grad for model in (vocoder, surrogate) for parameter in model.parameters())
