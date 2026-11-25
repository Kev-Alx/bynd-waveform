from dataclasses import replace

import pytest
import torch

from beyondwaveform.attacks import feature_attack, relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.trajectory_diagnostics import SNAPSHOT_STEPS, collect_trajectory, finalize_deltas


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.8, dtype=torch.float64))

    def forward(self, x):
        return x[:32].reshape(4, 8) * self.scale


class Vocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))

    def forward(self, q):
        return torch.tanh(q.flatten().repeat_interleave(4) * self.scale)


class Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
        self.calls = 0

    def nll(self, audio, text):
        assert text == "fixed reference"
        self.calls += 1
        return ((audio+.2)*self.scale).square().mean()


def fixture():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(128, generator=generator, dtype=torch.float64, requires_grad=True) * .15
    return x.detach().requires_grad_(), Encoder(), Vocoder(), Surrogate()


def config_pair():
    return AttackConfig(learning_rate=.05), SpectralConfig(n_fft=8, hop_length=2, win_length=8)


def test_collector_exactly_reproduces_original_attack_and_keeps_terminal_trace():
    x, encoder, vocoder, surrogate = fixture()
    config, spectral = config_pair()
    original_parameters = [p.detach().clone() for m in (encoder, vocoder, surrogate) for p in m.parameters()]
    torch.manual_seed(49)
    original = feature_attack(x, "fixed reference", encoder, vocoder, surrogate, config, spectral)
    surrogate.calls = 0
    torch.manual_seed(49)
    collection = collect_trajectory(x, "fixed reference", encoder, vocoder, surrogate, config, spectral)
    result = collection.result
    assert surrogate.calls == 50
    assert result.trace == original.trace
    for current, expected in ((result.audio, original.audio), (result.reconstruction, original.reconstruction),
                              (result.delta, original.delta)):
        torch.testing.assert_close(current, expected, rtol=0, atol=0)
    assert tuple(collection.snapshots) == SNAPSHOT_STEPS
    assert len(result.trace) == 50
    for step, delta in collection.snapshots.items():
        assert delta.device.type == "cpu" and not delta.requires_grad and delta.grad is None
        assert relative_radius(delta, collection.q).item() <= .100001
        assert relative_radius(delta, collection.q).item() == pytest.approx(result.trace[step-1]["radius_after_update"])
    torch.testing.assert_close(collection.snapshots[50], result.delta, rtol=0, atol=0)
    assert not collection.q.requires_grad and collection.q.grad is None
    assert x.grad is None
    for saved, parameter in zip(original_parameters, [p for m in (encoder, vocoder, surrogate) for p in m.parameters()]):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
        assert not parameter.requires_grad and parameter.grad is None


def test_finalization_mean_and_matched_control_are_independent_and_feasible():
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(7, 4, generator=generator)
    snapshots = {}
    for step in SNAPSHOT_STEPS:
        delta = torch.randn(q.shape, generator=generator)
        snapshots[step] = delta * (.1*q.norm()/delta.norm())
    original = {step: delta.clone() for step, delta in snapshots.items()}
    outputs = finalize_deltas(snapshots)
    expected_mean = torch.stack([snapshots[step] for step in SNAPSHOT_STEPS]).mean(0)
    assert set(outputs) == {"terminal", "average_41_50", "terminal_norm_matched"}
    torch.testing.assert_close(outputs["average_41_50"], expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(outputs["terminal"], snapshots[50], rtol=0, atol=0)
    assert outputs["terminal"].data_ptr() != snapshots[50].data_ptr()
    average, control = outputs["average_41_50"], outputs["terminal_norm_matched"]
    torch.testing.assert_close(control.norm(), average.norm(), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(control/control.norm(), snapshots[50]/snapshots[50].norm(), rtol=1e-6, atol=1e-7)
    assert all(relative_radius(delta, q) <= .100001 for delta in outputs.values())
    assert all(not delta.requires_grad and delta.grad is None for delta in outputs.values())
    for step in SNAPSHOT_STEPS:
        torch.testing.assert_close(snapshots[step], original[step], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_finalization_preserves_dtype_and_exact_mean_arithmetic_at_different_scales(scale):
    snapshots = {step: torch.tensor([[step*.01, .3], [-.2, step*.02]], dtype=torch.float64)*scale
                 for step in SNAPSHOT_STEPS}
    outputs = finalize_deltas(snapshots)
    assert all(delta.dtype == torch.float64 for delta in outputs.values())
    torch.testing.assert_close(outputs["average_41_50"], torch.stack(list(snapshots.values())).mean(0), rtol=0, atol=0)
    torch.testing.assert_close(outputs["average_41_50"].norm(), outputs["terminal_norm_matched"].norm(), rtol=1e-14, atol=0)


def test_all_zero_snapshots_and_zero_mean_have_explicit_controls():
    zero = {step: torch.zeros(2, 3) for step in SNAPSHOT_STEPS}
    assert all(value.count_nonzero() == 0 for value in finalize_deltas(zero).values())
    opposite = {step: torch.ones(2, 3)*(1 if step%2 else -1) for step in SNAPSHOT_STEPS}
    result = finalize_deltas(opposite)
    assert result["average_41_50"].count_nonzero() == result["terminal_norm_matched"].count_nonzero() == 0
    missing_direction = {step: torch.ones(2, 3) for step in SNAPSHOT_STEPS}
    missing_direction[50].zero_()
    with pytest.raises(ValueError, match="zero terminal"):
        finalize_deltas(missing_direction)


@pytest.mark.parametrize("update", [{"steps": 49}, {"rho": .2}, {"lambda_perc": .5},
    {"optimizer": "sgd"}, {"initialization": "normal"}, {"lr_schedule": "cosine"}])
def test_collector_rejects_changed_optimization_protocol_before_model_calls(update):
    x, encoder, vocoder, surrogate = fixture()
    config, spectral = config_pair()
    with pytest.raises(ValueError, match="50 constant-Adam"):
        collect_trajectory(x, "fixed reference", encoder, vocoder, surrogate, replace(config, **update), spectral)
    assert surrogate.calls == 0
    assert all(parameter.requires_grad for module in (encoder, vocoder, surrogate) for parameter in module.parameters())


def test_collector_rejects_changed_spectral_contract():
    x, encoder, vocoder, surrogate = fixture()
    config, spectral = config_pair()
    with pytest.raises(ValueError, match="cutoff=6000"):
        collect_trajectory(x, "fixed reference", encoder, vocoder, surrogate, config, replace(spectral, cutoff_hz=5000))


@pytest.mark.parametrize("kind", ["missing", "extra", "shape", "dtype", "nan"])
def test_invalid_snapshot_collections_are_rejected(kind):
    snapshots = {step: torch.ones(2, 3) for step in SNAPSHOT_STEPS}
    if kind == "missing":
        del snapshots[41]
    elif kind == "extra":
        snapshots[40] = torch.ones(2, 3)
    elif kind == "shape":
        snapshots[45] = torch.ones(3, 3)
    elif kind == "dtype":
        snapshots[45] = torch.ones(2, 3, dtype=torch.float64)
    else:
        snapshots[45][0, 0] = float("nan")
    with pytest.raises(ValueError):
        finalize_deltas(snapshots)
