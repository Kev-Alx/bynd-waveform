"""CPU checks of direct waveform optimization, gradients, and failure receipts."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from waveform_pgd import (
    CONTENT_WEIGHT, STFT_WEIGHT, PGDConfig, PGDFailure, WaveformPGD,
    sequential_objective_gradients,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyEncoder(nn.Module):
    def forward(self, waveform):
        return torch.stack((waveform[::2] + 0.3, waveform[1::2] - 0.2))


def audio():
    return torch.sin(torch.arange(800, dtype=torch.float64) * 0.037) * 0.12


def toy_nll(waveform, text):
    assert text == "The unchanged Reference."
    return (2 * waveform + 0.1 * waveform.square()).sum()


def losses():
    return (
        lambda x: (x.sin() + x.square()).sum(),
        lambda x: (x - 0.01).square().sum(),
        lambda x: (0.2 * x + 0.3).square().mean(),
    )


def test_sequential_gradient_matches_joint_autograd_and_finite_difference():
    value = torch.tensor([0.013, -0.011, 0.002], dtype=torch.float64)
    source, stft, content = losses()
    gradient, row = sequential_objective_gradients(value, source, stft, content, condition="full_r3")
    joint_input = value.clone().requires_grad_()
    joint = -source(joint_input) + STFT_WEIGHT * stft(joint_input) + CONTENT_WEIGHT * content(joint_input)
    expected, = torch.autograd.grad(joint, joint_input)
    assert torch.allclose(gradient, expected, atol=1e-12, rtol=1e-12)
    finite_difference = []
    for index in range(value.numel()):
        offset = torch.zeros_like(value)
        offset[index] = 1e-6
        def objective(point):
            return -source(point) + STFT_WEIGHT * stft(point) + CONTENT_WEIGHT * content(point)
        finite_difference.append((objective(value + offset) - objective(value - offset)) / 2e-6)
    assert torch.allclose(gradient, torch.stack(finite_difference), rtol=1e-9, atol=1e-9)
    assert row["weighted_content_gradient_norm"] > 0
    assert row["weighted_stft_gradient_norm"] > 0


def test_nll_control_has_only_negative_source_derivative_and_counts():
    value = torch.tensor([0.01, -0.02], dtype=torch.float64)
    source, stft, content = losses()
    counts = {}
    gradient, row = sequential_objective_gradients(value, source, stft, content,
                                                  condition="nll_only", counts=counts)
    expected = -(value.cos() + 2 * value)
    assert torch.allclose(gradient, expected)
    assert counts == {"source_forward_calls": 1, "source_backward_calls": 1,
                      "stft_forward_calls": 1, "stft_backward_calls": 0,
                      "content_forward_calls": 1, "content_backward_calls": 0}
    assert row["optimized_objective"] == -float(source(value))
    assert row["content_gradient_norm"] is None and row["weighted_content_gradient_norm"] == 0
    assert row["full_r3_objective"] != row["optimized_objective"]


@pytest.mark.parametrize("condition", ["full_r3", "nll_only"])
def test_real_penalties_fifty_updates_all_active_gradients_and_final_iterate(condition):
    original = audio()
    callbacks = []
    solver = WaveformPGD(toy_nll, ToyEncoder().eval(), condition=condition)
    result = solver.run(original, "The unchanged Reference.", on_step=callbacks.append)
    assert len(result.trace) == len(callbacks) == 50
    assert result.counts["completed_updates"] == 50
    for term in ("source", "stft", "content"):
        assert result.counts[f"{term}_forward_calls"] == 50
        assert result.counts[f"{term}_backward_calls"] == (50 if condition == "full_r3" or term == "source" else 0)
        assert result.counts[f"final_{term}_forward_calls"] == 1
    assert result.counts["clean_reference_forward_calls"] == 1
    assert result.final_metrics["selected_iterate"] == 50
    assert result.final_metrics["residual_linf"] <= 0.003 + 1e-14
    assert result.final_metrics["residual_rms"] <= 0.003 + 1e-14
    assert result.adversarial.shape == original.shape
    assert torch.equal(result.adversarial - original, result.residual)
    assert not result.adversarial.requires_grad and not result.residual.requires_grad
    assert result.trace[0]["stft_loss"] == result.trace[0]["content_loss"] == 0
    if condition == "full_r3":
        assert result.trace[0]["stft_gradient_norm"] == result.trace[0]["content_gradient_norm"] == 0
        assert any(row["weighted_content_gradient_norm"] > 0 for row in result.trace[1:])
    else:
        # Every source derivative is positive, so minimizing -NLL raises audio
        # monotonically to the positive constraint, rather than descending NLL.
        assert torch.allclose(result.residual, torch.full_like(original, 0.003), atol=1e-14)
        assert result.final_metrics["source_sum_nll"] > toy_nll(original, "The unchanged Reference.")


def test_clipped_peak_samples_and_cached_clean_reference():
    original = audio()
    original[0], original[1] = 1.0, -1.0
    encoder = ToyEncoder().eval()
    reference = encoder(original).detach()
    result = WaveformPGD(toy_nll, encoder, condition="nll_only").run(
        original, "The unchanged Reference.", clean_features=reference)
    assert result.counts["clean_reference_forward_calls"] == 0
    assert result.adversarial[0] == 1 and result.residual[0] == 0
    assert result.adversarial[1] > -1
    assert result.adversarial.abs().max() <= 1
    assert all(row["audio_peak"] <= 1 for row in result.trace)


@pytest.mark.parametrize("source,reason", [
    (lambda x, text: (x * 0).sum(), "zero"),
    (lambda x, text: x.sum() * float("nan"), "finite scalar"),
    (lambda x, text: torch.tensor(1.0, dtype=x.dtype), "autograd graph"),
])
def test_bad_source_never_substitutes_objective_or_audio(source, reason):
    with pytest.raises(PGDFailure, match=reason) as caught:
        WaveformPGD(source, ToyEncoder().eval()).run(audio(), "The unchanged Reference.")
    receipt = caught.value.receipt
    assert receipt["status"] == "failed" and receipt["stage"] == "gradient"
    assert receipt["counts"]["completed_updates"] == 0
    assert receipt["trace"] == []
    assert torch.equal(caught.value.adversarial, audio())
    assert torch.equal(caught.value.residual, torch.zeros_like(audio()))
    assert not caught.value.adversarial.requires_grad


def test_partial_failure_retains_exact_last_feasible_iterate_and_all_calls():
    calls = 0
    def source(waveform, text):
        nonlocal calls
        calls += 1
        return waveform.sum() * (float("nan") if calls == 3 else 1)
    with pytest.raises(PGDFailure, match="finite scalar") as caught:
        WaveformPGD(source, ToyEncoder().eval(), condition="nll_only").run(audio(), "Reference")
    failure = caught.value
    assert failure.receipt["counts"]["completed_updates"] == 2
    assert len(failure.receipt["trace"]) == 2
    assert failure.receipt["counts"]["source_forward_calls"] == 3
    assert failure.receipt["counts"]["source_backward_calls"] == 2
    assert torch.allclose(failure.residual, torch.full_like(audio(), 0.00024), atol=1e-14)
    assert torch.equal(failure.adversarial, audio() + failure.residual)


def test_unexpected_infrastructure_error_is_not_a_mathematical_failure():
    def source(waveform, text):
        raise RuntimeError("Simulated CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory") as caught:
        WaveformPGD(source, ToyEncoder().eval()).run(audio(), "Reference")
    assert not isinstance(caught.value, PGDFailure)


def test_storage_callback_errors_stop_the_job():
    def callback(row):
        raise ValueError("Simulated invalid status destination")
    with pytest.raises(ValueError, match="status destination"):
        WaveformPGD(toy_nll, ToyEncoder().eval(), condition="nll_only").run(
            audio(), "The unchanged Reference.", on_step=callback)


def test_nonfinite_backward_has_receipt_before_any_update():
    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum()
        @staticmethod
        def backward(ctx, gradient):
            return torch.full((800,), float("inf"), dtype=gradient.dtype)
    with pytest.raises(PGDFailure, match="nonfinite") as caught:
        WaveformPGD(lambda x, text: BadGradient.apply(x), ToyEncoder().eval()).run(audio(), "Reference")
    assert caught.value.receipt["counts"]["source_backward_calls"] == 1
    assert caught.value.receipt["counts"]["stft_forward_calls"] == 0


def test_exact_cancellation_of_combined_gradient_is_explicit_failure():
    value = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="Summed objective"):
        sequential_objective_gradients(value, lambda x: x.sum(),
            lambda x: x.sum() / STFT_WEIGHT,
            lambda x: (x * 0).sum(), condition="full_r3")


@pytest.mark.parametrize("override", [
    {"updates": 49}, {"step_size": 0.0001}, {"epsilon": 0.001},
    {"stft_weight": STFT_WEIGHT * 2}, {"content_weight": CONTENT_WEIGHT * 2},
])
def test_no_unregistered_optimizer_changes(override):
    with pytest.raises(ValueError, match="frozen"):
        PGDConfig(**override)
