import sys
from pathlib import Path

import pytest
import torch


def test_native_reflection_backward_out_alias_is_known_without_relaxing_other_errors():
    from scripts.probe_numerical_repeatability import known_determinism_failure
    message = "reflection_pad1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'."
    assert known_determinism_failure(RuntimeError(message))
    assert not known_determinism_failure(RuntimeError("CUDA out of memory"))
    assert not known_determinism_failure(RuntimeError("reflection_pad1d_backward_out_cuda invalid input"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.probe_numerical_repeatability import (
    backend_manifest, backend_mode, compare_gradients, full_derivative, known_determinism_failure,
    repeat_state, tensor_value_sha256,
)
from beyondwaveform.attacks import freeze
from beyondwaveform.config import AttackConfig, SpectralConfig


def test_value_hash_is_container_independent_and_binds_shape_dtype_and_signed_zero(tmp_path):
    tensor = torch.tensor([[1., -0.], [2., 3.]])
    view = tensor.t().contiguous().t()
    assert not view.is_contiguous()
    assert tensor_value_sha256(tensor) == tensor_value_sha256(view)
    assert tensor_value_sha256(tensor) != tensor_value_sha256(tensor.double())
    assert tensor_value_sha256(tensor) != tensor_value_sha256(tensor.flatten())
    opposite_zero = tensor.clone()
    opposite_zero[0, 1] = 0.
    assert torch.equal(opposite_zero, tensor)
    assert tensor_value_sha256(opposite_zero) != tensor_value_sha256(tensor)


def test_gradient_comparison_reports_values_norm_and_undefined_zero_cosine():
    a, b = torch.tensor([3., 4.]), torch.tensor([3., -4.])
    row = compare_gradients(a, b)
    assert row["max_abs_difference"] == row["l2_difference"] == 8
    assert row["left_l2_norm"] == row["right_l2_norm"] == 5
    assert row["cosine"] == pytest.approx(-7/25)
    assert not row["raw_value_sha256_equal"]
    zero = compare_gradients(torch.zeros(2), torch.zeros(2))
    assert zero["raw_value_sha256_equal"] and zero["relative_l2_difference_to_left"] == 0
    assert zero["cosine"] is None


@pytest.mark.parametrize("mode", ["auto", "math", "strict_math"])
def test_backend_context_sets_only_declared_controls_and_restores_even_on_failure(mode):
    before = backend_manifest()
    with pytest.raises(RuntimeError, match="deliberate fixture"):
        with backend_mode(mode):
            active = backend_manifest()
            if mode != "auto":
                assert active["math_sdp_enabled"]
                assert not active["flash_sdp_enabled"] and not active["efficient_sdp_enabled"] and not active["cudnn_sdp_enabled"]
            if mode == "strict_math":
                assert active["deterministic_algorithms"] and not active["deterministic_warn_only"]
            raise RuntimeError("deliberate fixture")
    assert backend_manifest() == before


def test_known_unsupported_classifier_does_not_hide_oom_or_unknown_failures():
    assert known_determinism_failure(RuntimeError("reflection_pad1d_backward_cuda does not have a deterministic implementation"))
    assert not known_determinism_failure(RuntimeError("CUDA out of memory"))
    assert not known_determinism_failure(RuntimeError("unknown_operation does not have a deterministic implementation"))
    assert not known_determinism_failure(ValueError("reflection_pad1d_backward_cuda does not have a deterministic implementation"))


def fixture():
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

        def nll(self, wave, text):
            self.calls += 1
            assert text == "fixed reference"
            return self.scale*(wave+.2).square().mean()

    generator = torch.Generator().manual_seed(11)
    q = torch.randn(4, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    x = torch.randn(128, generator=generator, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(4, 8, generator=generator, dtype=torch.float64)*.01
    return q, x, delta, freeze(Vocoder()), freeze(Surrogate()), AttackConfig(), SpectralConfig(n_fft=8, hop_length=2, win_length=8)


def test_single_total_derivative_preserves_expression_and_freezes_only_copied_delta():
    from beyondwaveform.losses import CleanReference
    q, x, delta, vocoder, surrogate, attack, spectral = fixture()
    measure, gradient = full_derivative(q, x, delta, "fixed reference", vocoder, surrogate, attack, spectral)
    live = delta.clone().requires_grad_()
    reference = CleanReference(q.detach(), x.detach(), spectral, attack.eps)
    wave = vocoder(q.detach()+live)
    nll = surrogate.nll(wave, "fixed reference")
    tv, hf = reference.terms(q.detach()+live, wave)
    expected, = torch.autograd.grad(-nll + attack.lambda_perc*(tv+attack.alpha_hf*hf), live)
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    assert measure["loss_total"] == pytest.approx(-measure["nll"]+measure["tv_ratio"]+measure["hf_ratio"])
    assert measure["model_parameters_frozen"] and not measure["parameter_gradients_present"]
    assert q.grad is None and x.grad is None and delta.grad is None
    assert all(p.grad is None for model in (vocoder, surrogate) for p in model.parameters())


def test_three_cpu_repeats_are_exact_and_save_reviewable_gradient_values(tmp_path):
    q, x, delta, vocoder, surrogate, attack, spectral = fixture()
    result = repeat_state(q, x, delta, "fixed reference", vocoder, surrogate, attack, spectral, 2026, tmp_path)
    assert surrogate.calls == 3
    assert len(result["repeats"]) == len(result["pairs"]) == 3
    assert result["numerical_bitwise_agreement"]
    assert all(pair["loss_values_bitwise_equal"] and pair["gradient"]["max_abs_difference"] == 0 for pair in result["pairs"])
    for row in result["repeats"]:
        saved = torch.load(row["gradient_artifact"]["path"], weights_only=True)
        assert tensor_value_sha256(saved) == row["gradient_value_sha256"]
