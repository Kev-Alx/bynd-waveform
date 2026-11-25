import pytest
import torch

from beyondwaveform.feature_transforms import transform_delta


MODES = ["identity", "smooth", "smooth_norm_matched", "attenuation_matched"]


@pytest.mark.parametrize("mode", MODES)
def test_identity_and_unit_window_are_exact(mode):
    delta = torch.tensor([[1e-12, -3.0], [4.0, 2e-9]], dtype=torch.float64)
    torch.testing.assert_close(transform_delta(delta, 1, mode), delta, rtol=0, atol=0)
    assert transform_delta(delta, 9, "identity") is delta


@pytest.mark.parametrize("window", [3, 5, 9])
@pytest.mark.parametrize("mode", MODES)
def test_constant_features_are_preserved_including_edges(window, mode):
    delta = torch.tensor([[-2.0, 0.0, 7.0]], dtype=torch.float64).expand(11, -1).clone()
    torch.testing.assert_close(transform_delta(delta, window, mode), delta, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("window", [3, 5, 9])
def test_smoothing_damps_alternating_noise_without_mixing_features(window):
    alternating = torch.tensor([1.0, -1.0], dtype=torch.float64).repeat(32)
    delta = torch.stack([alternating, torch.full_like(alternating, 4.0)], dim=1)
    smoothed = transform_delta(delta, window, "smooth")
    # Ignore replicated boundaries: constant features stay constant, while
    # a frame-alternating perturbation loses at least40% of its amplitude.
    # The narrowest specified Gaussian (window3, sigma.5) retains ~57.4%.
    assert smoothed[window:-window, 0].abs().max() < 0.6
    torch.testing.assert_close(smoothed[:, 1], delta[:, 1], rtol=1e-14, atol=1e-14)
    assert smoothed[:, 0].diff().square().sum() < delta[:, 0].diff().square().sum()


@pytest.mark.parametrize("window", [3, 5, 9])
@pytest.mark.parametrize("scale", [1.0, 1e-12, 1e12])
def test_matched_variants_control_norm_and_direction(window, scale):
    generator = torch.Generator().manual_seed(42)
    delta = torch.randn(17, 4, generator=generator, dtype=torch.float64) * scale
    original = delta.clone()
    smoothed = transform_delta(delta, window, "smooth")
    matched = transform_delta(delta, window, "smooth_norm_matched")
    attenuation = transform_delta(delta, window, "attenuation_matched")
    torch.testing.assert_close(matched.norm(), delta.norm(), rtol=1e-14, atol=0)
    torch.testing.assert_close(attenuation.norm(), smoothed.norm(), rtol=1e-14, atol=0)
    torch.testing.assert_close(matched / matched.norm(), smoothed / smoothed.norm(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(attenuation / attenuation.norm(), delta / delta.norm(), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(delta, original, rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("window", [1, 3, 9])
def test_zero_input_is_finite_and_unchanged(mode, window):
    delta = torch.zeros(2, 3)
    transformed = transform_delta(delta, window, mode)
    assert torch.isfinite(transformed).all()
    assert transformed.count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mode", MODES)
def test_window_larger_than_time_preserves_shape_dtype_and_device(dtype, mode):
    delta = torch.tensor([[0.0, 2.0], [1.0, -3.0]], dtype=dtype)
    transformed = transform_delta(delta, 9, mode)
    assert transformed.shape == delta.shape
    assert transformed.dtype == dtype
    assert transformed.device == delta.device
    assert torch.isfinite(transformed).all()


def test_gaussian_boundary_rule_is_replicate_not_zero_or_wrapping():
    # For window3, sigma=.5, so side weights are exp(-2) and center is1.
    delta = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    side = torch.exp(torch.tensor(-2.0, dtype=torch.float64))
    weight = side / (1 + 2 * side)
    expected = torch.stack([weight, 1 - weight])[:, None]
    torch.testing.assert_close(transform_delta(delta, 3, "smooth"), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("delta", [None, torch.zeros(3), torch.zeros(1, 3), torch.zeros(3, 0),
                                   torch.zeros(2, 3, 1), torch.ones(2, 3, dtype=torch.int64),
                                   torch.ones(2, 3, dtype=torch.complex64),
                                   torch.tensor([[float("nan")], [0.0]]),
                                   torch.tensor([[float("inf")], [0.0]])])
def test_invalid_delta_is_rejected(delta):
    with pytest.raises(ValueError):
        transform_delta(delta)


@pytest.mark.parametrize("window", [0, -1, 2, 4, 3.0, True, "3"])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="positive odd"):
        transform_delta(torch.ones(2, 3), window, "smooth")


@pytest.mark.parametrize("eps", [0, -1e-8, float("nan"), float("inf"), "1e-8"])
def test_invalid_epsilon_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        transform_delta(torch.ones(2, 3), eps=eps)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        transform_delta(torch.ones(2, 3), mode="smooth_q")
