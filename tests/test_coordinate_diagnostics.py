"""Independent checks of clean weights, invertible coordinates and projection."""
import pytest
import torch

from beyondwaveform.coordinate_diagnostics import coordinate_weights, physical_delta, project_coordinates_


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_time_weights_match_independent_receptive_field_energy(dtype):
    x = torch.cat([torch.full((320,), v, dtype=dtype) for v in (0., .01, .3, 1., .4)])
    q = torch.arange(32, dtype=dtype).reshape(4, 8)
    actual = coordinate_weights(q, x, 'time_energy')
    rms = torch.tensor([x[i*320:i*320+400].double().square().mean().sqrt() for i in range(4)])
    raw = (rms / rms.max()).clamp(min=.1, max=1.)
    expected = (raw / (raw.square().sum() / 4).sqrt()).to(dtype).reshape(4, 1)
    assert torch.equal(actual, expected)
    assert actual.shape == (4, 1) and not actual.requires_grad and actual.device.type == 'cpu'
    assert abs(actual.double().square().mean().item() - 1) < 1e-7
    assert actual.max() / actual.min() <= 10.00001


def test_channel_weights_use_only_clean_temporal_variance_with_population_denominator():
    q = torch.tensor([[1., 0., 0.], [1., 1., 10.], [1., 2., 20.]], dtype=torch.float64, requires_grad=True)
    weights = coordinate_weights(q, torch.full((3,), float('nan')), 'channel_std')
    deviations = (q.detach() - q.detach().mean(0)).square().sum(0).div(3).sqrt()
    raw = (deviations / deviations.max()).clamp(.1, 1.)
    torch.testing.assert_close(weights, (raw / raw.square().mean().sqrt()).unsqueeze(0), rtol=0, atol=0)
    assert not weights.requires_grad and q.grad is None
    assert weights[0, 0] == weights[0, 1] and weights[0, 2] > weights[0, 1]


@pytest.mark.parametrize('kind', ['channel_std', 'time_energy'])
def test_zero_statistic_is_explicit_identity_without_removing_coordinates(kind):
    q = torch.ones(3, 4)
    weights = coordinate_weights(q, torch.zeros(1040), kind)
    assert torch.equal(weights, torch.ones_like(weights))


@pytest.mark.parametrize('samples', [399, 1359, 1680])
def test_time_frame_mismatch_rejects_instead_of_interpolating(samples):
    with pytest.raises(ValueError, match='at least 400|geometry differ'):
        coordinate_weights(torch.ones(4, 8), torch.ones(samples), 'time_energy')


def test_unknown_and_nonfinite_coordinates_reject():
    with pytest.raises(ValueError, match='Unknown optimization'):
        coordinate_weights(torch.ones(3, 4), torch.zeros(1040), 'inverse_energy')
    with pytest.raises(ValueError, match='finite floating'):
        coordinate_weights(torch.full((3, 4), float('nan')), torch.zeros(1040), 'channel_std')


def test_physical_projection_matches_independent_euclidean_ball_and_keeps_leaf():
    q = torch.arange(1., 13., dtype=torch.float64).reshape(3, 4)
    weights = torch.tensor([[.2, .7, 1.1, 2.]], dtype=q.dtype)
    theta = (q * .5).clone().requires_grad_()
    before = theta.detach() * weights
    limit = .1 * (q.norm() + 1e-8)
    expected = before * (limit / before.norm())
    actual = project_coordinates_(theta, weights, q, .1)
    torch.testing.assert_close(actual, expected, rtol=2e-16, atol=1e-16)
    assert theta.is_leaf and theta.requires_grad
    # Every original feasible delta has a representable theta; no coordinates are masked.
    target = q * .01
    torch.testing.assert_close(physical_delta(target / weights, weights), target, rtol=1e-15, atol=1e-16)


def test_identity_path_aliases_original_parameter_and_preserves_interior():
    q = torch.ones(3, 4)
    theta = torch.full_like(q, .01, requires_grad=True)
    weights = torch.ones(1, 1)
    assert physical_delta(theta, weights, True) is theta
    assert project_coordinates_(theta, weights, q, .1, identity=True) is theta
    assert torch.equal(theta, torch.full_like(q, .01))
