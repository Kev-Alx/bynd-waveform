"""CPU mechanism and physical-invariant tests; no speech models or targets."""
import math

import pytest
import torch

from improvement.r2_geometry import LocalNeighborMixture, optimize_local_neighbors


def curved_features(dtype=torch.float64):
    angles = torch.tensor([-.20, -.07, .04, .18, .25, .12], dtype=dtype)
    scale = torch.tensor([.7, 1., 1.4, .8, 1.3, 1.1], dtype=dtype)
    return torch.stack([angles.cos(), angles.sin(), torch.full_like(angles, .2)], -1) * scale[:, None]


def test_exact_clean_start_and_frozen_current_utterance_bank():
    q = curved_features().requires_grad_()
    mixture = LocalNeighborMixture(q)
    assert torch.equal(mixture.features(), q.detach())
    assert mixture.a.count_nonzero() == 0
    assert not mixture.clean.requires_grad
    assert not mixture.bank.requires_grad
    assert set(dict(mixture.named_parameters())) == {"a", "logits"}
    for t in range(q.shape[0]):
        for slot in mixture.valid[t].nonzero().flatten().tolist():
            neighbor = int(mixture.indices[t, slot])
            assert abs(neighbor - t) <= 2
            if neighbor != t:
                cosine = torch.nn.functional.cosine_similarity(q[t], q[neighbor], dim=0)
                assert cosine >= .95
    assert torch.equal(mixture.valid[:, 2], torch.ones(q.shape[0], dtype=torch.bool))


def test_radius_projection_interpolates_the_same_convex_direction():
    q = curved_features()
    mixture = LocalNeighborMixture(q, rho=.002)
    with torch.no_grad():
        mixture.a.copy_(torch.tensor([.6, .7, .4, .9, .5, .8]))
        mixture.logits.copy_(torch.arange(30, dtype=q.dtype).reshape(6, 5) * .1)
    before_a = mixture.a.detach().clone()
    before_delta = mixture.physical_delta().detach().clone()
    assert before_delta.norm() / q.norm() > .002
    receipt = mixture.project_()
    delta = mixture.physical_delta().detach()
    assert delta.norm() <= .002 * (q.norm() + 1e-8) + 1e-12
    ratios = mixture.a.detach() / before_a
    assert torch.max(ratios) - torch.min(ratios) < 1e-12
    assert torch.allclose(delta, before_delta * ratios[0], atol=1e-12)
    assert receipt["global_interpolation_factor"] < 1
    assert receipt["minimum_cosine"] >= .95


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_extreme_updates_obey_physical_constraints_and_support(dtype):
    torch.manual_seed(17)
    q = curved_features(dtype)
    mixture = LocalNeighborMixture(q, rho=.1)
    for _ in range(12):
        with torch.no_grad():
            mixture.a.add_(torch.randn_like(mixture.a) * 3)
            mixture.logits.add_(torch.randn_like(mixture.logits) * 10)
        mixture.project_()
        z = mixture.features().detach()
        assert (z.double() - q.double()).norm() / (q.double().norm() + 1e-8) <= .1 + 1e-6
        cosine = torch.nn.functional.cosine_similarity(z.double(), q.double(), dim=-1)
        assert cosine.min() >= .95 - 1e-6
        assert torch.all((mixture.a >= 0) & (mixture.a <= 1))
        # Independent convex-support property: every linear probe remains
        # between extrema of the allowed CLEAN neighbor bank.
        probes = torch.randn(19, q.shape[-1], dtype=dtype)
        for t in range(q.shape[0]):
            values = mixture.bank[t, mixture.valid[t]] @ probes.T
            projected = z[t] @ probes.T
            assert torch.all(projected >= values.min(0).values - 2e-6)
            assert torch.all(projected <= values.max(0).values + 2e-6)


def test_zero_clean_frames_cannot_acquire_energy():
    q = torch.tensor([[1., 0.], [0., 0.], [1., .1]], dtype=torch.float64)
    mixture = LocalNeighborMixture(q)
    with torch.no_grad():
        mixture.a.fill_(1)
        mixture.logits.normal_()
    mixture.project_()
    assert torch.equal(mixture.features()[1], q[1])
    assert mixture.a[1] == 0
    assert mixture.diagnostics()["zero_frame_max_abs_delta"] == 0
    assert mixture.valid[1].sum() == 1


def test_nonzero_but_orthogonal_neighbors_produce_degenerate_bank_without_objective_query():
    q = torch.eye(3, dtype=torch.float64)
    calls = []
    result = optimize_local_neighbors(q, lambda z: calls.append(z))
    assert result.status == "degenerate_bank"
    assert calls == []
    assert result.state["objective_queries"] == 0
    assert torch.equal(result.features, q)
    assert result.diagnostics["neighbor_coverage"] == 0


@pytest.mark.parametrize("q", [torch.ones(4, 3), torch.zeros(4, 3), torch.ones(1, 3)])
def test_duplicate_or_all_zero_banks_are_named_failures(q):
    result = optimize_local_neighbors(q, lambda z: z.sum())
    assert result.status == "degenerate_bank"
    assert result.trace == []
    assert result.delta.count_nonzero() == 0
    assert result.diagnostics["effective_rank_max"] == 0


def test_canonical_exact_zero_start_escapes_on_asymmetric_geometry():
    q = torch.tensor([[1., 0.], [1., .10], [1., .16]], dtype=torch.float64)
    result = optimize_local_neighbors(q, lambda z: {"loss": -z[0, 1], "nll": z[0, 1]},
                                      steps=4, learning_rate=.05)
    assert result.status == "completed"
    assert result.state["condition"] == "canonical_zero_uniform"
    assert result.features[0, 1] > q[0, 1]
    assert result.trace[0]["amplitude_gradient_norm"] > 0
    assert result.trace[0]["logits_gradient_norm"] == 0
    assert result.trace[1]["logits_gradient_norm"] > 0
    assert result.state["objective_queries"] == 4
    assert result.state["bootstrap_frames"] == 0
    assert result.delta.norm() > 0


def test_symmetric_uniform_start_stall_is_not_hidden_as_bank_degeneracy():
    q = torch.tensor([[1., -.1], [1., 0.], [1., .1]], dtype=torch.float64)
    result = optimize_local_neighbors(q, lambda z: -z[1, 1], steps=5)
    assert result.status == "uniform_start_stalled_with_feasible_descent"
    assert result.diagnostics["neighbor_coverage"] == 1
    assert result.trace[0]["physical_gradient_norm"] == 1
    assert result.trace[0]["feasible_descent_frames"] == 1
    assert result.state["objective_queries"] == 1
    assert result.delta.count_nonzero() == 0


def test_explicit_gradient_bootstrap_repair_escapes_without_extra_objective_queries():
    q = torch.tensor([[1., -.1], [1., 0.], [1., .1]], dtype=torch.float64)
    calls = []

    def objective(z):
        calls.append(z.detach().clone())
        return -z[1, 1]

    result = optimize_local_neighbors(q, objective, steps=4, gradient_bootstrap=True)
    assert result.status == "completed"
    assert result.state["condition"] == "gradient_logit_bootstrap_repair"
    assert result.state["bootstrap_frames"] == 1
    assert torch.equal(calls[0], q)
    assert result.features[1, 1] > 0
    assert len(calls) == result.state["objective_queries"] == 4


def test_feature_jacobian_matches_directional_finite_difference_after_mixing():
    q = curved_features()
    mixture = LocalNeighborMixture(q)
    with torch.no_grad():
        mixture.a.fill_(.3)
        mixture.logits.copy_(torch.linspace(-1, 1, q.shape[0] * 5).reshape(q.shape[0], 5))
    probe = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q) / 10
    scalar = (mixture.features() * probe).sum()
    grad_a, grad_logits = torch.autograd.grad(scalar, (mixture.a, mixture.logits))
    direction_a = torch.linspace(-.1, .1, q.shape[0], dtype=q.dtype)
    direction_logits = torch.sin(torch.arange(q.shape[0] * 5, dtype=q.dtype)).reshape(q.shape[0], 5)
    analytic = (grad_a * direction_a).sum() + (grad_logits * direction_logits).sum()
    values = []
    saved_a, saved_logits = mixture.a.detach().clone(), mixture.logits.detach().clone()
    with torch.no_grad():
        for sign in (-1, 1):
            mixture.a.copy_(saved_a + sign * 1e-6 * direction_a)
            mixture.logits.copy_(saved_logits + sign * 1e-6 * direction_logits)
            values.append((mixture.features() * probe).sum())
    numeric = (values[1] - values[0]) / 2e-6
    assert torch.allclose(analytic, numeric, atol=1e-9, rtol=1e-6)
    assert grad_logits.norm() > 0


def test_all_nonnegative_feasible_derivatives_are_reported_as_no_descent():
    q = torch.tensor([[1., 0.], [1., .1], [1., .2]], dtype=torch.float64)
    result = optimize_local_neighbors(q, lambda z: z[0, 1], steps=5)
    assert result.status == "no_first_order_feasible_descent"
    assert result.trace[0]["feasible_descent_frames"] == 0
    assert result.delta.count_nonzero() == 0


def test_flat_or_disconnected_objective_stops_without_fabricated_gradient():
    q = curved_features()
    result = optimize_local_neighbors(q, lambda z: torch.tensor(1.))
    assert result.status == "flat_objective"
    assert result.trace[0]["physical_gradient_norm"] == 0
    disconnected = torch.tensor(1., requires_grad=True)
    result = optimize_local_neighbors(q, lambda z: disconnected)
    assert result.status == "flat_objective"


def test_zero_radius_preserves_exact_clean_without_queries():
    q = curved_features()
    result = optimize_local_neighbors(q, lambda z: -z.sum(), rho=0)
    assert result.status == "zero_radius"
    assert result.state["objective_queries"] == 0
    assert torch.equal(result.features, q)


def test_effective_rank_and_entropy_distinguish_a_line_from_diverse_offsets():
    q = torch.tensor([[1., -.1], [1., 0.], [1., .1]], dtype=torch.float64)
    geometry = LocalNeighborMixture(q)
    initial = geometry.diagnostics()
    assert initial["effective_rank_mean_active"] == pytest.approx(1., abs=1e-6)
    assert initial["weight_entropy_mean_active"] == pytest.approx(math.log(3))
    with torch.no_grad():
        geometry.logits.fill_(-10)
        geometry.logits[:, 2] = 10
    assert geometry.diagnostics()["weight_entropy_mean_active"] < .001


@pytest.mark.parametrize("invalid", [torch.ones(3), torch.empty(0, 3), torch.ones(3, 2, dtype=torch.int64), torch.full((3, 2), float("nan"))])
def test_invalid_clean_features_rejected(invalid):
    with pytest.raises(ValueError):
        LocalNeighborMixture(invalid)


def test_nonfinite_objective_and_parameter_updates_fail_explicitly():
    q = curved_features()
    with pytest.raises(FloatingPointError):
        optimize_local_neighbors(q, lambda z: z.sum() * float("inf"))
    geometry = LocalNeighborMixture(q)
    with torch.no_grad():
        geometry.a[0] = float("nan")
    with pytest.raises(FloatingPointError):
        geometry.project_()
