"""CPU checks of analytic quality projection and nonlinear-feasible continuation."""
from __future__ import annotations

import math
import pytest
import torch
from torch import nn

import constrained_refine as cr
from constrained_refine import (
    CONTENT_WEIGHT, STFT_WEIGHT, RefineConfig, RefineFailure, WaveformRefiner,
    constrained_direction, nonlinear_backtracking_step, project_two_halfspaces,
    sequential_component_gradients,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def vector(values):
    return torch.tensor(values, dtype=torch.float64)


def test_exact_two_active_planes_has_analytic_kkt_solution():
    u = vector([3., 2., 4.])
    normals = [vector([1., 0., 0.]), vector([.6, .8, 0.])]
    result, receipt = project_two_halfspaces(u, normals, [0., 0.])
    assert torch.allclose(result, vector([0., 0., 4.]), atol=1e-14)
    assert receipt['active_set'] == [0, 1]
    assert receipt['multipliers'] == pytest.approx([1.5, 2.5])
    assert receipt['active_subsets_examined'] == 4
    # Stationarity, dual feasibility, primal feasibility and complementarity.
    stationarity = result - u + sum(lam * normal for lam, normal in zip(receipt['multipliers'], normals))
    assert torch.linalg.vector_norm(stationarity) < 1e-14
    assert all(lam >= 0 for lam in receipt['multipliers'])
    assert max(receipt['halfspace_residuals_float64']) <= 1e-14


def test_single_active_plane_and_feasible_identity():
    u = vector([2., 3., 4.])
    normals = [vector([1., 0., 0.]), vector([.6, .8, 0.])]
    result, receipt = project_two_halfspaces(u, normals, [0., 0.])
    assert torch.allclose(result, vector([-.16, .12, 4.]), atol=1e-14)
    assert receipt['active_set'] == [1]
    identity, feasible = project_two_halfspaces(vector([-1., -2., 3.]), normals, [0., 0.])
    assert torch.equal(identity, vector([-1., -2., 3.]))
    assert feasible['active_set'] == []


def test_permuting_quality_terms_preserves_solution_and_distance():
    generator = torch.Generator().manual_seed(113)
    for _ in range(40):
        u = torch.randn(13, dtype=torch.float64, generator=generator)
        normals = [torch.randn(13, dtype=torch.float64, generator=generator) for _ in range(2)]
        normals = [a / a.norm() for a in normals]
        bounds = [-.01, -.02]
        result, receipt = project_two_halfspaces(u, normals, bounds)
        swapped, other = project_two_halfspaces(u, normals[::-1], bounds[::-1])
        assert result is not None and swapped is not None
        assert torch.allclose(result, swapped, atol=1e-13, rtol=1e-13)
        assert receipt['distance_squared'] == pytest.approx(other['distance_squared'], abs=1e-12)
        assert all(float(a @ result) <= b + 1e-12 for a, b in zip(normals, bounds))
        # Projection theorem: every feasible tested point lies on the inward
        # side of the supporting plane through the returned closest point.
        probes = torch.randn(60, 13, dtype=torch.float64, generator=generator)
        feasible = (probes @ normals[0] <= bounds[0]) & (probes @ normals[1] <= bounds[1])
        assert bool((((probes[feasible] - result) @ (u - result)) <= 1e-11).all())


@pytest.mark.parametrize('normals,bounds,expected', [
    ([[1., 0.], [1., 0.]], [.5, 1.], [.5, 3.]),
    ([[1., 0.], [-1., 0.]], [0., 0.], [0., 3.]),
    ([[0., 0.], [1., 0.]], [0., .4], [.4, 3.]),
])
def test_parallel_opposite_and_zero_normals(normals, bounds, expected):
    result, receipt = project_two_halfspaces(vector([2., 3.]), list(map(vector, normals)), bounds)
    assert torch.allclose(result, vector(expected), atol=1e-14)
    assert receipt['singular_subsets'] > 0


def test_infeasible_inward_parallel_planes_detected_not_pseudoinverse_fudged():
    result, receipt = project_two_halfspaces(vector([2., 3.]),
        [vector([1., 0.]), vector([-1., 0.])], [-.01, 0.])
    assert result is None and receipt['feasible'] is False
    zero, _ = project_two_halfspaces(vector([2., 3.]), [vector([0., 0.])], [-.01])
    assert zero is None


def test_free_mask_is_an_exact_coordinate_constraint():
    u = vector([5., 2., 3.])
    normal = vector([100., 1., 0.])
    result, _ = project_two_halfspaces(u, [normal], [0.], free_mask=torch.tensor([False, True, True]))
    assert torch.equal(result, vector([0., 0., 3.]))


def test_inward_conflict_margin_and_no_second_sign_operation():
    original, waveform = vector([0., 0.]), vector([.001, 0.])
    gradients = {'source': vector([1., 1.]), 'stft': vector([1., 0.]), 'content': vector([0., 0.])}
    result, receipt = constrained_direction(original, waveform, gradients)
    assert torch.allclose(result, vector([-.01, 1.]), atol=1e-14)
    assert receipt['inward_halfspace_bounds'] == [-.01]
    assert receipt['zero_margin_fallback'] is False
    assert result[0] != result[0].sign()


def test_infeasible_inward_margin_has_explicit_zero_margin_fallback():
    gradients = {'source': vector([1., 1.]), 'stft': vector([1., 0.]), 'content': vector([-1., 0.])}
    result, receipt = constrained_direction(vector([0., 0.]), vector([0., 0.]), gradients)
    assert torch.equal(result, vector([0., 1.]))
    assert receipt['zero_margin_fallback'] is True
    assert receipt['qp']['feasible'] is True


def test_source_zero_and_box_blocked_coordinates_are_legitimate_stationary_directions():
    gradients = {'source': vector([0., 0.]), 'stft': vector([1., 2.]), 'content': vector([0., 0.])}
    result, receipt = constrained_direction(vector([0., 0.]), vector([.003, -.003]), gradients)
    assert torch.equal(result, vector([0., 0.]))
    gradients['source'] = vector([1., -1.])
    result, receipt = constrained_direction(vector([0., 0.]), vector([.003, -.003]), gradients)
    assert torch.equal(result, vector([0., 0.]))
    assert receipt['free_coordinate_count'] == 0


def test_component_gradients_equal_joint_gradient_without_losing_each_derivative():
    value = vector([.02, -.01, .04])
    functions = (lambda x: (x.sin() + x.square()).sum(),
                 lambda x: (x - .03).square().mean(),
                 lambda x: (x * .2 + .1).square().sum())
    counts = {}
    gradients, row = sequential_component_gradients(value, *functions, counts=counts)
    variable = value.clone().requires_grad_()
    objective = -functions[0](variable) + STFT_WEIGHT * functions[1](variable) + CONTENT_WEIGHT * functions[2](variable)
    expected, = torch.autograd.grad(objective, variable)
    assert torch.allclose(-gradients['source'] + STFT_WEIGHT * gradients['stft'] + CONTENT_WEIGHT * gradients['content'], expected, atol=1e-13)
    assert set(gradients) == {'source', 'stft', 'content'}
    assert row['full_r3_objective'] == pytest.approx(float(objective.detach()))
    for name in cr.TERMS:
        assert counts[f'{name}_forward_calls'] == counts[f'{name}_backward_calls'] == 1
        assert counts[f'{name}_gradient_forward_calls'] == 1


def test_backtracking_first_rejects_curvature_then_accepts_half_step_and_counts():
    original = vector([0.])
    functions = (lambda x: x.sum(), lambda x: (x - .00005).square().sum(), lambda x: (x * 0).sum())
    caps = {'stft_loss': .00005 ** 2, 'content_loss': 0.}
    counts = {}
    result, metrics, trials = nonlinear_backtracking_step(original, original, vector([1.]), functions,
        caps=caps, current_source=0., counts=counts)
    assert result.item() == pytest.approx(.00006)
    assert len(trials) == 2 and not trials[0]['accepted'] and trials[1]['accepted']
    assert trials[1]['factor'] == .5
    assert metrics['stft_loss'] <= caps['stft_loss']
    assert all(t['forward_calls'] == {'source': 1, 'stft': 1, 'content': 1} for t in trials)
    for name in cr.TERMS:
        assert counts[f'{name}_forward_calls'] == counts[f'{name}_trial_forward_calls'] == 2


def test_all_ten_invalid_trials_retain_exact_last_feasible_audio():
    original = vector([0.])
    functions = (lambda x: x.sum(), lambda x: 1e9 * x.square().sum(), lambda x: (x * 0).sum())
    counts = {}
    result, metrics, trials = nonlinear_backtracking_step(original, original, vector([1.]), functions,
        caps={'stft_loss': 0., 'content_loss': 0.}, current_source=0., counts=counts)
    assert torch.equal(result, original) and metrics is None
    assert [t['factor'] for t in trials] == [2. ** -i for i in range(10)]
    assert all(not t['accepted'] for t in trials)
    assert counts['source_forward_calls'] == counts['stft_forward_calls'] == counts['content_forward_calls'] == 10


def test_nonlinear_source_nondecrease_is_checked_after_physical_projection():
    original = vector([0.])
    functions = (lambda x: -x.sum(), lambda x: (x * 0).sum(), lambda x: (x * 0).sum())
    result, metrics, trials = nonlinear_backtracking_step(original, original, vector([1.]), functions,
        caps={'stft_loss': 0., 'content_loss': 0.}, current_source=0., counts={})
    assert torch.equal(result, original) and metrics is None
    assert len(trials) == 10 and all(not t['source_nondecrease'] for t in trials)


def test_nonfinite_trial_is_rejected_and_counted_without_hiding_forward_calls():
    original = vector([0.])
    functions = (lambda x: x.sum(),
        lambda x: x.sum() * (float('nan') if float(x.max()) > .0001 else 0.),
        lambda x: (x * 0).sum())
    counts = {}
    result, metrics, trials = nonlinear_backtracking_step(original, original, vector([1.]), functions,
        caps={'stft_loss': 0., 'content_loss': 0.}, current_source=0., counts=counts)
    assert result.item() == pytest.approx(.00006)
    assert 'invalid_reason' in trials[0]
    assert trials[0]['forward_calls'] == {'source': 1, 'stft': 1, 'content': 0}
    assert trials[1]['forward_calls'] == {'source': 1, 'stft': 1, 'content': 1}
    assert counts['source_forward_calls'] == counts['stft_forward_calls'] == 2
    assert counts['content_forward_calls'] == 1


def test_tiny_content_cap_does_not_receive_unit_scaled_absolute_slack():
    x = torch.zeros(1, dtype=torch.float32)
    assert cr._cap_tolerance(x, 1e-8) == 1e-12
    assert cr._cap_tolerance(x, .1) == pytest.approx(32 * torch.finfo(x.dtype).eps * .1)


class ToyEncoder(nn.Module):
    def forward(self, x):
        return x + .2


def cheap_penalties(monkeypatch):
    monkeypatch.setattr(cr, 'multiresolution_stft_loss', lambda a, o: (a - o).square().mean())


def test_constrained_full_fifty_updates_preserve_caps_and_last_iterate(monkeypatch):
    cheap_penalties(monkeypatch)
    original = torch.zeros(800, dtype=torch.float64)
    initial = torch.full_like(original, .001)
    initial[400:] *= -1
    weights = torch.linspace(.1, 1., 800, dtype=torch.float64)
    callbacks = []
    result = WaveformRefiner(lambda x, text: (x * weights).sum(), ToyEncoder().eval()).run(
        original, 'Reference unchanged.', initial_waveform=initial, on_step=callbacks.append)
    assert len(result.trace) == len(callbacks) == 50
    assert result.counts['completed_updates'] == 50
    assert result.counts['accepted_updates'] + result.counts['noop_updates'] == 50
    assert result.final_metrics['selected_iterate'] == 100
    assert result.final_metrics['stft_cap_satisfied'] and result.final_metrics['content_cap_satisfied']
    assert result.final_metrics['source_nondecrease_from_initial']
    assert torch.equal(result.residual, result.adversarial - original)
    assert not result.adversarial.requires_grad
    for name in cr.TERMS:
        assert result.counts[f'{name}_backward_calls'] == 50
        assert result.counts[f'{name}_gradient_forward_calls'] == 50
        assert result.counts[f'{name}_initial_forward_calls'] == result.counts[f'{name}_final_forward_calls'] == 1
    for row in result.trace:
        for trial in row['line_search']:
            if trial['accepted']:
                assert trial['stft_cap_satisfied'] and trial['content_cap_satisfied'] and trial['source_nondecrease']


def test_zero_source_stationary_point_completes_noops_without_failure(monkeypatch):
    cheap_penalties(monkeypatch)
    original = torch.zeros(800, dtype=torch.float64)
    result = WaveformRefiner(lambda x, text: (x * 0).sum(), ToyEncoder().eval()).run(
        original, 'Reference.', initial_waveform=original)
    assert torch.equal(result.adversarial, original)
    assert result.counts['noop_updates'] == 50 and result.counts['accepted_updates'] == 0
    assert result.counts['line_search_trials'] == 0
    assert all(row['source_gradient_norm'] == 0 for row in result.trace)


def test_weighted_control_exactly_continues_frozen_combined_sign_steps(monkeypatch):
    cheap_penalties(monkeypatch)
    original = torch.sin(torch.arange(800, dtype=torch.float64) * .03) * .1
    initial = original + torch.cos(torch.arange(800, dtype=torch.float64) * .07) * .001
    source = lambda x, text: (x.sin() + .1 * x.square()).sum()
    encoder = ToyEncoder().eval()
    result = WaveformRefiner(source, encoder, condition='weighted_control').run(
        original, 'Reference.', initial_waveform=initial)
    reference = encoder(original).detach()
    expected = initial.clone()
    for _ in range(50):
        g, _ = sequential_component_gradients(expected,
            lambda x: source(x, 'Reference.'),
            lambda x: (x - original).square().mean(),
            lambda x: cr.relative_feature_loss(encoder(x), reference))
        combined = -g['source'] + STFT_WEIGHT * g['stft'] + CONTENT_WEIGHT * g['content']
        expected = original + cr.project_waveform_residual(original,
            expected - original - .00012 * combined.sign(), .003, .003)
    assert torch.equal(result.adversarial, expected)
    assert result.counts['line_search_trials'] == 0
    assert result.counts['source_backward_calls'] == result.counts['stft_backward_calls'] == result.counts['content_backward_calls'] == 50
    assert result.counts['source_forward_calls'] == 52
    assert result.final_metrics['objective_condition'] == 'weighted_control'


def test_partial_mathematical_failure_keeps_last_feasible_audio(monkeypatch):
    cheap_penalties(monkeypatch)
    calls = 0
    def source(x, text):
        nonlocal calls
        calls += 1
        return x.sum() * (float('nan') if calls == 4 else 1.)
    original = torch.zeros(800, dtype=torch.float64)
    with pytest.raises(RefineFailure, match='finite scalar') as caught:
        WaveformRefiner(source, ToyEncoder().eval(), condition='weighted_control').run(
            original, 'Reference.', initial_waveform=original)
    failure = caught.value
    assert failure.receipt['counts']['completed_updates'] == 2
    assert failure.receipt['counts']['source_forward_calls'] == 4
    assert failure.receipt['counts']['source_backward_calls'] == 2
    assert torch.allclose(failure.adversarial, torch.full_like(original, .00024), atol=1e-15)
    assert failure.receipt['stage'] == 'gradient'


def test_infrastructure_and_callback_errors_propagate(monkeypatch):
    cheap_penalties(monkeypatch)
    original = torch.zeros(800, dtype=torch.float64)
    def source(x, text):
        raise RuntimeError('CUDA out of memory simulation')
    with pytest.raises(RuntimeError, match='out of memory') as caught:
        WaveformRefiner(source, ToyEncoder().eval()).run(original, 'Reference.', initial_waveform=original)
    assert not isinstance(caught.value, RefineFailure)
    def callback(row):
        raise cr.InvalidRefinement('status storage failed')
    with pytest.raises(cr.InvalidRefinement, match='storage'):
        WaveformRefiner(lambda x, text: (x * 0).sum(), ToyEncoder().eval()).run(
            original, 'Reference.', initial_waveform=original, on_step=callback)


@pytest.mark.parametrize('override', [{'additional_updates': 49}, {'initial_updates': 0},
    {'step_size': .0001}, {'epsilon': .01}, {'inward_fraction': .02}, {'backtracking_trials': 11},
    {'stft_weight': STFT_WEIGHT + 1}])
def test_no_unregistered_method_hyperparameter_changes(override):
    with pytest.raises(ValueError, match='frozen'):
        RefineConfig(**override)


def test_nll_roundoff_does_not_accumulate_across_fifty_accepted_steps(monkeypatch):
    cheap_penalties(monkeypatch)
    original = torch.zeros(800, dtype=torch.float64)
    initial = torch.full_like(original, .001)
    # Tiny source slope makes each quality-inward step reduce NLL by less than
    # the numerical tolerance. Reusing only the previous value as the floor
    # would let those small reductions accumulate beyond one-time tolerance.
    result = WaveformRefiner(lambda x, text: 1. + 1e-4 * x.mean(), ToyEncoder().eval()).run(
        original, 'Reference.', initial_waveform=initial)
    initial_nll = result.final_metrics['initial_metrics']['source_sum_nll']
    assert result.final_metrics['source_sum_nll'] >= initial_nll - cr._source_tolerance(initial, initial_nll)
    assert result.final_metrics['source_nondecrease_from_initial']
    assert any(row['source_sum_nll'] < initial_nll - 1e-8 for row in result.trace)
    assert all(row['source_eligibility_floor'] >= initial_nll for row in result.trace)


def test_physical_rejection_records_zero_model_forwards(monkeypatch):
    original = vector([0.])
    native_bounds = cr.physical_bounds
    attempts = 0
    def reject_first(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise cr.InvalidRefinement('Simulated failed physical candidate check')
        return native_bounds(*args, **kwargs)
    monkeypatch.setattr(cr, 'physical_bounds', reject_first)
    functions = (lambda x: x.sum(), lambda x: (x * 0).sum(), lambda x: (x * 0).sum())
    counts = {}
    result, metrics, trials = nonlinear_backtracking_step(original, original, vector([1.]), functions,
        caps={'stft_loss': 0., 'content_loss': 0.}, current_source=0., counts=counts)
    assert result.item() == pytest.approx(.00006)
    assert trials[0]['forward_calls'] == {'source': 0, 'stft': 0, 'content': 0}
    assert not trials[0]['accepted'] and 'invalid_reason' in trials[0]
    assert trials[1]['forward_calls'] == {'source': 1, 'stft': 1, 'content': 1}
    assert all(counts[f'{term}_trial_forward_calls'] == 1 for term in cr.TERMS)
