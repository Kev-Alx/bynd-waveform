"""Corruption checks for the independent saved-receipt audit."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('saved_refinement_auditor', Path(__file__).with_name('audit_refinement.py'))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def completed_noop_receipt():
    metrics = {'source_sum_nll': 2., 'stft_loss': .1, 'content_loss': .2,
               'full_r3_objective': -2 + AUDIT.STFT_WEIGHT * .1 + AUDIT.CONTENT_WEIGHT * .2}
    bounds = {'residual_linf': .001, 'residual_rms': .0005, 'audio_peak': .5}
    trace = []
    for step in range(1, 51):
        trace.append({**metrics, **bounds, 'step': step, 'total_pgd_step': 50 + step,
                      'loss_evaluation_iterate': 49 + step, 'objective_condition': 'constrained',
                      'source_gradient_norm': 0., 'stft_gradient_norm': 0., 'content_gradient_norm': 0.,
                      'qp': {'feasible': True, 'active_subsets_examined': 1, 'singular_subsets': 0,
                             'active_set': [], 'multipliers': [], 'halfspace_residuals_float64': [],
                             'distance_squared': 0., 'numerical_tolerance': 1e-12},
                      'quality_directions': [], 'inward_halfspace_bounds': [],
                      'zero_margin_fallback': False, 'free_coordinate_count': 1000,
                      'preferred_norm': 0., 'direction_norm': 0., 'source_directional_derivative': 0.,
                      'line_search': [], 'accepted_metrics': None, 'accepted_backtracking_factor': None,
                      'source_eligibility_floor': 2.,
                      'accepted': False, 'changed': False})
    caps = {k: metrics[k] for k in ('stft_loss', 'content_loss')}
    final = {**metrics, **bounds, 'initial_metrics': dict(metrics), 'quality_caps': caps,
             'quality_cap_roundoff_tolerance': {k: AUDIT.cap_tolerance(v) for k, v in caps.items()},
             'selected_iterate': 100, 'initial_updates': 50, 'additional_updates': 50,
             'stft_cap_satisfied': True, 'content_cap_satisfied': True,
             'source_nondecrease_from_initial': True}
    counts = {'clean_reference_forward_calls': 0, 'completed_updates': 50, 'accepted_updates': 0,
              'noop_updates': 50, 'line_search_trials': 0, 'zero_margin_fallbacks': 0}
    for term in ('source', 'stft', 'content'):
        counts.update({term + '_backward_calls': 50, term + '_gradient_forward_calls': 50,
                       term + '_initial_forward_calls': 1, term + '_final_forward_calls': 1,
                       term + '_forward_calls': 52})
    receipt = {'status': 'completed', 'condition': 'constrained', 'trace': trace,
               'counts': counts, 'final_metrics': final}
    return receipt, {'linf': .001, 'rms': .0005, 'peak': .5}


def inspect(receipt, actual):
    AUDIT.ISSUES.clear()
    AUDIT.audit_refinement_receipt(receipt, 'fixture', actual)
    return list(AUDIT.ISSUES)


def test_complete_zero_gradient_noop_is_valid():
    assert inspect(*completed_noop_receipt()) == []


@pytest.mark.parametrize('mutation,phrase', [
    (lambda r: r['counts'].__setitem__('source_forward_calls', 53), 'unaccounted source actual forward'),
    (lambda r: r['final_metrics']['quality_caps'].__setitem__('stft_loss', .2), 'cap differs from fixed initial'),
    (lambda r: r['trace'][4].__setitem__('total_pgd_step', 99), 'total steps not'),
    (lambda r: r['trace'][0].__setitem__('changed', True), 'rejected step moved waveform'),
    (lambda r: r['trace'][0].__setitem__('full_r3_objective', 1.), 'weighted-loss arithmetic'),
    (lambda r: r['final_metrics'].__setitem__('stft_loss', .12), 'exceeds initial cap'),
    (lambda r: r['trace'][0].__setitem__('source_eligibility_floor', 1.9), 'global source eligibility floor'),
])
def test_tampered_receipt_rejected(mutation, phrase):
    receipt, actual = completed_noop_receipt()
    mutation(receipt)
    assert any(phrase in issue for issue in inspect(receipt, actual))


def test_negative_qp_multiplier_is_rejected():
    receipt, actual = completed_noop_receipt()
    row = receipt['trace'][0]
    row.update(quality_directions=['stft'], inward_halfspace_bounds=[0.])
    row['qp'].update(active_subsets_examined=2, active_set=[0], multipliers=[-1.],
                     halfspace_residuals_float64=[0.])
    assert any('dual-sign' in issue for issue in inspect(receipt, actual))


def add_invalid_trial(receipt, calls):
    receipt['trace'][0]['line_search'] = [{'trial': 1, 'factor': 1., 'accepted': False,
        'invalid_reason': 'deliberate STFT scalar failure', 'forward_calls': calls}]
    receipt['counts']['line_search_trials'] = 1
    for term, count in calls.items():
        receipt['counts'][term + '_trial_forward_calls'] = count
        receipt['counts'][term + '_forward_calls'] += count


def test_invalid_trial_prefix_counts_actual_calls_without_inventing_content_forward():
    receipt, actual = completed_noop_receipt()
    add_invalid_trial(receipt, {'source': 1, 'stft': 1, 'content': 0})
    assert inspect(receipt, actual) == []


def test_nonprefix_invalid_trial_calls_rejected():
    receipt, actual = completed_noop_receipt()
    add_invalid_trial(receipt, {'source': 0, 'stft': 1, 'content': 0})
    assert any('sequential 0/1 prefix' in issue for issue in inspect(receipt, actual))


def test_global_initial_floor_rejects_cumulative_local_tolerance_drift():
    receipt, actual = completed_noop_receipt()
    current = 2.
    for index, next_nll in enumerate((2. - 1e-6, 2. - 2.5e-6)):
        row = receipt['trace'][index]
        row['source_sum_nll'] = current
        row['full_r3_objective'] += 2. - current
        trial = {k: row[k] for k in ('source_sum_nll', 'stft_loss', 'content_loss', 'full_r3_objective',
                                     'residual_linf', 'residual_rms', 'audio_peak')}
        trial.update(source_sum_nll=next_nll, full_r3_objective=row['full_r3_objective'] + current - next_nll,
                     trial=1, factor=1., accepted=True, source_nondecrease=True,
                     stft_cap_satisfied=True, content_cap_satisfied=True,
                     forward_calls={'source': 1, 'stft': 1, 'content': 1})
        row.update(line_search=[trial], accepted_metrics={k: trial[k] for k in ('source_sum_nll', 'stft_loss', 'content_loss', 'full_r3_objective')},
                   accepted=True, changed=True, accepted_backtracking_factor=1.)
        assert next_nll >= current - AUDIT.source_tolerance(current)
        current = next_nll
    for row in receipt['trace'][2:]:
        row['source_sum_nll'] = current
        row['full_r3_objective'] += 2. - current
    receipt['final_metrics']['source_sum_nll'] = current
    receipt['final_metrics']['full_r3_objective'] += 2. - current
    receipt['final_metrics']['source_nondecrease_from_initial'] = False
    receipt['counts'].update(line_search_trials=2, accepted_updates=2, noop_updates=48)
    for term in ('source', 'stft', 'content'):
        receipt['counts'][term + '_trial_forward_calls'] = 2
        receipt['counts'][term + '_forward_calls'] = 54
    issues = inspect(receipt, actual)
    assert any('acceptance flag source_nondecrease' in issue for issue in issues)
    assert any('source fell below fixed initial floor' in issue for issue in issues)
