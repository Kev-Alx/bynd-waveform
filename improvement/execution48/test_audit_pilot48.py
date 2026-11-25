"""Check audit rejects missing optimizer calls and infeasible saved waveforms."""
import json

import audit_pilot48 as audit


def receipt(condition='full_r3'):
    rows = []
    for step in range(1, 51):
        rows.append({'step': step, 'loss_evaluation_iterate': step - 1,
            'objective_condition': condition, 'source_sum_nll': 1.0,
            'stft_loss': 0.0, 'content_loss': 0.0, 'full_r3_objective': -1.0,
            'optimized_objective': -1.0, 'source_gradient_norm': 1.0,
            'combined_gradient_norm': 1.0, 'stft_gradient_norm': 0.0 if condition == 'full_r3' else None,
            'content_gradient_norm': 0.0 if condition == 'full_r3' else None,
            'weighted_stft_gradient_norm': 0.0, 'weighted_content_gradient_norm': 0.0,
            'stft_active': condition == 'full_r3', 'content_active': condition == 'full_r3',
            'residual_linf': 0.0, 'residual_rms': 0.0, 'audio_peak': .5})
    counts = {'completed_updates': 50, 'clean_reference_forward_calls': 0}
    for term in ('source', 'stft', 'content'):
        counts[term + '_forward_calls'] = 50
        counts[term + '_backward_calls'] = 50 if term == 'source' or condition == 'full_r3' else 0
        counts['final_' + term + '_forward_calls'] = 1
    return {'status': 'completed', 'condition': condition, 'trace': rows, 'counts': counts,
        'final_metrics': {'selected_iterate': 50, 'objective_condition': condition,
            'source_sum_nll': 1.0, 'stft_loss': 0.0, 'content_loss': 0.0,
            'full_r3_objective': -1.0, 'optimized_objective': -1.0,
            'residual_linf': 0.0, 'residual_rms': 0.0, 'audio_peak': .5}}


def run_receipt(tmp_path, value, variant='pgd_full'):
    audit.ISSUES.clear()
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(value))
    return audit.audit_pgd(path, 'uid', variant, {'linf': 0.0, 'rms': 0.0, 'peak': .5},
                           {'stft': 65.0, 'content': 186.0})


def test_complete_full_and_nll_receipts(tmp_path):
    for condition, variant in (('full_r3', 'pgd_full'), ('nll_only', 'pgd_nll')):
        result = run_receipt(tmp_path, receipt(condition), variant)
        assert result['updates'] == 50 and not audit.ISSUES


def test_missing_backward_rejected(tmp_path):
    value = receipt()
    value['counts']['content_backward_calls'] = 49
    run_receipt(tmp_path, value)
    assert any('call counts' in item for item in audit.ISSUES)


def test_changed_final_iterate_rejected(tmp_path):
    value = receipt()
    value['final_metrics']['selected_iterate'] = 49
    run_receipt(tmp_path, value)
    assert any('selected iterate' in item for item in audit.ISSUES)


def test_nll_penalty_gradient_rejected(tmp_path):
    value = receipt('nll_only')
    value['trace'][3]['content_gradient_norm'] = 1.0
    run_receipt(tmp_path, value, 'pgd_nll')
    assert any('penalty gradient' in item for item in audit.ISSUES)


def test_saved_bounds_rejected():
    import numpy as np
    audit.ISSUES.clear()
    clean = np.zeros(720, dtype=np.float32)
    attack = np.full(720, .004, dtype=np.float32)
    audit.physical(clean, attack, 'uid', 'pgd_full')
    assert any('Linf violation' in item for item in audit.ISSUES)
    assert any('RMS violation' in item for item in audit.ISSUES)


def test_only_mathematical_failures_may_be_retained(tmp_path):
    value = receipt()
    value.update(status='failed', failure_type='mathematical', reason='nonfinite final loss',
                 last_feasible_bounds={'residual_linf': 0.0, 'residual_rms': 0.0, 'audio_peak': .5})
    run_receipt(tmp_path, value)
    assert not audit.ISSUES
    value['failure_type'] = 'contract'
    run_receipt(tmp_path, value)
    assert any('contract failures must stop' in item for item in audit.ISSUES)


def test_independent_text_edit_count():
    assert audit.normalize('THREE ticks—of THE clock!') == 'three ticks of the clock'
    assert audit.word_distance('three ticks of the clock', 'three ticks at the clock') == 1
    assert audit.word_distance('one two three', '') == 3
