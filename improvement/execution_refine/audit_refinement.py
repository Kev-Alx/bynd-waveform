"""Independent CPU-only audit of saved waveform-refinement experiments.

No neural inference or protected audio reads. Exit0 means artifact consistency,
exit1 means an observed contradiction, exit2 means the full cohort is pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unicodedata

import numpy as np
import soundfile as sf
import torch

ROOT = Path('/workspace/improvement')
CYCLE = ROOT / 'execution_refine'
VARIANTS = ('clean', 'fv50', 'full50', 'weighted100', 'constrained100', 'fv100')
EPS = np.finfo(np.float32).eps
STFT_WEIGHT, CONTENT_WEIGHT = 65.08367870148994, 186.6123439723859
ISSUES, HASHES = [], {}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(condition, message):
    if not condition:
        ISSUES.append(message)
    return bool(condition)


def bind(path, expected):
    path = Path(path)
    if not path.exists():
        return verify(False, f'Missing bound file: {path}')
    measured = HASHES.setdefault(str(path), sha(path))
    return verify(measured == expected, f'Hash mismatch: {path}')


def close(actual, expected, label, *, atol=1e-7, rtol=1e-6):
    return verify(math.isfinite(float(actual)) and math.isfinite(float(expected))
                  and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), label)


def finite(row, fields, label):
    for name in fields:
        verify(isinstance(row[name], (int, float)) and math.isfinite(row[name]), f'{label}: nonfinite {name}')


def objective(row, label):
    finite(row, ('source_sum_nll', 'stft_loss', 'content_loss', 'full_r3_objective'), label)
    expected = -row['source_sum_nll'] + STFT_WEIGHT * row['stft_loss'] + CONTENT_WEIGHT * row['content_loss']
    close(row['full_r3_objective'], expected, f'{label}: incorrect weighted-loss arithmetic', atol=1e-10, rtol=1e-10)


def saved_bounds(row, label):
    finite(row, ('residual_linf', 'residual_rms', 'audio_peak'), label)
    verify(0 <= row['residual_linf'] <= .003 + 4 * EPS, f'{label}: Linf violation')
    verify(0 <= row['residual_rms'] <= .003 + 4 * EPS, f'{label}: RMS violation')
    verify(0 <= row['audio_peak'] <= 1, f'{label}: peak violation')


def cap_tolerance(cap):
    return max(1e-12, 32 * EPS * abs(cap))


def source_tolerance(nll):
    return max(1e-7, 8 * EPS * abs(nll))


def qp_checks(row, label):
    qp = row['qp']
    verify(qp['feasible'] is True, f'{label}: final quality-direction QP infeasible')
    m = len(row['quality_directions'])
    verify(m <= 2 and set(row['quality_directions']) <= {'stft', 'content'}, f'{label}: unknown quality directions')
    verify(qp['active_subsets_examined'] == 2 ** m, f'{label}: QP active subsets not all examined')
    active, multipliers = qp['active_set'], qp['multipliers']
    verify(len(active) == len(set(active)) == len(multipliers) and set(active) <= set(range(m)), f'{label}: invalid active set')
    residuals, tolerance = qp['halfspace_residuals_float64'], qp['numerical_tolerance']
    verify(len(residuals) == m and math.isfinite(tolerance) and tolerance > 0, f'{label}: invalid QP scalar receipt')
    verify(all(math.isfinite(x) and x <= tolerance for x in residuals), f'{label}: QP primal residual violation')
    verify(all(math.isfinite(x) and x >= -tolerance for x in multipliers), f'{label}: QP dual-sign violation')
    for index, multiplier in zip(active, multipliers):
        # The solver's arithmetic has a small accumulation error on long audio;
        # use its registered feasibility tolerance for active-plane residuals.
        verify(abs(residuals[index]) <= tolerance, f'{label}: active-plane complementarity violation')
        verify(abs(multiplier * residuals[index]) <= max(tolerance, tolerance * abs(multiplier)),
               f'{label}: multiplier/residual complementarity violation')
    verify(math.isfinite(qp['distance_squared']) and qp['distance_squared'] >= 0, f'{label}: invalid projection distance')
    verify(len(row['inward_halfspace_bounds']) == m and all(x <= 0 and math.isfinite(x) for x in row['inward_halfspace_bounds']),
           f'{label}: wrong inward quality bounds')
    finite(row, ('preferred_norm', 'direction_norm', 'source_directional_derivative'), label)
    verify(row['preferred_norm'] >= 0 and row['direction_norm'] >= 0 and row['free_coordinate_count'] >= 0,
           f'{label}: invalid direction norm/free-coordinate count')


def audit_refinement_receipt(receipt, label, actual):
    condition = receipt.get('condition', receipt.get('objective_condition'))
    verify(condition in ('constrained', 'weighted_control'), f'{label}: wrong refinement condition')
    trace, counts = receipt['trace'], receipt['counts']
    complete = receipt['status'] in ('complete', 'completed')
    verify(complete, f'{label}: refinement is not completed for all registered updates')
    verify(len(trace) == counts['completed_updates'] == 50, f'{label}: must retain50 added updates')
    verify([r['step'] for r in trace] == list(range(1, 51)), f'{label}: added steps not contiguous')
    verify([r['total_pgd_step'] for r in trace] == list(range(51, 101)), f'{label}: total steps not51..100')
    final = receipt['final_metrics']
    initial, caps = final['initial_metrics'], final['quality_caps']
    objective(initial, label + '/initial')
    for name in ('stft_loss', 'content_loss'):
        close(caps[name], initial[name], f'{label}: cap differs from fixed initial {name}', atol=0, rtol=0)
        close(final['quality_cap_roundoff_tolerance'][name], cap_tolerance(caps[name]), f'{label}: wrong cap tolerance', atol=0, rtol=1e-12)
    current, trials_total, accepted_total, noops, fallbacks = initial, 0, 0, 0, 0
    trial_forward_totals = {term: 0 for term in ('source', 'stft', 'content')}
    for step, row in enumerate(trace, 1):
        tag = f'{label}/step{step}'
        objective(row, tag)
        saved_bounds(row, tag)
        verify(row['objective_condition'] == condition and row['loss_evaluation_iterate'] == 49 + step,
               f'{tag}: wrong objective or loss-evaluation iterate')
        for name in ('source', 'stft', 'content'):
            verify(math.isfinite(row[name + '_gradient_norm']) and row[name + '_gradient_norm'] >= 0,
                   f'{tag}: invalid component gradient norm')
        if condition == 'constrained':
            for key in ('source_sum_nll', 'stft_loss', 'content_loss'):
                close(row[key], current[key], f'{tag}: previous accepted state mismatch', atol=1e-7, rtol=2e-5)
            qp_checks(row, tag)
            floor = max(row['source_sum_nll'], initial['source_sum_nll'])
            close(row['source_eligibility_floor'], floor, f'{tag}: incorrect global source eligibility floor', atol=0, rtol=0)
            fallbacks += int(row['zero_margin_fallback'])
            trials = row['line_search']
            verify(len(trials) <= 10 and [r['trial'] for r in trials] == list(range(1, len(trials) + 1)),
                   f'{tag}: invalid backtracking trial inventory')
            verify(not any(t['accepted'] for t in trials[:-1]), f'{tag}: continued line search after first accepted trial')
            for index, trial in enumerate(trials):
                close(trial['factor'], 2. ** (-index), f'{tag}: wrong frozen halving sequence', atol=0, rtol=0)
                calls = trial['forward_calls']
                verify(set(calls) == set(trial_forward_totals), f'{tag}: missing/extra trial forward terms')
                prefix = [calls[term] for term in ('source', 'stft', 'content')]
                verify(all(type(n) is int and n in (0, 1) for n in prefix)
                       and prefix[0] >= prefix[1] >= prefix[2], f'{tag}: trial forward calls are not a sequential 0/1 prefix')
                for term in trial_forward_totals:
                    trial_forward_totals[term] += calls[term]
                if 'invalid_reason' in trial:
                    verify(not trial['accepted'], f'{tag}: invalid trial accepted')
                    continue
                verify(prefix == [1, 1, 1], f'{tag}: finite complete trial did not call all3 terms')
                objective(trial, tag + '/trial')
                saved_bounds(trial, tag + '/trial')
                conditions = {
                    'source_nondecrease': trial['source_sum_nll'] >= floor - source_tolerance(floor),
                    'stft_cap_satisfied': trial['stft_loss'] <= caps['stft_loss'] + cap_tolerance(caps['stft_loss']),
                    'content_cap_satisfied': trial['content_loss'] <= caps['content_loss'] + cap_tolerance(caps['content_loss']),
                }
                for key, value in conditions.items():
                    verify(trial[key] == value, f'{tag}: incorrect nonlinear acceptance flag {key}')
                verify(trial['accepted'] == all(conditions.values()), f'{tag}: incorrect trial acceptance')
            accepted = bool(trials and trials[-1]['accepted'])
            verify(row['accepted'] == accepted, f'{tag}: accepted flag differs from actual trial')
            verify(row['accepted_backtracking_factor'] == (trials[-1]['factor'] if accepted else None),
                   f'{tag}: wrong selected backtracking factor')
            if accepted:
                verify(row['accepted_metrics'] is not None, f'{tag}: accepted state has no metrics')
                for key in ('source_sum_nll', 'stft_loss', 'content_loss', 'full_r3_objective'):
                    close(row['accepted_metrics'][key], trials[-1][key], f'{tag}: accepted metrics differ', atol=0, rtol=0)
                current = row['accepted_metrics']
            else:
                verify(row['accepted_metrics'] is None and not row['changed'], f'{tag}: rejected step moved waveform')
            trials_total += len(trials)
        else:
            verify(row['line_search'] == [], f'{tag}: weighted control used line search')
            verify(math.isfinite(row['combined_gradient_norm']) and row['combined_gradient_norm'] >= 0,
                   f'{tag}: invalid weighted gradient norm')
        accepted_total += int(row['accepted'] and row['changed'])
        noops += int(not (row['accepted'] and row['changed']))
    objective(final, label + '/final')
    saved_bounds(final, label + '/final')
    verify(final['selected_iterate'] == 100 and final['initial_updates'] == final['additional_updates'] == 50,
           f'{label}: final iterate is not fixed100')
    verify(counts['accepted_updates'] == accepted_total and counts['noop_updates'] == noops,
           f'{label}: acceptance/noop counters differ')
    verify(counts['line_search_trials'] == trials_total and counts['zero_margin_fallbacks'] == fallbacks,
           f'{label}: search/fallback counters differ')
    verify(counts['clean_reference_forward_calls'] == 0, f'{label}: clean reference unexpectedly recomputed')
    for term in ('source', 'stft', 'content'):
        verify(counts[term + '_backward_calls'] == 50 and counts[term + '_gradient_forward_calls'] == 50,
               f'{label}: wrong {term} gradient/backward count')
        verify(counts[term + '_initial_forward_calls'] == counts[term + '_final_forward_calls'] == 1,
               f'{label}: wrong {term} boundary measurement count')
        verify(counts.get(term + '_trial_forward_calls', 0) == trial_forward_totals[term],
               f'{label}: wrong {term} line-search forward count')
        verify(counts[term + '_forward_calls'] == 52 + trial_forward_totals[term],
               f'{label}: unaccounted {term} actual forward calls')
    for name in ('stft_loss', 'content_loss'):
        passed = final[name] <= caps[name] + cap_tolerance(caps[name])
        verify(final[name.replace('_loss', '_cap_satisfied')] == passed, f'{label}: wrong final cap flag')
        if condition == 'constrained':
            verify(passed, f'{label}: constrained final {name} exceeds initial cap')
    if condition == 'constrained':
        source_passed = final['source_sum_nll'] >= initial['source_sum_nll'] - source_tolerance(initial['source_sum_nll'])
        verify(final['source_nondecrease_from_initial'] == source_passed, f'{label}: incorrect final source nondecrease flag')
        verify(source_passed, f'{label}: constrained final source fell below fixed initial floor')
        for key in ('source_sum_nll', 'stft_loss', 'content_loss'):
            close(final[key], current[key], f'{label}: final state differs from last accepted iterate', atol=1e-7, rtol=2e-5)
    for actual_key, saved_key in (('linf', 'residual_linf'), ('rms', 'residual_rms'), ('peak', 'audio_peak')):
        close(actual[actual_key], final[saved_key], f'{label}: saved final waveform {saved_key} mismatch', atol=2e-7)
    return {'condition': condition, 'updates': len(trace), 'counts': counts,
            'fixed_initial_quality_caps': caps, 'final_metrics': final,
            'actual_waveform_bounds': actual,
            'qp_audit_scope': 'Scalar primal feasibility, dual signs and active-plane complementarity; neural gradients, direction vectors and QP stationarity not independently replayed.'}


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in text).split())


def word_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def waveform(asset, label):
    path = Path(asset['path']).resolve()
    verify(path.is_relative_to(ROOT), f'{label}: new waveform outside improvement')
    bind(path, asset['sha256'])
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    info = sf.info(path)
    verify(rate == 16000 and audio.shape[1] == 1 and audio.shape[0] > 0, f'{label}: invalid audio metadata')
    verify(info.format == 'WAV' and info.subtype == 'FLOAT', f'{label}: not float WAV')
    verify(np.isfinite(audio).all(), f'{label}: nonfinite waveform')
    verify(asset['samples'] == len(audio) and asset['sample_rate'] == rate, f'{label}: asset sample metadata differs')
    close(float(np.abs(audio).max()), asset['peak'], f'{label}: asset peak differs', atol=1e-8)
    return audio[:, 0]


def physical(clean, candidate, label):
    verify(clean.shape == candidate.shape, f'{label}: input length changed')
    if clean.shape != candidate.shape:
        raise ValueError(f'{label}: waveform cannot be aligned to original')
    residual = candidate.astype(np.float64) - clean.astype(np.float64)
    actual = {'linf': float(np.abs(residual).max()), 'rms': float(np.sqrt(np.mean(residual ** 2))),
              'peak': float(np.abs(candidate).max())}
    verify(actual['linf'] <= .003 + 4 * EPS and actual['rms'] <= .003 + 4 * EPS
           and actual['peak'] <= 1, f'{label}: saved waveform physical constraint violation')
    return actual


def tensor(asset, label):
    bind(asset['path'], asset['sha256'])
    value = torch.load(asset['path'], map_location='cpu', weights_only=True)
    verify(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
           f'{label}: invalid saved tensor')
    return value


def audit_fv(diag, query, uid, variant, *, old=None):
    steps = {'fv50': 50, 'fv100': 100}[variant]
    delta = tensor(diag['delta'], f'{uid}/{variant}/delta')
    verify(delta.shape == query.shape, f'{uid}/{variant}: feature shape mismatch')
    radius = float(delta.double().norm() / query.double().norm())
    verify(radius <= .1 + 5e-7, f'{uid}/{variant}: relative feature-ball violation')
    close(radius, diag['relative_radius'], f'{uid}/{variant}: recorded radius mismatch', atol=5e-7)
    bind(diag['trace'], diag['trace_sha256'])
    trace = read(diag['trace'])
    verify(len(trace) == steps and [r['step'] for r in trace] == list(range(1, steps + 1)),
           f'{uid}/{variant}: missing FV trajectory updates')
    verify(diag['source_backward_calls'] == steps, f'{uid}/{variant}: wrong source backward budget')
    for row in trace:
        finite(row, ('loss', 'nll', 'tv', 'hf', 'relative_radius'), f'{uid}/{variant}')
        verify(row['relative_radius'] <= .1 + 5e-7, f'{uid}/{variant}: trajectory radius violation')
        close(row['loss'], -row['nll'] + row['tv'] + row['hf'], f'{uid}/{variant}: FV TV/HF arithmetic differs',
              atol=2e-5, rtol=2e-6)
    close(radius, trace[-1]['relative_radius'], f'{uid}/{variant}: final radius differs from trace', atol=5e-7)
    if old is not None:
        prior = tensor(old['diagnostics']['fv']['delta'], f'{uid}/historical-fv50')
        if steps == 50:
            verify(torch.equal(delta, prior), f'{uid}: FV50 is not an exact tensor replay of registered control')
        verify(diag['pilot50_exact_replay'] is True, f'{uid}/{variant}: pilot replay was not asserted')
    else:
        verify(diag['pilot50_exact_replay'] is False, f'{uid}/{variant}: validation wrongly claims historical replay')
    return {'id': uid, 'variant': variant, 'steps': steps, 'relative_radius': radius,
            'source_backward_calls': diag['source_backward_calls']}, trace


def audit_scores(run, cohort, records, generation, plan, previous):
    registration, summary = read(run / 'evaluation-registration.json'), read(run / 'source-summary.json')
    scores = rows(run / 'source-scores.jsonl')
    count, allowed = len(cohort), {r['id']: r for r in cohort}
    assets = {r['id']: r['audio'] for r in records}
    bind(run / 'source-scores.jsonl', summary['scores_sha256'])
    bind(registration['generated_manifest'], registration['generated_manifest_sha256'])
    bind(ROOT / 'source_evaluate_refine.py', registration['evaluator_sha256'])
    verify(registration['generated_manifest_sha256'] == sha(run / 'generated.jsonl'), 'Scored a different generated manifest')
    verify(registration['cycle_plan_sha256'] == sha(CYCLE / 'plan.json'), 'Evaluation uses different experiment plan')
    verify(registration['cohort_count'] == count and registration['cohort'] == run.name,
           'Evaluator cohort count/name differs')
    verify(registration['allowed_development_ids'] == [r['id'] for r in records], 'Evaluator cohort identity/order differs')
    verify(registration['source_only'] is True and registration['target_models_evaluated'] == [], 'Evaluator model scope changed')
    expected_backend = read(registration['historical_backend'])['evaluation_backend']
    verify(registration['backend'] == summary['evaluation_backend'] == expected_backend, 'Native source policy mismatch')
    verify(summary['status'] == 'complete' and summary['source_only'] is True
           and summary['target_models_evaluated'] == [] and summary['baseline_policy_exactly_matched'] is True,
           'Incomplete or changed source evaluation')
    verify(summary['scorer_manifest'] == registration['scorer_manifest'] == previous['quality'], 'Quality scorer specification changed')
    source = read('/workspace/reproduction-runs/paper-quality-stable-execution/pilot/strong-sum-lr-0.1/run.json')['surrogate']
    verify(summary['source_metadata'] == generation['source_metadata'] == source, 'Source checkpoint/decoder specification changed')
    expected_pairs = {(uid, variant) for uid in allowed for variant in VARIANTS}
    verify(len(scores) == count * len(VARIANTS)
           and Counter((r['id'], r['variant']) for r in scores) == Counter(expected_pairs),
           'Scores omit or duplicate cohort/variant pairs')
    for score in scores:
        uid, variant = score['id'], score['variant']
        verify(read(run / 'score-records' / f'{uid}--{variant}.json') == score, f'{uid}/{variant}: per-score receipt differs')
        verify(score['reference'] == allowed[uid]['text'], f'{uid}/{variant}: reference text changed')
        reference, hypothesis = normalize(score['reference']), normalize(score['hypothesis'])
        verify(score['reference_normalized'] == reference and score['hypothesis_normalized'] == hypothesis,
               f'{uid}/{variant}: normalized text differs')
        distance = word_distance(reference.split(), hypothesis.split())
        verify(distance == score['substitutions'] + score['deletions'] + score['insertions'],
               f'{uid}/{variant}: independent word-distance recount differs')
        verify(len(reference.split()) == score['reference_units'], f'{uid}/{variant}: reference denominator differs')
        verify(score['audio_sha256'] == assets[uid][variant]['sha256'], f'{uid}/{variant}: wrong waveform scored')
        finite(score['quality'], ('utmos', 'dnsmos', 'nisqa'), f'{uid}/{variant}/MOS')
    recomputed = {}
    for variant in VARIANTS:
        pool = [r for r in scores if r['variant'] == variant]
        verify(len(pool) == count, f'{variant}: partial cohort denominator')
        totals = {key: sum(r[key] for r in pool) for key in ('substitutions', 'deletions', 'insertions', 'reference_units')}
        totals['error_percent'] = 100 * sum(totals[k] for k in ('substitutions', 'deletions', 'insertions')) / totals['reference_units']
        quality = {key: sum(r['quality'][key] for r in pool) / count for key in ('utmos', 'dnsmos', 'nisqa')}
        saved = summary['summary'][variant]
        verify(saved['count'] == count, f'{variant}: wrong aggregate count')
        for key, value in totals.items():
            close(saved[key], value, f'{variant}: aggregate {key} differs', atol=1e-12, rtol=1e-12)
        for key, value in quality.items():
            close(saved['quality'][key], value, f'{variant}: mean {key} differs', atol=1e-12, rtol=1e-12)
        recomputed[variant] = {'count': count, **totals, 'quality': quality}
    return recomputed


def audit(cohort_name):
    plan, previous = read(CYCLE / 'plan.json'), read(ROOT / 'execution/plan.json')
    verify(plan['source_only'] is True and plan['target_models_evaluated'] == []
           and plan['confirmation_or_final_inference'] is False, 'Registered model scope changed')
    verify(tuple(plan['variants']) == VARIANTS, 'Variant registration changed')
    verify(plan['method']['additional_updates'] == 50 and plan['method']['step_size'] == .00012
           and plan['method']['epsilon'] == plan['method']['rms_limit'] == .003
           and plan['method']['inward_fraction'] == .01 and plan['method']['backtracking_trials'] == 10,
           'Registered refinement algorithm/budget changed')
    verify(plan['fv']['learning_rate'] == .02 and plan['fv']['steps'] == 100 and plan['fv']['rho'] == .1
           and plan['fv']['full_original_tv_hf'] is True and plan['fv']['save_steps'] == [50, 100],
           'Registered FV control algorithm/budget changed')
    for path, expected in plan['bindings'].items():
        bind(path, expected)
    for path, expected in previous['frozen_files'].items():
        bind(path, expected)
    info, run = plan['cohorts'][cohort_name], CYCLE / cohort_name
    count = {'pilot': 48, 'validation': 76}[cohort_name]
    verify(info['count'] == count, 'Registered cohort size changed')
    bind(info['manifest'], info['sha256'])
    cohort = rows(info['manifest'])
    allowed = {r['id']: r for r in cohort}
    verify(len(cohort) == len(allowed) == count, 'Manifest is not the complete unique cohort')
    if cohort_name == 'validation':
        selection = read(CYCLE / 'validation-76.receipt.json')
        verify([r['id'] for r in cohort] == selection['ids_in_order'], 'Validation selection/order changed')
        bind(selection['script'], selection['script_sha256'])
        for item in selection['input_files'].values():
            bind(item['path'], item['sha256'])
        freeze = read(CYCLE / 'validation-freeze.json')
        verify(freeze['status'] == 'validate', 'Validation proceeded without earning its preregistered gate')
        for path, expected in freeze['bindings'].items():
            bind(path, expected)
    generation = read(run / 'generation-result.json')
    verify(generation['status'] == 'completed' and generation['cohort'] == cohort_name
           and generation['actual_count'] == generation['count'] == count, 'Generation is not complete for the entire cohort')
    verify(generation['source_only'] is True and generation['target_models_evaluated'] == [], 'Generation model scope changed')
    historical_generation = read(ROOT / 'execution48/mechanism-pilot48/generation-result.json')
    verify(generation['backend'] == historical_generation['backend'], 'Strict generation execution policy changed')
    verify(generation['encoder_metadata'] == historical_generation['encoder_metadata'], 'Content checkpoint/layer/adapter changed')
    if cohort_name == 'validation':
        verify(freeze['candidate'] in ('constrained100', 'weighted100') and freeze['control'] in ('fv50', 'fv100'),
               'Validation candidate/control was not the frozen registered choice')
        verify(datetime.fromisoformat(freeze['created_at_utc']) < datetime.fromisoformat(generation['started_at_utc']),
               'Candidate freeze did not precede validation generation')
    verify(generation['plan_sha256'] == sha(CYCLE / 'plan.json'), 'Generation experiment binding changed')
    bind(run / 'generated.jsonl', generation['generated_sha256'])
    records = rows(run / 'generated.jsonl')
    verify(len(records) == count and [r['id'] for r in records] == [r['id'] for r in cohort], 'Generated cohort count/order changed')
    historical = {r['id']: r for r in rows(ROOT / 'execution48/mechanism-pilot48/generated.jsonl')}
    # Reuse the earlier independently written CPU PGD-receipt auditor. It has no
    # imports from the optimizer implementation and performs no neural work.
    helper_path = ROOT / 'execution48/audit_pilot48.py'
    spec = importlib.util.spec_from_file_location('independent_pgd_receipt_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ISSUES, helper.CHECKED_FILES = ISSUES, HASHES
    HASHES[str(helper_path)] = sha(helper_path)
    refinement_audits, fv_audits, physical_audits = [], [], []
    for record in records:
        uid = record['id']; row = allowed[uid]
        verify(read(run / 'records' / f'{uid}.json') == record, f'{uid}: individual record differs')
        verify(record['text'] == row['text'] and record['speaker'] == row['speaker']
               and record['input_audio_sha256'] == row['audio_sha256'], f'{uid}: input/reference/speaker changed')
        verify(record['source_only'] is True and record['role'] == cohort_name
               and record['count'] == count and record['plan_sha256'] == sha(CYCLE / 'plan.json'), f'{uid}: record scope changed')
        verify(set(record['audio']) == set(VARIANTS), f'{uid}: missing or extra variants')
        bind(row['audio'], row['audio_sha256'])
        original, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
        verify(rate == 16000 and original.shape == (row['samples'], 1) and .045 <= len(original) / rate <= 30,
               f'{uid}: source metadata/duration changed')
        audio = {variant: waveform(record['audio'][variant], f'{uid}/{variant}') for variant in VARIANTS}
        verify(np.array_equal(original[:, 0], audio['clean']), f'{uid}: clean waveform was transformed')
        q = tensor(record['diagnostics']['q'], f'{uid}/clean-q')
        verify(q.ndim == 2 and q.shape[1] == 1024 and float(q.double().norm()) > 0, f'{uid}: invalid clean feature reference')
        prior = historical.get(uid) if cohort_name == 'pilot' else None
        if prior:
            prior_q = tensor(prior['clean_features'], f'{uid}/prior-q')
            verify(torch.equal(q, prior_q), f'{uid}: pilot clean features differ from frozen reference')
            verify(record['audio']['full50'] == prior['audio']['pgd_full'], f'{uid}: full50 initialization asset changed')
            verify(record['diagnostics']['full50'] == prior['diagnostics']['pgd']['pgd_full'], f'{uid}: full50 prior receipt changed')
        fv_traces = {}
        for variant in ('fv50', 'fv100'):
            result, fv_traces[variant] = audit_fv(record['diagnostics'][variant], q, uid, variant, old=prior)
            fv_audits.append(result)
        verify(fv_traces['fv100'][:50] == fv_traces['fv50'], f'{uid}: FV50 and FV100 were not the same saved trajectory')
        initial_diag = record['diagnostics']['full50']
        bind(initial_diag['receipt'], initial_diag['sha256'])
        initial_receipt = read(initial_diag['receipt'])
        initial_bounds = physical(audio['clean'], audio['full50'], f'{uid}/full50')
        helper.audit_pgd(initial_diag['receipt'], uid, 'pgd_full', initial_bounds,
                         {'stft': STFT_WEIGHT, 'content': CONTENT_WEIGHT})
        initial_metrics = initial_receipt['final_metrics']
        for variant in ('weighted100', 'constrained100'):
            diag = record['diagnostics'][variant]
            bind(diag['receipt'], diag['sha256'])
            receipt = read(diag['receipt'])
            verify(diag['status'] == receipt['status'], f'{uid}/{variant}: refinement status differs')
            verify(receipt['condition'] == {'weighted100': 'weighted_control', 'constrained100': 'constrained'}[variant],
                   f'{uid}/{variant}: wrong continuation objective')
            actual = physical(audio['clean'], audio[variant], f'{uid}/{variant}')
            physical_audits.append({'id': uid, 'variant': variant, **actual})
            if receipt['status'] == 'completed':
                result = audit_refinement_receipt(receipt, f'{uid}/{variant}', actual)
                for key in ('source_sum_nll', 'stft_loss', 'content_loss'):
                    close(receipt['final_metrics']['initial_metrics'][key], initial_metrics[key],
                          f'{uid}/{variant}: optimizer did not begin from saved full50 metrics', atol=1e-7, rtol=2e-5)
                refinement_audits.append({'id': uid, 'variant': variant, 'status': 'completed', **result})
            else:
                verify(receipt['status'] == 'failed' and bool(receipt['reason']), f'{uid}/{variant}: unknown failure status')
                verify(0 <= len(receipt['trace']) == receipt['counts']['completed_updates'] <= 50,
                       f'{uid}/{variant}: invalid partial refinement inventory')
                refinement_audits.append({'id': uid, 'variant': variant, 'status': 'failed',
                                          'reason': receipt['reason'], 'counts': receipt['counts']})
    declared = {(r['id'], r['variant']) for r in generation['failures']}
    failures = [r for r in refinement_audits if r['status'] != 'completed']
    verify(declared == {(r['id'], r['variant']) for r in failures}, 'Declared refinement failures differ from saved receipts')
    recomputed = audit_scores(run, cohort, records, generation, plan, previous)
    verify(not torch.cuda.is_initialized(), 'CPU saved audit initialized CUDA')
    return {'cohort_count': count, 'variants': list(VARIANTS), 'recomputed_summary': recomputed,
            'refinement_receipts': refinement_audits, 'refinement_failures': failures,
            'fv_geometry': fv_audits, 'waveform_bounds': physical_audits,
            'checked_file_hashes': HASHES,
            'scope_limits': 'CPU saved-tensor geometry, waveform/hash/trace/loss/scalar-QP consistency and independent word-distance/MOS aggregation. No neural inference, gradient-vector replay, direction-vector reconstruction, full QP stationarity proof, human intelligibility or transfer evaluation. Validation uses historically exposed dev-clean utterances with shared speakers.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=('pilot', 'validation'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    run = CYCLE / args.cohort
    required = [CYCLE / 'plan.json', run / 'generated.jsonl', run / 'generation-result.json',
                run / 'evaluation-registration.json', run / 'source-summary.json', run / 'source-scores.jsonl']
    missing = [str(p) for p in required if not p.exists()]
    if args.cohort == 'validation' and not (CYCLE / 'validation-freeze.json').exists():
        missing.append(str(CYCLE / 'validation-freeze.json'))
    generation = read(run / 'generation-result.json') if (run / 'generation-result.json').exists() else {}
    failure_path = run / 'source-evaluation-failure.json'
    fatal = generation.get('status') in ('failed_infrastructure', 'negative_initialization') or failure_path.exists()
    if fatal:
        ISSUES.append('Generation/evaluation did not complete the registered cohort: '
                      + generation.get('error', read(failure_path).get('error', '') if failure_path.exists() else ''))
        value, exit_code = {}, 1
    elif missing or generation.get('status') != 'completed':
        value, exit_code = {'generation_status': generation.get('status', 'not_started')}, 2
    else:
        try:
            value = audit(args.cohort)
        except Exception as error:
            ISSUES.append(type(error).__name__ + ': ' + str(error))
            value = {'checked_file_hashes': HASHES}
        exit_code = 1 if ISSUES else 0
    value.update(status='pending' if exit_code == 2 else ('failed' if ISSUES else 'pass'),
                 cohort=args.cohort, issues=ISSUES, missing=missing,
                 cpu_only=True, model_inference=False, cuda_initialized=torch.cuda.is_initialized(),
                 scientific_attack_success_established_by_audit=False,
                 audit_script_sha256=sha(__file__), created_at_utc=datetime.now(timezone.utc).isoformat())
    run.mkdir(parents=True, exist_ok=True)
    (run / 'audit.json').write_text(json.dumps(value, indent=2) + '\n')
    lines = [f'# Saved-artifact audit: {args.cohort}', '', f'Status: **{value["status"]}**.', '',
             'This CPU artifact-consistency audit runs no neural inference and does not establish attack improvement.']
    if missing:
        lines += ['', 'Pending artifacts:'] + ['- `' + name + '`' for name in missing]
    if ISSUES:
        lines += ['', 'Observed issues:'] + ['- ' + issue for issue in ISSUES]
    if 'recomputed_summary' in value:
        lines += ['', '| Condition | n | Source WER % | UTMOS | DNSMOS | NISQA |',
                  '|---|---:|---:|---:|---:|---:|']
        for variant in VARIANTS:
            entry = value['recomputed_summary'][variant]
            quality = entry['quality']
            lines.append(f'| {variant} | {entry["count"]} | {entry["error_percent"]:.6f} | {quality["utmos"]:.6f} | {quality["dnsmos"]:.6f} | {quality["nisqa"]:.6f} |')
        lines += ['', f'Explicit retained refinement failures: {len(value["refinement_failures"])}.', '', value['scope_limits']]
    (run / 'audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'cohort', 'issues', 'missing')}, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
