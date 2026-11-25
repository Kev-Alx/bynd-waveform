"""Independent CPU-only audit of the registered 48-point saved-artifact cycle.

No arguments, model imports, inference, GPU work, or held-out audio reads.
Exit 0: internally consistent completed artifacts; 1: audit failure; 2: pending.
Mathematical attack failures remain explicit and never become completed attacks.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata

import numpy as np
import soundfile as sf
import torch

ROOT = Path('/workspace/improvement')
CYCLE = ROOT / 'execution48'
RUN = CYCLE / 'mechanism-pilot48'
VARIANTS = ('clean', 'reconstruction', 'fv_quality', 'r3_initial', 'r3_trained', 'pgd_full', 'pgd_nll')
BOUND_TOLERANCE = 4 * np.finfo(np.float32).eps
ISSUES = []
CHECKED_FILES = {}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
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
        ISSUES.append(f'Missing bound file: {path}')
        return False
    measured = CHECKED_FILES.setdefault(str(path), sha(path))
    return verify(measured == expected, f'Hash mismatch: {path}')


def close(actual, expected, message, *, atol=1e-7, rtol=1e-6):
    return verify(math.isfinite(float(actual)) and math.isfinite(float(expected))
                  and math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol), message)


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in text).split())


def word_distance(left, right):
    left, right = left.split(), right.split()
    previous = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def waveform(asset, uid, variant):
    path = Path(asset['path']).resolve()
    verify(path.is_relative_to(ROOT), f'{uid}/{variant}: output outside improvement')
    bind(path, asset['sha256'])
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    info = sf.info(path)
    verify(rate == 16000 and audio.shape[1] == 1, f'{uid}/{variant}: wrong audio rate/channels')
    verify(info.format == 'WAV' and info.subtype == 'FLOAT', f'{uid}/{variant}: wrong saved audio format')
    verify(np.isfinite(audio).all() and len(audio) > 0, f'{uid}/{variant}: invalid saved audio')
    verify(asset['samples'] == len(audio) and asset['sample_rate'] == rate, f'{uid}/{variant}: wrong asset sample metadata')
    close(float(np.max(np.abs(audio))), asset['peak'], f'{uid}/{variant}: saved peak differs', atol=1e-8)
    return audio[:, 0]


def physical(clean, attack, uid, variant):
    verify(attack.shape == clean.shape, f'{uid}/{variant}: waveform length changed')
    if attack.shape != clean.shape:
        raise ValueError(f'{uid}/{variant}: cannot compare misaligned waveform')
    residual = attack.astype(np.float64) - clean.astype(np.float64)
    stats = {'linf': float(np.max(np.abs(residual))),
             'rms': float(np.sqrt(np.mean(residual ** 2))),
             'peak': float(np.max(np.abs(attack))), 'samples': int(len(attack))}
    verify(stats['linf'] <= .003 + BOUND_TOLERANCE, f'{uid}/{variant}: Linf violation')
    verify(stats['rms'] <= .003 + BOUND_TOLERANCE, f'{uid}/{variant}: RMS violation')
    verify(stats['peak'] <= 1, f'{uid}/{variant}: full-scale peak violation')
    return stats


def finite_scalars(row, keys, name):
    for key in keys:
        verify(isinstance(row[key], (int, float)) and math.isfinite(row[key]), f'{name}: invalid {key}')


def audit_pgd(path, uid, variant, actual, weights):
    receipt = read(path)
    condition = {'pgd_full': 'full_r3', 'pgd_nll': 'nll_only'}[variant]
    verify(receipt['condition'] == condition, f'{uid}/{variant}: wrong objective condition')
    trace, counts = receipt['trace'], receipt['counts']
    complete = receipt['status'] == 'completed'
    verify(receipt['status'] in ('completed', 'failed'), f'{uid}/{variant}: unclassified status')
    updates = counts['completed_updates']
    verify(0 <= updates <= 50 and len(trace) == updates, f'{uid}/{variant}: missing or extra PGD trace')
    verify([r['step'] for r in trace] == list(range(1, updates + 1)), f'{uid}/{variant}: noncontiguous PGD trace')
    for row in trace:
        label = f'{uid}/{variant}/step{row["step"]}'
        finite_scalars(row, ('source_sum_nll', 'stft_loss', 'content_loss',
            'full_r3_objective', 'optimized_objective', 'source_gradient_norm',
            'combined_gradient_norm', 'residual_linf', 'residual_rms', 'audio_peak'), label)
        verify(row['loss_evaluation_iterate'] == row['step'] - 1, label + ': wrong loss iterate')
        verify(row['objective_condition'] == condition, label + ': wrong objective')
        verify(row['source_gradient_norm'] > 0 and row['combined_gradient_norm'] > 0, label + ': zero required gradient')
        verify(row['stft_active'] == (condition == 'full_r3') and row['content_active'] == (condition == 'full_r3'), label + ': wrong active penalties')
        expected = -row['source_sum_nll'] + weights['stft'] * row['stft_loss'] + weights['content'] * row['content_loss']
        close(row['full_r3_objective'], expected, label + ': wrong full objective', atol=1e-10, rtol=1e-10)
        close(row['optimized_objective'], expected if condition == 'full_r3' else -row['source_sum_nll'], label + ': wrong optimized objective', atol=1e-10, rtol=1e-10)
        verify(row['residual_linf'] <= .003 + BOUND_TOLERANCE and row['residual_rms'] <= .003 + BOUND_TOLERANCE and row['audio_peak'] <= 1, label + ': infeasible recorded update')
        if condition == 'nll_only':
            verify(row['stft_gradient_norm'] is None and row['content_gradient_norm'] is None,
                   label + ': NLL-only condition used penalty gradient')
            verify(row['weighted_stft_gradient_norm'] == 0 and row['weighted_content_gradient_norm'] == 0,
                   label + ': NLL-only penalty weight active')
        else:
            finite_scalars(row, ('stft_gradient_norm', 'content_gradient_norm'), label)
            close(row['weighted_stft_gradient_norm'], weights['stft'] * row['stft_gradient_norm'], label + ': wrong weighted STFT gradient')
            close(row['weighted_content_gradient_norm'], weights['content'] * row['content_gradient_norm'], label + ': wrong weighted content gradient')
    verify(counts['clean_reference_forward_calls'] == 0, f'{uid}/{variant}: unexpected clean-reference recomputation')
    if complete:
        verify(updates == 50, f'{uid}/{variant}: completed condition lacks 50 updates')
        expected_counts = {'completed_updates': 50, 'clean_reference_forward_calls': 0}
        for term in ('source', 'stft', 'content'):
            expected_counts[term + '_forward_calls'] = 50
            expected_counts[term + '_backward_calls'] = 50 if term == 'source' or condition == 'full_r3' else 0
            expected_counts['final_' + term + '_forward_calls'] = 1
        verify(counts == expected_counts, f'{uid}/{variant}: completed forward/backward call counts differ')
        final = receipt['final_metrics']
        verify(final['selected_iterate'] == 50 and final['objective_condition'] == condition,
               f'{uid}/{variant}: wrong selected iterate/condition')
        finite_scalars(final, ('source_sum_nll', 'stft_loss', 'content_loss', 'full_r3_objective', 'optimized_objective'), f'{uid}/{variant}/final')
        objective = -final['source_sum_nll'] + weights['stft'] * final['stft_loss'] + weights['content'] * final['content_loss']
        close(final['full_r3_objective'], objective, f'{uid}/{variant}: wrong final objective', atol=1e-10, rtol=1e-10)
        close(final['optimized_objective'], objective if condition == 'full_r3' else -final['source_sum_nll'], f'{uid}/{variant}: wrong final optimized objective', atol=1e-10, rtol=1e-10)
    else:
        verify(receipt['failure_type'] == 'mathematical' and bool(receipt['reason']),
               f'{uid}/{variant}: retained failure must be mathematical; contract failures must stop generation')
        for term in ('source', 'stft', 'content'):
            forwards, backwards = counts[term + '_forward_calls'], counts[term + '_backward_calls']
            verify(updates <= forwards <= min(50, updates + 1), f'{uid}/{variant}: invalid partial forward count')
            verify(0 <= backwards <= forwards, f'{uid}/{variant}: invalid partial backward count')
            if term != 'source' and condition == 'nll_only':
                verify(backwards == 0, f'{uid}/{variant}: failed NLL-only penalty backward')
        final = receipt['last_feasible_bounds']
    if final is not None:
        for actual_key, saved_key in (('linf', 'residual_linf'), ('rms', 'residual_rms'), ('peak', 'audio_peak')):
            close(actual[actual_key], final[saved_key], f'{uid}/{variant}: saved waveform differs from final {saved_key}', atol=2e-7)
    return {'id': uid, 'variant': variant, 'status': receipt['status'], 'updates': updates,
            'counts': counts, 'bounds': actual, **({} if complete else {'reason': receipt['reason'], 'failure_type': receipt['failure_type']})}


def audit():
    torch.set_num_threads(4)
    verify(not torch.cuda.is_initialized(), 'CUDA initialized before CPU audit')
    plan = read(CYCLE / 'plan.json')
    verify(plan['pilot_count'] == 48 and plan['target_models_evaluated'] == [] and plan['confirmation_or_final_inference'] is False,
           'Registered cohort/source-only scope changed')
    verify(tuple(plan['variants']) == VARIANTS, 'Registered variant set changed')
    for path, expected in plan['bindings'].items():
        bind(path, expected)
    previous = read(plan['prior_plan'])
    for path, expected in previous['frozen_files'].items():
        bind(path, expected)
    verify(previous['base_config']['surrogate']['checkpoint'] == 'artifacts/whisper/small.pt', 'Unregistered source recognizer')
    verify(previous['base_config']['surrogate']['nll_reduction'] == 'sum', 'Changed NLL reduction')
    cohort = rows(plan['development_manifest'])
    allowed = {r['id']: r for r in cohort}
    verify(len(cohort) == len(allowed) == 48, 'Cohort is not 48 unique IDs')
    bind(plan['development_manifest'], plan['manifest_sha256'])
    selection = read(CYCLE / 'development-48.receipt.json')
    bind(selection['script'], selection['script_sha256'])
    for item in selection['input_files'].values():
        bind(item['path'], item['sha256'])
    verify([r['id'] for r in cohort] == selection['old_ids_in_order'] + selection['new_ids_in_order'], 'Selected cohort order changed')
    bind(plan['generator_checkpoint'], plan['generator_sha256'])
    training = read(ROOT / 'execution/r3-waveform-autograd-repair/training.json')
    verify(plan['generator_checkpoint'] == training['selected_checkpoint']
           and Path(plan['generator_checkpoint']).name == 'update-01500.pt', 'Frozen selected checkpoint changed')
    verify(plan['fv_quality'] == {'steps': 50, 'optimizer': 'Adam', 'learning_rate': .02,
        'rho': .1, 'lambda_perc': 1, 'alpha_hf': 1, 'initialization': 'zero',
        'objective': 'original uppercase lexicalSUM NLL with full TV/HF'}, 'FV control contract changed')
    verify(plan['pgd']['steps'] == 50 and plan['pgd']['initialization'] == 'zero'
           and plan['pgd']['step_size'] == 2 * .003 / 50
           and plan['pgd']['linf'] == plan['pgd']['rms_limit'] == .003, 'PGD contract changed')
    weights = plan['pgd']['weights']
    verify(weights == training['weights'], 'PGD weights differ from frozen R3 loss')
    generation = read(RUN / 'generation-result.json')
    verify(generation['status'] == 'completed' and generation['actual_count'] == generation['pilot_count'] == 48,
           'Generation is not completed for all48')
    verify(generation['source_only'] is True and generation['target_models_evaluated'] == [], 'Generation model scope changed')
    bind(RUN / 'generated.jsonl', generation['generated_sha256'])
    verify(generation['plan_sha256'] == sha(CYCLE / 'plan.json'), 'Generation plan binding changed')
    generated = rows(RUN / 'generated.jsonl')
    verify(len(generated) == 48 and [r['id'] for r in generated] == list(allowed), 'Generated cohort count/order changed')
    geometry, pgd, waveform_bounds = [], [], []
    assets = {}
    for record in generated:
        uid = record['id']
        row = allowed[uid]
        verify(read(RUN / 'records' / f'{uid}.json') == record, f'{uid}: per-record manifest differs')
        verify(record['text'] == row['text'] and record['input_audio_sha256'] == row['audio_sha256'], f'{uid}: reference/input changed')
        verify(record['pilot_count'] == 48 and record['source_only'] is True and record['role'] == 'pilot', f'{uid}: cohort/scope metadata changed')
        verify(record['plan_sha256'] == sha(CYCLE / 'plan.json'), f'{uid}: wrong plan binding')
        verify(set(record['audio']) == set(VARIANTS), f'{uid}: missing or extra condition')
        bind(row['audio'], row['audio_sha256'])
        original, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
        verify(rate == 16000 and original.shape == (row['samples'], 1), f'{uid}: source metadata changed')
        verify(.045 <= len(original) / rate <= 30, f'{uid}: inadmissible source duration')
        audio = {variant: waveform(record['audio'][variant], uid, variant) for variant in VARIANTS}
        verify(np.array_equal(audio['clean'], original[:, 0]), f'{uid}: clean waveform transformed')
        verify(audio['fv_quality'].shape == audio['reconstruction'].shape, f'{uid}: FV and reconstruction output lengths differ')
        assets[uid] = record['audio']
        query_asset = record['clean_features']
        bind(query_asset['path'], query_asset['sha256'])
        query = torch.load(query_asset['path'], map_location='cpu', weights_only=True)
        delta_asset = record['diagnostics']['fv']['delta']
        bind(delta_asset['path'], delta_asset['sha256'])
        delta = torch.load(delta_asset['path'], map_location='cpu', weights_only=True)
        verify(query.ndim == 2 and query.shape[1] == 1024 and delta.shape == query.shape,
               f'{uid}: invalid q/delta feature shape')
        verify(query.dtype == delta.dtype == torch.float32 and bool(torch.isfinite(query).all())
               and bool(torch.isfinite(delta).all()), f'{uid}: invalid q/delta dtype or value')
        qnorm, dnorm = float(query.double().norm()), float(delta.double().norm())
        verify(qnorm > 0, f'{uid}: zero clean-feature norm')
        radius = dnorm / qnorm
        verify(radius <= .1 + 5e-7, f'{uid}: feature relative-ball violation')
        close(radius, record['diagnostics']['fv']['relative_radius'], f'{uid}: wrong recorded FV radius', atol=5e-7)
        bind(record['diagnostics']['fv']['trace'], record['diagnostics']['fv']['trace_sha256'])
        fv_trace = read(record['diagnostics']['fv']['trace'])
        verify(len(fv_trace) == 50 and [r['update'] for r in fv_trace] == list(range(1, 51)), f'{uid}: missing FV updates')
        verify(record['diagnostics']['fv']['source_backward_calls'] == 50, f'{uid}: wrong FV backward count')
        for entry in fv_trace:
            finite_scalars(entry, ('loss', 'nll', 'tv', 'hf', 'gradient_norm', 'relative_radius'), f'{uid}/FV')
            verify(entry['relative_radius'] <= .1 + 5e-7, f'{uid}/FV: trace radius violation')
            close(entry['loss'], -entry['nll'] + entry['tv'] + entry['hf'], f'{uid}/FV: wrong full TV/HF loss', atol=2e-5, rtol=2e-6)
        close(radius, fv_trace[-1]['relative_radius'], f'{uid}: final FV radius/trace mismatch', atol=5e-7)
        geometry.append({'id': uid, 'query_shape': list(query.shape), 'query_norm': qnorm, 'delta_norm': dnorm,
                         'relative_radius': radius, 'updates': len(fv_trace)})
        for variant in ('r3_initial', 'r3_trained', 'pgd_full', 'pgd_nll'):
            bounds = physical(audio['clean'], audio[variant], uid, variant)
            waveform_bounds.append({'id': uid, 'variant': variant, **bounds})
            if variant.startswith('r3_'):
                diag = record['diagnostics']['generator'][variant]
                verify(diag['samples'] == bounds['samples'], f'{uid}/{variant}: wrong generator length')
                close(bounds['linf'], diag['linf'], f'{uid}/{variant}: generator Linf mismatch', atol=2e-7)
                close(bounds['rms'], diag['rms'], f'{uid}/{variant}: generator RMS mismatch', atol=2e-7)
                finite_scalars(diag, ('sum_nll', 'stft_loss', 'content_loss'), f'{uid}/{variant}')
            else:
                diag = record['diagnostics']['pgd'][variant]
                bind(diag['receipt'], diag['sha256'])
                item = audit_pgd(diag['receipt'], uid, variant, bounds, weights)
                verify(item['status'] == diag['status'], f'{uid}/{variant}: status differs from record')
                pgd.append(item)
    failures = [item for item in pgd if item['status'] != 'completed']
    declared = {(r['id'], r['variant']) for r in generation['pgd_mathematical_failures']}
    verify(declared == {(r['id'], r['variant']) for r in failures}, 'Mathematical failure inventory differs')
    registration = read(RUN / 'evaluation-registration.json')
    summary = read(RUN / 'source-summary.json')
    scores = rows(RUN / 'source-scores.jsonl')
    bind(RUN / 'source-scores.jsonl', summary['scores_sha256'])
    bind(registration['generated_manifest'], registration['generated_manifest_sha256'])
    bind(ROOT / 'source_evaluate48.py', registration['evaluator_sha256'])
    verify(registration['cycle_plan_sha256'] == sha(CYCLE / 'plan.json'), 'Evaluation plan binding changed')
    verify(registration['pilot_count'] == 48 and registration['source_only'] is True
           and registration['target_models_evaluated'] == [], 'Evaluator cohort/model scope changed')
    expected_backend = read(registration['historical_backend'])['evaluation_backend']
    verify(registration['backend'] == summary['evaluation_backend'] == expected_backend,
           'Native source evaluation backend differs from historical policy')
    verify(summary['status'] == 'complete' and summary['source_only'] is True
           and summary['target_models_evaluated'] == [] and summary['baseline_policy_exactly_matched'] is True,
           'Source evaluation incomplete or wrong model scope')
    verify(summary['scorer_manifest'] == registration['scorer_manifest'] == previous['quality'], 'Quality scorer manifests differ')
    original_source = read(Path('/workspace/reproduction-runs/paper-quality-stable-execution/pilot/strong-sum-lr-0.1/run.json'))['surrogate']
    verify(summary['source_metadata'] == generation['source_metadata'] == original_source, 'Source checkpoint/decoding metadata changed')
    expected_pairs = {(uid, variant) for uid in allowed for variant in VARIANTS}
    verify(len(scores) == 336 and Counter((r['id'], r['variant']) for r in scores) == Counter(expected_pairs),
           'Source scores omit or duplicate cohort/condition pairs')
    recomputed = {}
    for score in scores:
        uid, variant = score['id'], score['variant']
        verify(read(RUN / 'score-records' / f'{uid}--{variant}.json') == score, f'{uid}/{variant}: individual score differs')
        verify(score['reference'] == allowed[uid]['text'], f'{uid}/{variant}: scorer changed reference')
        reference, hypothesis = normalize(score['reference']), normalize(score['hypothesis'])
        verify(score['reference_normalized'] == reference and score['hypothesis_normalized'] == hypothesis,
               f'{uid}/{variant}: normalized scoring text differs')
        errors = word_distance(reference, hypothesis)
        verify(errors == score['substitutions'] + score['deletions'] + score['insertions'], f'{uid}/{variant}: edit-distance recount differs')
        verify(len(reference.split()) == score['reference_units'], f'{uid}/{variant}: reference word count differs')
        verify(score['audio_sha256'] == assets[uid][variant]['sha256'], f'{uid}/{variant}: scorer used wrong waveform')
        finite_scalars(score['quality'], ('utmos', 'dnsmos', 'nisqa'), f'{uid}/{variant}/quality')
    for variant in VARIANTS:
        pool = [r for r in scores if r['variant'] == variant]
        verify(len(pool) == 48, f'{variant}: scorer denominator is not48')
        totals = {key: sum(r[key] for r in pool) for key in ('substitutions', 'deletions', 'insertions', 'reference_units')}
        totals['error_percent'] = 100 * (totals['substitutions'] + totals['deletions'] + totals['insertions']) / totals['reference_units']
        quality = {k: sum(r['quality'][k] for r in pool) / 48 for k in ('utmos', 'dnsmos', 'nisqa')}
        saved = summary['summary'][variant]
        verify(saved['count'] == 48, f'{variant}: summary count is not48')
        for key, value in totals.items():
            close(value, saved[key], f'{variant}: summary {key} differs', atol=1e-12, rtol=1e-12)
        for key, value in quality.items():
            close(value, saved['quality'][key], f'{variant}: mean {key} differs', atol=1e-12, rtol=1e-12)
        recomputed[variant] = {'count': 48, **totals, 'quality': quality}
    verify(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return {'pilot_count': 48, 'variants': list(VARIANTS), 'saved_audio_count': 336,
        'source_score_count': len(scores), 'recomputed_summary': recomputed,
        'feature_geometry': geometry, 'waveform_bounds': waveform_bounds,
        'pgd_receipts': pgd, 'pgd_failures': failures,
        'native_backend_exactly_matched': summary['evaluation_backend'] == expected_backend,
        'checked_file_hashes': CHECKED_FILES,
        'scope_limits': 'Saved CPU geometry/waveform/hash/trace/edit-distance/MOS aggregation only; no neural gradient or model-output replay. Fresh uniform strict E6 is not compared with old cached queries. Human intelligibility/held-out transfer remain unestablished.'}


def write_result(value):
    value.update(created_at_utc=datetime.now(timezone.utc).isoformat(), cpu_only=True,
                 cuda_initialized=torch.cuda.is_initialized(), model_inference=False,
                 audit_script_sha256=sha(__file__))
    CYCLE.mkdir(parents=True, exist_ok=True)
    (CYCLE / 'audit-pilot48.json').write_text(json.dumps(value, indent=2) + '\n')
    lines = ['# Saved-artifact audit: development48', '', f'Status: **{value["status"]}**.', '',
             'This is an artifact-consistency audit, not a finding that the attacks succeeded. It runs on CPU and loads no models.']
    if 'missing' in value:
        lines += ['', 'Awaiting:'] + ['- `' + name + '`' for name in value['missing']]
    if value.get('issues'):
        lines += ['', 'Issues:'] + ['- ' + issue for issue in value['issues']]
    if 'recomputed_summary' in value:
        lines += ['', '| Condition | n | Source WER % | UTMOS | DNSMOS | NISQA |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for variant in VARIANTS:
            row = value['recomputed_summary'][variant]
            q = row['quality']
            lines.append(f'| {variant} | 48 | {row["error_percent"]:.6f} | {q["utmos"]:.6f} | {q["dnsmos"]:.6f} | {q["nisqa"]:.6f} |')
        lines += ['', f'Recorded PGD failures: {len(value["pgd_failures"])}. Failed attempts remain included among all48 and are not completed attacks.', '', value['scope_limits']]
    (CYCLE / 'audit-pilot48.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: value[k] for k in ('status', 'issues', 'missing') if k in value}, indent=2))


def main():
    if len(sys.argv) != 1:
        raise SystemExit('audit_pilot48.py accepts no arguments')
    required = [CYCLE / 'plan.json', RUN / 'generation-result.json', RUN / 'generated.jsonl',
                RUN / 'evaluation-registration.json', RUN / 'source-summary.json', RUN / 'source-scores.jsonl']
    missing = [str(p) for p in required if not p.exists()]
    generation = read(RUN / 'generation-result.json') if (RUN / 'generation-result.json').exists() else {}
    if generation.get('status') == 'failed_infrastructure':
        write_result({'status': 'failed', 'issues': ['Generation failed: ' + generation.get('error', 'unspecified')], 'missing': missing})
        return 1
    evaluation_failure = RUN / 'source-evaluation-failure.json'
    if evaluation_failure.exists():
        failure = read(evaluation_failure)
        write_result({'status': 'failed', 'issues': ['Source evaluation failed: ' + failure.get('error', 'unspecified')], 'missing': missing})
        return 1
    if missing or generation.get('status') != 'completed':
        write_result({'status': 'pending', 'issues': [], 'missing': missing,
                      'generation_status': generation.get('status', 'not_started')})
        return 2
    try:
        value = audit()
    except Exception as error:
        ISSUES.append(type(error).__name__ + ': ' + str(error))
        value = {'checked_file_hashes': CHECKED_FILES}
    value['issues'] = ISSUES
    value['status'] = 'failed' if ISSUES else 'pass'
    value['scientific_attack_success_established_by_audit'] = False
    write_result(value)
    return 1 if ISSUES else 0


if __name__ == '__main__':
    raise SystemExit(main())
