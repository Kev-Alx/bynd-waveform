#!/usr/bin/env python3
"""CPU-only replay of saved R3 waveform/decoding failure evidence.

No model inference, imports of project runtimes, or reads of held-out audio.
--assert-attack-signal is deliberately red on the saved near-identity failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path('/workspace/improvement')
DEFAULT_RUN = ROOT / 'execution/r3-waveform-autograd-repair'


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()))


def word_distance(reference, hypothesis):
    left, right = reference.split(), hypothesis.split()
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def verify_score_summary(scores, saved):
    rebuilt = {}
    for variant in ('clean', 'initial-generator', 'adversarial'):
        rows = [r for r in scores if r['variant'] == variant]
        words = sum(len(r['reference_normalized'].split()) for r in rows)
        distance = sum(word_distance(r['reference_normalized'], r['hypothesis_normalized']) for r in rows)
        errors = sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in rows)
        quality = {key: float(np.mean([r['quality'][key] for r in rows])) for key in ('utmos', 'dnsmos', 'nisqa')}
        actual = dict(count=len(rows), word_distance=distance, reference_units=words, error_percent=100 * distance / words, quality=quality)
        expected = saved[variant]
        if distance != errors or words != expected['reference_units'] or len(rows) != expected['count']:
            raise ValueError(f'Independent score recount failed: {variant}')
        if abs(actual['error_percent'] - expected['error_percent']) > 1e-12 or any(abs(quality[k] - expected['quality'][k]) > 1e-12 for k in quality):
            raise ValueError(f'Score-summary aggregation mismatch: {variant}')
        rebuilt[variant] = actual
    return rebuilt


def waveform_stats(clean, adv, sample_rate, linf_limit, rms_limit):
    r = adv.astype(np.float64) - clean.astype(np.float64)
    power = float(np.mean(r * r))
    rms = float(np.sqrt(power))
    linf = float(np.max(np.abs(r)))
    spectrum = np.abs(np.fft.rfft(r)) ** 2
    if len(r) % 2 == 0:
        spectrum[1:-1] *= 2
    else:
        spectrum[1:] *= 2
    frequencies = np.fft.rfftfreq(len(r), 1 / sample_rate)
    spectral_power = max(float(spectrum.sum()), np.finfo(float).tiny)
    band_edges = [(0, 100), (100, 1000), (1000, 4000), (4000, 8001)]
    bands = {f'{lo}-{hi if hi != 8001 else 8000}Hz':
             float(spectrum[(frequencies >= lo) & (frequencies < hi)].sum() / spectral_power)
             for lo, hi in band_edges}
    edge_power = {}
    for edge in (64, 256, 1024):
        mask = np.zeros(len(r), dtype=bool)
        mask[:edge] = True
        mask[-edge:] = True
        edge_power[f'first_last_{edge}_samples'] = float(np.sum(r[mask] ** 2) / max(np.sum(r ** 2), np.finfo(float).tiny))
    return dict(linf=linf, rms=rms, mean_residual=float(r.mean()),
                dc_power_fraction=float(r.mean() ** 2 / max(power, np.finfo(float).tiny)),
                linf_budget_fraction=linf / linf_limit, rms_budget_fraction=rms / rms_limit,
                samples_above_one_percent_linf_budget=float(np.mean(np.abs(r) > .01 * linf_limit)),
                output_peak=float(np.max(np.abs(adv))), band_power_fraction=bands,
                edge_power_fraction=edge_power)


def trace_diagnostics(run, weights):
    trace = load_jsonl(run / 'training-trace.jsonl')
    by_clip, by_epoch = {}, {}
    for row in trace:
        by_clip.setdefault(row['id'], []).append(row)
        by_epoch.setdefault(row['epoch'], []).append(row)
    keys = ('source_sum_nll', 'stft_loss', 'content_loss')
    epochs = []
    for epoch, rows in sorted(by_epoch.items()):
        means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        epochs.append(dict(epoch=epoch, count=len(rows), distinct_clips=len({r['id'] for r in rows}),
                           means=means,
                           weighted_penalty_mean=weights['stft'] * means['stft_loss'] + weights['content'] * means['content_loss']))
    paired = []
    for clip, rows in sorted(by_clip.items()):
        first, last = rows[0], rows[-1]
        paired.append(dict(id=clip, observations=len(rows), first_epoch=first['epoch'], last_epoch=last['epoch'],
                           first={k: first[k] for k in keys}, last={k: last[k] for k in keys},
                           last_minus_first={k: last[k] - first[k] for k in keys}))
    gradients = [r['preclip_gradient_norm'] for r in trace if r.get('preclip_gradient_norm') is not None]
    return dict(microbatches=len(trace), distinct_training_clips=len(by_clip), epoch_summaries=epochs,
                same_training_clip_last_minus_first={k: distribution([r['last_minus_first'][k] for r in paired]) for k in keys},
                same_training_clip_comparisons=paired,
                update_preclip_joint_gradient_norm=distribution(gradients),
                gradient_clip_threshold=1.0,
                fraction_updates_joint_norm_above_clip_threshold=float(np.mean(np.asarray(gradients) > 1.0)),
                limitations=['First/last observations hold clip identity fixed but are not clean/adversarial pairs.',
                             'Across-example NLL means do not measure an attack gain; references and duration vary.',
                             'Only joint parameter gradient norms were logged. Component norms, pairwise gradient cosines, and clean teacher-forced NLL were not saved.',
                             'Weighted scalar loss magnitudes cannot establish which gradient dominates.'])


def checkpoint_diagnostics(run):
    import torch
    rows = []
    for source in sorted((run / 'checkpoints').glob('update-*.source.json')):
        item = json.loads(source.read_text())
        item['receipt'] = str(source)
        checkpoint = source.with_name(source.name.replace('.source.json', '.pt'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        tensors = saved['generator']
        head = tensors['head.weight'].double().numpy()
        bias = tensors['head.bias'].double().numpy()
        item['parameter_evidence'] = dict(checkpoint_sha256=sha(checkpoint),
            parameter_count=sum(t.numel() for t in tensors.values()),
            all_finite=all(bool(torch.isfinite(t).all()) for t in tensors.values()),
            head_weight_l2=float(np.linalg.norm(head)),
            head_weight_maxabs=float(np.abs(head).max()),
            head_bias_values=bias.tolist(),
            generator_total_parameter_l2=float(np.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))),
            layer_weight_norms={k: float(t.double().norm()) for k, t in tensors.items() if k.endswith('weight')},
            interpretation='Parameter norms prove stored nonzero weights, not useful attack directions. The head bias alone does not determine the waveform mean.')
        rows.append(item)
    return rows


def hypotheses():
    return [
        dict(rank=1, mechanism='Quality penalties favor identity and overwhelm useful attack directions near the learned solution.',
             evidence='The generator removes most initial residual energy; STFT/content penalties shrink. Calibration balanced separate waveform-gradient norms at one random 1e-4 probe, which does not balance their joint parameter-space gradients throughout training.',
             falsification='At fixed per-example budget, 50-step NLL-only PGD succeeds on the same 48 clips but the identical optimizer with the registered STFT/content penalties fails. If both succeed comparably, penalty dominance alone is insufficient.'),
        dict(rank=2, mechanism='Amortization across utterances cancels source attack directions or prevents the shared generator learning them.',
             evidence='The trained generator is nearly identity despite finite updates and repeated training exposure. No source/penalty component gradient cosines or cross-utterance cosines were saved, so cancellation remains unproven.',
             falsification='Same-objective per-example PGD succeeds on 48 fixed clips while the frozen generator fails under the same budget. If the per-example objective also fails, amortization alone is not the explanation.'),
        dict(rank=3, mechanism='The generator initialization and interpolation architecture favor a DC or boundary response over input-specific adversarial structure.',
             evidence='Initial saved residuals have almost all power at DC; final residual spectra and boundary energy are measured below. Existing weights and finite gradients do not establish expressive attack directions.',
             falsification='After the PGD control, fit the unchanged generator to the fixed successful per-example residuals without ASR or quality losses. Failure to fit them under the same budget supports a representation or optimization bottleneck; successful fitting weakens it. This is a separate later experiment, not a retune of current coefficients.'),
        dict(rank=4, mechanism='Teacher-forced lexical NLL can change without altering native beam decoding.',
             evidence='Every saved decoded transcript is unchanged. Paired clean/adversarial teacher-forced NLL was not saved; training NLL comparisons across examples cannot demonstrate this mismatch.',
             falsification='On the 48 fixed clips, record paired clean/adversarial sum NLL and native beam-5 WER at initialization and final PGD. Large paired NLL gains with unchanged decoding support the mismatch; decoded WER growth refutes it as the sole cause.'),
        dict(rank=5, mechanism='The fixed 0.003 waveform box is too small for this source/model/domain at 50 updates.',
             evidence='The generator uses far less than the allowed budget, so its failure cannot establish that the budget is intrinsically insufficient.',
             falsification='NLL-only per-example PGD at Linf/RMS 0.003 reaches large budget fractions yet fails to raise paired source WER. Successful PGD under this exact budget rejects budget insufficiency as the sole cause. Do not increase the budget in this diagnostic comparison.')]


def next_experiment(weights):
    return dict(name='48-point paired source-only mechanism diagnostic',
        points=48, selection='Use one frozen outcome-independent 48-point pilot manifest; identical IDs/reference text in all arms. Do not append whichever old failures look hardest.',
        objective_weights=weights, retuning=False,
        controls=[
            dict(name='clean', optimization='none', purpose='Paired source decoding, teacher-forced NLL, and quality reference.'),
            dict(name='frozen R3 generator', optimization='none; use the saved selected update-1500 checkpoint', purpose='Check whether the original failure persists on the 48-point pilot.'),
            dict(name='R3-objective per-example waveform PGD', optimization='50 steps; ascend sum NLL minus registered STFT/content penalties; direct waveform residual with identical Linf/RMS/peak constraints', purpose='Separate the objective and budget from shared-generator optimization.'),
            dict(name='NLL-only per-example waveform PGD', optimization='Same initialization, 50 steps, optimizer, source forward, constraints and source-call budget as the previous arm; only remove the two soft losses', purpose='Isolate the effect of the quality penalties.'),
            dict(name='original B2 feature-delta baseline', optimization='Frozen original baseline optimizer/configuration on the same 48 examples; regenerate and native-score with matching policy', purpose='Positive-control source sensitivity and reference attack/quality tradeoff; feature-space and waveform budgets differ, so this is not a matched perturbation comparison.')],
        required_records=['Registration and code/model/data hashes before predictions.',
                          'Per clip clean/initial/final teacher-forced sum NLL, STFT, content, Linf, RMS, saturation fractions, waveform peak.',
                          'At matched saved iterates, raw waveform gradients of source/STFT/content and their cosines; do not infer dominance from loss magnitude.',
                          'Native beam-5 temperature-0 source transcripts, exact edit counts and all three existing quality scorers on saved float32 WAVs.',
                          'No held-out target models or fresh final-reserve data during mechanism diagnosis.'],
        interpretation='This controlled comparison identifies a bottleneck; it is not evidence of a better attack or transfer. PGD step size/initialization must be registered once by the runner owner, not selected after outcomes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'execution48')
    parser.add_argument('--assert-attack-signal', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_dir.resolve()
    if ROOT not in output.parents:
        raise ValueError('Output must stay under /workspace/improvement')
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads((run / 'training.json').read_text())
    generation = load_jsonl(run / 'generated.jsonl')
    scores = load_jsonl(run / 'source-scores.jsonl')
    summary = json.loads((run / 'source-summary.json').read_text())
    index = {(r['id'], r['variant']): r for r in scores}
    expected_ids = {r['id'] for r in generation}
    if len(expected_ids) != len(generation) or len(generation) != 24:
        raise ValueError('This saved failure replay is specifically the existing 24-point run')
    if len(index) != 3 * len(generation) or set(index) != {(clip, variant) for clip in expected_ids for variant in ('clean', 'initial-generator', 'adversarial')}:
        raise ValueError('Missing or duplicated score identities')
    independently_recounted = verify_score_summary(scores, summary['summary'])
    per_clip = []
    hashes = {}
    for row in generation:
        values = {}
        for variant in ('clean', 'initial-generator', 'adversarial'):
            meta = row['audio'][variant]
            path = Path(meta['path'])
            actual_sha = sha(path)
            if actual_sha != meta['sha256']:
                raise ValueError(f'WAV binding mismatch: {path}')
            wav, sr = sf.read(path, dtype='float32', always_2d=False)
            if wav.ndim != 1 or sr != 16000 or len(wav) != meta['samples']:
                raise ValueError(f'WAV shape mismatch: {path}')
            values[variant] = wav
            hashes[str(path)] = actual_sha
        item = dict(id=row['id'])
        for variant in ('initial-generator', 'adversarial'):
            clean_score = index[row['id'], 'clean']
            adv_score = index[row['id'], variant]
            item[variant] = waveform_stats(values['clean'], values[variant], sr,
                training['generator']['linf'], training['generator']['rms_limit'])
            item[variant]['exact_hypothesis_equal_clean'] = adv_score['hypothesis'] == clean_score['hypothesis']
            item[variant]['normalized_hypothesis_equal_clean'] = (
                adv_score['hypothesis_normalized'] == clean_score['hypothesis_normalized'])
            if item[variant]['linf_budget_fraction'] > 1.00001 or item[variant]['rms_budget_fraction'] > 1.00001 or item[variant]['output_peak'] > 1:
                raise ValueError(f'Residual constraint violation: {row["id"]} {variant}')
        per_clip.append(item)
    condition_stats = {}
    for variant in ('initial-generator', 'adversarial'):
        stats = {}
        for key in ('linf', 'rms', 'mean_residual', 'dc_power_fraction', 'linf_budget_fraction',
                    'rms_budget_fraction', 'samples_above_one_percent_linf_budget'):
            stats[key] = distribution([r[variant][key] for r in per_clip])
        stats['exact_hypotheses_equal_clean'] = sum(r[variant]['exact_hypothesis_equal_clean'] for r in per_clip)
        stats['normalized_hypotheses_equal_clean'] = sum(r[variant]['normalized_hypothesis_equal_clean'] for r in per_clip)
        stats['band_power_fraction_utterance_mean'] = {
            band: float(np.mean([r[variant]['band_power_fraction'][band] for r in per_clip]))
            for band in per_clip[0][variant]['band_power_fraction']}
        stats['edge_power_fraction_utterance_mean'] = {
            edge: float(np.mean([r[variant]['edge_power_fraction'][edge] for r in per_clip]))
            for edge in per_clip[0][variant]['edge_power_fraction']}
        condition_stats[variant] = stats
    delta_wer = summary['summary']['adversarial']['error_percent'] - summary['summary']['clean']['error_percent']
    near_identity = condition_stats['adversarial']['rms_budget_fraction']['median'] < .01
    no_decoding_effect = condition_stats['adversarial']['normalized_hypotheses_equal_clean'] == len(per_clip)
    verdict = 'FAIL_NO_ATTACK_SIGNAL' if no_decoding_effect and delta_wer <= 0 else 'ATTACK_SIGNAL_PRESENT'
    trace = trace_diagnostics(run, training['weights'])
    checkpoints = checkpoint_diagnostics(run)
    calibration = json.loads((run / 'penalty-calibration.json').read_text())
    ranked = hypotheses()
    result = dict(scope='CPU-only saved 24-point failure replay; no fresh pilot or inference',
                  deterministic_signal=dict(verdict=verdict, normalized_hypotheses_unchanged=no_decoding_effect,
                      source_wer_delta_percentage_points=delta_wer,
                      median_rms_below_one_percent_budget=near_identity),
                  previous_pilot_points=len(per_clip), future_pilot_requirement=48,
                  generator_budget=dict(linf=training['generator']['linf'], rms=training['generator']['rms_limit']),
                  conditions=condition_stats, per_clip=per_clip,
                  trace_diagnostics=trace, checkpoint_diagnostics=checkpoints,
                  calibration=calibration,
                  hypotheses_ranked=ranked, recommended_next_experiment=next_experiment(training['weights']),
                  source_summary=summary['summary'], input_audio_sha256=hashes,
                  independently_recounted_score_summary=independently_recounted,
                  input_json_sha256={str(run/n): sha(run/n) for n in ('training.json', 'generated.jsonl', 'source-scores.jsonl', 'source-summary.json', 'training-trace.jsonl', 'penalty-calibration.json')})
    (output / 'r3-failure-diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Saved R3 failure replay', '',
             f'Verdict: **{verdict}**. This replays the existing 24 examples; future pilots must contain 48.', '',
             f'All {len(per_clip)} normalized adversarial transcripts equal clean transcripts. Source WER changes by {delta_wer:.8f} percentage points.', '',
             f"The median residual RMS uses {100 * condition_stats['adversarial']['rms_budget_fraction']['median']:.6f}% of the fixed 0.003 RMS budget.", '',
             'Run `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/execution48/diagnostics_saved_r3.py --assert-attack-signal`.', '',
             'Exit code 1 records the failure symptom. It does not mean that a dependency or the diagnostic program failed.', '']
    lines.extend(['## Measured failure', '',
                  '| Saved condition | Median RMS | RMS budget used | Mean DC power fraction | Mean edge power fraction, first/last 256 samples |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for variant, stats in condition_stats.items():
        lines.append(f"| {variant} | {stats['rms']['median']:.8g} | {100 * stats['rms_budget_fraction']['median']:.6f}% | {100 * stats['dc_power_fraction']['mean']:.4f}% | {100 * stats['edge_power_fraction_utterance_mean']['first_last_256_samples']:.4f}% |")
    lines.extend(['', f"Training completed {training['updates']} optimizer updates / {training['completed_microbatches']} microbatches in {training['training_wall_seconds']:.2f} seconds. "
                       f"{100 * trace['fraction_updates_joint_norm_above_clip_threshold']:.2f}% of logged update gradients exceeded the fixed norm-1 clipping threshold.", '',
                  'The initial generator mainly emits a constant offset. The trained generator suppresses almost all of its RMS energy. The remaining residual is too small to support a meaningful test of the allowed waveform budget.', '',
                  'The checkpoint source WER remains exactly 3.578947% at updates 500, 1000 and 1500. Mean STFT loss falls from '
                  f"{checkpoints[0]['mean_stft_loss']:.8g} to {checkpoints[-1]['mean_stft_loss']:.8g}; the selected checkpoint is update 1500 because WER ties and the registered selector then favors smaller STFT loss. "
                  'That selection rule rewards preservation when every checkpoint fails to attack.', '',
                  'Same-clip first/last training observations:', '',
                  '| Quantity | Mean change | Median change | Range |', '| --- | ---: | ---: | ---: |'])
    for key, stats in trace['same_training_clip_last_minus_first'].items():
        lines.append(f"| {key} | {stats['mean']:.8g} | {stats['median']:.8g} | {stats['minimum']:.8g} to {stats['maximum']:.8g} |")
    lines.extend(['', 'These comparisons control clip identity but compare changing generator parameters over training. They do **not** measure clean-to-adversarial NLL gain. '
                  'The saved trace has no paired clean NLL and no source/penalty component gradient vectors; neither objective dominance nor gradient cancellation can be established from it.', '',
                  f"The fixed weights are STFT={training['weights']['stft']:.12g}, content={training['weights']['content']:.12g}. "
                  'Calibration equalized each separate waveform-gradient norm to source NLL at the median of three random probes. The sum of two penalties and their parameter-space alignment can behave differently near identity. '
                  'The calibration is a starting convention, not evidence of balanced optimization.', '',
                  '## Ranked hypotheses and tests', ''])
    for item in ranked:
        lines.extend([f"{item['rank']}. **{item['mechanism']}** {item['evidence']} Test: {item['falsification']}", ''])
    lines.extend(['## Recommended controlled comparison', '',
                  'Use exactly 48 fixed pilot examples for every arm: clean; the frozen selected R3 generator; per-example waveform PGD with the identical R3 objective; the same PGD with NLL alone; and the frozen original B2 feature-delta baseline. '
                  'Use the existing coefficients without retuning, unchanged 0.003 Linf/RMS waveform constraints, identical initialization and 50 updates for both PGD arms, and the same frozen source/evaluation policy. '
                  'The runner must freeze the step size and all remaining optimizer choices before outcomes. B2 is an attack positive control; its feature-space ball is not the same threat model as the waveform box.', '',
                  'Record paired clean/initial/final teacher-forced NLL, objective components, component waveform-gradient norms/cosines, residual budget use, native beam decoding, exact edits, and the three existing quality scores. '
                  'If the full objective works per example, focus on amortization. If only NLL-only PGD works, focus on penalty constraints. If neither works despite budget use, examine the source NLL/decoding mismatch and fixed budget before training another generator. '
                  'Successful NLL-only attack audio still needs its quality judged; it is a diagnostic result, not a promotion.', '',
                  'No held-out model, 48-point pilot inference, new source inference, or final-reserve audio was read by this diagnostic. The JSON contains waveform hashes, full per-clip residual/spectral summaries, same-clip trace comparisons, and checkpoint parameter norms.'])
    (output / 'r3-failure-diagnostics.md').write_text('\n'.join(lines))
    print(json.dumps(result['deterministic_signal'], indent=2))
    if args.assert_attack_signal and verdict == 'FAIL_NO_ATTACK_SIGNAL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
