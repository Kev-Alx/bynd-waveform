"""Create source-only48-point findings from completed, independently audited data."""
from pathlib import Path
import json
from datetime import datetime, timezone, timedelta
import numpy as np

OUT = Path('/workspace/improvement')
RUN = OUT / 'execution48'
FOLDER = RUN / 'mechanism-pilot48'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    audit = read(RUN / 'audit-pilot48.json')
    if audit['status'] != 'pass':
        raise ValueError('Do not publish verified results until independent audit passes')
    plan = read(RUN / 'plan.json')
    generated = read(FOLDER / 'generation-result.json')
    native = read(FOLDER / 'source-summary.json')
    cohort = rows(plan['development_manifest'])
    scores = rows(FOLDER / 'source-scores.jsonl')
    summaries = native['summary']
    if len(cohort) != 48 or any(value['count'] != 48 for value in summaries.values()):
        raise ValueError('A new pilot must report every one of48 utterances for every variant')
    reference = summaries['fv_quality']
    floors = lambda item: (item['quality']['utmos'] >= 3.5 and item['quality']['dnsmos'] >= 3
                           and item['quality']['nisqa'] >= 3.75)
    findings = {}
    for variant in ('pgd_full', 'pgd_nll', 'r3_trained'):
        item = summaries[variant]
        dwer = item['error_percent'] - reference['error_percent']
        dutmos = item['quality']['utmos'] - reference['quality']['utmos']
        failures = [f for f in generated['pgd_mathematical_failures'] if f['variant'] == variant]
        findings[variant] = {'source_wer_delta_points': dwer, 'utmos_delta': dutmos,
            'automatic_quality_floors_pass': floors(item), 'primary_control_quality_pass': floors(reference),
            'mathematical_failures': failures,
            'g1_source_quality_signal': floors(item) and floors(reference) and not failures
                and (dwer >= 3 or dutmos >= .05 and dwer >= -2)}

    # Paired speaker-cluster bootstrap; descriptive source/development endpoints.
    speakers = sorted({r['speaker'] for r in cohort})
    ids = {r['id']: r for r in cohort}
    byvariant = {v: {r['id']: r for r in scores if r['variant'] == v} for v in summaries}
    rng = np.random.default_rng(20261022)
    multiplicities = rng.multinomial(len(speakers), [1 / len(speakers)] * len(speakers), size=10000)
    totals = {}
    for variant in byvariant:
        totals[variant] = np.asarray([[sum(byvariant[variant][r['id']][k]
            for r in cohort if r['speaker'] == speaker) for k in ('substitutions', 'deletions', 'insertions', 'reference_units')]
            + [sum(byvariant[variant][r['id']]['quality']['utmos'] for r in cohort if r['speaker'] == speaker),
               sum(r['speaker'] == speaker for r in cohort)] for speaker in speakers], dtype=np.float64)
    for variant, finding in findings.items():
        candidate = multiplicities @ totals[variant]
        control = multiplicities @ totals['fv_quality']
        delta_wer = 100 * candidate[:, :3].sum(1) / candidate[:, 3] - 100 * control[:, :3].sum(1) / control[:, 3]
        delta_mos = candidate[:, 4] / candidate[:, 5] - control[:, 4] / control[:, 5]
        finding['descriptive_speaker_bootstrap_95'] = {
            'source_wer_delta_points': np.quantile(delta_wer, [.025, .975]).tolist(),
            'utmos_delta': np.quantile(delta_mos, [.025, .975]).tolist(),
            'replicates': 10000, 'seed': 20261022, 'speakers': len(speakers),
            'scope': 'Exposed-development/source sampling only; no training-seed, human or transfer inference'}
    result = {'status': 'complete', 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'pilot_count': 48, 'speakers': len(speakers), 'source_only': True,
        'target_models_evaluated': [], 'confirmation_or_final_inference': False,
        'summaries': summaries, 'findings': findings,
        'generation_wall_seconds': generated['wall_seconds'],
        'evaluation_resources': native['resources'], 'audit_status': audit['status'],
        'no_new_generator_training': True}
    (RUN / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    names = {'clean': 'Clean', 'reconstruction': 'Reconstruction', 'fv_quality': 'Original FV quality control',
        'r3_initial': 'Untrained waveform generator', 'r3_trained': 'Frozen trained waveform generator',
        'pgd_full': 'Per-example waveform PGD, full R3 loss', 'pgd_nll': 'Per-example waveform PGD, NLL only'}
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# 48-point pilot results', '', f'Updated **{stamp} (Asia/Taipei)**. '
        'Execution and independent saved-artifact audit completed.', '',
        f"Every condition contains the same48 utterances / {reference['reference_units']} reference words / {len(speakers)} speakers. "
        'Higher attacked source WER means more recognition errors. These are exposed-development source-only results.', '',
        '| Condition | Source WER % | DNSMOS | NISQA | UTMOS |', '|---|---:|---:|---:|---:|']
    for variant in plan['variants']:
        value = summaries[variant]; q = value['quality']
        lines.append(f"| {names[variant]} | {value['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} |")
    lines += ['', '## Registered development screen', '']
    for variant, value in findings.items():
        interval = value['descriptive_speaker_bootstrap_95']['source_wer_delta_points']
        lines.append(f"- {names[variant]}: {value['source_wer_delta_points']:+.4f} WER points and "
            f"{value['utmos_delta']:+.4f} UTMOS versus the FV quality control; screen "
            f"**{'passes' if value['g1_source_quality_signal'] else 'fails'}**. "
            f"Descriptive paired speaker interval for source-WER difference: [{interval[0]:.4f}, {interval[1]:.4f}].")
    full_effect = summaries['pgd_full']['error_percent'] - summaries['clean']['error_percent']
    nll_effect = summaries['pgd_nll']['error_percent'] - summaries['clean']['error_percent']
    lines += ['', '## Mechanism evidence', '',
        f'Full-objective per-example PGD changes source WER by {full_effect:+.4f} points relative to clean; '
        f'NLL-only PGD changes it by {nll_effect:+.4f} points. The generator checkpoint and coefficients '
        'were frozen before this pilot. The per-example conditions separate a shared generator from '
        'direct waveform optimization; the loss pair isolates the penalty terms under this optimizer. '
        'No result proves a globally optimal attack or identifies all causes of amortized-training failure.', '',
        f"Generation took {generated['wall_seconds']:.2f} seconds, native source/MOS evaluation "
        f"{native['resources']['elapsed_wall_seconds']:.2f} seconds. All additional model passes are disclosed in saved traces. "
        'No lifetime-efficiency superiority is inferred from these timings.', '',
        'No confirmation/final audio, sealed recognizers or human raters were used. A screen pass is '
        'exploratory and still needs mechanism/content validation and untouched evaluation. The bootstrap '
        'intervals describe only this exposed source cohort; the single learned seed does not estimate training variance.', '',
        '[Protocol](07_pilot48_protocol.md) · [Machine-readable results](execution48/comparison.json) · '
        '[Independent audit](execution48/audit-pilot48.md) · [Live job status](JOB_STATUS.md)', '']
    (OUT / '08_pilot48_results.md').write_text('\n'.join(lines))
    print(json.dumps({'status': 'complete', 'pilot_count': 48, 'findings': findings}, indent=2))


if __name__ == '__main__':
    main()
