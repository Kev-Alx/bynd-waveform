"""Freeze a useful method before larger source-only validation; no target access."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import hashlib
import numpy as np

OUT = Path('/workspace/improvement'); RUN = OUT / 'execution_refine'


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def floors(value):
    q = value['quality']
    return q['utmos'] >= 3.5 and q['dnsmos'] >= 3 and q['nisqa'] >= 3.75


def compare(cohort):
    plan = read(RUN / 'plan.json'); folder = RUN / cohort
    audit = read(folder / 'audit.json')
    if audit['status'] != 'pass': raise ValueError('Independent audit has not passed')
    native = read(folder / 'source-summary.json'); generation = read(folder / 'generation-result.json')
    summaries = native['summary']; info = plan['cohorts'][cohort]
    data = rows(info['manifest']); scores = rows(folder / 'source-scores.jsonl')
    if any(value['count'] != info['count'] for value in summaries.values()):
        raise ValueError('Incomplete paired cohort')
    if cohort == 'pilot':
        admissible = [v for v in ('fv50', 'fv100') if floors(summaries[v])]
        control = min(admissible, key=lambda v: (-summaries[v]['error_percent'],
            -summaries[v]['quality']['utmos'], v != 'fv50')) if admissible else 'fv50'
    else:
        control = read(RUN / 'validation-freeze.json')['control']
    baseline = summaries[control]; start = summaries['full50']; findings = {}
    for variant in ('constrained100', 'weighted100'):
        value = summaries[variant]
        dwer = value['error_percent'] - baseline['error_percent']
        dutmos = value['quality']['utmos'] - baseline['quality']['utmos']
        failures = [item for item in generation['failures'] if item['variant'] == variant]
        admissible = floors(value) and not failures
        g1 = admissible and floors(baseline) and (dwer >= 3 or dutmos >= .05 and dwer >= -2)
        gain = value['error_percent'] - start['error_percent']
        quality_gain = value['quality']['utmos'] - start['quality']['utmos']
        mechanism = admissible and ((gain >= 3 and quality_gain >= -.05)
            or (quality_gain >= .05 and gain >= -2))
        findings[variant] = {'source_wer_delta_points': dwer, 'utmos_delta': dutmos,
            'automatic_quality_floors_pass': floors(value), 'failures': failures,
            'primary_source_quality_screen': g1, 'source_gain_vs_full50': gain,
            'utmos_gain_vs_full50': quality_gain, 'mechanism_frontier_signal': mechanism,
            'earns_larger_validation': g1 or mechanism}
    speakers = sorted({row['speaker'] for row in data})
    weights = np.random.default_rng(20261022).multinomial(len(speakers),
        [1 / len(speakers)] * len(speakers), size=10000)
    lookup = {v: {r['id']: r for r in scores if r['variant'] == v} for v in summaries}
    aggregates = {}
    for variant in lookup:
        aggregates[variant] = np.asarray([[sum(lookup[variant][r['id']][key]
            for r in data if r['speaker'] == speaker) for key in
            ('substitutions', 'deletions', 'insertions', 'reference_units')]
            + [sum(lookup[variant][r['id']]['quality']['utmos'] for r in data if r['speaker'] == speaker),
               sum(r['speaker'] == speaker for r in data)] for speaker in speakers])
    reference = weights @ aggregates[control]
    for variant, finding in findings.items():
        candidate = weights @ aggregates[variant]
        dwer = 100 * candidate[:, :3].sum(1) / candidate[:, 3] - 100 * reference[:, :3].sum(1) / reference[:, 3]
        dutmos = candidate[:, 4] / candidate[:, 5] - reference[:, 4] / reference[:, 5]
        finding['descriptive_speaker_95'] = {'source_wer_delta': np.quantile(dwer, [.025, .975]).tolist(),
            'utmos_delta': np.quantile(dutmos, [.025, .975]).tolist(), 'replicates': 10000,
            'seed': 20261022, 'scope': 'Source/cohort variation only; no training/human/transfer inference'}
    result = {'status': 'complete', 'cohort': cohort, 'count': info['count'], 'speakers': len(speakers),
        'summaries': summaries, 'control': control, 'findings': findings,
        'generation_wall_seconds': generation['wall_seconds'], 'evaluation_resources': native['resources'],
        'source_only': True, 'target_models_evaluated': [], 'human_intelligibility_established': False,
        'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    if cohort == 'validation':
        result['frozen_primary_candidate'] = read(RUN / 'validation-freeze.json')['candidate']
        result['candidate_reselected_on_validation'] = False
    (folder / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if cohort == 'pilot':
        eligible = [v for v in findings if findings[v]['earns_larger_validation']]
        selected = min(eligible, key=lambda v: (-summaries[v]['error_percent'],
            -summaries[v]['quality']['utmos'], v != 'constrained100')) if eligible else None
        selection = {'status': 'validate' if selected else 'stop', 'candidate': selected, 'control': control,
            'selection_scope': 'Frozen source-only48-point decision before validation76',
            'reason': 'Primary screen or preregistered mechanism frontier signal; the latter is exploratory, not primary superiority',
            'findings': findings, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
        selection_path = RUN / 'selection.json'
        if selection_path.exists(): raise FileExistsError('Preserve the original source-only method selection')
        selection_path.write_text(json.dumps(selection, indent=2) + '\n'); selection_path.chmod(0o444)
        bound = [RUN / 'plan.json', folder / 'source-summary.json', folder / 'audit.json',
            folder / 'comparison.json', selection_path, Path(plan['cohorts']['validation']['manifest'])]
        freeze = {**selection, 'bindings': {str(path): digest(path) for path in bound}}
        freeze_path = RUN / 'validation-freeze.json'
        freeze_path.write_text(json.dumps(freeze, indent=2) + '\n'); freeze_path.chmod(0o444)
    write_report()
    print(json.dumps({'status': 'complete', 'cohort': cohort, 'control': control, 'findings': findings}, indent=2))


def write_report():
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# Refinement and larger-validation results', '', f'Updated **{stamp} (Asia/Taipei)**.', '',
        'All results use the frozen native source recognizer and automatic quality scorers. '
        'These measurements do not establish human intelligibility, black-box transfer or novelty.', '']
    for cohort in ('pilot', 'validation'):
        path = RUN / cohort / 'comparison.json'
        if not path.exists():
            lines += [f'## {cohort.capitalize()}', '', 'Not completed. See live job status.', '']; continue
        value = read(path)
        lines += [f"## {cohort.capitalize()}: {value['count']} utterances / {value['speakers']} speakers", '',
            '| Condition | Source WER % | DNSMOS | NISQA | UTMOS |', '|---|---:|---:|---:|---:|']
        for variant in ('clean', 'fv50', 'fv100', 'full50', 'weighted100', 'constrained100'):
            s = value['summaries'][variant]; q = s['quality']
            lines.append(f"| {variant} | {s['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} |")
        lines += ['', f"The primary quality-admissible control fixed on the pilot is **{value['control']}**.", '']
        for variant, item in value['findings'].items():
            lines.append(f"- {variant}: {item['source_wer_delta_points']:+.4f} source-WER points and "
                f"{item['utmos_delta']:+.4f} UTMOS versus that control; primary source/quality screen "
                f"**{'passes' if item['primary_source_quality_screen'] else 'fails'}**. "
                f"Gain over the full50 initializer: {item['source_gain_vs_full50']:+.4f} WER points.")
        lines += ['', f"Generation {value['generation_wall_seconds']:.2f}s; native source/MOS evaluation "
            f"{value['evaluation_resources']['elapsed_wall_seconds']:.2f}s. Constrained line-search forwards "
            'are additional cost;100 source backward calls does not mean equal total runtime.', '']
    selection = RUN / 'selection.json'
    if selection.exists():
        selected = read(selection)
        lines += [f"Validation decision: **{selected['status']}**; fixed candidate **{selected['candidate']}**. "
            'A mechanism signal alone justifies exploratory validation; it is not a relaxed primary-improvement claim.', '']
    lines += ['The validation76 has no utterance/hash/text overlap with the48-point pilot, but shares speakers '
        'and has historical study exposure. No untouched confirmation/final set or held-out recognizer was used. '
        'Caps constrain two proxy distortions, not MOS or human semantics. No final superiority claim follows.', '',
        '[Protocol](09_refinement_protocol.md) · [Live status](JOB_STATUS.md) · '
        '[Prior-work check](execution_refine/prior-work.md) · [Earlier48-point results](08_pilot48_results.md)', '']
    (OUT / '10_refinement_results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['pilot', 'validation'], required=True)
    compare(parser.parse_args().cohort)
