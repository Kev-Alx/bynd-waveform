"""Report all conditions and freeze one predetermined candidate before validation."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path('/workspace/improvement'); RUN = OUT / 'execution_envelope'
CANDIDATE, CONTROL = 'envelope_nll50', 'fv50'


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def floors(value):
    q = value['quality']; return q['dnsmos'] >= 3 and q['nisqa'] >= 3.75 and q['utmos'] >= 3.5


def decision(summary, failures):
    candidate, control = summary[CANDIDATE], summary[CONTROL]
    full, matched = summary['global_full50'], summary['attenuated_global_nll50']
    dw = candidate['error_percent'] - control['error_percent']
    du = candidate['quality']['utmos'] - control['quality']['utmos']
    admissible = floors(candidate) and not failures
    primary = admissible and floors(control) and (dw >= 3 or du >= .05 and dw >= -2)
    mechanism = admissible and candidate['error_percent'] >= full['error_percent'] - 2 \
        and candidate['quality']['nisqa'] >= full['quality']['nisqa'] + .10 \
        and candidate['quality']['utmos'] >= full['quality']['utmos'] - .05 \
        and candidate['error_percent'] >= matched['error_percent'] + 3
    return {'automatic_quality_floors_pass': floors(candidate), 'failures': failures,
        'primary_source_quality_screen': bool(primary), 'exploratory_shape_signal': bool(mechanism),
        'earns_larger_validation': bool(primary or mechanism), 'source_wer_delta_points': dw,
        'utmos_delta': du, 'source_wer_delta_vs_rms_matched': candidate['error_percent'] - matched['error_percent'],
        'nisqa_delta_vs_global_full50': candidate['quality']['nisqa'] - full['quality']['nisqa']}


def compare(cohort):
    folder = RUN / cohort; plan = read(RUN / 'plan.json')
    if read(folder / 'audit.json')['status'] != 'pass': raise ValueError('Audit has not passed')
    generation = read(folder / 'generation-result.json'); native = read(folder / 'source-summary.json')
    summary = native['summary']; info = plan['cohorts'][cohort]
    if any(v['count'] != info['count'] for v in summary.values()): raise ValueError('Partial-cohort report forbidden')
    findings = decision(summary, generation['failures'])
    data = rows(info['manifest']); scores = rows(folder / 'source-scores.jsonl')
    speakers = sorted({r['speaker'] for r in data})
    weights = np.random.default_rng(20261022).multinomial(len(speakers), [1 / len(speakers)] * len(speakers), size=10000)
    lookup = {(r['id'], r['variant']): r for r in scores}
    aggregates = {}
    for variant in (CANDIDATE, CONTROL, 'attenuated_global_nll50'):
        aggregates[variant] = np.asarray([[sum(lookup[r['id'], variant][key] for r in data if r['speaker'] == speaker)
            for key in ('substitutions', 'deletions', 'insertions', 'reference_units')]
            + [sum(lookup[r['id'], variant]['quality']['utmos'] for r in data if r['speaker'] == speaker),
               sum(r['speaker'] == speaker for r in data)] for speaker in speakers])
    candidate = weights @ aggregates[CANDIDATE]
    intervals = {}
    for variant in (CONTROL, 'attenuated_global_nll50'):
        reference = weights @ aggregates[variant]
        dw = 100 * candidate[:, :3].sum(1) / candidate[:, 3] - 100 * reference[:, :3].sum(1) / reference[:, 3]
        du = candidate[:, 4] / candidate[:, 5] - reference[:, 4] / reference[:, 5]
        intervals[variant] = {'source_wer_delta_points': np.quantile(dw, [.025, .975]).tolist(),
            'utmos_delta': np.quantile(du, [.025, .975]).tolist(), 'replicates': 10000, 'seed': 20261022,
            'scope': 'Descriptive speaker bootstrap on exposed-development/source model only'}
    value = {'status': 'complete', 'cohort': cohort, 'count': info['count'], 'speakers': len(speakers),
        'candidate': CANDIDATE, 'control': CONTROL, 'candidate_reselected': False, 'summaries': summary,
        'findings': findings, 'descriptive_intervals95': intervals, 'source_only': True, 'target_models_evaluated': [],
        'generation_wall_seconds': generation['wall_seconds'], 'evaluation_wall_seconds': native['resources']['elapsed_wall_seconds'],
        'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    (folder / 'comparison.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if cohort == 'pilot':
        if (RUN / 'selection.json').exists(): raise FileExistsError('Preserve first envelope decision')
        selected = {'status': 'validate' if findings['earns_larger_validation'] else 'stop',
            'candidate': CANDIDATE, 'control': CONTROL, 'findings': findings,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Fixed primary candidate; mechanism-only continuation is exploratory, not primary superiority'}
        path = RUN / 'selection.json'; path.write_text(json.dumps(selected, indent=2) + '\n'); path.chmod(0o444)
        bound = [RUN / 'plan.json', path, folder / 'comparison.json', folder / 'source-summary.json',
            folder / 'audit.json', Path(plan['cohorts']['validation']['manifest'])]
        freeze = {**selected, 'bindings': {str(p): digest(p) for p in bound}}
        path = RUN / 'validation-freeze.json'; path.write_text(json.dumps(freeze, indent=2) + '\n'); path.chmod(0o444)
    else:
        import envelope_experiment as cycle
        cycle.validation_freeze()
    report(); print(json.dumps({'cohort': cohort, 'findings': findings}, indent=2))


def report():
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# Clean-energy waveform experiment results', '', f'Updated **{stamp} (Asia/Taipei)**.', '',
        'The fixed primary candidate is envelope_nll50. All scores use the same frozen public source '
        'recognizer and native automatic quality evaluation. No human, held-out transfer or novelty claim follows.', '']
    for cohort in ('pilot', 'validation'):
        path = RUN / cohort / 'comparison.json'
        if not path.exists():
            stop = (RUN / 'selection.json').exists() and read(RUN / 'selection.json')['status'] == 'stop'
            lines += [f'## {cohort.capitalize()}', '', 'Not run: the pilot failed its frozen advancement rule.' if stop
                else 'Pending; see live status.', '']; continue
        value = read(path)
        lines += [f"## {cohort.capitalize()}: {value['count']} utterances / {value['speakers']} speakers", '',
            '| Condition | Source WER % | DNSMOS | NISQA | UTMOS |', '|---|---:|---:|---:|---:|']
        for variant, score in value['summaries'].items():
            q = score['quality']
            lines.append(f"| {variant} | {score['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} |")
        f = value['findings']
        lines += ['', f"Primary screen: **{'passes' if f['primary_source_quality_screen'] else 'fails'}**. "
            f"Exploratory shape signal: **{'passes' if f['exploratory_shape_signal'] else 'fails'}**. "
            f"Candidate vsFV50: {f['source_wer_delta_points']:+.4f} WER points / {f['utmos_delta']:+.4f} UTMOS. "
            f"Candidate vsRMS-matched global NLL: {f['source_wer_delta_vs_rms_matched']:+.4f} WER points.", '',
            f"Generation {value['generation_wall_seconds']:.2f}s; native source/MOS evaluation {value['evaluation_wall_seconds']:.2f}s.", '']
    if (RUN / 'selection.json').exists():
        lines += [f"Validation decision: **{read(RUN / 'selection.json')['status']}**. Candidate selection remains fixed.", '']
    lines += ['The76-utterance validation is disjoint in utterance/hash/text from the48-point pilot, '
        'but has historical study exposure and shared speakers. Confirmation48/final200 and held-out ASRs remain unused. '
        'The clean-energy bounds protect registered-frame energy; they are not auditory masking or a human-content guarantee.', '',
        '[Explanation and goals](11_changes_and_goals.md) · [Protocol](12_envelope_protocol.md) · [Live status](JOB_STATUS.md)', '']
    (OUT / '13_envelope_results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--cohort', choices=['pilot', 'validation'], required=True)
    compare(p.parse_args().cohort)
