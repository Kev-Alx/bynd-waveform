"""Durable CPU post-processing after the detached, bounded GPU trial finishes.

Raw registrations, results and checkpoints are never changed. This creates a
derived comparison/report and runs independent saved-artifact validation.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path('/workspace/improvement')
RUN = OUT / 'execution'
R3 = RUN / 'r3-waveform-autograd-repair'
JOB = RUN / 'jobs/r3-waveform-autograd-repair-bounded-run/job.json'
PYTHON = '/opt/venv/bin/python'
TAIPEI = timezone(timedelta(hours=8))


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_report():
    primary = read(RUN / 'resume-audit.json').get('primary_quality_control', {})
    conditions = []
    if primary:
        conditions.append({'method': 'Original quality control, LR 0.02',
                           'provenance': 'historical same24 native-policy saved score audit', **primary})
    for folder, variants in [
        ('r2-canonical-zero-uniform-dev24', [('clean', 'Clean'), ('reconstruction', 'Zero-edit reconstruction'),
           ('baseline_frozen', 'Original strong control, LR 0.1'), ('adversarial', 'R2 local geometry')]),
        ('r1-deletion-development24', [('adversarial', 'R1 with one deletion candidate')]),
        ('r3-waveform-autograd-repair', [('initial-generator', 'R3 untrained seed11 control'),
                                       ('adversarial', 'R3 trained seed11')])]:
        summary = read(RUN / folder / 'source-summary.json')
        for variant, label in variants:
            if variant in summary.get('summary', {}):
                conditions.append({'method': label, 'provenance': str(RUN / folder / 'source-summary.json'),
                                   'variant': variant, **summary['summary'][variant]})
    for condition in conditions:
        quality = condition['quality']
        condition['automatic_quality_floors_pass'] = (quality['utmos'] >= 3.5
            and quality['dnsmos'] >= 3 and quality['nisqa'] >= 3.75)
        if primary:
            delta_wer = condition['error_percent'] - primary['error_percent']
            delta_mos = quality['utmos'] - primary['quality']['utmos']
            condition['versus_primary_control'] = {'source_wer_points': delta_wer, 'utmos': delta_mos}
            condition['g1_source_quality_signal'] = condition['automatic_quality_floors_pass'] and (
                delta_wer >= 3 or delta_mos >= .05 and delta_wer >= -2)
        condition['human_content_established'] = False
        condition['heldout_transfer_established'] = False
    training = read(R3 / 'training.json')
    source = read(R3 / 'source-summary.json')
    completed = training.get('status') == 'completed' and source.get('status') == 'complete'
    result = {'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_source_only_initial_cycle' if completed else 'incomplete',
        'source_only': True, 'target_models_evaluated': [], 'confirmation_or_final_inference': False,
        'cohort_count': 24, 'reference_words': 475, 'conditions': conditions,
        'r3_training': training, 'r3_job': read(JOB),
        'r3_generation_resources': read(R3 / 'generation-resource.json'),
        'r3_final_evaluation_resources': source.get('resources'),
        'limits': ['Previously exposed development cohort; no final/confirmation inference.',
                   'One learned seed; training-seed variance is not estimated.',
                   'No human transcription; automatic quality is a proxy.',
                   'No matched per-example waveform-PGD or lifetime-cost superiority claim.',
                   'R1/R2 scientific rejections remain separate from R3 implementation failures.']}
    atomic(RUN / 'initial-comparison.json', result)
    stamp = datetime.now(TAIPEI).isoformat(timespec='seconds')
    lines = ['# Initial development results', '', f'Updated **{stamp} (Asia/Taipei)**. '
        f"Status: **{result['status']}**.", '',
        'All numerical rows below use the same exposed 24 development clips / 475 reference words '
        'and the frozen native source-recognition policy. Higher attacked machine WER indicates '
        'more recognition errors; it does not establish human content preservation or transfer.', '',
        '| Method | Source WER % | DNSMOS | NISQA | UTMOS | Automatic quality floors |',
        '|---|---:|---:|---:|---:|---|']
    for row in conditions:
        q = row['quality']
        lines.append(f"| {row['method']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | "
                     f"{q['nisqa']:.4f} | {q['utmos']:.4f} | {'pass' if row['automatic_quality_floors_pass'] else 'fail'} |")
    lines += ['', 'R1 canonical beam risk stopped on an all-word-correct candidate set. Its one '
        'candidate-set repair completed but failed the UTMOS floor. R2 retained quality while '
        'eliminating nearly all attack effect. Both fail the registered comparison with the '
        'quality-admissible original control.', '']
    if training:
        lines += [f"R3 completed **{training.get('updates')} optimizer updates / "
            f"{training.get('completed_microbatches')} accumulated microbatches** in "
            f"**{training.get('training_wall_seconds', 0):.2f} seconds**. Stop reason: "
            f"`{training.get('stop_reason')}`. Selection: {training.get('checkpoint_selection')}. "
            'Checkpoint source/MOS evaluations are charged to that training wall-time cap.', '']
    else:
        lines += ['R3 training/final source evaluation has not produced a completed summary yet. '
            'Inspect the live status and preserved job/failure receipts; no trained result is inferred.', '']
    trained = next((r for r in conditions if r['method'] == 'R3 trained seed11'), None)
    if trained:
        comparison = trained['versus_primary_control']
        lines += [f"R3 differs from the primary original quality control by "
            f"**{comparison['source_wer_points']:+.4f} source-WER points** and "
            f"**{comparison['utmos']:+.4f} UTMOS**. Its automatic source/quality development screen "
            f"**{'passes' if trained['g1_source_quality_signal'] else 'fails'}**. "
            'This is a one-seed feasibility result. It is not final superiority or evidence of a publishable contribution.', '']
    lines += ['No target recognizers, confirmation48, final200, human raters or defense training '
        'were used in this initial cycle. No sampling or training-seed confidence interval is claimed. '
        'A promising source-only result would still need the registered mechanism controls, genuine '
        'additional seeds, content screen and untouched evaluation before an improvement claim.', '',
        '[Machine-readable comparison](execution/initial-comparison.json) · '
        '[Detailed execution report](05_execution_report.md) · '
        '[Reproduction guide](execution/reproduction-guide.md) · [Job status](JOB_STATUS.md)', '']
    audit = RUN / 'r3-completed-audit.md'
    if audit.exists():
        lines += ['[Independent R3 saved-artifact audit](execution/r3-completed-audit.md)', '']
    (OUT / '06_initial_results.md').write_text('\n'.join(lines))
    return result


def main():
    with (RUN / 'finalization.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        while read(JOB).get('status') not in {'complete', 'failed', 'interrupted'}:
            atomic(RUN / 'finalization-status.json', {'status': 'waiting_for_detached_trial',
                'pid': os.getpid(), 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                'waiting_for': str(JOB)})
            if time.monotonic() - started > 7200:
                raise TimeoutError('Two-hour postprocessor wait exceeded; preserve running job and inspect status')
            time.sleep(20)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        checks = []
        for script, log in [(RUN / 'resume_audit.py', RUN / 'resume-audit-refresh.log'),
                            (RUN / 'audit_r3_completed.py', RUN / 'r3-completed-audit.log')]:
            with log.open('ab') as stream:
                process = subprocess.run([PYTHON, str(script)], cwd=OUT,
                    stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=300)
            checks.append({'script': str(script), 'returncode': process.returncode, 'log': str(log)})
        result = write_report()
        report = OUT / '05_execution_report.md'
        text = report.read_text()
        text = text.replace('**This report is in progress.**',
            '**The bounded source-only cycle has finished; see [final measured results](06_initial_results.md).**'
            if result['status'] == 'completed_source_only_initial_cycle' else
            '**The detached attempt stopped; see [measured results and execution status](06_initial_results.md).**')
        report.write_text(text)
        with (RUN / 'package-validation-final.log').open('ab') as stream:
            validation = subprocess.run([PYTHON, str(OUT / 'validate_package.py')], cwd=OUT,
                stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=300)
        checks.append({'script': str(OUT / 'validate_package.py'), 'returncode': validation.returncode})
        summary = {'status': 'complete' if all(c['returncode'] == 0 for c in checks) else 'needs_attention',
            'pid': os.getpid(), 'checks': checks, 'result_status': result['status'],
            'updated_at_utc': datetime.now(timezone.utc).isoformat()}
        atomic(RUN / 'finalization-status.json', summary)
        state = read(OUT / 'JOB_STATUS.json')
        state['postprocessing'] = summary
        atomic(OUT / 'JOB_STATUS.json', state)
        (OUT / 'JOB_STATUS.md').write_text((OUT / 'JOB_STATUS.md').read_text()
            + f"\nPost-processing: **{summary['status']}**. [Measured results]({OUT}/06_initial_results.md).\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
