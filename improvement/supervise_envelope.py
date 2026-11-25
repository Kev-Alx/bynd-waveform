"""Detached clean-energy experiment and conditional validation; shared GPU lock."""
from datetime import datetime, timezone, timedelta
import fcntl
import os
from pathlib import Path
import signal
import time
import supervise_pilot48 as jobs

OUT = Path('/workspace/improvement'); RUN = OUT / 'execution_envelope'
CURRENT = 'pilot'


def heartbeat():
    state = jobs.STATE
    folder = RUN / CURRENT
    expected = 48 if CURRENT == 'pilot' else 76
    state.update(updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        timezone='Asia/Taipei', cohort=CURRENT, count=expected,
        generation=jobs.read(folder / 'generation-status.json'),
        scored_variant_waveforms=len(list((folder / 'score-records').glob('*.json'))),
        total_variant_waveforms=7 * expected,
        independent_audit_status=jobs.read(folder / 'audit.json').get('status', 'pending'),
        selection=jobs.read(RUN / 'selection.json'))
    jobs.atomic(RUN / 'progress.json', state); jobs.atomic(OUT / 'JOB_STATUS.json', state)
    lines = ['# Clean-energy experiment job status', '',
        f"Updated **{state['updated_at']} (Asia/Taipei)**. Status: **{state['status']}**. "
        f"Current phase: **{state['phase']}**.", '',
        'Detached sequential queue with30-second heartbeats.48-point screen; automatic larger '
        'validation76 for a qualifying frozen candidate. All work remains source-only.', '',
        '| Stage | Status | Exit code |', '|---|---|---:|']
    for stage in state['stages']:
        lines.append(f"| {stage['name']} | {stage['status']} | {stage.get('exit_code', '—')} |")
    p = state['generation']
    if p:
        lines += ['', 'Generation: ' + ', '.join(f'{k}={p[k]}' for k in
            ('completed', 'clip', 'total', 'id', 'variant', 'update') if k in p) + '.']
    lines += ['', f"Scored waveform variants: {state['scored_variant_waveforms']}/{7 * expected}; "
        f"independent audit: {state['independent_audit_status']}.", '',
        'The earlier runs are preserved. Operational completion and scientific improvement are reported separately. '
        'Untouched confirmation/final data and sealed recognizers remain unused.', '',
        f'[Results]({OUT}/13_envelope_results.md) · [Protocol]({OUT}/12_envelope_protocol.md) · '
        f'[Prior48-point results]({OUT}/08_pilot48_results.md)', '']
    temporary = OUT / 'JOB_STATUS.md.tmp'; temporary.write_text('\n'.join(lines)); temporary.replace(OUT / 'JOB_STATUS.md')


def main():
    global CURRENT
    RUN.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution/continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = RUN / 'previous-JOB_STATUS.json'
        if not previous.exists(): jobs.atomic(previous, jobs.read(OUT / 'JOB_STATUS.json'))
        jobs.RUN = RUN
        jobs.heartbeat = heartbeat
        jobs.STATE.clear()
        jobs.STATE.update(status='starting', phase='register', source_only=True, target_models_evaluated=[],
            confirmation_or_final_inference=False, maximum_cycle_wall_seconds=7200,
            heartbeat_seconds=30, supervisor_pid=os.getpid(), stages=[],
            started_at_utc=datetime.now(timezone.utc).isoformat())
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT): signal.signal(sig, jobs.interrupted)
        deadline = time.monotonic() + 7200
        if jobs.stage('registration', [jobs.PYTHON, str(OUT / 'envelope_experiment.py'), 'register'], deadline, cpu=True):
            jobs.STATE.update(status='needs_attention', phase='registration_failed'); heartbeat(); return 1
        for cohort in ('pilot', 'validation'):
            if cohort == 'validation' and jobs.read(RUN / 'validation-freeze.json').get('status') != 'validate':
                code = jobs.stage('package-validation', [jobs.PYTHON, str(OUT / 'validate_package.py')], deadline, cpu=True)
                jobs.STATE.update(status='complete' if code == 0 else 'needs_attention', phase='no_candidate_earned_validation',
                    scientific_status='No qualifying improvement; stop this method rather than repeat pilots')
                heartbeat(); return code
            CURRENT = cohort
            commands = [
                ('generation', [jobs.PYTHON, '-u', str(OUT / 'envelope_experiment.py'), 'generate', '--cohort', cohort], True, False),
                ('native-evaluation', [jobs.PYTHON, '-u', str(OUT / 'source_evaluate_envelope.py'),
                    '--name', cohort, '--output-dir', str(RUN / cohort)], False, False),
                ('independent-audit', [jobs.PYTHON, str(RUN / 'audit_envelope.py'), '--cohort', cohort], False, True),
                ('selection' if cohort == 'pilot' else 'report', [jobs.PYTHON, str(OUT / 'postprocess_envelope.py'),
                    '--cohort', cohort], False, True)]
            for stage, command, strict, cpu in commands:
                code = jobs.stage(f'{cohort}-{stage}', command, deadline, strict=strict, cpu=cpu)
                if code:
                    jobs.STATE.update(status='needs_attention', phase=f'{cohort}-{stage}_failed',
                        error='Scheduled stage failed or thetwo-hour continuation ceiling was reached; inspect its receipt')
                    heartbeat(); return code
        code = jobs.stage('package-validation', [jobs.PYTHON, str(OUT / 'validate_package.py')], deadline, cpu=True)
        jobs.STATE.update(status='complete' if code == 0 else 'needs_attention', phase='envelope_validation_finished',
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        heartbeat(); return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        jobs.STATE.update(status='failed', error=str(error), error_type=type(error).__name__)
        heartbeat()
        raise
