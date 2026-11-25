"""Detached sequential GPU queue, heartbeat and postprocessing for pilot48."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OUT = Path('/workspace/improvement')
RUN = OUT / 'execution48'
ROOT = OUT.parent
FOLDER = RUN / 'mechanism-pilot48'
PYTHON = '/opt/venv/bin/python'
STATE = {'status': 'starting', 'phase': 'register_pilot48', 'pilot_count': 48,
    'source_only': True, 'target_models_evaluated': [], 'confirmation_or_final_inference': False,
    'heartbeat_seconds': 30, 'maximum_cycle_wall_seconds': 7200,
    'supervisor_pid': os.getpid(), 'stages': [],
    'historical_failures': 'Earlier R3 cache/autograd preflight failures were resolved; preserved in execution/. Previous R1/R2/R3 scientific screen failures remain negative results.'}
ACTIVE = None


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def heartbeat():
    STATE['updated_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    STATE['timezone'] = 'Asia/Taipei'
    STATE['generation'] = read(FOLDER / 'generation-status.json')
    STATE['scored_variant_waveforms'] = len(list((FOLDER / 'score-records').glob('*.json')))
    STATE['total_variant_waveforms'] = 48 * 7
    STATE['scientific_results'] = read(RUN / 'comparison.json').get('findings')
    STATE['independent_audit_status'] = read(RUN / 'audit-pilot48.json').get('status', 'pending')
    atomic(RUN / 'progress.json', STATE)
    atomic(OUT / 'JOB_STATUS.json', STATE)
    progress = STATE['generation']
    lines = ['#48-point pilot job status', '', f"Updated **{STATE['updated_at']} (Asia/Taipei)**. "
        f"Status: **{STATE['status']}**. Current phase: **{STATE['phase']}**.", '',
        'Detached process with no controlling terminal; updates every30 seconds while running. '
        'All pilot conditions contain exactly48 paired utterances.', '',
        '| Stage | Status | Exit code |', '|---|---|---:|']
    for stage in STATE['stages']:
        lines.append(f"| {stage['name']} | {stage['status']} | {stage.get('exit_code', '—')} |")
    if progress:
        lines += ['', 'Generation: ' + ', '.join(f'{k}={progress[k]}' for k in
            ('completed', 'clip', 'total', 'id', 'variant', 'update') if k in progress) + '.']
    lines += ['', f"Scored variant waveforms: {STATE['scored_variant_waveforms']}/336. "
        f"Independent audit: {STATE['independent_audit_status']}.", '', STATE['historical_failures'], '',
        'No confirmation/final evaluation or held-out recognizer loading is permitted.', '',
        f'[Protocol]({OUT}/07_pilot48_protocol.md) · [48-point results]({OUT}/08_pilot48_results.md) · '
        f'[Historical24-point results]({OUT}/06_initial_results.md)', '']
    temporary = OUT / 'JOB_STATUS.md.tmp'
    temporary.write_text('\n'.join(lines).replace('#48-point', '# 48-point'))
    temporary.replace(OUT / 'JOB_STATUS.md')


def terminate_active():
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
        try:
            ACTIVE.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ACTIVE.kill()
            ACTIVE.wait()


def interrupted(signum, _frame):
    STATE.update(status='interrupted', signal=signum)
    terminate_active()
    heartbeat()
    raise SystemExit(128 + signum)


def stage(name, command, deadline, *, strict=False, cpu=False):
    global ACTIVE
    folder = RUN / 'jobs' / name
    folder.mkdir(parents=True, exist_ok=True)
    job = folder / 'job.json'
    if job.exists():
        raise FileExistsError('Preserve prior stage receipts; do not silently launch twice')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', BEYONDWAVEFORM_NOTIFY_DISCORD='0')
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    if strict:
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    wrapped = [PYTHON, '-u', str(ROOT / 'scripts/run_logged.py'),
        '--log', str(folder / 'stage.log'), '--status', str(job), '--', *command]
    value = {'name': name, 'status': 'running', 'command': command,
        'job_file': str(job), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    STATE['stages'].append(value)
    STATE.update(status='running', phase=name)
    heartbeat()
    with (folder / 'supervisor-output.log').open('ab') as stream:
        ACTIVE = subprocess.Popen(wrapped, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, env=env)
        value['wrapper_pid'] = ACTIVE.pid
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_active()
                value.update(status='time_limit', exit_code=124)
                STATE.update(status='needs_attention', error='Two-hour cycle ceiling reached')
                heartbeat()
                return 124
            try:
                code = ACTIVE.wait(timeout=min(30, remaining))
                break
            except subprocess.TimeoutExpired:
                heartbeat()
        ACTIVE = None
    value.update(status='complete' if code == 0 else 'failed', exit_code=code,
                 finished_at_utc=datetime.now(timezone.utc).isoformat(), receipt=read(job))
    heartbeat()
    return code


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution/continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = RUN / 'previous-JOB_STATUS.json'
        if not previous.exists():
            atomic(previous, read(OUT / 'JOB_STATUS.json'))
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
        deadline = time.monotonic() + 7200
        STATE['started_at_utc'] = datetime.now(timezone.utc).isoformat()
        commands = [
            ('registration48', [PYTHON, str(OUT / 'pilot48_runner.py'), 'register'], False, True),
            ('generation48', [PYTHON, '-u', str(OUT / 'pilot48_runner.py'), 'generate'], True, False),
            ('native-evaluation48', [PYTHON, '-u', str(OUT / 'source_evaluate48.py'),
                '--name', 'mechanism-pilot48', '--output-dir', str(FOLDER)], False, False),
            ('independent-audit48', [PYTHON, str(RUN / 'audit_pilot48.py')], False, True),
            ('report48', [PYTHON, str(OUT / 'postprocess_pilot48.py')], False, True),
            ('package-validation48', [PYTHON, str(OUT / 'validate_package.py')], False, True)]
        for name, command, strict, cpu in commands:
            code = stage(name, command, deadline, strict=strict, cpu=cpu)
            if code:
                STATE.update(status='needs_attention', phase=name + '_failed')
                heartbeat()
                return code
        STATE.update(status='complete', phase='pilot48_cycle_finished',
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        heartbeat()
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        STATE.update(status='failed', error=str(error), error_type=type(error).__name__)
        heartbeat()
        raise
