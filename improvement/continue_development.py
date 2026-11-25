"""Detached, source-only continuation queue for the already registered trials.

No scientific implementation or frozen input is changed by this supervisor.
One process owns the queue lock, and one GPU stage runs at a time. Every stage
uses the repository's existing run_logged wrapper, with notifications disabled.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/workspace')
OUT = ROOT / 'improvement'
RUN = OUT / 'execution'
PYTHON = '/opt/venv/bin/python'
TAIPEI = timezone(timedelta(hours=8))
ACTIVE = None
STATE = {'status': 'starting', 'phase': 'resume', 'source_only': True,
         'target_models_evaluated': [], 'confirmation_or_final_inference': False,
         'maximum_initial_gpu_hours': 6, 'maximum_r3_training_seconds': 3600,
         'supervisor_pid': os.getpid(), 'stages': [], 'heartbeat_seconds': 30}


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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heartbeat():
    STATE['updated_at'] = datetime.now(TAIPEI).isoformat(timespec='seconds')
    STATE['timezone'] = 'Asia/Taipei'
    STATE['r1_generation'] = {key: value for key, value in
        read(RUN / 'r1-deletion-development24/result.json').items()
        if key in ('status', 'actual_count', 'planned_count', 'seconds')}
    STATE['r1_native_summary'] = read(RUN / 'r1-deletion-development24/source-summary.json').get('summary')
    STATE['r2_native_summary'] = read(RUN / 'r2-canonical-zero-uniform-dev24/source-summary.json').get('summary')
    STATE['r3_status'] = read(RUN / STATE.get('r3_name', 'r3-waveform') / 'status.json')
    STATE['r3_training'] = read(RUN / STATE.get('r3_name', 'r3-waveform') / 'training.json')
    STATE['r3_native_summary'] = read(RUN / STATE.get('r3_name', 'r3-waveform') / 'source-summary.json').get('summary')
    STATE['completed_r3_training_microbatches'] = 0
    trace = RUN / STATE.get('r3_name', 'r3-waveform') / 'training-trace.jsonl'
    if trace.exists():
        with trace.open() as stream:
            for line in stream:
                if line.strip():
                    try:
                        STATE['latest_r3_microbatch'] = json.loads(line)
                        STATE['completed_r3_training_microbatches'] += 1
                    except ValueError:
                        pass  # An append may be in progress; inspect again next heartbeat.
    atomic(RUN / 'progress.json', STATE)
    atomic(OUT / 'JOB_STATUS.json', STATE)
    lines = ['# Improvement experiment status', '',
        f"Updated: **{STATE['updated_at']} (Asia/Taipei)**. Status: **{STATE['status']}**.", '',
        f"Current phase: **{STATE['phase']}**. Supervisor PID: {os.getpid()}.", '',
        'The supervisor is detached from the chat terminal. Updates every 30 seconds while running.', '',
        '| Stage | State | Exit code |', '|---|---|---:|']
    for stage in STATE['stages']:
        lines.append(f"| {stage['name']} | {stage['status']} | {stage.get('exit_code', '—')} |")
    latest = STATE.get('latest_r3_microbatch')
    if latest:
        lines += ['', f"R3: {latest['updates']} optimizer updates, {latest['microbatches']} microbatches; "
                  f"{latest['elapsed_training_seconds']:.1f} measured training seconds at last completed microbatch."]
    lines += ['', 'All phases use development data and the source recognizer only. '
        'Confirmation48, final200, held-out recognizers, and human studies remain unused.', '',
        f'[Detailed status]({OUT}/JOB_STATUS.json) · [Execution report]({OUT}/05_execution_report.md)', '']
    destination = OUT / 'JOB_STATUS.md'
    temporary = destination.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(destination)


def run_stage(name, command, strict=False):
    global ACTIVE
    directory = RUN / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'job.json'
    if read(status_path).get('status') == 'complete':
        raise RuntimeError(f'Preserve completed job {name}; do not launch it twice')
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       BEYONDWAVEFORM_NOTIFY_DISCORD='0')
    environment.pop('CUBLAS_WORKSPACE_CONFIG', None)
    if strict:
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    wrapped = [PYTHON, '-u', str(ROOT / 'scripts/run_logged.py'),
               '--log', str(directory / 'stage.log'), '--status', str(status_path), '--', *command]
    record = {'name': name, 'status': 'running', 'command': command,
              'strict_generation_policy': strict, 'job_file': str(status_path),
              'started_at': datetime.now(timezone.utc).isoformat()}
    STATE['stages'].append(record)
    STATE.update(status='running', phase=name)
    heartbeat()
    print(f'START {name}', flush=True)
    with (directory / 'supervisor-output.log').open('ab') as log:
        ACTIVE = subprocess.Popen(wrapped, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, env=environment)
        record['wrapper_pid'] = ACTIVE.pid
        while True:
            try:
                code = ACTIVE.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                heartbeat()
        ACTIVE = None
    record.update(status='complete' if code == 0 else 'failed', exit_code=code,
                  finished_at=datetime.now(timezone.utc).isoformat())
    record['wrapper_receipt'] = read(status_path)
    heartbeat()
    print(f'FINISH {name} exit={code}', flush=True)
    return code


def interrupted(signum, _frame):
    STATE.update(status='interrupted', signal=signum)
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
        try:
            ACTIVE.wait(timeout=15)
        except subprocess.TimeoutExpired:
            ACTIVE.kill()
            ACTIVE.wait()
    heartbeat()
    raise SystemExit(128 + signum)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'continuation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Continuation already running; no duplicate launched')
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
        STATE['started_at'] = datetime.now(timezone.utc).isoformat()
        STATE['supervisor_sha256'] = sha(__file__)
        STATE['r3_name'] = 'r3-waveform'
        heartbeat()
        generation = read(RUN / 'r1-deletion-development24/result.json')
        if generation.get('status') != 'completed' or generation.get('actual_count') != 24:
            raise RuntimeError('R1 full24 generation must be complete before this resume queue')
        if not read(RUN / 'r1-deletion-development24/source-summary.json').get('status') == 'complete':
            run_stage('r1-deletion-native-evaluation', [PYTHON, '-u', str(OUT / 'source_evaluate.py'),
                      '--name', 'r1-deletion-development24'])
        else:
            STATE['stages'].append({'name': 'r1-deletion-native-evaluation', 'status': 'previously_complete'})
        # Default cache-contract conformance must be attempted and preserved.
        canonical = RUN / 'r3-waveform'
        previous = read(canonical / 'preflight.json')
        if not previous:
            preflight_code = run_stage('r3-native-reference-preflight',
                [PYTHON, '-u', str(OUT / 'r3_real_runner.py'), 'preflight'], strict=True)
        else:
            preflight_code = 0 if previous.get('status') == 'passed' else 1
        if preflight_code:
            failed = read(canonical / 'preflight.json')
            items = failed.get('items', [])
            repair_allowed = (failed.get('mechanism') == 'cached content representation mismatch'
                and failed.get('calibration_executed') is False and failed.get('training_executed') is False
                and len(items) == 3 and all(item.get('native_adapter_bitwise_equal')
                    and item.get('strict_repeat_bitwise_equal') for item in items))
            if not repair_allowed:
                STATE.update(status='needs_attention', phase='r3_preflight_failed_no_registered_repair')
                heartbeat()
                return 1
            STATE['r3_name'] = 'r3-waveform-reference-repair'
            STATE['r3_reference_repair_reason'] = 'cached-reference mismatch; same native/adapter strict values and repeated reference verified'
        name = STATE['r3_name']
        command = [PYTHON, '-u', str(OUT / 'r3_real_runner.py'), 'run', '--name', name]
        if name == 'r3-waveform-reference-repair':
            command.append('--allow-new-content-reference')
        if not read(RUN / name / 'source-summary.json').get('status') == 'complete':
            code = run_stage(name + '-bounded-run', command, strict=True)
        else:
            code = 0
        failures = [stage for stage in STATE['stages'] if stage['status'] == 'failed'
                    and stage['name'] != 'r3-native-reference-preflight']
        STATE.update(status='complete' if code == 0 and not failures else 'needs_attention',
                     phase='initial_development_cycle_finished',
                     finished_at=datetime.now(timezone.utc).isoformat())
        heartbeat()
        return 0 if STATE['status'] == 'complete' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        STATE.update(status='failed', error=str(error), error_type=type(error).__name__)
        heartbeat()
        raise
