"""Detach-safe continuation of the preserved R3 preflight execution failure."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
import signal

import continue_development as supervisor


def main():
    run, out = supervisor.RUN, supervisor.OUT
    with (run / 'continuation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An existing continuation owns the GPU queue')
        prior = supervisor.read(out / 'JOB_STATUS.json')
        archive = run / 'continuation-first-attempt.json'
        if not archive.exists():
            supervisor.atomic(archive, prior)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, supervisor.interrupted)
        supervisor.STATE.update({
            'status': 'running', 'phase': 'registered_autograd_execution_repair',
            'started_at': datetime.now(timezone.utc).isoformat(),
            'supervisor_pid': os.getpid(), 'supervisor_sha256': supervisor.sha(__file__),
            'helper_sha256': supervisor.sha(out / 'continue_development.py'),
            'prior_attempt_receipt': str(archive), 'stages': prior.get('stages', []),
            'r3_name': 'r3-waveform-autograd-repair',
            'execution_repair': 'Value-preserving positional-convolution input clone; no method/optimizer/loss/budget change'})
        supervisor.heartbeat()
        registration = {
            'registered_at_utc': datetime.now(timezone.utc).isoformat(),
            'reason': 'Actual pinned WavLM backward failed before calibration/training because an inplace addition invalidated the positional Conv1d saved input',
            'same_forward_weights_layer_normalization_loss_optimizer_and_budget': True,
            'preserved_failures': [str(run / 'r3-waveform/preflight.json'),
                                  str(run / 'r3-waveform-reference-repair/run-failure.json')],
            'source_only': True, 'target_models_evaluated': [],
            'files': {str(path): supervisor.sha(path) for path in [
                out / 'r3_encoder_repair.py', out / 'r3_real_runner_v2.py',
                out / 'test_r3_encoder_repair.py', run / 'r3-encoder-repair-tests.log',
                run / 'plan.json', out / 'run_autograd_repair.py']}}
        receipt = run / 'r3-autograd-execution-repair-registration.json'
        if receipt.exists():
            raise RuntimeError('Preserve the registered repair; do not silently restart it')
        supervisor.atomic(receipt, registration)
        code = supervisor.run_stage('r3-waveform-autograd-repair-bounded-run', [
            supervisor.PYTHON, '-u', str(out / 'r3_real_runner_v2.py'), 'run',
            '--name', 'r3-waveform-autograd-repair', '--allow-new-content-reference'], strict=True)
        supervisor.STATE.update(status='complete' if code == 0 else 'needs_attention',
            phase='initial_development_cycle_finished',
            execution_failure_history_preserved=True,
            finished_at=datetime.now(timezone.utc).isoformat())
        supervisor.heartbeat()
        return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        supervisor.STATE.update(status='failed', error=str(error), error_type=type(error).__name__)
        supervisor.heartbeat()
        raise
