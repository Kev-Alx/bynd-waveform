"""Account recorded research-process wall time without counting chat idle time.

Whole detached-job timers take precedence over the timers of nested operations.
This is conservative scheduled time, not a GPU utilization measurement.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/improvement')
OUT = ROOT / 'execution_refine'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def field(value, key):
    for part in key.split('.'):
        value = value[part]
    return value


def main():
    entries = []

    def add(name, rel, key=None, kind='recorded operation wall timer', job=False):
        record = read(rel)
        if job:
            start = datetime.fromisoformat(record['started_at'])
            stop = datetime.fromisoformat(record['finished_at'])
            seconds = (stop - start).total_seconds()
            timer = 'finished_at - started_at'
        else:
            seconds, timer = float(field(record, key)), key
        assert seconds >= 0
        entries.append({'name': name, 'seconds': seconds, 'hours': seconds / 3600,
                        'timer_scope': kind, 'evidence': str(ROOT / rel),
                        'evidence_sha256': sha(ROOT / rel), 'evidence_field': timer})

    baseline_rel = 'execution/baseline-preflight/result.json'
    baseline = read(baseline_rel)
    entries.append({'name': 'Baseline exact-control preflight:150 updates and final waveforms',
                    'seconds': sum(float(r['seconds']) for r in baseline['items']),
                    'timer_scope': 'Sum of3 nonoverlapping clip optimization regions; model/Python startup not recorded',
                    'evidence': str(ROOT / baseline_rel), 'evidence_sha256': sha(ROOT / baseline_rel),
                    'evidence_field': 'sum(items[].seconds)'})
    for name, folder in (
        ('R1 canonical calibration', 'r1-calibration'),
        ('R1 canonical initial3 mechanism check', 'r1-calibrated-pilot3'),
        ('R1 canonical development24 flat-risk stop', 'r1-calibrated-development24'),
        ('R1 deletion calibration', 'r1-deletion-calibration'),
        ('R1 deletion initial3 mechanism check', 'r1-deletion-pilot3'),
        ('R1 deletion development24 generation', 'r1-deletion-development24'),
    ):
        add(name, f'execution/{folder}/result.json', 'seconds',
            'Runner wall timer, including model load; excludes Python import/startup')
    for name, folder in (
        ('R2 initial3 geometry check', 'r2-canonical-zero-uniform-smoke3'),
        ('R2 development24 generation', 'r2-canonical-zero-uniform-dev24'),
    ):
        add(name, f'execution/{folder}/generation-status.json', 'wall_seconds',
            'Runner wall timer including model load; excludes Python import/startup')
    add('R2 native source recognition plus all MOS scoring',
        'execution/r2-canonical-zero-uniform-dev24/source-summary.json', 'resources.elapsed_wall_seconds',
        'Complete evaluator timed region including source recognition and CPU MOS work')
    for folder in (
        'r1-deletion-native-evaluation',
        'r3-native-reference-preflight',
        'r3-waveform-reference-repair-bounded-run',
        'r3-waveform-autograd-repair-bounded-run',
    ):
        add(folder, f'execution/jobs/{folder}/job.json',
            kind='Whole detached child-process wall interval including startup, failures, and nested CPU/source scoring', job=True)
    for folder in ('generation48', 'native-evaluation48'):
        add(folder, f'execution48/jobs/{folder}/job.json',
            kind='Whole detached child-process wall interval including startup and all CPU/source scoring', job=True)
    for row in entries:
        row['hours'] = row['seconds'] / 3600
    seconds = sum(r['seconds'] for r in entries)
    original_seconds = sum(r['seconds'] for r in entries if '/execution48/' not in r['evidence'])
    r3 = read('execution/r3-waveform-autograd-repair/training.json')
    gaps = [
        'Some early baseline/R1/R2 runs have operation-region timers rather than externally recorded whole-process start/stop; Python imports/startup and baseline model loading are not fully covered.',
        'An early R1 failure was rerun with logging-only receipt changes; only the currently preserved result timer is counted. Any overwritten earlier invocation timer cannot be recovered from these JSON receipts.',
        'The scope is newly produced improvement/execution and improvement/execution48 research work, not the older historical reproduction-runs study or unrecorded commands.',
    ]
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'recorded_time_reconciled_with_explicit_coverage_gaps',
        'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'definition': 'Sum nonoverlapping recorded process/operation wall intervals for scheduled GPU research jobs. Charge all source inference and CPU MOS wall time inside those jobs. Chat/limit idle gaps do not count. This is not profiler-measured GPU active time.',
        'entries': entries, 'recorded_scheduled_seconds': seconds,
        'recorded_scheduled_hours': seconds / 3600,
        'initial_development24_and_r3_seconds': original_seconds,
        'pilot48_cycle_seconds': seconds - original_seconds,
        'original_six_hour_envelope_seconds': 21600,
        'six_hour_envelope_minus_recorded_seconds': 21600 - seconds,
        'budget_caveat': 'This subtraction is recorded headroom, not a certified remaining balance: coverage gaps prevent an exact historical total. Reserve additional time for unknown early startup/repeated attempts. Later jobs need explicit detached whole-process timers.',
        'coverage_gaps': gaps,
        'nested_timers_excluded_from_sum': [
            {'path': str(ROOT / 'execution/r3-waveform-autograd-repair/training.json'),
             'training_wall_seconds': r3['training_wall_seconds'],
             'reason': 'Contained in whole R3 bounded-run process; includes3 checkpoint source/MOS evaluations. Never add the one-hour limit or nested checkpoint durations again.'},
            {'path': str(ROOT / 'execution/r3-waveform-autograd-repair/source-summary.json'),
             'reason': 'Final evaluation already inside whole R3 process.'},
            {'path': str(ROOT / 'execution/r3-waveform-autograd-repair/preflight.json'),
             'reason': 'Successful encoder conformance/calibration/fit already inside whole R3 process.'},
            {'path': str(ROOT / 'execution/r1-deletion-development24/source-summary.json'),
             'reason': 'Already inside whole R1 native-evaluation process.'},
            {'path': str(ROOT / 'execution48/mechanism-pilot48/generation-result.json'),
             'reason': 'Already inside generation48 process.'},
            {'path': str(ROOT / 'execution48/mechanism-pilot48/source-summary.json'),
             'reason': 'Already inside native-evaluation48 process.'},
        ],
        'cpu_only_operations_excluded': ['Dataset downloads/extraction/metadata selection', 'CPU unit tests and artifact audits', 'Report generation and package validation', 'Chat idle time and user-limit interruption'],
        'uncounted_configured_caps': ['6-hour initial research envelope', '2-hour pilot48 cycle cap', '1-hour R3 training ceiling; count actual whole process instead'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'compute-ledger.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = '\n'.join(f"| {r['name']} | {r['seconds']:.3f} | {r['timer_scope']} |" for r in entries)
    note = f'''# Recorded research compute ledger

Recorded scheduled time totals **{seconds / 3600:.3f} hours ({seconds:.3f} seconds)**: {original_seconds / 3600:.3f} hours for the original development24/R3 work and {(seconds - original_seconds) / 3600:.3f} hours for the later48-utterance generation/evaluation. The original six-hour envelope minus this recorded subtotal is **{(21600 - seconds) / 3600:.3f} hours**. This is recorded headroom, not a certified exact balance, because early timer coverage is incomplete.

The ledger sums actual run timers rather than elapsed time between chats. Whole detached process intervals take precedence over nested timers. Source recognition and all CPU MOS time count as scheduled job time, even when the GPU is idle. R3's entire detached process is counted once; its {r3['training_wall_seconds']:.3f}-second training region and nested checkpoint/final evaluations are not added again. Configuration caps are not treated as actual compute consumed.

| Job | Recorded seconds | Timer coverage |
|---|---:|---|
{rows}

Some early baseline/R1/R2 receipts omit Python startup and the baseline model load. One early R1 failure had a logging-only rerun whose earlier timer was not preserved separately. These gaps prevent a strict exact-total claim; reserve additional allowance for them. No idle gap caused by the user's limit is charged. Historical reproduction runs outside improvement/, CPU data preparation, unit tests, audits and report rendering are outside this GPU research ledger.

The [JSON ledger](/workspace/improvement/execution_refine/compute-ledger.json) binds each evidence file by SHA256, specifies the field used, and lists nested receipts excluded to avoid double counting. [Rebuild script](/workspace/improvement/execution_refine/build_compute_ledger.py).
'''
    (OUT / 'compute-ledger.md').write_text(note)
    print(json.dumps({k: result[k] for k in ('status', 'recorded_scheduled_seconds', 'recorded_scheduled_hours', 'six_hour_envelope_minus_recorded_seconds')}, indent=2))


if __name__ == '__main__':
    main()
