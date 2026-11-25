"""Detached continuation: baseline -> prespecified development probe -> test.

Stops at any failed numerical/integrity gate. Never changes an attack setting in
response to the final test result. Each GPU stage gets its own detached session.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reproduction-runs/mandarin-cer-corrected'


def read(path):
    return json.loads(path.read_text())


def write(state):
    state.update(updated_at=datetime.now(timezone.utc).isoformat(),sequence_pid=os.getpid())
    temp=OUT/'sequence.json.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(OUT/'sequence.json')


def wait_job(pid):
    while True:
        job=read(OUT/'job.json')
        if job['supervisor_pid']!=pid:
            raise RuntimeError('Tracked job was replaced; refusing to follow an unrelated job')
        if job['status'] in ['complete','failed','interrupted','stopped']:
            if job['status']!='complete' or job.get('exit_code')!=0:
                raise RuntimeError(f"Tracked stage ended with {job['status']}")
            return
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
            # Allow the supervisor one polling interval to publish its final
            # receipt, without restarting or inferring success from a timeout.
            time.sleep(10)
            terminal=read(OUT/'job.json')
            if terminal.get('supervisor_pid')==pid and terminal.get('status')=='complete' and terminal.get('exit_code')==0:
                return
            raise RuntimeError('Tracked supervisor is no longer live without a successful completion receipt')
        time.sleep(10)


def audit(stage,candidate):
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_mandarin_redo.py'),'--stage',stage,'--candidate',candidate],cwd=ROOT,check=True)
    return read(OUT/stage/candidate/'independent-audit.json')


def start(stage,candidate):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/start_mandarin_redo.py'),'--stage',stage,'--candidate',candidate],cwd=ROOT,check=True,capture_output=True,text=True)
    receipt=json.loads(result.stdout)
    assert receipt.get('detached') is True,receipt
    pid=receipt['supervisor_pid']
    for _ in range(30):
        if read(OUT/'job.json').get('supervisor_pid')==pid:
            return pid
        time.sleep(1)
    raise RuntimeError('Detached stage did not publish its supervisor receipt; no duplicate started')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',required=True,type=int)
    args=parser.parse_args()
    state={'status':'running','phase':'waiting-for-full-clean-baseline','tracked_supervisor_pid':args.wait_for,
           'next_development_candidate':'paper-lr025','final_test_count':5000,
           'stopping_rule':'Full clean baseline and 48-clip development pilot must meet declared tolerances before final generation. Final test results are never used to tune settings.'}
    write(state)
    try:
        wait_job(args.wait_for)
        baseline=audit('test-clean','paper')
        if not baseline['paper_comparison']['clean']['within_declared_tolerance']:
            state.update(status='needs_review',phase='full-clean-baseline-outside-tolerance',baseline=baseline)
            write(state);return
        pid=start('pilot','paper-lr025')
        state.update(phase='running-prespecified-development-probe',tracked_supervisor_pid=pid)
        write(state)
        wait_job(pid)
        pilot=audit('pilot','paper-lr025')
        if not all(v['within_declared_tolerance'] for v in pilot['paper_comparison'].values()):
            state.update(status='needs_review',phase='development-probe-outside-tolerance',pilot=pilot)
            write(state);return
        # Freeze this development choice before any attacked final-test output
        # is decoded. The full clean baseline is a health gate, not a fitting
        # target for the optimizer parameter.
        promotion={'candidate':'paper-lr025','selection_data':'fixed 48 development clips only',
                   'pilot_audit':pilot,'baseline_health_audit':baseline,
                   'frozen_at':datetime.now(timezone.utc).isoformat()}
        path=OUT/'promotion.json'
        if path.exists():
            raise RuntimeError('A promotion already exists; manual review required before replacing it')
        path.write_text(json.dumps(promotion,indent=2)+'\n')
        pid=start('test','paper-lr025')
        state.update(phase='running-frozen-5000-record-test',tracked_supervisor_pid=pid,promotion=str(path))
        write(state)
        wait_job(pid)
        final=audit('test','paper-lr025')
        met=all(v['within_declared_tolerance'] for v in final['paper_comparison'].values())
        state.update(status='criteria_met' if met else 'needs_review',phase='final-test-verified',final_audit=final)
        write(state)
    except BaseException as error:
        state.update(status='failed',error=str(error))
        write(state)
        raise


if __name__=='__main__':
    main()
