"""Start one corrected AISHELL stage in a detached process session."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction-runs/mandarin-cer-corrected'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['pilot-clean', 'pilot', 'test-clean', 'test'])
    parser.add_argument('--candidate', default='selected', choices=['selected', 'paper', 'paper-lr025'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    job_file = OUT/'job.json'
    if job_file.exists():
        prior = json.loads(job_file.read_text())
        if prior['status'] not in ['complete','failed','interrupted','stopped']:
            pid = prior.get('supervisor_pid')
            print(json.dumps({'status':'already_running','supervisor_pid':pid,
                              'live_process':Path(f'/proc/{pid}/stat').exists()}))
            return
        previous_command = prior.get('command',[])
        previous_stage = previous_command[previous_command.index('--stage')+1] if '--stage' in previous_command else 'unknown'
        previous_candidate = previous_command[previous_command.index('--candidate')+1] if '--candidate' in previous_command else 'unknown'
        history = OUT/previous_stage/previous_candidate/'supervisor-job.json'
        history.parent.mkdir(parents=True,exist_ok=True)
        history.write_text(json.dumps(prior,indent=2)+'\n')
    command = ['/opt/venv/bin/python','-u',str(ROOT/'scripts/run_logged.py'),
               '--log',str(OUT/'suite.log'),'--status',str(job_file),'--',
               '/opt/venv/bin/python','-u',str(ROOT/'scripts/run_mandarin_redo.py'),
               '--stage',args.stage,'--candidate',args.candidate]
    env = {**os.environ,'BEYONDWAVEFORM_NOTIFY_DISCORD':'0','OMP_NUM_THREADS':'8',
           'MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    with (OUT/'launcher.log').open('ab') as log:
        child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
    sid = os.getsid(child.pid)
    receipt = {'launched_at':datetime.now(timezone.utc).isoformat(),'supervisor_pid':child.pid,
               'session_id':sid,'detached':sid==child.pid,'command':command,
               'stage':args.stage,'candidate':args.candidate}
    assert receipt['detached']
    with (OUT/'launches.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
