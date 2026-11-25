#!/usr/bin/env python3
"""One sequential, bounded GPU queue with durable status and no notifications."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'runtime'),str(ROOT.parent)]
from defense_lab.common import configure_runtime,atomic_json,stamp
configure_runtime()


def main():
    folder=ROOT/'runs/pilot-v1'
    lock=(folder/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'status':'running','started_at':stamp(),'supervisor_pid':os.getpid(),
            'phase':'waiting-for-inference48','stages':[],'maximum_seconds':7200,
            'notifications':False}
    started=time.time()
    def save():
        status['updated_at']=stamp();status['elapsed_seconds']=time.time()-started
        atomic_json(ROOT/'JOB_STATUS.json',status)
        text=['# Defense job status','',f"Status: **{status['status']}**",f"Phase: **{status['phase']}**",
              f"Updated: {status['updated_at']}",f"Supervisor PID: {status['supervisor_pid']}",'']
        for stage in status['stages']:text.append(f"- {stage['name']}: {stage['status']} (exit {stage.get('exit_code','pending')})")
        (ROOT/'JOB_STATUS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save()
    wait_started=time.time()
    while True:
        path=folder/'inference48/summary.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':break
        if time.time()-wait_started>900:
            status.update(status='failed',phase='inference48-not-complete');save();return 1
        time.sleep(15);save()
    stages=[('clean-ft',['train','--condition','clean_ft']),
            ('reconstruction-ft',['train','--condition','reconstruction_ft']),
            ('union-at',['train','--condition','union_at']),
            ('adaptive',['adaptive']),('report',['report'])]
    for name,args in stages:
        status['phase']=name
        log=folder/'jobs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        command=['/opt/venv/bin/python','-u',str(ROOT/'run.py'),*args]
        stage={'name':name,'status':'running','started_at':stamp(),'command':command,'log':str(log)}
        status['stages'].append(stage)
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd='/workspace',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   env=os.environ.copy(),start_new_session=True)
            stage['pid']=child.pid;save()
            while child.poll() is None:
                if time.time()-started>status['maximum_seconds']:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    stage.update(status='time-limit',exit_code=child.returncode)
                    status.update(status='time-limit',phase=name);save();return 2
                time.sleep(15);save()
            stage.update(status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode,finished_at=stamp())
            save()
            if child.returncode:
                status.update(status='failed',phase=name);save();return child.returncode
    status.update(status='complete',phase='pilot-complete',finished_at=stamp());save();return 0


if __name__=='__main__':
    raise SystemExit(main())
