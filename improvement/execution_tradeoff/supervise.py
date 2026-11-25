"""Detached bounded cycle; 30-second heartbeats and explicit exit receipts."""
import sys,pathlib,os,subprocess,time,json,fcntl,signal
from datetime import datetime,timezone
D=pathlib.Path('/workspace/improvement/execution_tradeoff');PY='/opt/venv/bin/python';CEILING=2700

def stamp():return datetime.now(timezone.utc).isoformat()
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 start=time.monotonic();lock=pathlib.Path('/workspace/improvement/execution/continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'status':'running','pid':os.getpid(),'started_at_utc':stamp(),'maximum_wall_seconds':CEILING,'heartbeat_seconds':30,'stages':[]};child=None;samples=[]
 def interrupted(signum,frame):raise RuntimeError(f'Supervisor interrupted: {signum}')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  for stage,script,strict in [('generation','pilot.py',True),('evaluation','score.py',False)]:
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','BEYONDWAVEFORM_NOTIFY_DISCORD':'0'}
   if strict:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
   else:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
   entry={'stage':stage,'started_at_utc':stamp(),'status':'running'};state['stages'].append(entry);stage_start=time.monotonic();log=(D/f'{stage}_launcher.log').open('ab')
   child=subprocess.Popen([PY,'-u','/workspace/scripts/run_logged.py','--log',str(D/f'{stage}.log'),'--status',str(D/f'{stage}_job.json'),'--',PY,'-u',str(D/script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd='/workspace');entry['wrapper_pid']=child.pid;last=-30
   while child.poll() is None:
    elapsed=time.monotonic()-start
    if elapsed>CEILING:raise TimeoutError('Declared 2700-second cycle ceiling reached')
    now=time.monotonic()
    if now-last>=30:
     state.update(updated_at_utc=stamp(),elapsed_wall_seconds=elapsed,stage=stage)
     if (D/'progress.json').exists():state['progress']=json.loads((D/'progress.json').read_text())
     try:
      raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip();util,mem,total=map(float,raw.split(','));samples.append({'elapsed_seconds':elapsed,'stage':stage,'utilization_percent':util,'device_memory_mib':mem,'device_total_mib':total});save(D/'gpu_samples.json',samples)
     except Exception as e:state['gpu_sampling_error']=type(e).__name__
     save(D/'status.json',state);print(json.dumps(state),flush=True);last=now
    time.sleep(2)
   code=child.returncode;log.close();entry.update(status='complete' if code==0 else 'failed',exit_code=code,wall_seconds=time.monotonic()-stage_start,finished_at_utc=stamp());save(D/'status.json',state)
   if code:raise RuntimeError(f'{stage} exited {code}')
  state.update(status='complete',exit_code=0)
 except BaseException as error:
  if child is not None and child.poll() is None:
   # Terminate child first so run_logged retains its completion/exit receipt.
   job=D/f'{state.get("stage","generation")}_job.json'
   if job.exists():
    d=json.loads(job.read_text());pid=d.get('child_pid')
    if pid:
     try:os.kill(pid,signal.SIGTERM)
     except ProcessLookupError:pass
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  state.update(status='time_ceiling' if isinstance(error,TimeoutError) else 'failed',exit_code=124 if isinstance(error,TimeoutError) else 1,error_type=type(error).__name__,error=str(error))
 finally:
  state.update(finished_at_utc=stamp(),updated_at_utc=stamp(),elapsed_wall_seconds=time.monotonic()-start);save(D/'status.json',state);save(D/'exit_receipt.json',state)
 return state['exit_code']
if __name__=='__main__':raise SystemExit(main())
