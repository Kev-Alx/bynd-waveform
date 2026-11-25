"""Detached reporting continuation: wait for the bounded experiment, then verify delivery."""
import pathlib,subprocess,json,os,time,sys
from datetime import datetime,timezone
D=pathlib.Path('/workspace/improvement/execution_tradeoff');start=time.monotonic();P=D.parent;PY='/opt/venv/bin/python'
def stamp():return datetime.now(timezone.utc).isoformat()
def save(state):
 state['updated_at_utc']=stamp();t=D/'delivery_status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(D/'delivery_status.json')
def main():
 state={'status':'waiting_for_experiment','started_at_utc':stamp(),'pid':os.getpid(),'maximum_delivery_wall_seconds':3600,'heartbeat_seconds':30,'stages':[]};save(state)
 try:
  while not (D/'exit_receipt.json').exists():
   if time.monotonic()-start>3000:raise TimeoutError('Experiment wait ceiling reached')
   state['experiment']=json.loads((D/'status.json').read_text());save(state);time.sleep(30)
  exp=json.loads((D/'exit_receipt.json').read_text());assert exp['status']=='complete',f'Experiment ended {exp["status"]}; preserve failure and do not publish partial scores'
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','BEYONDWAVEFORM_NOTIFY_DISCORD':'0'};env.pop('CUBLAS_WORKSPACE_CONFIG',None)
  qa=P/'audio_browser_qa';(D/'qa_tmp').mkdir(exist_ok=True);env.update(PLAYWRIGHT_BROWSERS_PATH=str(qa/'browsers'),LD_LIBRARY_PATH=str(qa/'libs/usr/lib/x86_64-linux-gnu'),TMPDIR=str(D/'qa_tmp'))
  tasks=[('analysis',[PY,str(D/'analyze.py')]),('prepared_data_forecast',[PY,str(D/'forecast_prepared_data.py')]),('report',[PY,str(D/'write_report.py')]),('reviewed_catalogue',[PY,str(D/'refresh_interface.py')]),('atomic_interface_build',[PY,str(D/'build_interface.py')]),('browser_QA',['/usr/bin/node',str(D/'check_browser.cjs')])]
  for label,cmd in tasks:
   state.update(status='running',stage=label);entry={'stage':label,'started_at_utc':stamp(),'command':cmd};state['stages'].append(entry);save(state);t=time.monotonic()
   with (D/f'delivery_{label}.log').open('ab') as log:
    child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd='/workspace');entry['pid']=child.pid
    while child.poll() is None:
     if time.monotonic()-start>3600 or time.monotonic()-t>600:child.terminate();child.wait(timeout=15);raise TimeoutError(f'{label} stage ceiling reached')
     save(state);time.sleep(3)
   entry.update(exit_code=child.returncode,finished_at_utc=stamp(),wall_seconds=time.monotonic()-t);save(state)
   if child.returncode:raise RuntimeError(f'{label} exited {child.returncode}; see persistent stage log')
  state.update(status='complete',exit_code=0)
 except BaseException as error:state.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error))
 state.update(finished_at_utc=stamp(),elapsed_wall_seconds=time.monotonic()-start);save(state);(D/'delivery_exit_receipt.json').write_text(json.dumps(state,indent=2)+'\n');return state['exit_code']
if __name__=='__main__':raise SystemExit(main())
