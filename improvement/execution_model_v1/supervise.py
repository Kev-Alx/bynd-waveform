"""Single detached queue with13-hour overall ceiling, real locks and gated stages."""
import pathlib,os,sys,time,json,subprocess,fcntl,signal,math
from datetime import datetime,timezone
D=pathlib.Path(__file__).resolve().parent;P=D.parent;PY='/opt/venv/bin/python';MAX=46800

def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
class BudgetStop(Exception):pass

def main():
 started=time.monotonic();deadline=started+MAX;state={'status':'queued','pid':os.getpid(),'started_at_utc':stamp(),'deadline_at_utc':datetime.fromtimestamp(time.time()+MAX,timezone.utc).isoformat(),'maximum_overall_seconds':MAX,'queue_counts_against_ceiling':True,'heartbeat_seconds':30,'stages':[],'queue_wait_seconds':0,'active_wall_seconds':0};child=None;gpu_locks=[];samples=[];active_start=None;gpu_active=False;queue_finished=False;last=-30
 def heartbeat():
  nonlocal last
  now=time.monotonic();state.update(updated_at_utc=stamp(),overall_wall_seconds=now-started)
  if gpu_active:state['active_wall_seconds']=now-active_start
  elif not queue_finished:state['queue_wait_seconds']=now-started
  if (D/'progress.json').exists():state['progress']=read(D/'progress.json')
  if now-last>=30:
   try:
    data=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip();util,mem=map(float,data.split(','));sample={'updated_at_utc':stamp(),'elapsed_seconds':now-started,'owned_gpu_phase':bool(gpu_active),'stage':state.get('stage','queue'),'utilization_percent':util,'memory_used_mib':mem};samples.append(sample);save(D/'gpu_samples.json',samples);state['gpu']=sample
   except Exception as e:state['gpu_sampling_error']=type(e).__name__
   print(json.dumps({k:v for k,v in state.items() if k!='stages'}),flush=True);last=now
  save(D/'status.json',state)
 def remaining():return deadline-time.monotonic()-30
 def stage(name,script,args=(),strict=True,cpu=False,cap=None):
  nonlocal child
  if remaining()<60:raise BudgetStop('Insufficient total time to start another complete stage')
  limit=min(remaining(),cap if cap else remaining());entry={'name':name,'status':'running','started_at_utc':stamp(),'maximum_stage_seconds':limit};state['stages'].append(entry);state.update(status='running',stage=name);env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','BEYONDWAVEFORM_NOTIFY_DISCORD':'0'}
  (D/'cache').mkdir(exist_ok=True);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(D/'cache/huggingface'),XDG_CACHE_HOME=str(D/'cache'),MPLCONFIGDIR=str(D/'cache/matplotlib'))
  if strict and not cpu:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
  else:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
  if cpu:
   env['CUDA_VISIBLE_DEVICES']='';env['PLAYWRIGHT_BROWSERS_PATH']='/workspace/improvement/audio_browser_qa/browsers';env['LD_LIBRARY_PATH']='/workspace/improvement/audio_browser_qa/libs/usr/lib/x86_64-linux-gnu';(D/'qa_tmp').mkdir(exist_ok=True);env['TMPDIR']=str(D/'qa_tmp')
  command=[PY,'-u',str(D/script),*args] if script.endswith('.py') else ['/usr/bin/node',str(D/script),*args];entry['command']=command;folder=D/'jobs'/name;folder.mkdir(parents=True,exist_ok=False);log=(folder/'launcher.log').open('ab');child=subprocess.Popen([PY,'-u','/workspace/scripts/run_logged.py','--log',str(folder/'stage.log'),'--status',str(folder/'exit.json'),'--',*command],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd='/workspace');entry['wrapper_pid']=child.pid;phase_start=time.monotonic()
  while child.poll() is None:
   heartbeat()
   if time.monotonic()-phase_start>limit or remaining()<=0:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    entry.update(status='time_ceiling',exit_code=124,finished_at_utc=stamp(),wall_seconds=time.monotonic()-phase_start);log.close();raise BudgetStop(f'{name} reached its declared ceiling')
   time.sleep(3)
  log.close();entry.update(status='complete' if child.returncode==0 else 'failed_execution',exit_code=child.returncode,finished_at_utc=stamp(),wall_seconds=time.monotonic()-phase_start);heartbeat()
  if child.returncode:raise RuntimeError(f'{name} exited{child.returncode}; see persistent stage/exit logs')
 def interrupted(sig,frame):raise RuntimeError(f'Controller interrupted by signal{sig}')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
 own=(D/'controller.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (D/'status.json').exists(),'Preserve the previous queue';heartbeat()
 try:
  # Existing external lock is opened without creating/modifying an outside artifact.
  paths=[P.parent/'reproduction-runs/auxiliary-gpu.lock',P/'execution/continuation.lock'];quiet=0
  while True:
   if time.monotonic()-started>14400:raise BudgetStop('Four-hour GPU queue subceiling reached')
   acquired=[]
   try:
    for p in paths:
     f=p.open('r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired.append(f)
   except (BlockingIOError,OSError):
    for f in acquired:f.close()
    if 'f' in locals() and not f.closed:f.close()
    quiet=0;heartbeat();time.sleep(15);continue
   data=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip();util,mem=map(float,data.split(','));quiet=quiet+1 if util<=5 and mem<128 else 0;heartbeat()
   if quiet>=3:gpu_locks=acquired;break
   for f in acquired:f.close()
   time.sleep(15)
  state['queue_wait_seconds']=time.monotonic()-started;active_start=time.monotonic();gpu_active=True;queue_finished=True;state['gpu_acquired_at_utc']=stamp();heartbeat()
  stage('engineering-profile','feature_runtime.py',['profile'],cap=900);profile=read(D/'profile.json');assert profile['status']=='pass'
  forecast_need=profile['forecast_feature_preprocessing_seconds']*1.5+profile['forecast_warmup_seconds']*1.3+profile['forecast_two_training_arms_seconds']*1.3+profile['forecast_pilot_scoring_seconds']+600
  save(D/'admission_forecast.json',{'created_at_utc':stamp(),'measured_profile':str(D/'profile.json'),'forecast_initial_experiment_seconds_with_margins':forecast_need,'remaining_overall_seconds':remaining(),'training_admitted':forecast_need<remaining(),'no_performance_claim_from_engineering':True})
  if forecast_need>=remaining():raise BudgetStop('Measured initial experiment forecast does not fit the remaining13-hour envelope')
  stage('preprocessing','feature_runtime.py',['cache'],cap=1800);stage('warmup','feature_runtime.py',['train','--arm','warmup'],cap=1260);stage('training-source','feature_runtime.py',['train','--arm','source'],cap=7260);stage('training-quality','feature_runtime.py',['train','--arm','quality'],cap=7260)
  stage('pilot-generation','feature_inference.py',['pilot'],cap=900);stage('pilot-native-scoring','score.py',['pilot'],strict=False,cap=7200);stage('pilot-selection','decide.py',['pilot'],cpu=True,cap=180)
  selected=read(D/'selection.json')
  if selected['status']!='validate':state.update(status='scientific_stop_no_joint_pass',exit_code=0)
  else:
   eval_seconds=read(D/'pilot/source-summary.json')['resources']['wall_seconds'];estimate_validation=76*19.65+eval_seconds/(48*8)*(76*4)*1.5+300;save(D/'validation_forecast.json',{'forecast_seconds':estimate_validation,'remaining_seconds':remaining(),'basis':'measured pilot scoring plus historical measured original-control FV50 latency; fresh inference/control overhead margin included','candidate_frozen':selected['candidate']})
   if estimate_validation>=remaining():raise BudgetStop('Qualifying candidate frozen; validation forecast does not fit remaining clock')
   stage('validation-generation','feature_inference.py',['validation'],cap=3600);stage('validation-native-scoring','score.py',['validation'],strict=False,cap=5400);stage('validation-gate','decide.py',['validation'],cpu=True,cap=180)
   if read(D/'transfer_freeze.json')['status']!='evaluate_transfer':state.update(status='scientific_stop_validation_joint_failure',exit_code=0)
   elif remaining()<600:raise BudgetStop('Candidate passed validation; insufficient time left to profile transfer')
   else:
    stage('transfer-profile','transfer.py',['profile'],strict=False,cap=1200);tp=read(D/'transfer/profile.json');n=min(5,int((remaining()-600)/tp['forecast_seconds_per_complete76_condition_all4targets']));save(D/'transfer/execution_plan.json',{'created_at_utc':stamp(),'conditions':['raw','lpf6000','waveguard8000','lpf4000','waveguard12000'][:max(0,n)],'forecast_per_condition_seconds':tp['forecast_seconds_per_complete76_condition_all4targets'],'remaining_seconds':remaining(),'selection_depends_only_on_compute':True,'full_coverage_required_for_preservation_pass':True})
    if n<1:raise BudgetStop('Transfer profile complete; a full raw76 condition does not fit remaining clock')
    stage('transfer-and-defenses','transfer.py',['evaluate'],strict=False,cap=remaining()-300);state.update(status='complete',exit_code=0)
 except BudgetStop as e:state.update(status='budget_stop',exit_code=124,error=str(e),scientific_failure=False)
 except BaseException as e:
  state.update(status='failed_execution',exit_code=1,error_type=type(e).__name__,error=str(e))
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
   except Exception:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
 finally:
  if active_start:state['active_wall_seconds']=time.monotonic()-active_start
  else:state['queue_wait_seconds']=time.monotonic()-started
  gpu_active=False;queue_finished=True
  state.update(finished_at_utc=stamp(),overall_wall_seconds=time.monotonic()-started);save(D/'status.json',state);save(D/'gpu_exit_receipt.json',state)
  for f in gpu_locks:f.close()
 # CPU reports/builds are also bounded by the same overall script deadline.
 try:
  if remaining()>60:
   scientific_status=state['status'];state['scientific_status']=scientific_status;stage('reporting','report.py',cpu=True,cap=300)
   if (D/'pilot/source-summary.json').exists() and remaining()>90:
    stage('interface-build','build_interface.py',cpu=True,cap=300);stage('interface-QA','check_browser.cjs',cpu=True,cap=180)
   state['status']=scientific_status;state['delivery_status']='complete';state['delivery_exit_code']=0
  else:state.update(delivery_status='skipped_time_ceiling',delivery_exit_code=124)
 except BaseException as e:state.update(delivery_status='failed',delivery_error=str(e),delivery_exit_code=1)
 finally:
  state['status']=state.get('scientific_status',state['status']);state.setdefault('delivery_status','interrupted')
 state.update(finished_at_utc=stamp(),overall_wall_seconds=time.monotonic()-started);save(D/'status.json',state);save(D/'exit_receipt.json',state);return state.get('exit_code',1)
if __name__=='__main__':raise SystemExit(main())
