"""Launch one detached, externally bounded controller; no GPU overlap."""
import pathlib,subprocess,os,json,datetime,hashlib
D=pathlib.Path(__file__).resolve().parent;PY='/opt/venv/bin/python'
def main():
 assert (D/'plan.json').exists() and not (D/'launch_receipt.json').exists()
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','BEYONDWAVEFORM_NOTIFY_DISCORD':'0','PYTHONUNBUFFERED':'1'}
 command=[PY,'-u','/workspace/scripts/run_logged.py','--log',str(D/'controller.log'),'--status',str(D/'controller_exit.json'),'--','/usr/bin/timeout','--signal=TERM','--kill-after=15s','46785s',PY,'-u',str(D/'supervise.py')]
 now=datetime.datetime.now(datetime.timezone.utc)
 with (D/'detached_stdout.log').open('ab') as log:
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd='/workspace',start_new_session=True)
 receipt={'status':'launched_detached','wrapper_pid':process.pid,'started_at_utc':now.isoformat(),'hard_ceiling_including_queue_seconds':46800,'external_term_seconds':46785,'external_kill_grace_seconds':15,'maximum_end_at_utc':(now+datetime.timedelta(seconds=46800)).isoformat(),'command':command,'plan_sha256':hashlib.sha256((D/'plan.json').read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'gpu_must_wait_for_existing_auxiliary_and_continuation_locks':True,'discord_notifications':False}
 (D/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
