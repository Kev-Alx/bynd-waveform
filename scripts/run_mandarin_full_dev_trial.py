"""Detached, conditional full-development fine-tuning and validation-CER trial.

No official test transcription is read to rank checkpoints. All epochs are
checked on 48 validation recordings before full validation CER selection.
"""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction-runs/mandarin-cer-full-dev'
UPSTREAM=ROOT/'reproduction-runs/mandarin-cer-checkpoint-selection'
TAIPEI=timezone(timedelta(hours=8))


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(2**20):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def freeze(path,value):
    if path.exists():assert read(path)==value,f'Frozen trial differs: {path}'
    else:save(path,value)


class Trial:
    def __init__(self,upstream_pid):
        self.upstream_pid=upstream_pid
        self.data=read(OUT/'data-plan.json')
        self.state={'status':'running','scope':'Conditional full-development Whisper-small fine-tuning, followed by 48-clip pilots and validation-only epoch CER selection',
                    'started_at':datetime.now(TAIPEI).isoformat(timespec='seconds')}

    def progress(self,phase,done=0,total=0,remaining=None,**extra):
        self.state.update(phase=phase,updated_at=datetime.now(TAIPEI).isoformat(timespec='seconds'),
                          clip_progress={'completed':done,'total':total,'percent':round(100*done/total,1) if total else 0},**extra)
        for key in ['estimated_remaining_seconds','estimated_remaining_range_seconds','estimated_completion_taipei']:
            self.state.pop(key,None)
        if remaining is not None:
            self.state.update(estimated_remaining_seconds=round(max(0,remaining)),
                              estimated_remaining_range_seconds=[round(max(0,remaining)*.7),round(max(0,remaining)*1.8)],
                              estimated_completion_taipei=(datetime.now(TAIPEI)+timedelta(seconds=max(0,remaining))).isoformat(timespec='seconds'))
        save(OUT/'progress.json',self.state)
        eta=(f"Estimated remaining in current stage: **{remaining/3600:.2f} hours**; completion **{self.state['estimated_completion_taipei']} (Asia/Taipei)**."
             if remaining is not None else 'Current-stage ETA is not available yet; waiting or calibrating on observed work.')
        (OUT/'JOB_STATUS.md').write_text(f"# Full-development Mandarin surrogate trial\n\nUpdated: {self.state['updated_at']} (Asia/Taipei).\n\nStatus: **{self.state['status']}**; phase: **{phase}**.\n\nProgress: **{done}/{total}** observations in the current stage.\n\n{eta}\n\n{self.state.get('eta_basis','')}\n\nFine-tuning uses all 14,326 official development recordings, raw Whisper-small initialization and the original fixed three-epoch optimizer recipe. The eight checkpoint-validation speakers and eight attack-pilot speakers come only from the official training split and are excluded from surrogate fine-tuning. Test data never enters checkpoint ranking.\n\nEvery epoch gets a 48-record validation pilot before 1,435-record validation CER selection. This trial does not itself claim a completed 5,000-record attack reproduction.\n")
        main_status=ROOT/'reproduction-runs/mandarin-cer-corrected/WORKFLOW_STATUS.md'
        training_started=(OUT/'training/training.json').exists()
        main_status.write_text(f"# Corrected AISHELL workflow status\n\nUpdated: {self.state['updated_at']} (Asia/Taipei). The requested goal is **not complete**.\n\nThe corrected existing-checkpoint clean result on all 5,000 test recordings is **8.576592% CER**, versus paper **6.29%**. It passed the transcript/reference audit but missed the declared comparison gate. The paper-rho/lambda 48-clip attack pilot measured **78.561151% CER**, versus paper **72.25%**.\n\nCurrent full-development trial status: **{self.state['status']}**; phase **{phase}**. Progress **{done}/{total}** in this stage. New training started: **{training_started}**.\n\n{eta}\n\nThe whole-task ETA remains unknown until the validation-selected model passes a separate 48-record attack pilot. A consistent final 5,000-record attack evaluation is still required.\n\nThe conditional trial uses all 14,326 development recordings and validation speakers from the official training split, with no test-based checkpoint ranking. The paper does not disclose its exact fine-tuning partition or optimizer recipe; this is an explicit data-coverage replication interpretation.\n\n- [Live full-development trial status]({OUT}/JOB_STATUS.md)\n- [Trial progress]({OUT}/progress.json)\n- [Frozen data plan]({OUT}/data-plan.json)\n- [Original checkpoint comparison]({UPSTREAM}/JOB_STATUS.md)\n- [Completed 5,000-record clean audit]({main_status.parent}/test-clean/paper/independent-audit.json)\n")
        sys.path.insert(0,str(ROOT/'scripts'))
        from update_job_status import update
        update(ROOT/'reproduction-runs','paper-quality-coordinate-research')

    def wait_upstream(self):
        while True:
            job=read(UPSTREAM/'job.json')
            assert job['supervisor_pid']==self.upstream_pid,'Tracked upstream job was replaced'
            if job['status'] in {'complete','failed','interrupted','stopped'}:
                assert job['status']=='complete' and job.get('exit_code')==0,'Upstream checkpoint comparison did not complete'
                break
            stat=Path(f'/proc/{self.upstream_pid}/stat')
            assert stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z','Upstream supervisor is no longer live'
            current=read(UPSTREAM/'progress.json')
            self.progress('waiting-for-existing-checkpoint-validation',eta_basis='Conditional trial waits for the already-running validation-CER decision',upstream_progress=current.get('clip_progress'))
            time.sleep(30)
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_mandarin_checkpoint_selection.py'),'--stage','validation'],cwd=ROOT,check=True)
        return read(UPSTREAM/'validation-results.json')['selection']

    def verify_inputs(self):
        for path,digest in self.data['manifest_sha256'].items():assert sha(path)==digest
        assert sha(OUT/'training-config.json')==self.data['training_config_sha256']
        assert sha(ROOT/'artifacts/whisper/small.pt')==self.data['raw_whisper_small_sha256']
        config=read(OUT/'training-config.json')
        assert config['epochs']==3 and config['learning_rate']==1e-5
        assert config['model']['decoding']['without_timestamps'] is True
        assert config['save_epoch_checkpoints'] is True
        source_paths=[ROOT/'beyondwaveform'/name for name in ['training.py','models/whisper.py','losses.py','io.py','data.py','config.py']]
        self.source_hashes={str(p):sha(p) for p in source_paths}
        plan={'status':'frozen','data_plan_sha256':sha(OUT/'data-plan.json'),'training_config_sha256':sha(OUT/'training-config.json'),
              'source_sha256':self.source_hashes,'upstream_validation_supervisor_pid':self.upstream_pid,
              'scope':self.state['scope'],'activation_rule':'Only when the unchanged existing best checkpoint wins validation CER; otherwise review the selected alternative first',
              'selection_rule':self.data['selection_rule'],'interpretation':self.data['paper_interpretation']}
        freeze(OUT/'plan.json',plan)

    def assert_sources(self):
        assert all(sha(p)==expected for p,expected in self.source_hashes.items()),'Frozen training source changed'

    def wait_gpu(self):
        minimum_mib=14*1024
        pilot_path=OUT/'memory-pilot/results.json'
        if pilot_path.exists():
            pilot=read(pilot_path)
            assert pilot['status']=='passed' and pilot['count']==48 and pilot['optimizer_updates']==6
            assert pilot['checkpoint_sha256']==self.data['raw_whisper_small_sha256']
            assert pilot['training_config_sha256']==self.data['training_config_sha256']
            assert pilot['dtype']=='fp32' and pilot['temporary_weights_discarded'] is True
            minimum_mib=math.ceil(pilot['recommended_minimum_free_mib']/1024)*1024
            self.state.update(memory_pilot_sha256=sha(pilot_path),gpu_minimum_free_mib=minimum_mib)
        while True:
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free=int(result.stdout.splitlines()[0].strip())
            self.state['gpu_free_mib']=free
            if free>=minimum_mib:return
            self.progress('waiting-for-gpu-capacity',eta_basis=f'Training start waits for {minimum_mib/1024:.1f} GiB free; uses measured 48-record peak plus margin when available',gpu_free_mib=free)
            time.sleep(30)

    def phase(self,name,command,progress_file=None):
        folder=OUT/'phases'/name;folder.mkdir(parents=True,exist_ok=True)
        log=folder/'stage.log'
        wrapped=[sys.executable,'-u',str(ROOT/'scripts/run_logged.py'),'--log',str(log),'--status',str(folder/'job.json'),'--',*command]
        started=time.monotonic();self.progress(name)
        child=subprocess.Popen(wrapped,cwd=ROOT,stdin=subprocess.DEVNULL,env={**os.environ,'BEYONDWAVEFORM_NOTIFY_DISCORD':'0'})
        try:
            while True:
                if progress_file and progress_file.exists():
                    s=read(progress_file);c=s.get('clip_progress',{})
                    self.progress(f"{name}:{s.get('phase','starting')}",c.get('completed',0),c.get('total',0),s.get('estimated_remaining_seconds'),eta_basis=s.get('eta_basis','Child-stage timing'),phase_pid=child.pid)
                elif name=='training' and log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
                    matches=list(re.finditer(r'(Fine-tune|Validating) epoch (\d+):[^\r\n]*?(\d+)/(\d+)\s*\[',tail))
                    if matches:
                        kind,epoch,pos,_=matches[-1].groups();epoch=int(epoch);pos=int(pos)
                        done=(epoch-1)*14326+pos if kind=='Fine-tune' else epoch*14326
                        pending_val=(3-epoch)*1435+(1435 if kind=='Fine-tune' else 1435-pos)
                        rate=(time.monotonic()-started)/max(1,done)
                        remaining=((42978-done)*rate+pending_val*rate/2) if done>=48 else None
                        self.progress(f'training-epoch-{epoch}' if kind=='Fine-tune' else f'training-validation-{epoch}',done,42978,remaining,eta_basis='Observed training progress; validation overhead approximate; excludes later decoded-CER selection and attacks',phase_pid=child.pid)
                try:
                    code=child.wait(timeout=20);break
                except subprocess.TimeoutExpired:continue
            assert code==0,f'{name} exited {code}; inspect {log}'
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()

    def run(self):
        self.verify_inputs()
        selected=self.wait_upstream()
        if selected!='best':
            self.progress('existing-alternative-selected-review-before-training',status='needs_review',selected_existing_checkpoint=selected)
            return
        with (ROOT/'reproduction-runs/auxiliary-gpu.lock').open('a+') as gpu:
            while True:
                try:
                    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    self.progress('waiting-for-existing-auxiliary-gpu-job',eta_basis='Another cooperating GPU job still holds the shared lock')
                    time.sleep(30)
            self.wait_gpu();self.assert_sources()
            self.phase('training',[sys.executable,'-u',str(ROOT/'scripts/finetune.py'),'--config',str(OUT/'training-config.json'),
                       '--train-manifest',str(OUT/'manifests/finetune.jsonl'),'--validation-manifest',str(OUT/'manifests/checkpoint-validation.jsonl'),
                       '--output',str(OUT/'training'),'--device','cuda'])
            self.assert_sources()
        training=read(OUT/'training/training.json')
        assert training['status']=='complete' and len(training['epoch_checkpoints'])==3
        snapshots=sorted(training['epoch_checkpoints'],key=lambda r:r['epoch'])
        for record in snapshots:assert sha(record['path'])==record['sha256']
        selection=OUT/'checkpoint-selection'
        protocol={'output':str(selection),'validation_manifest':str(OUT/'manifests/checkpoint-validation.jsonl'),'validation_split':'train',
                  'pilot_manifest':str(OUT/'manifests/checkpoint-pilot-48.jsonl'),'training_manifest':str(OUT/'manifests/finetune.jsonl'),
                  'attack_manifest':str(OUT/'manifests/attack-pilot-48.jsonl'),'checkpoints':{f"epoch-{r['epoch']:03d}":r['path'] for r in snapshots},
                  'scope':'Full-development Whisper-small epoch CER selection on separate non-test speakers',
                  'pilot_selection':'Prepared seed-2026 duration-stratified six records per each of eight checkpoint-validation speakers',
                  'selection_rule':'Lowest corpus CER on the fixed 1,435 checkpoint-validation recordings; earliest epoch on a tie. Each epoch passes a 48-record execution pilot first. No test or attack CER enters ranking.'}
        freeze(OUT/'selection-protocol.json',protocol)
        for stage in ['pilot','validation']:
            self.phase(f'checkpoint-{stage}',[sys.executable,'-u',str(ROOT/'scripts/select_mandarin_checkpoint_cer.py'),'--protocol',str(OUT/'selection-protocol.json'),'--stage',stage],selection/'progress.json')
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_mandarin_checkpoint_selection.py'),'--root',str(selection),'--stage',stage],cwd=ROOT,check=True)
        result=read(selection/'validation-results.json');winner=result['selection'];target=read(selection/f'{winner}-target.json')
        selected_meta=read(selection/'plan.json')['targets'][winner]
        freeze(OUT/'surrogate-freeze.json',{'status':'frozen','checkpoint':target['checkpoint'],'checkpoint_sha256':selected_meta['checkpoint_sha256'],
               'selected_epoch':winner,'selection_rule':protocol['selection_rule'],'validation_results_sha256':sha(selection/'validation-results.json'),
               'data_plan_sha256':sha(OUT/'data-plan.json'),'training_metadata_sha256':sha(OUT/'training/training.json'),
               'validation_cer':result['summaries'][winner]['error_percent'],'target':target})
        save(OUT/'results.json',{'status':'surrogate-frozen; full attack evaluation pending','metric':'validation CER','summary':result['summaries']})
        (OUT/'report.md').write_text(f"# Full-development Mandarin surrogate\n\nSelected **{winner}** using validation CER **{result['summaries'][winner]['error_percent']:.6f}%**.\n\nThis is a validation result, not final test CER. A separate 48-record attack pilot and consistent 5,000-record evaluation are still required.\n")
        self.progress('surrogate-frozen-awaiting-48-record-attack-pilot',status='complete',selected_epoch=winner,validation_cer=result['summaries'][winner]['error_percent'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-for',required=True,type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'trial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trial=Trial(args.wait_for)
        try:trial.run()
        except BaseException as error:
            trial.progress('failed',status='failed',error=str(error));raise
