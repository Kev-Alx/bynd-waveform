"""Detached48-hour expanded research queue with explicit scientific gates."""
from datetime import datetime,timezone,timedelta
import fcntl
import os
import signal
import time
from pathlib import Path
import supervise_pilot48 as jobs
import scaled_scope as scope

OUT=scope.OUT;RUN=scope.RUN;CURRENT='qualification'


def heartbeat():
    s=jobs.STATE;info=scope.COHORTS.get(CURRENT,scope.COHORTS['validation']);folder=RUN/CURRENT
    s.update(updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),timezone='Asia/Taipei',
        cohort=CURRENT,count=info['count'],generation=jobs.read(folder/'generation-status.json'),
        transfer_progress=jobs.read(RUN/'transfer/progress.json'),
        scored_variant_waveforms=len(list((folder/'score-records').glob('*.json'))),
        total_variant_waveforms=info['count']*len(info['variants']),decoder_decision=jobs.read(RUN/'decoder-decision.json'),
        selection=jobs.read(RUN/'selection.json'),transfer_decision=jobs.read(RUN/'transfer-freeze.json'))
    jobs.atomic(RUN/'progress.json',s);jobs.atomic(OUT/'JOB_STATUS.json',s)
    lines=['# Larger-model research job status','',f"Updated **{s['updated_at']} (Asia/Taipei)**. "
        f"Status: **{s['status']}**. Phase: **{s['phase']}**.",'',
        'Detached single-GPU queue;30-second heartbeats;48-hour ceiling. Larger Whisper/Data2Vec sources, '
        '500-step ensembles, up to1500-step source-call controls, and an independently qualified112M BigVGAN decoder.','',
        '| Stage | Status | Exit |','|---|---|---:|']
    for st in s['stages']:lines.append(f"| {st['name']} | {st['status']} | {st.get('exit_code','—')} |")
    if s['generation']:lines+=['','Generation: '+', '.join(f'{k}={s["generation"][k]}' for k in
        ('completed','clip','total','id','variant','update') if k in s['generation'])+'.']
    if s['transfer_progress']:lines+=['','Held-out evaluation: '+str(s['transfer_progress'])]
    lines+=['',f"Scored native variants: {s['scored_variant_waveforms']}/{s['total_variant_waveforms']}.",'',
        'SourceWER>75 and quality are separate from transfer/defense preservation. Held-out checkpoints are opened '
        'only after fixed76-point validation qualifies. Unknown paper defenses/human intelligibility remain unmeasured.','',
        f'[Protocol]({OUT}/14_scaled_research_protocol.md) · [Results]({OUT}/15_scaled_results.md) · '
        f'[Explanation]({OUT}/11_changes_and_goals.md)','']
    p=OUT/'JOB_STATUS.md.tmp';p.write_text('\n'.join(lines));p.replace(OUT/'JOB_STATUS.md')


def main():
    global CURRENT
    with (OUT/'execution/continuation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'jobs/registration/job.json').exists():raise FileExistsError('Inspect previous expanded queue before relaunch')
        jobs.atomic(RUN/'previous-JOB_STATUS.json',jobs.read(OUT/'JOB_STATUS.json'))
        jobs.RUN=RUN;jobs.heartbeat=heartbeat;jobs.STATE.clear()
        jobs.STATE.update(status='starting',phase='registration',source_only_generation=True,
            target_models_evaluated=[],maximum_cycle_wall_seconds=172800,heartbeat_seconds=30,
            supervisor_pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),stages=[])
        for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,jobs.interrupted)
        for name in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','MPLCONFIGDIR'):
            path=RUN/'runtime-cache'/name.lower();path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
        os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        deadline=time.monotonic()+172800
        def stage(name,script,*args,strict=False,cpu=False):
            code=jobs.stage(name,[jobs.PYTHON,'-u',str(script),*args],deadline,strict=strict,cpu=cpu)
            if code:
                jobs.STATE.update(status='needs_attention',phase=name+'_failed',
                    error='Stage failed or48-hour ceiling reached; preserve receipts and partial outputs')
                heartbeat();raise RuntimeError(f'{name} exited{code}')
        def cohort(name):
            global CURRENT
            CURRENT=name
            stage(name+'-generation',OUT/'scaled_run.py','generate','--phase',name,strict=True)
            stage(name+'-native-scoring',OUT/'scaled_source_evaluate.py','--name',name,'--output-dir',str(RUN/name))
            stage(name+'-independent-audit',RUN/'audit_scaled.py','--phase',name,cpu=True)
            stage(name+'-report',OUT/'scaled_postprocess.py','report',cpu=True)
        stage('registration',OUT/'scaled_run.py','register',cpu=True)
        cohort('qualification');stage('decoder-gate',OUT/'scaled_postprocess.py','decoder',cpu=True)
        cohort('fvpilot')
        if jobs.read(RUN/'decoder-decision.json').get('status')=='pass':cohort('melpilot')
        else:
            jobs.STATE['stages'].append({'name':'melpilot','status':'skipped_reconstruction_gate'})
            heartbeat()
        stage('source-only-selection',OUT/'scaled_postprocess.py','select',cpu=True)
        if jobs.read(RUN/'selection.json').get('status')=='validate':
            CURRENT='validation'
            stage('validation-feature-cache',OUT/'scaled_run.py','cache-validation',strict=True)
            cohort('validation');stage('validation-gate',OUT/'scaled_postprocess.py','validation',cpu=True)
            if jobs.read(RUN/'transfer-freeze.json').get('status')=='evaluate_transfer':
                CURRENT='transfer'
                stage('held-out-transfer-and-preprocessing',OUT/'scaled_transfer_evaluate.py')
                jobs.STATE['target_models_evaluated']=list(jobs.read(RUN/'plan.json')['target_configs'])
                stage('independent-target-audit',RUN/'audit_scaled.py','--phase','transfer',cpu=True)
                stage('transfer-report',OUT/'scaled_postprocess.py','report',cpu=True)
        stage('package-check',OUT/'validate_package.py',cpu=True)
        jobs.STATE.update(status='complete',phase='expanded_cycle_finished',finished_at_utc=datetime.now(timezone.utc).isoformat(),
            scientific_status='Inspect separate source, quality, transfer and defense endpoints; completion is not proof of success')
        heartbeat()


if __name__=='__main__':
    try:main()
    except Exception as error:
        jobs.STATE.update(status='needs_attention',error=str(error),error_type=type(error).__name__)
        heartbeat();raise
