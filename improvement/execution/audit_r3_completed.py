"""CPU-only audit for completed R3 artifacts; never imports experiment code.

Exit codes: 0 verified saved-artifact scope; 1 discrepancies/error; 2 pending.
Outputs are always execution/r3-completed-audit.{json,md}. --self-test exercises
only small pure helpers and writes no report. No neural inference is performed.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse, hashlib, json, math, random, statistics, sys, traceback, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(1)
ROOT=Path('/workspace'); OUT=ROOT/'improvement'; RUN=OUT/'execution'
LOSS_KEYS=('source_sum_nll','stft_loss','content_loss')
QUALITY_KEYS=('dnsmos','nisqa','utmos')
COUNT_KEYS=('substitutions','deletions','insertions','reference_units')

def read(path):return json.loads(Path(path).read_text())
def rows(path):return[json.loads(line)for line in Path(path).read_text().splitlines()if line.strip()]
def norm(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return' '.join(''.join(' 'if unicodedata.category(c).startswith('P')else c for c in text).split())
def edit(reference,hypothesis):
    a,b=norm(reference).split(),norm(hypothesis).split();previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1],len(a)
def finite_tree(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite_tree(x)for x in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(x)for x in value)
    return True
def is_admissible(q):return q['utmos']>=3.5 and q['dnsmos']>=3 and q['nisqa']>=3.75
def safe_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError(f'New-artifact path outside improvement: {path}')
    return path
def relative_error(actual,expected):return abs(actual-expected)/max(1,abs(expected))

class Audit:
    def __init__(self,folder):
        self.folder=folder;self.issues=[];self.hashes={};self.checkpoints=[]
        self.report={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'pending','run':str(folder),'cpu_only':True,'neural_inference':False,
            'cuda_initialised':False,'issues':self.issues,'file_sha256':self.hashes}
    def need(self,condition,message):
        if not condition:self.issues.append(message)
    def sha(self,path):
        path=Path(path)
        if str(path)not in self.hashes:
            h=hashlib.sha256()
            with path.open('rb')as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            self.hashes[str(path)]=h.hexdigest()
        return self.hashes[str(path)]
    def binding(self,path,expected):
        path=Path(path);self.need(path.exists(),f'Missing bound file {path}')
        if path.exists():self.need(self.sha(path)==expected,f'Hash mismatch {path}')
    def asset(self,asset):
        safe_path(asset['path']);self.binding(asset['path'],asset.get('sha256',asset.get('file_sha256')))
    def counts(self,reference,hypothesis,row,label):
        distance,n=edit(reference,hypothesis)
        self.need(distance==sum(row[k]for k in COUNT_KEYS[:3])and n==row['reference_units'],f'Word edit/reference counts {label}')
        self.need(norm(reference)==row['reference_normalized']and norm(hypothesis)==row['hypothesis_normalized'],f'Word normalization {label}')
    def waveform(self,asset,label):
        self.asset(asset);audio,sr=sf.read(asset['path'],dtype='float32')
        self.need(audio.ndim==1 and sr==asset['sample_rate']==16000 and len(audio)==asset['samples'],f'Waveform format/length {label}')
        self.need(np.isfinite(audio).all(),f'Nonfinite waveform {label}')
        self.need(abs(float(np.abs(audio).max())-asset['peak'])<1e-7,f'Waveform peak metadata {label}')
        return audio
    def checkpoint(self,path,registration,calibration,expected_updates=None):
        path=safe_path(path);state=torch.load(path,map_location='cpu',weights_only=True)
        self.sha(path)
        self.need(state['source_only']is True and state['manifest_sha256']==registration['train_manifest_sha256'],f'Checkpoint training scope {path}')
        self.need(state['architecture']==registration['generator']and state['weights']==calibration['weights']and state['calibration']==calibration,f'Checkpoint architecture/calibration {path}')
        config=state['config'];updates=state['updates']
        self.need(config['seed']==11 and config['learning_rate']==1e-4 and config['weight_decay']==0 and config['accumulation_steps']==4 and config['gradient_clip']==1 and config['checkpoint_interval']==500 and config['max_updates']==12000 and config['max_gpu_hours']==1,f'Checkpoint optimizer configuration {path}')
        self.need(isinstance(updates,int)and 0<=updates<=12000 and state['microbatches']==4*updates,f'Checkpoint counters {path}')
        if expected_updates is not None:self.need(updates==expected_updates,f'Checkpoint update binding {path}')
        generator=state['generator'];parameters=sum(t.numel()for t in generator.values())
        self.need(parameters==registration['generator']['parameters']==1590833 and parameters<=3000000,f'Generator parameter count {path}')
        self.need(all(isinstance(t,torch.Tensor)and t.device.type=='cpu'and torch.isfinite(t).all().item()for t in generator.values()),f'Generator nonfinite/non-CPU state {path}')
        optimizer=state['optimizer'];groups=optimizer['param_groups']
        self.need(len(groups)==1 and len(groups[0]['params'])==len(generator),f'Optimizer parameter coverage {path}')
        group=groups[0]
        self.need(group['lr']==1e-4 and group['betas']==(.9,.999)and group['weight_decay']==0 and group['eps']==1e-8,f'AdamW settings {path}')
        self.need(len(optimizer['state'])==(len(generator)if updates else 0),f'AdamW initialized states {path}')
        for pid,values in optimizer['state'].items():
            self.need(float(values['step'])==updates,f'AdamW step counter {path}/{pid}')
            self.need(all(not isinstance(t,torch.Tensor)or(t.device.type=='cpu'and torch.isfinite(t).all().item())for t in values.values()),f'AdamW finite state {path}/{pid}')
            self.need(values['exp_avg'].shape==values['exp_avg_sq'].shape,f'AdamW moment shapes {path}/{pid}')
        result={'path':str(path),'sha256':self.sha(path),'updates':updates,'microbatches':state['microbatches'],'generator_parameters':parameters,'state_tensors':len(generator),'best_checkpoint_key':state['best_checkpoint_key']}
        self.checkpoints.append(result)
        return result
    def evaluate(self,summary_path,dev,expected_backend,source_metadata,expected_variants=None):
        d=read(summary_path);score_path=summary_path.parent/'source-scores.jsonl'
        self.need(d['status']=='complete'and d['source_only']and d['target_models_evaluated']==[],f'Native evaluation scope {summary_path}')
        self.need(d['source_metadata']==source_metadata and d['evaluation_backend']==expected_backend and d['baseline_policy_exactly_matched'],f'Native source/backend identity {summary_path}')
        self.binding(score_path,d['scores_sha256']);scores=rows(score_path)
        registration=read(summary_path.parent/'evaluation-registration.json')
        self.need(registration['backend']==expected_backend,f'Native pre-inference policy {summary_path}')
        self.need(registration['source_only']is True and registration['target_models_evaluated']==[],f'Native evaluation registration scope {summary_path}')
        self.binding(OUT/'source_evaluate.py',registration['evaluator_sha256'])
        self.binding(registration['generated_manifest'],registration['generated_manifest_sha256'])
        generated=rows(registration['generated_manifest']);gr={r['id']:r for r in generated}
        self.need(registration['allowed_development_ids']==[r['id']for r in generated],f'Native registered cohort {summary_path}')
        for score in scores:
            uid=score['id'];self.need(uid in dev and score['reference']==dev[uid]['text'],f'Non-development or changed reference {summary_path}/{uid}')
            self.counts(score['reference'],score['hypothesis'],score,f'{summary_path}/{uid}/{score["variant"]}')
            self.need(score['audio_sha256']==gr[uid]['audio'][score['variant']]['sha256'],f'Native waveform binding {summary_path}/{uid}')
            self.need(all(math.isfinite(score['quality'][k])for k in QUALITY_KEYS),f'Native nonfinite quality {summary_path}/{uid}')
        variants=sorted({score['variant']for score in scores});aggregates={}
        if expected_variants is not None:self.need(set(variants)==set(expected_variants),f'Final native variant coverage {summary_path}')
        for variant in variants:
            pool=[r for r in scores if r['variant']==variant]
            self.need(len(pool)==len({r['id']for r in pool})==24 and set(r['id']for r in pool)==set(dev),f'Native cohort coverage {summary_path}/{variant}')
            a={'count':len(pool),**{key:sum(r[key]for r in pool)for key in COUNT_KEYS}}
            a['error_percent']=100*sum(a[k]for k in COUNT_KEYS[:3])/a['reference_units']
            a['quality']={k:statistics.fmean(r['quality'][k]for r in pool)for k in QUALITY_KEYS}
            self.need(all(a[k]==d['summary'][variant][k]for k in a if k!='quality')and all(abs(a['quality'][k]-d['summary'][variant]['quality'][k])<1e-12 for k in QUALITY_KEYS),f'Native aggregate {summary_path}/{variant}')
            aggregates[variant]=a
        self.need(d['human_intelligibility_established']is False,f'Unsupported human-content declaration {summary_path}')
        return{'path':str(summary_path),'score_rows':len(scores),'summary':aggregates,'native_backend_exactly_matches_historical':True}
    def run(self):
        required=['registration.json','preflight.json','penalty-calibration.json','training.json',
            'training-trace.jsonl','generated.jsonl','generation-resource.json',
            'model-provenance.json','source-summary.json','source-scores.jsonl','evaluation-registration.json']
        missing=[str(self.folder/name)for name in required if not(self.folder/name).exists()]
        if missing:
            self.report['pending_files']=missing;return 2
        reg=read(self.folder/'registration.json');pre=read(self.folder/'preflight.json')
        training=read(self.folder/'training.json');native=read(self.folder/'source-summary.json')
        if training.get('status')!='completed'or native.get('status')!='complete':
            self.report['pending_status']={'training':training.get('status'),'native':native.get('status')};return 2
        plan=read(RUN/'plan.json');calibration=read(self.folder/'penalty-calibration.json')
        provenance=read(self.folder/'model-provenance.json')
        devrows=rows(plan['development_manifest']);dev={r['id']:r for r in devrows}
        trainrows=rows(reg['train_manifest']);trainbyid={r['id']:r for r in trainrows}
        self.need(len(dev)==24 and len(trainbyid)==len(trainrows)==286,'Registered train/development cohort counts')
        self.need(reg['source_only']and reg['target_models_evaluated']==[]and not reg['confirmation_or_final_inference'],'Registered source-only scope')
        self.need(reg['seed']==11 and reg['batch_size']==1 and reg['accumulation']==4 and reg['maximum_training_seconds']==3600 and reg['maximum_updates']==12000 and reg['training_crops']is None,'Registered training configuration')
        self.binding(reg['train_manifest'],reg['train_manifest_sha256'])
        for path,h in reg['implementation_hashes'].items():self.binding(path,h)
        for path,h in plan['frozen_files'].items():self.binding(path,h)
        repair=reg.get('autograd_execution_repair',{})
        if repair:
            self.binding(repair['prior_failed_preflight'],repair['prior_failure_sha256'])
            self.binding(repair['cpu_regression'],repair['cpu_regression_sha256'])
            self.need(repair['forward_values_weights_layer_and_losses_unchanged']is True,'Autograd execution repair declaration')
        self.need(not set(trainbyid)&set(dev),'Training/development utterance overlap')
        self.need(not{r['speaker']for r in trainrows}&{r['speaker']for r in devrows},'Training/development speaker overlap')
        self.need(not{r['audio_sha256']for r in trainrows}&{r['audio_sha256']for r in devrows},'Training/development audio-hash overlap')
        self.need(not{norm(r['text'])for r in trainrows}&{norm(r['text'])for r in devrows},'Training/development normalized-reference overlap')
        self.need(pre['status']=='passed'and pre['source_only']and not pre['cached_q_parity_claimed'],'Preflight execution/parity declarations')
        self.asset(pre['calibration']);self.need(calibration['implementation_hashes']==reg['implementation_hashes'],'Calibration implementation binding')
        self.need(calibration['status']=='completed'and calibration['seed']==11 and calibration['probe_amplitude']==1e-4,'Calibration scope')
        self.need([r['id']for r in calibration['clips']]==[r['id']for r in rows(plan['calibration_manifest'])],'Calibration cohort/order')
        recalibrated={}
        for key,normkey in [('stft','stft_gradient_norm'),('content','content_gradient_norm')]:
            ratios=[r['source_sum_nll_gradient_norm']/r[normkey]for r in calibration['clips']]
            recalibrated[key]=statistics.median(ratios)
            self.need(recalibrated[key]==calibration['weights'][key]and all(math.isfinite(v)and v>0 for v in ratios),f'Calibration saved-scalar ratio {key}')
        fit=pre['longest_training_clip_fit'];longest=max(trainrows,key=lambda r:r['duration'])
        self.need(fit['id']==longest['id']and fit['duration']==longest['duration']and fit['parameter_count']==1590833 and fit['optimizer_updates']==1 and len(fit['microbatches'])==4,'Longest registered training fit')
        self.need(fit['first_parameter_gradients']['all_finite']and fit['bounds']['finite']and fit['bounds']['original_samples']==fit['bounds']['output_samples'],'Longest-fit finite/duration flags')
        self.need(all(r['native_adapter_bitwise_equal']and r['strict_repeat_bitwise_equal']for r in pre['encoder_conformance']),'Native adapter/strict repeat flags')
        for path,h in calibration['implementation_hashes'].items():self.binding(path,h)
        self.binding(provenance['source']['checkpoint'],provenance['source']['sha256'])
        self.binding(provenance['content']['checkpoint_path'],provenance['content']['checkpoint_sha256'])
        self.need(provenance['content']['layer']==6 and provenance['content']['vocoder_loaded']is False and provenance['source']['nll_reduction']=='sum','Source/content model scope')

        trace=rows(self.folder/'training-trace.jsonl');self.binding(self.folder/'training-trace.jsonl',training['trace_sha256'])
        self.need(training['seed']==11 and training['maximum_training_wall_seconds']==3600 and 0<training['training_wall_seconds']<=3600,'Actual training seed/time ceiling')
        self.need(training['training_crops']is None and not training['cached_q_parity_claimed']and not training['human_admissibility_established'],'Training declarations')
        self.need(training['training_manifest_sha256']==reg['train_manifest_sha256']and training['weights']==calibration['weights']and training['generator']==reg['generator'],'Training frozen configuration')
        self.need(training['updates']==sum(bool(r['optimizer_step'])for r in trace)and training['completed_microbatches']==4*training['updates']and len(trace)==training['completed_microbatches']+training['discarded_partial_microbatches']and training['discarded_partial_microbatches']==len(trace)%4,'Actual accumulation/update counts')
        self.need(0<training['updates']<=12000,'Actual optimizer update ceiling')
        prng=random.Random(11);expected=[];epoch=0
        while len(expected)<len(trace):
            order=list(range(len(trainrows)));prng.shuffle(order);epoch+=1
            expected.extend((epoch,trainrows[i]['id'])for i in order)
        previous_elapsed=0;loss_totals=Counter()
        for i,row in enumerate(trace,1):
            uid=row['id'];self.need(finite_tree(row),f'Nonfinite trace row {i}')
            self.need((row['epoch'],uid)==expected[i-1],f'Seeded epoch training order {i}')
            self.need(uid in trainbyid and row['duration']==trainbyid[uid]['duration'],f'Unregistered/truncated training row {i}')
            self.need(row['microbatches']==i and row['updates']==i//4 and row['optimizer_step']==(i%4==0),'Microbatch/update progression '+str(i))
            self.need(row['weights']==calibration['weights']and row['batch_size']==1 and row['source_backward_passes']==row['content_backward_passes']==1 and row['generator_forward_passes']==3,'Training loss/backward configuration '+str(i))
            self.need(all(row[k]>=0 for k in LOSS_KEYS),'Negative loss component '+str(i))
            self.need((row['preclip_gradient_norm']is not None)==row['optimizer_step'],'Gradient-clipping schedule '+str(i))
            if row['preclip_gradient_norm']is not None:self.need(row['preclip_gradient_norm']>=0,'Gradient norm '+str(i))
            self.need(row['wall_seconds']>0 and previous_elapsed<=row['elapsed_training_seconds']<=training['training_wall_seconds'],'Training timing progression '+str(i))
            previous_elapsed=row['elapsed_training_seconds']
            for key in LOSS_KEYS:loss_totals[key]+=row[key]
        self.need(training['epochs_started']==epoch,'Epoch counter')
        for key in LOSS_KEYS:self.need(relative_error(loss_totals[key]/len(trace),training['mean_microbatch_losses'][key])<1e-12,'Mean training loss '+key)
        self.need(sum(r['wall_seconds']for r in trace)<=training['training_wall_seconds']+1e-6,'Microbatch span exceeds total training wall time')
        self.report['training']={k:training[k]for k in ['updates','completed_microbatches','discarded_partial_microbatches','epochs_started','training_wall_seconds','maximum_training_wall_seconds','cuda_event_microbatch_span_seconds','peak_allocated_bytes','peak_reserved_bytes','stop_reason','checkpoint_selection','weights','mean_microbatch_losses']}
        self.report['training']['trace_rows']=len(trace)
        self.report['training']['seeded_manifest_order_independently_replayed']=True
        self.report['calibration']={'saved_scalar_median_weights':recalibrated,'neural_gradients_recomputed':False}
        self.asset(training['terminal_checkpoint'])
        self.checkpoint(training['terminal_checkpoint']['path'],reg,calibration,training['updates'])
        source_evaluations=[];selection_candidates=[];checkpoint_gaps=[]
        old_summary=read(Path(plan['baseline_run'])/'evaluation/summary.json');backend=old_summary['evaluation_backend']
        for path in sorted((self.folder/'checkpoints').glob('update-*.pt')):
            update=int(path.stem.removeprefix('update-'));self.need(update%500==0 and update<=training['updates'],f'Periodic checkpoint schedule {path}')
            self.checkpoint(path,reg,calibration,update)
            source_path=path.with_suffix('.source.json')
            if not source_path.exists():checkpoint_gaps.append(str(path));continue
            source=read(source_path);receipt=Path(source['native_policy_receipt']);self.binding(receipt,source['native_policy_receipt_sha256'])
            evaluated=self.evaluate(receipt,dev,backend,provenance['source'],['adversarial']);source_evaluations.append(evaluated)
            scores=evaluated['summary']['adversarial']
            self.need(all(source[k]==scores[k]for k in COUNT_KEYS+('count','error_percent'))and all(abs(source['quality'][k]-scores['quality'][k])<=1e-12 for k in QUALITY_KEYS),'Checkpoint source score metadata '+str(path))
            self.need(source['source_only']and not source['human_admissibility_established']and source['automatic_proxy_admissible']==is_admissible(scores['quality']),'Checkpoint selection scope '+str(path))
            self.need(math.isfinite(source['mean_stft_loss'])and source['mean_stft_loss']>=0,'Checkpoint STFT statistic '+str(path))
            if source['automatic_proxy_admissible']:selection_candidates.append(((-scores['error_percent'],source['mean_stft_loss'],update),path))
        selected=safe_path(training['selected_checkpoint'])
        expected_selected=min(selection_candidates,key=lambda item:item[0])[1]if selection_candidates else Path(training['terminal_checkpoint']['path'])
        self.need(selected==expected_selected.resolve(),'Highest admissible source-only checkpoint selection')
        self.report['checkpoints']=self.checkpoints
        self.report['checkpoint_selection']={'selected':str(selected),'expected_selected':str(expected_selected),'periodic_checkpoints_without_completed_selection_scores':checkpoint_gaps,'completed_checkpoint_native_evaluations':len(source_evaluations),'selected_sha256':self.sha(selected)}

        generated=rows(self.folder/'generated.jsonl');resources=read(self.folder/'generation-resource.json')
        self.binding(self.folder/'generated.jsonl',resources['generated_manifest_sha256'])
        self.need([r['id']for r in generated]==[r['id']for r in devrows]and len(generated)==24,'Final generated development coverage/order')
        self.need(resources['count']==24 and resources['generator_forwards_per_clip']==2 and resources['initial_generator_control_included']and resources['identity_control_equals_clean']and resources['source_model_calls']==resources['content_model_calls']==resources['vocoder_calls']==0,'Generation resource/scope declarations')
        bound_details=[]
        for row in generated:
            uid=row['id'];original=dev[uid]
            self.need(row['text']==original['text']and row['input_audio_sha256']==original['audio_sha256']and row['source_only']and row['seed']==11 and row['identity_control_equals_clean'],'Final generated input/scope '+uid)
            self.need(set(row['audio'])=={'clean','initial-generator','adversarial'},'Final waveform variants '+uid)
            waves={v:self.waveform(a,uid+'/'+v)for v,a in row['audio'].items()}
            self.binding(original['audio'],original['audio_sha256']);source_audio,sr=sf.read(original['audio'],dtype='float32')
            self.need(sr==16000 and source_audio.ndim==1 and np.array_equal(source_audio,waves['clean']),'Clean waveform differs from registered original '+uid)
            for variant in ['initial-generator','adversarial']:
                waveform=waves[variant];self.need(waveform.shape==source_audio.shape,'Output duration '+uid+'/'+variant)
                residual=waveform.astype('float64')-source_audio.astype('float64')
                linf=float(np.abs(residual).max());rms=float(np.sqrt(np.mean(residual**2)));peak=float(np.abs(waveform).max())
                self.need(linf<=reg['generator']['linf']+1e-7 and rms<=reg['generator']['rms_limit']+1e-7 and peak<=1,'Waveform physical bounds '+uid+'/'+variant)
                if variant=='adversarial':
                    b=row['bounds'];self.need(b['finite']and b['input_samples']==b['output_samples']==len(source_audio)and abs(linf-b['linf'])<1e-7 and abs(rms-b['rms'])<1e-7 and abs(peak-b['peak'])<1e-7,'Saved waveform bounds metadata '+uid)
                bound_details.append({'id':uid,'variant':variant,'samples':len(waveform),'linf':linf,'rms':rms,'peak':peak})
        self.report['generated']={'count':len(generated),'saved_waveforms':3*len(generated),'bounds_independently_recomputed':bound_details,'clean_samples_equal_registered_original':True}
        final_evaluation=self.evaluate(self.folder/'source-summary.json',dev,backend,provenance['source'],['clean','initial-generator','adversarial'])
        self.report['final_native_evaluation']=final_evaluation;self.report['checkpoint_native_evaluations']=source_evaluations
        for path,h in native['scorer_manifest']['hashes'].items():self.binding(path,h)
        self.need(native['scorer_manifest']==plan['quality'],'MOS scorer manifest differs from frozen quality policy')
        primary_path=ROOT/'reproduction-runs/paper-quality-stable-execution/pilot/quality-sum-lr-0.02/evaluation/summary.json'
        primary=read(primary_path)['summary']['adversarial'];a=final_evaluation['summary']['adversarial'];dwer=a['error_percent']-primary['error_percent'];dutmos=a['quality']['utmos']-primary['quality']['utmos'];admissible=is_admissible(a['quality'])
        self.report['primary_source_quality_screen']={'control':{'path':str(primary_path),'source_wer_percent':primary['error_percent'],'quality':primary['quality']},'delta_source_wer_points':dwer,'delta_utmos':dutmos,'automatic_quality_admissible':admissible,'source_quality_screen_passes':admissible and(dwer>=3 or(dutmos>=.05 and dwer>=-2)),'efficiency_screen_assessed':False,'human_or_heldout_transfer_established':False}
        self.report['cuda_initialised']=torch.cuda.is_initialized();self.need(not torch.cuda.is_initialized(),'CUDA initialized during CPU audit')
        self.report['status']='pass'if not self.issues else'issues_found'
        return 0 if not self.issues else 1
    def write(self):
        self.report['limits']=['Saved-artifact verification only; no neural inference or optimizer replay.','Trace finiteness verifies reported scalar losses/gradient norms, not all intermediate activation/gradient tensors.','Checkpoint tensors are loaded weights_only on CPU; terminal hash is producer-bound, while periodic checkpoint hashes are computed at audit time.','Generation is not reexecuted: checkpoint-to-WAV causal correspondence relies on registered producer code and receipts.','Source transcript use is constrained by frozen IDs and implementation; full training input transcripts are not repeated in trace rows.','MOS and source WER do not establish human intelligibility, held-out ASR transfer or improved generalization.','No final-reserve audio or held-out-target artifact is accessed. The earlier independent cohort audit covers sealed-reserve exclusion.']
        self.report['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (RUN/'r3-completed-audit.json').write_text(json.dumps(self.report,indent=2,allow_nan=False)+'\n')
        lines=['# Independent CPU audit of completed R3','',f"Updated {self.report['created_at_utc']}. Status: **{self.report['status']}**.",'',f'Run: `{self.folder}`. No neural model was loaded or executed; CUDA initialized: {torch.cuda.is_initialized()}.']
        if self.report['status']=='pending':lines+=['','Required completed artifacts are still pending. No successful completion audit is claimed.',*['- '+x for x in self.report.get('pending_files',[])]]
        if'training'in self.report:
            t=self.report['training'];lines+=['',f"Training: {t['updates']} AdamW updates, {t['completed_microbatches']} completed microbatches, {t['discarded_partial_microbatches']} discarded partial microbatches, {t['training_wall_seconds']:.3f} training wall seconds. Registered manifest order was replayed with Python seed 11. Accumulation, finite reported losses, fixed weights, completed checkpoint states and source-only selection were checked."]
        if'generated'in self.report:lines+=['',f"Generated {self.report['generated']['count']} development examples / {self.report['generated']['saved_waveforms']} WAVs. Clean samples equal registered source audio exactly; duration, peak, Linf and RMS bounds were independently checked for initial and trained generators."]
        if'final_native_evaluation'in self.report:
            lines+=['','| Variant | N | Source WER % | DNSMOS | NISQA | UTMOS |','|---|---:|---:|---:|---:|---:|']
            for v,a in self.report['final_native_evaluation']['summary'].items():
                q=a['quality'];lines.append(f"| {v} | {a['count']} | {a['error_percent']:.6f} | {q['dnsmos']:.6f} | {q['nisqa']:.6f} | {q['utmos']:.6f} |")
        if'primary_source_quality_screen'in self.report:
            s=self.report['primary_source_quality_screen'];lines+=['',f"Against B2 LR=.02: delta source WER {s['delta_source_wer_points']:+.6f} percentage points, delta UTMOS {s['delta_utmos']:+.6f}. Automatic quality admissible: {s['automatic_quality_admissible']}; registered source/quality screen passes: {s['source_quality_screen_passes']}. This screen excludes efficiency, human intelligibility and held-out transfer."]
        lines+=['','## Issues','',*['- '+x for x in(self.issues or['No discrepancy found in the completed scope.'if self.report['status']=='pass'else'Completion has not been verified.'])],'','## Limits','',*['- '+x for x in self.report['limits']]]
        (RUN/'r3-completed-audit.md').write_text('\n'.join(lines)+'\n')

def self_test():
    assert norm(' Ａ, B!  ')== 'a b'
    assert edit('A B C','A C')==(1,3)
    assert edit('A','')==(1,1)
    assert edit('a b','b a')==(2,2)
    assert not finite_tree({'x':[1,float('nan')]})
    assert finite_tree({'x':[1,None,True,2.0]})
    assert is_admissible({'utmos':3.5,'dnsmos':3,'nisqa':3.75})
    assert not is_admissible({'utmos':3.49,'dnsmos':3,'nisqa':3.75})
    assert safe_path(RUN/'r3-completed-audit.json').is_relative_to(OUT)
    assert not torch.cuda.is_initialized()
    print('R3 auditor helper self-test passed; no audit of live artifacts and no model inference.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='r3-waveform-autograd-repair')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    if Path(args.name).name!=args.name or not args.name.startswith('r3-'):parser.error('Use a simple R3 execution directory name')
    auditor=Audit(safe_path(RUN/args.name))
    try:result=auditor.run()
    except Exception as error:
        auditor.issues.append(f'Audit could not finish: {type(error).__name__}: {error}')
        auditor.report['status']='audit_error';auditor.report['traceback']=traceback.format_exc();result=1
    auditor.write()
    print(json.dumps({'status':auditor.report['status'],'issues':auditor.issues,'pending_files':auditor.report.get('pending_files',[]),'output':str(RUN/'r3-completed-audit.json'),'cuda_initialised':torch.cuda.is_initialized()},indent=2),flush=True)
    return result
if __name__=='__main__':sys.exit(main())
