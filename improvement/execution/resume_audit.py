"""Independent CPU audit of the resumed R1 deletion run and native score files.

No experiment implementation, model weights, final-reserve audio or neural
inference is imported/executed. Re-running refreshes only resume-audit.{json,md}.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import hashlib, json, math, statistics, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from whisper.tokenizer import get_tokenizer

torch.set_num_threads(1)
ROOT=Path('/workspace'); OUT=ROOT/'improvement'; RUN=OUT/'execution'
folder=RUN/'r1-deletion-development24'
issues=[]; hashes={}; counters=Counter(); maxima=Counter()
def need(ok, message):
    if not ok: issues.append(message)
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def sha(p):
    p=Path(p)
    if str(p)not in hashes:
        h=hashlib.sha256()
        with p.open('rb')as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        hashes[str(p)]=h.hexdigest()
    return hashes[str(p)]
def binding(p,h):
    p=Path(p);need(p.exists(),f'Missing binding {p}')
    if p.exists():need(sha(p)==h,f'Hash mismatch {p}')
def asset(a):
    if isinstance(a,dict)and'path'in a:
        h=a.get('sha256',a.get('file_sha256'))
        if h:binding(a['path'],h)
def value_sha(t):
    t=t.detach().contiguous()
    description=json.dumps({'dtype':str(t.dtype),'shape':list(t.shape)},sort_keys=True).encode()
    return hashlib.sha256(description+b'\0'+t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def norm(s):
    s=unicodedata.normalize('NFKC',s).lower()
    return' '.join(''.join(' 'if unicodedata.category(c).startswith('P')else c for c in s).split())
def edit(a,b):
    a,b=norm(a).split(),norm(b).split();previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1],len(a)
def count_check(reference,hypothesis,c,label):
    dist,n=edit(reference,hypothesis)
    need(dist==sum(c[k]for k in ['substitutions','deletions','insertions']),f'Word edit distance {label}')
    key='reference_units'if'reference_units'in c else'reference_words'
    need(n==c[key],f'Reference count {label}')
    if'reference_normalized'in c:need(norm(reference)==c['reference_normalized']and norm(hypothesis)==c['hypothesis_normalized'],f'Normalization {label}')
    return dist,n

plan=read(RUN/'plan.json');registration=read(RUN/'r1-native-plus-deletion-registration.json')
result=read(folder/'result.json');calibration=read(RUN/'r1-deletion-calibration/result.json')
development=rows(plan['development_manifest']);dev={r['id']:r for r in development}
cache={r['id']:r for r in read(plan['cache_audit'])['items']}
binding(RUN/'plan.json',registration['plan_sha256'])
binding(RUN/'r1-calibrated-development24/result.json',registration['canonical_stop_sha256'])
binding(RUN/'r1-native-plus-deletion-registration.json',result['bindings']['repair_registration_sha256'])
binding(RUN/'r1-deletion-calibration/result.json',result['calibration_sha256'])
for p,h in registration['code_hashes'].items():binding(p,h)
for p,h in plan['frozen_files'].items():binding(p,h)
for p,h in result['bindings']['code_hashes'].items():binding(p,h)
need(registration['created_at_utc']<calibration['created_at_utc']<result['created_at_utc'],'Registration order')
need(registration['maximum_repairs']==1 and registration['target_models_permitted']==[],'Repair scope')
need(result['status']=='completed'and result['actual_count']==result['planned_count']==24,'R1 run not complete24')
need(result['source_only']and result['target_models_evaluated']==[],'R1 source-only scope')
probes=calibration['calibration']['probes']
ratios=[p['nll_gradient_norm']/p['risk_gradient_norm']for p in probes]
scale=statistics.median(ratios)
need(scale==calibration['calibration']['scale']==result['scale']and len(probes)==3 and all(math.isfinite(x)and x>0 for x in ratios),'Calibration arithmetic')
need([p['clip_id']for p in probes]==registration['calibration_clips'],'Calibration cohort')
records=rows(folder/'generated.jsonl');need(len(records)==len({r['id']for r in records})==24,'R1 generated count')
need([r['id']for r in records]==[r['id']for r in development],'R1 cohort/order changed')
tokenizer=get_tokenizer(multilingual=True,language='en',task='transcribe')
details=[]
for r in records:
    uid=r['id'];need(r==read(folder/'records'/f'{uid}.json'),f'Record differs {uid}')
    need(r['text']==dev[uid]['text']and r['input_audio_sha256']==dev[uid]['audio_sha256'],f'Input identity {uid}')
    need(r['actual_optimization_updates']==len(r['trace'])==50,f'Update count {uid}')
    need(r['scale']==scale,f'Scale changed {uid}')
    need([b['step']for b in r['candidate_refreshes']]==list(range(0,50,5)),f'Refresh schedule {uid}')
    sets={}
    for b in r['candidate_refreshes']:
        asset(b);cset=read(b['path']);step=b['step'];cs=cset['candidates'];sets[step]=cset
        need(cset['reference']==r['text']and cset['refreshed_at']==step,f'Candidate cohort/step {uid}/{step}')
        origins=Counter(c['origin']for c in cs);counters.update({f'origin:{k}':v for k,v in origins.items()})
        need(origins=={'native_beam':4,'reference':1,'synthetic_source_best_middle_word_deletion':1},f'Candidate origin counts {uid}/{step}')
        need(cset['decoded_count']==4 and len({tuple(c['tokens'])for c in cs})==len(cs),f'Candidate uniqueness {uid}/{step}')
        native=[c for c in cs if c['origin']=='native_beam']
        need(all(native[i]['search_score']>=native[i+1]['search_score']for i in range(len(native)-1)),f'Native rank order {uid}/{step}')
        policy=cset['search_options'];repair=policy['deletion_repair']
        need(policy['beam_size']==4 and policy['temperature']==0 and policy['without_timestamps']and policy['lexical_or_eos_only']and policy['rank_normalization']=='lexical_length+1',f'Native policy {uid}/{step}')
        source=next(c for c in native if c['text'].split());words=source['text'].split();index=len(words)//2
        synthetic=' '.join(words[:index]+words[index+1:])
        need(repair['source_text']==source['text']and repair['source_tokens']==source['tokens']and repair['deleted_word_index']==index and repair['deleted_word']==words[index]and repair['synthetic_text']==synthetic,f'Deletion construction {uid}/{step}')
        need(repair['synthetic_tokens']==tokenizer.encode(synthetic)and repair['added_after_deduplication'],f'Deletion tokenization {uid}/{step}')
        need(not repair['posterior_beam_hypothesis']and not repair['additional_fallbacks']and repair['maximum_synthetic_candidates']==1,f'Deletion scope {uid}/{step}')
        syn=next(c for c in cs if c['origin']=='synthetic_source_best_middle_word_deletion')
        need(syn['tokens']==repair['synthetic_tokens'],f'Wrong synthetic candidate {uid}/{step}')
        for c in cs:
            need(all(0<=t<tokenizer.eot for t in c['tokens']),f'Nonlexical token {uid}/{step}')
            need(tokenizer.decode(c['tokens']).strip()==c['text'],f'Decoded text mismatch {uid}/{step}')
            dist,n=count_check(r['text'],c['text'],c,f'{uid}/{step}/{c["origin"]}')
            need(abs(min(dist/max(1,n),1)-c['cost'])<1e-12,f'Candidate cost {uid}/{step}')
            counters['candidate_rows']+=1
        counters['candidate_refreshes']+=1
    for i,t in enumerate(r['trace']):
        label=f'{uid}/{i}';need(t['step']==i+1 and t['candidate_step_zero_based']==i and t['refreshed_at']==i//5*5,f'Trace ordering {label}')
        cs=sets[i//5*5]['candidates'];costs=np.asarray([c['cost']for c in cs]);scores=np.asarray(t['scores']);p=np.exp(scores-scores.max());p/=p.sum()
        need(len(scores)==len(costs)==t['candidate_count'],f'Trace candidate count {label}')
        need(np.max(np.abs(costs-np.asarray(t['costs'])))<1e-7,f'Trace candidate cost {label}')
        need(np.max(np.abs(p-np.asarray(t['posterior'])))<1e-6 and abs(float(p@costs)-t['risk'])<1e-6,f'Risk arithmetic {label}')
        entropy=float(-(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum())
        need(abs(entropy-t['posterior_entropy'])<1e-6,f'Posterior entropy {label}')
        loss=-scale*t['risk']+t['tv']+t['hf'];err=abs(loss-t['loss']);maxima['loss_scalar_roundoff']=max(maxima['loss_scalar_roundoff'],err)
        need(err<5e-5,f'Full loss arithmetic {label}: {err}')
        need(math.isfinite(t['risk_gradient_norm'])and t['risk_gradient_norm']>0 and math.isfinite(t['total_gradient_norm'])and t['radius']<=.100001,f'Trace finite gradient/radius {label}')
        need((t['nll']is not None)==(i%5==0 or i==49),f'NLL diagnostic schedule {label}')
        need(t['unique_token_paths']==len(cs)and t['unique_normalized_texts']==len({norm(c['text'])for c in cs})and t['empty_candidate_count']==sum(not c['tokens']for c in cs),f'Trace candidate diagnostics {label}')
        counters['optimizer_updates']+=1
    asset(r['delta']);asset(r['optimizer']);qinfo=cache[uid];binding(qinfo['q_path'],qinfo['q_sha256'])
    delta=torch.load(r['delta']['path'],map_location='cpu',weights_only=True);q=torch.load(qinfo['q_path'],map_location='cpu',weights_only=True)
    need(torch.isfinite(delta).all().item()and delta.shape==q.shape,f'Delta shape/finite {uid}')
    need(value_sha(delta)==r['delta']['value_sha256']==r['trace'][-1]['delta_value_sha256'],f'Final delta value binding {uid}')
    radius=float(np.linalg.norm(delta.numpy().astype('float64'))/(np.linalg.norm(q.numpy().astype('float64'))+1e-8))
    need(radius<=.100001 and abs(radius-r['actual_relative_radius'])<1e-7 and abs(radius-r['trace'][-1]['radius'])<1e-7,f'Final physical radius {uid}')
    optimizer=torch.load(r['optimizer']['path'],map_location='cpu',weights_only=True)
    need(len(optimizer['state'])==1 and float(optimizer['state'][0]['step'])==50 and optimizer['param_groups'][0]['lr']==.1,f'Optimizer final count/LR {uid}')
    for key,tensor in optimizer['state'][0].items():need(value_sha(tensor)==r['optimizer']['value_receipt']['state']['0'][key]['tensor_sha256'],f'Optimizer tensor binding {uid}/{key}')
    waves={}
    for variant,a in r['audio'].items():
        asset(a);x,sr=sf.read(a['path'],dtype='float32');waves[variant]=x
        need(sr==a['sample_rate']==16000 and x.ndim==1 and len(x)==a['samples']and np.isfinite(x).all(),f'WAV format {uid}/{variant}')
        need(abs(float(np.abs(x).max())-a['peak'])<1e-7,f'WAV peak {uid}/{variant}')
        counters['waveforms']+=1
    need(waves['reconstruction'].shape==waves['adversarial'].shape,f'Vocoder output shape {uid}')
    count_check(r['text'],r['source_hypothesis'],r['source_error_counts'],f'{uid}/generation-source-diagnostic')
    details.append({'id':uid,'updates':50,'candidate_refreshes':10,'physical_radius':radius,'clean_samples':len(waves['clean']),'vocoder_samples':len(waves['adversarial'])})

# Native evaluation is checked only once its complete receipt has been written.
# Live incomplete outputs are reported as pending, never as successful results.
evaluations={};expected_backend=read(Path(plan['baseline_run'])/'evaluation/summary.json')['evaluation_backend']
for summary_path in sorted(RUN.rglob('source-summary.json')):
    if not any(part.startswith(('r1-deletion','r2-canonical','r3-'))for part in summary_path.parts):continue
    summary=read(summary_path);score_path=summary_path.parent/'source-scores.jsonl'
    if summary.get('status')!='complete'or not score_path.exists():continue
    binding(score_path,summary['scores_sha256']);scores=rows(score_path)
    need(summary['source_only']and summary['target_models_evaluated']==[],f'Evaluation source-only {summary_path}')
    need(summary['evaluation_backend']==expected_backend,f'Native backend changed {summary_path}')
    er=read(summary_path.parent/'evaluation-registration.json');need(er['backend']==expected_backend,f'Evaluation registration policy {summary_path}')
    binding(er['generated_manifest'],er['generated_manifest_sha256']);gr={r['id']:r for r in rows(er['generated_manifest'])}
    aggregates={}
    for score in scores:
        uid=score['id'];need(uid in dev and score['reference']==dev[uid]['text'],f'Non-development or altered reference {summary_path}/{uid}')
        count_check(score['reference'],score['hypothesis'],score,f'{summary_path}/{uid}')
        need(score['audio_sha256']==gr[uid]['audio'][score['variant']]['sha256'],f'Score WAV binding {summary_path}/{uid}')
    for variant in sorted({s['variant']for s in scores}):
        rs=[s for s in scores if s['variant']==variant]
        a={'count':len(rs),**{k:sum(s[k]for s in rs)for k in ['substitutions','deletions','insertions','reference_units']}}
        a['error_percent']=100*sum(a[k]for k in ['substitutions','deletions','insertions'])/a['reference_units'];a['quality']={k:statistics.fmean(s['quality'][k]for s in rs)for k in ['dnsmos','nisqa','utmos']}
        need(all(a[k]==summary['summary'][variant][k]for k in a if k!='quality')and all(abs(a['quality'][k]-summary['summary'][variant]['quality'][k])<1e-12 for k in a['quality']),f'Score aggregate mismatch {summary_path}/{variant}')
        aggregates[variant]=a
    evaluations[str(summary_path)]={'score_rows':len(scores),'summary':aggregates,'native_backend_matches_old_policy':True}

control_path=ROOT/'reproduction-runs/paper-quality-stable-execution/pilot/quality-sum-lr-0.02/evaluation/summary.json'
control=read(control_path)['summary']['adversarial'];screen={}
for path,e in evaluations.items():
    a=e['summary'].get('adversarial')
    if a is None or a['count']!=24:continue
    quality=a['quality'];admissible=quality['utmos']>=3.5 and quality['dnsmos']>=3 and quality['nisqa']>=3.75
    dwer=a['error_percent']-control['error_percent'];dutmos=quality['utmos']-control['quality']['utmos']
    screen[path]={'primary_control':str(control_path),'control_source_wer_percent':control['error_percent'],'control_utmos':control['quality']['utmos'],'delta_source_wer_points':dwer,'delta_utmos':dutmos,'automatic_quality_admissible':admissible,'source_quality_screen_passes':admissible and(dwer>=3 or(dutmos>=.05 and dwer>=-2)),'efficiency_screen_assessed':False,'human_or_transfer_claim_supported':False}
limitations=['CPU saved-artifact arithmetic and bindings only; no neural inference or optimizer trajectory replay.','Only the terminal delta and optimizer tensors are saved. Intermediate gradient/delta hashes cannot reconstruct those vectors.','Calibration norm ratios are recomputed from producer-saved scalar norms, not independently recomputed neural gradients.','Tokenization and decoding checks use the installed frozen public Whisper tokenizer; no ASR model is loaded.','Independent edit dynamic programming verifies total minimum edit distance; individual S/D/I alignments can have ties.','Feature-ball validity, automatic MOS and source recognition do not establish human intelligibility or held-out ASR transfer.','No final-reserve audio or held-out-target artifact is accessed by this audit. This snapshot can be refreshed when pending native evaluations finish.']
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'pass'if not issues else'issues_found','cpu_only':True,'cuda_initialised':torch.cuda.is_initialized(),'neural_inference':False,'issues':issues,'scope':'R1 repaired full24 saved candidates, traces, native tokenizer, final tensors, WAV integrity, preregistration bindings and completed native score arithmetic.','counts':dict(counters),'maximum_scalar_roundoff':dict(maxima),'r1_calibration':{'scale':scale,'ratios':ratios,'gradient_vectors_recomputed':False},'r1_development':details,'native_evaluations':evaluations,'source_quality_screen':screen,'primary_quality_control':control,'file_sha256':hashes,'limits':limitations}
need(not report['cuda_initialised'],'CUDA unexpectedly initialized')
report['status']='pass'if not issues else'issues_found'
(RUN/'resume-audit.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Independent CPU audit after resumption','',f"Updated {report['created_at_utc']}. Status: **{report['status']}**.",'',f"R1 repaired generation: {len(records)} development clips, {counters['optimizer_updates']} updates, {counters['candidate_refreshes']} candidate refreshes, {counters['candidate_rows']} candidate rows, {counters['waveforms']} saved WAVs. All registered IDs/references, one middle-word deletion per refresh, lexical tokens, detached word costs, score/posterior/risk/loss arithmetic, terminal feature radii, optimizer state and saved WAV bindings were checked independently on CPU.",'',f'Fixed gradient scale recomputed from saved scalar norms: {scale:.12f}; ratios {ratios}. CUDA initialized: {torch.cuda.is_initialized()}. Neural models loaded/executed: none.','', '| Run / variant | N | WER % | DNSMOS | NISQA | UTMOS |','|---|---:|---:|---:|---:|---:|']
for path,e in evaluations.items():
    for v,a in e['summary'].items():
        q=a['quality'];lines.append(f"| {Path(path).parent.name} / {v} | {a['count']} | {a['error_percent']:.6f} | {q['dnsmos']:.6f} | {q['nisqa']:.6f} | {q['utmos']:.6f} |")
lines+=['',f"Primary quality-admissible comparator is B2 LR=.02, source WER {control['error_percent']:.6f}%, UTMOS {control['quality']['utmos']:.6f}. Attenuated strong-baseline output is a norm diagnostic; it does not replace that comparator.",'']
for path,s in screen.items():lines.append(f"{Path(path).parent.name}: delta source WER {s['delta_source_wer_points']:+.6f} percentage points; delta UTMOS {s['delta_utmos']:+.6f}. Automatic quality admissible: {s['automatic_quality_admissible']}; registered source/quality screen passes: {s['source_quality_screen_passes']}. Efficiency, human intelligibility and transfer are not established.")
if not any('r1-deletion-development24'in p for p in evaluations):lines+=['','Repaired R1 native evaluation is pending; generation-time strict-backend source transcripts are diagnostic and are not substituted for formal matched-policy results.']
lines+=['','## Limits','',*['- '+x for x in limitations],'','## Discrepancies','',*['- '+x for x in(issues or['No discrepancy found in this saved-artifact scope.'])],'','Refresh with `PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python /workspace/improvement/execution/resume_audit.py`.']
(RUN/'resume-audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'issues':issues,'counts':dict(counters),'native_evaluations':list(evaluations),'cuda_initialised':torch.cuda.is_initialized()},indent=2),flush=True)
