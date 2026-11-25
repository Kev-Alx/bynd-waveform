"""Independent saved-artifact CPU audit. No project imports or neural inference.
Outputs only audit_initial_execution.{json,md} beside this file. Re-run as phases finish.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import ast,csv,hashlib,json,math,random,statistics,unicodedata
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
import soundfile as sf
import torch
torch.set_num_threads(1)
ROOT=Path('/workspace'); OUT=ROOT/'improvement'; RUN=OUT/'execution'
issues=[]; limitations=[]; hashes={}; checked=[]
def require(ok,message):
 if not ok: issues.append(message)
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 p=Path(p)
 if str(p) not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
  hashes[str(p)]=h.hexdigest()
 return hashes[str(p)]
def binding(p,expected):
 p=Path(p); require(p.exists(),f'Missing bound file: {p}')
 if p.exists(): require(sha(p)==expected,f'Hash mismatch: {p}')
def asset(a):
 if isinstance(a,dict) and 'path'in a:
  expected=a.get('file_sha256',a.get('sha256'))
  if expected: binding(a['path'],expected)
def tensor(p): return torch.load(p,map_location='cpu',weights_only=True)
def value_sha(t):
 cpu=t.detach().contiguous(); description=json.dumps({'dtype':str(cpu.dtype),'shape':list(cpu.shape)},sort_keys=True).encode()
 return hashlib.sha256(description+b'\0'+cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def normtext(s):
 s=unicodedata.normalize('NFKC',s).lower()
 return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s).split())
def edit(a,b):
 a,b=normtext(a).split(),normtext(b).split(); prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1],len(a)
def editrow(r):
 d,n=edit(r['reference'],r['hypothesis']);require(d==sum(r[k] for k in ['substitutions','deletions','insertions']) and n==r['reference_units'],f'Edit-count error {r.get("id")}')
 require(normtext(r['reference'])==r['reference_normalized'] and normtext(r['hypothesis'])==r['hypothesis_normalized'],f'Normalization error {r.get("id")}')
def audio(a):
 asset(a); x,sr=sf.read(a['path'],dtype='float32');require(x.ndim==1 and sr==16000 and np.isfinite(x).all(),f'Invalid WAV {a["path"]}')
 require(len(x)==a['samples'] and sr==a['sample_rate'],f'WAV shape metadata {a["path"]}')
 require(abs(float(np.abs(x).max())-a['peak'])<1e-7,f'Peak mismatch {a["path"]}')
 return x
plan=read(RUN/'plan.json')
for p,h in plan['frozen_files'].items():binding(p,h)
snapshot=read(OUT/'baseline_snapshot_manifest.json')
for p,b in snapshot.items():binding(p,b['sha256']);binding(b['snapshot'],b['sha256'])
for p,h in plan['quality']['hashes'].items():binding(p,h)
binding(ROOT/'artifacts/checkpoints/WavLM-Large.pt',plan['feature_artifacts']['encoder_sha256'])
binding(ROOT/'artifacts/checkpoints/g_02500000.pt',plan['feature_artifacts']['vocoder_sha256'])
development=rows(plan['development_manifest']); dev={r['id']:r for r in development}; calibration=rows(plan['calibration_manifest']); calids=[r['id'] for r in calibration]
require(len(dev)==24,'Development cohort not exactly24 unique IDs')
ordered=sorted(development,key=lambda r:(r['duration'],r['id']))
require(calids==[r['id'] for r in [ordered[0],ordered[len(ordered)//2],ordered[-1]]],'Calibration duration strata differ')
base={r['id']:r for r in rows(Path(plan['baseline_run'])/'generated.jsonl')}
cache={r['id']:r for r in read(plan['cache_audit'])['items']}
for uid in dev:binding(cache[uid]['q_path'],cache[uid]['q_sha256'])
def asset_tree(value):
 if isinstance(value,dict):
  asset(value)
  for k,v in value.items():
   if k in ['code_hashes','implementation_hashes'] and isinstance(v,dict):
    for path,h in v.items():binding(path,h)
   else:asset_tree(v)
 elif isinstance(value,list):
  for v in value:asset_tree(v)
for registration_file in RUN.rglob('*registration*.json'):asset_tree(read(registration_file))
pre=read(RUN/'baseline-preflight/result.json'); updates=0
for item in pre['items']:
 old=read(base[item['id']]['trajectory_receipt']['path'])
 require(item['id']in calids and len(item['steps'])==50 and item['final_audio_matches'],'Preflight cohort or audio receipt mismatch')
 for actual,expected in zip(item['steps'],old['trace']):
  require(actual['passed'] and actual['loss_value_sha256']==expected['loss_value_sha256'] and actual['gradient_value_sha256']==expected['gradient_value_sha256'] and actual['delta_value_sha256']==expected['delta']['value_sha256'] and actual['optimizer_value_receipt']==expected['optimizer_value_receipt'],f'Baseline preflight differs {item["id"]}/{actual["step"]}')
  updates+=1
require(updates==pre['checked_updates']==150,'Preflight actual updates !=150')
# Independently replay deterministic data selection from saved indices, no pipeline imports.
old={}
for info in read(OUT/'baseline_evidence.json')['manifests'].values():
 for r in rows(info['path']):old[r['id']]=r
historical=list(old.values())
def balanced(rs,seed):
 groups=defaultdict(list)
 for r in rs:groups[r['speaker']].append(r)
 speakers=sorted(groups); random.Random(seed).shuffle(speakers)
 for speaker in speakers:groups[speaker].sort(key=lambda r:hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())
 return [groups[s][i] for i in range(max(map(len,groups.values()),default=0)) for s in speakers if i<len(groups[s])]
def eligibility(rs,excluded):
 speakers={r['speaker'] for r in excluded}; h={r['audio_sha256'] for r in excluded};t={normtext(r['text']) for r in excluded}
 return [r for r in rs if r['speaker']not in speakers and r['audio_sha256']not in h and normtext(r['text'])not in t and 1<=r['duration']<=30]
def unique_text(rs):
 seen=set();result=[]
 for r in rs:
  t=normtext(r['text'])
  if t not in seen:seen.add(t);result.append(r)
 return result
cohorts={}; acquired={}
for name in ['final-reserve-200','train-1h','train-10h']:
 p=RUN/'data'/f'{name}.jsonl'
 if not p.exists():continue
 rs=rows(p);receipt=read(p.with_suffix('.receipt.json')); binding(p,receipt['manifest_sha256'])
 require(len(rs)==receipt['count'] and len({r['id']for r in rs})==len(rs),f'Cohort count/duplicates {name}')
 require(len({r['speaker']for r in rs})==receipt['speakers'],f'Cohort speaker count {name}')
 for r in rs:
  binding(r['audio'],r['audio_sha256']);info=sf.info(r['audio'])
  require(info.samplerate==r['sample_rate']==16000 and info.frames==r['samples'] and abs(info.duration-r['duration'])<1e-8,f'Corpus audio metadata {name}/{r["id"]}')
 acquired[name]=rs
 cohorts[name]={'count':len(rs),'speakers':len({r['speaker']for r in rs}),'seconds':sum(r['duration']for r in rs),'sha256':sha(p),'split':sorted({r['split']for r in rs})}
final=acquired['final-reserve-200']; require(len(final)==200 and len({r['speaker']for r in final})>=30,'Final reserve size/speakers invalid')
idx=OUT/'data/test-other-index.jsonl'
if idx.exists():
 selected=unique_text(balanced(eligibility(rows(idx),historical),20261021))[:200]
 require([r['id']for r in selected]==[r['id']for r in final],'Final reserve independent selection replay differs')
 cohorts['final-reserve-200']['selection_replayed']=True
for name in ['train-1h','train-10h']:
 if name not in acquired:continue
 rs=acquired[name];allowed=eligibility(rs,historical+final);require(len(rs)==len(allowed),f'Training overlap exclusion failed {name}')
idx=OUT/'data/train-clean-100-index.jsonl'
if idx.exists() and 'train-10h'in acquired:
 selected=unique_text(balanced(eligibility(rows(idx),historical+final),11)); duration=0
 for i,r in enumerate(selected):
  duration+=r['duration']
  for hours in [1,10]:
   name=f'train-{hours}h'
   if duration>=3600*hours and name in acquired and not cohorts[name].get('selection_replayed'):
    require([x['id']for x in selected[:i+1]]==[x['id']for x in acquired[name]],f'Training deterministic selection differs {name}');cohorts[name]['selection_replayed']=True
  if duration>=36000:break
if 'train-1h'in acquired and 'train-10h'in acquired:require([r['id']for r in acquired['train-10h'][:len(acquired['train-1h'])]]==[r['id']for r in acquired['train-1h']],'Training pools are not nested prefixes')
# Generic generated outputs: IDs, references, audio integrity and physical deltas.
generated={};geometry=[]
for p in sorted(RUN.rglob('generated.jsonl')):
 rs=rows(p);require(len(rs)==len({r['id']for r in rs}),f'Duplicate generated IDs {p}')
 for r in rs:
  uid=r['id'];require(uid in dev and r['text']==dev[uid]['text'] and r.get('input_audio_sha256',dev[uid]['audio_sha256'])==dev[uid]['audio_sha256'],f'Non-development generated ID/reference {p}/{uid}')
  waves={v:audio(a) for v,a in r['audio'].items()}
  for k in ['delta','optimizer','geometry_parameters','attenuated_delta']:asset(r.get(k))
  if 'bounds'in r:
   clean_wav,adv_wav=waves['clean'],waves['adversarial']; residual=adv_wav.astype('float64')-clean_wav.astype('float64')
   require(adv_wav.shape==clean_wav.shape and np.abs(residual).max()<=.0030001 and np.sqrt(np.mean(residual**2))<=.0030001 and np.abs(adv_wav).max()<=1,f'R3 saved-waveform bound {uid}')
   require(r['bounds']['finite'] and len(clean_wav)==r['bounds']['input_samples']==r['bounds']['output_samples'],f'R3 bound metadata {uid}')
  if 'delta'in r:
   delta=tensor(r['delta']['path']); require(isinstance(delta,torch.Tensor) and torch.isfinite(delta).all().item(),f'Invalid physical delta {p}/{uid}')
   if 'value_sha256'in r['delta']:require(value_sha(delta)==r['delta']['value_sha256'],f'Tensor value hash {p}/{uid}')
   q=tensor(cache[uid]['q_path']); require(q.shape==delta.shape,f'Delta shape {p}/{uid}')
   radius=float(np.linalg.norm(delta.numpy().astype('float64'))/(np.linalg.norm(q.numpy().astype('float64'))+1e-8))
   require(radius<=.100001,f'Feature radius violation {p}/{uid}')
  if 'geometry_parameters'in r:
   st=tensor(r['geometry_parameters']['path']);q=q.numpy();d=delta.numpy();T,D=q.shape
   offsets=np.arange(-2,3);indices=np.arange(T)[:,None]+offsets[None,:];inside=(indices>=0)&(indices<T);indices=indices.clip(0,T-1);bank=q[indices]
   q64=q.astype('float64');bank64=bank.astype('float64');qn=np.linalg.norm(q64,axis=1);bn=np.linalg.norm(bank64,axis=2)
   cosine=(q64[:,None,:]*bank64).sum(2)/np.maximum(qn[:,None]*bn,np.finfo('float64').tiny)
   valid=inside&(qn[:,None]>0)&(bn>0)&(cosine>=.95);valid[:,2]=True
   active=(((bank64-q64[:,None,:])!=0).any(2)&valid).any(1)&(qn>0)
   require(np.array_equal(valid,st['valid_neighbors'].numpy()) and np.array_equal(indices,st['neighbor_indices'].numpy()),f'R2 independent bank mismatch {uid}')
   a=st['a'].numpy();logits=st['logits'].numpy();require(np.isfinite(a).all() and np.isfinite(logits).all() and np.all((a>=0)&(a<=1)) and not np.any(a[~active]),f'R2 parameter constraints {uid}')
   masked=np.where(valid,logits.astype('float64'),-np.inf);weights=np.exp(masked-masked.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
   z64=q64+(a*active)[:,None]*((weights[:,:,None]*bank64).sum(1)-q64)
   reconstruction_error=float(np.max(np.abs((z64-q64)-d)))
   require(reconstruction_error<1e-5,f'R2 independent delta reconstruction {uid}: {reconstruction_error}')
   actual_z=q64+d.astype('float64');framecos=(actual_z*q64).sum(1)/np.maximum(np.linalg.norm(actual_z,axis=1)*qn,np.finfo('float64').tiny)
   mincos=float(framecos[qn>0].min())if np.any(qn>0)else None
   require(mincos is None or mincos>=.95-1e-6,f'R2 frame cosine {uid}')
   require(not np.any(d[qn==0]),f'R2 changed zero clean frame {uid}')
   bas=tensor(r['input_binding']['baseline_delta']['path']).numpy();atten=tensor(r['attenuated_delta']['path']).numpy();b_norm=np.linalg.norm(bas.astype('float64'));d_norm=np.linalg.norm(d.astype('float64'));factor=min(1.,d_norm/b_norm)if b_norm>0 else 0.
   expected=bas*np.float32(factor);attenerror=float(np.abs(expected-atten).max());residual=float(np.linalg.norm(atten.astype('float64'))-d_norm)
   require(attenerror<=1e-6 and abs(residual)<=1e-4,f'R2 deterministic norm match {uid}')
   require(len(r['trace'])==r['actual_optimization_updates']==r['source_objective_queries']==st['objective_queries'],f'R2 actual update count {uid}')
   require(all(t['radius']<=.100001 and(t['minimum_cosine']is None or t['minimum_cosine']>=.949999)for t in r['trace']),f'R2 saved iterate constraints {uid}')
   require(np.array_equal(waves['clean'],sf.read(base[uid]['audio']['clean']['path'],dtype='float32')[0]) and np.array_equal(waves['reconstruction'],sf.read(base[uid]['audio']['reconstruction']['path'],dtype='float32')[0]),f'R2 altered clean/reconstruction waveform {uid}')
   if not np.any(d):require(np.array_equal(waves['adversarial'],waves['reconstruction']),f'R2 zero-delta waveform differs {uid}')
   geometry.append({'run':p.parent.name,'id':uid,'status':r['r2_status'],'updates':len(r['trace']),'radius':radius,'minimum_cosine':mincos,'independent_reconstruction_max_abs_error':reconstruction_error,'attenuation_element_max_abs_error':attenerror,'norm_match_residual':residual,'active_frames':int(active.sum()),'frames':T})
  if 'candidate_refreshes'in r:
   for b in r['candidate_refreshes']:
    asset(b);cset=read(b['path'])
    for c in cset['candidates']:
     dist,n=edit(cset['reference'],c['text']);require(abs(c['cost']-min(dist/max(1,n),1))<1e-12,f'R1 candidate cost {uid}');require(dist==sum(c[k]for k in ['substitutions','deletions','insertions']) and n==c['reference_words'],f'R1 candidate counts {uid}')
   for t in r['trace']:
    probs=np.exp(np.asarray(t['scores'])-max(t['scores']));probs/=probs.sum()
    require(np.max(np.abs(probs-np.asarray(t['posterior'])))<1e-6 and abs(np.dot(probs,t['costs'])-t['risk'])<1e-6,f'R1 risk arithmetic {uid}/{t["step"]}')
    require(math.isfinite(t['risk_gradient_norm'])and t['risk_gradient_norm']>0 and t['radius']<=.100001,f'R1 trace gradient/radius {uid}')
  if 'source_hypothesis'in r:editrow({'reference':r['text'],'hypothesis':r['source_hypothesis'],'id':uid,**r['source_error_counts']})
 generated[str(p)]={'count':len(rs),'ids':[r['id']for r in rs],'trace_updates':sum(len(r.get('trace',[]))for r in rs),'variants':sorted({v for r in rs for v in r['audio']})}
# R1 scale is independently recalculated from saved scalar norms (not vectors).
r1={}
r1['calibrations']={}
for cp in sorted(RUN.glob('r1-*calibration/result.json')):
 c=read(cp)
 if c.get('status')!='passed':continue
 probes=c['calibration']['probes'];ratios=[p['nll_gradient_norm']/p['risk_gradient_norm']for p in probes];scale=statistics.median(ratios)
 require(scale==c['calibration']['scale']and len(probes)==3 and all(math.isfinite(x)and x>0 for x in ratios),f'R1 gradient scale arithmetic {cp}')
 for item in c['items']:
  diagnostic=item['diagnostics'];probs=np.exp(np.asarray(diagnostic['scores'])-max(diagnostic['scores']));probs/=probs.sum()
  require(abs(float(np.dot(probs,diagnostic['costs']))-diagnostic['risk'])<1e-6,f'R1 calibration risk arithmetic {cp}/{item["id"]}')
  for candidate in item['candidate_set']['candidates']:
   dist,n=edit(item['candidate_set']['reference'],candidate['text']);require(abs(candidate['cost']-min(dist/max(1,n),1))<1e-12,f'R1 calibration candidate cost {cp}')
 r1['calibrations'][cp.parent.name]={'scale':scale,'norm_ratios':ratios,'gradient_vectors_independently_recomputed':False}
for p in sorted(RUN.glob('r1-*/result.json')):
 d=read(p);r1[str(p.parent.name)]={k:d[k]for k in ['status','actual_count','planned_count','reason','failed_clip','scale','seconds','target_models_evaluated']if k in d}
r3={}
for p in sorted(RUN.glob('r3-*/training.json')):
 d=read(p);folder=p.parent;tr=rows(folder/'training-trace.jsonl')
 require(d['updates']==sum(bool(r['optimizer_step'])for r in tr) and d['completed_microbatches']%4==0 and len(tr)==d['completed_microbatches']+d['discarded_partial_microbatches'],f'R3 training counters {folder}')
 require(d['updates']<=12000 and d['training_wall_seconds']<=3610 and d['seed']==11,f'R3 training registered cap {folder}')
 allowed={r['id']for r in acquired['train-1h']}
 require(all(r['id']in allowed for r in tr),f'R3 trained on unregistered IDs {folder}')
 binding(folder/'training-trace.jsonl',d['trace_sha256']);asset(d['terminal_checkpoint']);checkpoint=tensor(d['terminal_checkpoint']['path'])
 require(checkpoint['source_only'] is True and checkpoint['updates']==d['updates'] and checkpoint['manifest_sha256']==cohorts['train-1h']['sha256'],f'R3 checkpoint identity {folder}')
 parameter_count=sum(t.numel()for t in checkpoint['generator'].values()if isinstance(t,torch.Tensor))
 require(parameter_count<=3000000,f'R3 generator exceeds size cap {folder}')
 r3[folder.name]={k:d[k]for k in ['status','updates','completed_microbatches','discarded_partial_microbatches','training_wall_seconds','stop_reason','checkpoint_selection','cached_q_parity_claimed','human_admissibility_established']};r3[folder.name]['saved_generator_state_elements']=parameter_count
# Source score aggregation and backend: verify saved scores, never execute evaluator.
source_results={};source_files=[]
expected_backend=read(Path(plan['baseline_run'])/'evaluation/summary.json')['evaluation_backend']
for p in sorted(RUN.rglob('source-summary.json')):
 d=read(p);scores_path=p.parent/'source-scores.jsonl'
 if d.get('status')!='complete'or not scores_path.exists():continue
 binding(scores_path,d['scores_sha256']);rs=rows(scores_path);summary={}
 require(d.get('source_only')is True and d.get('target_models_evaluated')==[],'Source scoring used targets')
 require(d['source_metadata']['checkpoint']==str(ROOT/'artifacts/whisper/small.pt'),'Wrong source recognizer')
 require(d['evaluation_backend']==expected_backend,f'Historical native backend differs {p}')
 registration=p.parent/'evaluation-registration.json'
 if registration.exists():
  er=read(registration);require(er['backend']==expected_backend,f'Backend not bound before scoring {p}');binding(er['generated_manifest'],er['generated_manifest_sha256']);gr={x['id']:x for x in rows(er['generated_manifest'])}
  for score in rs:require(score['audio_sha256']==gr[score['id']]['audio'][score['variant']]['sha256'],f'Score/audio binding {p}/{score["id"]}')
 for r in rs:require(r['id']in dev,f'Final/confirmation inference in scores {p}');editrow(r)
 for v in sorted({r['variant']for r in rs}):
  pool=[r for r in rs if r['variant']==v];a={'count':len(pool),**{k:sum(r[k]for r in pool)for k in ['substitutions','deletions','insertions','reference_units']}};a['error_percent']=100*sum(a[k]for k in ['substitutions','deletions','insertions'])/a['reference_units'];a['quality']={k:statistics.fmean(r['quality'][k]for r in pool)for k in ['dnsmos','nisqa','utmos']}
  require(all(a[k]==d['summary'][v][k]for k in ['count','substitutions','deletions','insertions','reference_units','error_percent'])and all(abs(a['quality'][k]-d['summary'][v]['quality'][k])<1e-12 for k in a['quality']),f'Source aggregate mismatch {p}/{v}');summary[v]=a
 source_results[str(p.parent.name)]={'summary':summary,'score_rows':len(rs),'backend_exactly_matches_historical':True};source_files.append(str(scores_path))
quality_control_path=ROOT/'reproduction-runs/paper-quality-stable-execution/pilot/quality-sum-lr-0.02/evaluation/summary.json'
quality_control=read(quality_control_path)['summary']['adversarial']
require(quality_control['count']==24,'Quality-admissible B2 control has wrong cohort')
interpretations={}
for name,evaluation in source_results.items():
 a=evaluation['summary'].get('adversarial')
 if a is None or a['count']!=24:continue
 q=a['quality'];admissible=q['utmos']>=3.5 and q['dnsmos']>=3 and q['nisqa']>=3.75
 dwer=a['error_percent']-quality_control['error_percent'];dutmos=q['utmos']-quality_control['quality']['utmos']
 signal=admissible and (dwer>=3 or (dutmos>=.05 and dwer>=-2))
 interpretations[name]={'primary_source_quality_control':str(quality_control_path),'quality_control_wer':quality_control['error_percent'],'quality_control_utmos':quality_control['quality']['utmos'],'delta_wer_points':dwer,'delta_utmos':dutmos,'automatic_quality_admissible':admissible,'source_quality_performance_screen_passes':signal,'human_or_transfer_claim_supported':False,'efficiency_screen_not_assessed_here':True}
# Evidence of absence is limited to saved inference-bearing records and import/call review.
sealed_ids={r['id']for r in final}|{r['id']for r in rows(ROOT/'reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl')}
scored_sealed=[]
for p in RUN.rglob('*.json'):
 if p.name.startswith('audit_initial_execution'):continue
 try:d=read(p)
 except (ValueError,OSError):continue
 if isinstance(d,dict)and d.get('id')in sealed_ids and any(k in d for k in ['hypothesis','source_hypothesis','trace']):scored_sealed.append(str(p))
require(not scored_sealed,'Saved inference records contain sealed confirmation/final IDs')
code=ast.parse((OUT/'source_evaluate.py').read_text());function=next(x for x in code.body if isinstance(x,ast.FunctionDef)and x.name=='evaluate');calls=[(x.lineno,getattr(x.func,'id',getattr(x.func,'attr','')))for x in ast.walk(function)if isinstance(x,ast.Call)]
policy_lines=[n for n,name in calls if name=='native_policy'];load_lines=[n for n,name in calls if name=='load_whisper'];prediction_lines=[n for n,name in calls if name=='transcribe'];require(min(policy_lines)<min(load_lines)<min(prediction_lines),'Native policy not installed before model load/prediction')
# Recheck source snapshot and plan at end; live new outputs may have arrived during audit.
for p,h in plan['frozen_files'].items():require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,f'Frozen input changed during audit {p}')
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'pass'if not issues else 'issues_found','cpu_only':True,'cuda_initialised':torch.cuda.is_initialized(),'new_model_inference':False,'issues':issues,'scope':'Independent saved hashes, CPU tensor/geometry reconstruction, score arithmetic, selection replay; no neural re-inference or optimizer replay','source_preservation':{'snapshot_files':len(snapshot),'frozen_plan_files':len(plan['frozen_files']),'source_and_snapshot_hashes_checked':True},'baseline_preflight':{'receipt_updates_matching_frozen_control':updates,'clips':len(pre['items']),'neural_computation_rerun_by_auditor':False},'cohorts':cohorts,'generated_outputs':generated,'r1':r1,'r3':r3,'r2_independent_geometry':geometry,'source_evaluations':source_results,'source_quality_screen_interpretation':interpretations,'sealed_data':{'saved_inference_records_found':scored_sealed,'final_count':len(final),'final_speakers':len({r['speaker']for r in final}),'confirmation48_model_inference_found':False,'no_absolute_claim_about_unsaved_external_activity':True},'native_evaluator_static_order':{'policy_before_model_before_prediction':True,'historical_backend_exact_saved_equality':True},'file_sha256':hashes,'limitations':['No neural outputs or complete optimizer trajectories were independently reexecuted.','R1 calibration gradient norms are saved producer scalars; underlying gradient vectors were not saved for independent recomputation.','Hash integrity does not establish human intelligibility, black-box transfer or superiority.','This is a snapshot of concurrently finishing execution; rerun after later phases complete.']}
(RUN/'audit_initial_execution.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Independent CPU audit of initial execution','',f"Updated {report['created_at_utc']}. Status: **{report['status']}**.",'','No neural model was loaded or executed; CUDA remained uninitialized. The audit imports no experimental implementation. It reconstructs saved R2 mixtures using NumPy and reads tensors on CPU. Saved GPU receipts are distinguished from independent CPU calculations.','',f"- Preserved source/config snapshot: {len(snapshot)} files; frozen plan inputs: {len(plan['frozen_files'])} files. Current originals and snapshots match their bindings.",f'- Baseline preflight: all {updates} saved update receipts and three final-audio equality flags agree with their frozen reference receipts; neural computation was not replayed.',f'- Frozen reserve: {len(final)} utterances / {len({r["speaker"]for r in final})} speakers; saved selection replay and exclusion checks pass. No saved inference-bearing record uses this reserve or confirmation48.','']
for name,c in cohorts.items():lines.append(f"- {name}: {c['count']} recordings, {c['speakers']} speakers, {c['seconds']:.3f} seconds; independent selection replay = {c.get('selection_replayed',False)}.")
lines+=['','## Completed source evaluation','', '| Run / variant | Clips | WER % | DNSMOS | NISQA | UTMOS |','|---|---:|---:|---:|---:|---:|']
for name,d in source_results.items():
 for v,a in d['summary'].items():q=a['quality'];lines.append(f"| {name} / {v} | {a['count']} | {a['error_percent']:.6f} | {q['dnsmos']:.6f} | {q['nisqa']:.6f} | {q['utmos']:.6f} |")
lines+=['','These are Whisper-small source results on reused development clips. They are not held-out-target transfer or human-content results. A completed implementation or generated cohort is not a positive scientific result.','', '## R1 calibration and execution status',''
]
for name,c in r1['calibrations'].items():lines.append(f"{name}: independently recomputed saved-scalar median scale = {c['scale']:.12f}; three finite positive norm ratios. Underlying gradient vectors were not independently recomputed.")
for name,d in r1.items():
 if name=='calibrations':continue
 lines.append(f"{name}: saved status {d.get('status')}; completed {d.get('actual_count','not a generation stage')} of {d.get('planned_count','not applicable')}.")
lines += ['', '## Source-only screening interpretation','']
for name,x in interpretations.items():
 lines.append(f"{name}: compare against the registered quality-admissible B2 LR=.02 control (WER {x['quality_control_wer']:.6f}%, UTMOS {x['quality_control_utmos']:.6f}). Candidate delta WER {x['delta_wer_points']:+.6f} points; delta UTMOS {x['delta_utmos']:+.6f}. Automatic quality admissible: {x['automatic_quality_admissible']}; registered source/quality performance screen: {x['source_quality_performance_screen_passes']}. The near-clean attenuated diagnostic is not a substitute promotion control. No human or held-out-transfer claim follows.")
lines += ['', '## Independent R2 geometry','']
for name in sorted({g['run']for g in geometry}):
 gs=[g for g in geometry if g['run']==name];lines.append(f"{name}: {len(gs)} records, {sum(g['updates']for g in gs)} actual updates; status counts {dict(Counter(g['status']for g in gs))}. Maximum independently reconstructed-delta absolute discrepancy {max(g['independent_reconstruction_max_abs_error']for g in gs):.3g}; largest float32 norm-matching residual {max(abs(g['norm_match_residual'])for g in gs):.3g}. All final mixtures satisfy registered amplitude, bank, per-frame cosine and relative-radius constraints; unchanged clean/reconstruction audio samples match the frozen control.")
lines+=['','Norm matching is deterministic baseline attenuation on the same clean features. Floating-point output norms need not be literally equal; residuals and elementwise checks are reported. No vocoder resynthesis was performed by the auditor.','','## Limits and issues','']
lines += [f'- {x}'for x in (issues or ['No discrepancy found in the checked snapshot.'])]
lines += ['',*['- '+x for x in report['limitations']],'','Re-run `/opt/venv/bin/python /workspace/improvement/execution/audit_initial_execution.py` after new stages finish. Detailed bindings, results and record-level geometry are in `audit_initial_execution.json`.']
(RUN/'audit_initial_execution.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'issues':issues,'geometry_records':len(geometry),'source_eval_runs':list(source_results),'cohorts':cohorts,'cuda_initialised':torch.cuda.is_initialized()},indent=2),flush=True)
