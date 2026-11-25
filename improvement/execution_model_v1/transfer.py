"""Frozen model transfer and exact named defenses; budget selection uses cost only."""
import pathlib,sys,json,time,gc,os,statistics
D=pathlib.Path(__file__).resolve().parent;P=D.parent;sys.path[:0]=[str(D),str(P),'/workspace','/workspace/scripts']
import torch,numpy as np
import execution_runtime as rt
from feature_runtime import check,status,measured
from source_evaluate_refine import native_policy
from scaled_transfer_evaluate import preprocess
from beyondwaveform.config import load_config
from beyondwaveform.models.targets import load_target
from beyondwaveform.evaluation import error_counts,aggregate_errors,QualityScorer
TARGETS=['whisper_base','whisper_medium','wav2vec2_large','hubert_large'];CONDITIONS=['raw','lpf6000','waveguard8000','lpf4000','waveguard12000'];VARIANTS=['clean','reconstruction','candidate','control']
def prepare():
 check();freeze=rt.read(D/'transfer_freeze.json');assert freeze['status']=='evaluate_transfer'
 for p,h in freeze['bindings'].items():assert rt.digest(p)==h
 expected=rt.read(rt.BASE/'evaluation/summary.json')['evaluation_backend'];native_policy(expected);rows=rt.read_jsonl(D/'validation/generated.jsonl');assert len(rows)==76;return rows,expected

def profile():
 rows,backend=prepare();folder=D/'transfer';folder.mkdir(exist_ok=False);costs={};scores=[];began=time.perf_counter();ordered=sorted(rows,key=lambda r:len(rt.load_audio(r['audio']['clean']['path'])));chosen=[ordered[0],ordered[len(ordered)//2],ordered[-1]];durations={r['id']:len(rt.load_audio(r['audio']['clean']['path']))/16000 for r in rows};mean_duration=statistics.mean(durations.values())
 for name in TARGETS:
  config=load_config(rt.ROOT/f'configs/targets/{name}.yaml');start=time.perf_counter();target,meta=load_target(config,'cuda');target.eval().requires_grad_(False);loading=time.perf_counter()-start;times=[];audio_seconds=[]
  for r in chosen:
   for v,a in r['audio'].items():
    x=rt.load_audio(a['path']).cuda();hyp,t=measured(lambda:target.transcribe(x));times.append(t['wall_seconds']);audio_seconds.append(durations[r['id']]);item={'id':r['id'],'speaker':r['speaker'],'target':name,'defense':'raw','variant':v,'reference':r['text'],'hypothesis':hyp,'input_audio_sha256':a['sha256'],'evaluated_audio_sha256':a['sha256'],**error_counts(r['text'],hyp)};scores.append(item)
  costs[name]={'mean_seconds_per_decode':statistics.mean(times),'duration_adjusted_seconds_per_decode':max(statistics.mean(times),sum(times)/sum(audio_seconds)*mean_duration),'decode_seconds_per_audio_second':sum(times)/sum(audio_seconds),'loading_seconds':loading,'metadata':meta,'engineering_profile_rows':12};status(stage='transfer_profile',target=name,completed=len(costs),total=4);del target;gc.collect();torch.cuda.empty_cache()
 preprocessing={}
 for cond in CONDITIONS:
  samples=[]
  for r in chosen:
   for a in r['audio'].values():
    x=rt.load_audio(a['path']);t=time.perf_counter();preprocess(x,cond);samples.append(time.perf_counter()-t)
  preprocessing[cond]=statistics.mean(samples)
 validation=rt.read(D/'validation/source-summary.json')['resources'];qper=validation['quality_wall_seconds']/(76*4);reload_cost=sum(v['loading_seconds'] for v in costs.values());per_condition=(76*4*(sum(v['duration_adjusted_seconds_per_decode'] for v in costs.values())+qper+max(preprocessing.values()))+reload_cost+120)*1.5;forecast={'status':'pass','costs':costs,'profile_record_ids':[r['id'] for r in chosen],'profile_audio_seconds':[durations[r['id']] for r in chosen],'validation_mean_audio_seconds':mean_duration,'preprocessing_seconds_per_asset':preprocessing,'measured_target_reload_seconds_charged_per_condition':reload_cost,'unmeasured_quality_loading_reserve_seconds':120,'margin':1.5,'quality_seconds_per_asset_from_validation':qper,'forecast_seconds_per_complete76_condition_all4targets':per_condition,'forecast_full5conditions_seconds':per_condition*5,'wall_seconds':time.perf_counter()-began,'profile_only_no_performance_claim':True,'backend':backend};rt.write_jsonl(folder/'profile_scores.jsonl',scores);rt.atomic_json(folder/'profile.json',forecast)

def evaluate():
 rows,backend=prepare();folder=D/'transfer';registered=rt.read(folder/'execution_plan.json');conditions=registered['conditions'];assert conditions==CONDITIONS[:len(conditions)] and conditions;old=rt.read(rt.RUN/'plan.json');began=time.perf_counter();assets={};quality={};scores=[];peaks=[];decode_costs=[];preprocessing_seconds=0.;quality_seconds=0.;target_loading_seconds=0.
 import run_quality_research as historical
 historical.configure_scorers(old['quality']);scorer=QualityScorer('/workspace/artifacts')
 for cond in conditions:
  for i,r in enumerate(rows):
   for v,a in r['audio'].items():
    key=(cond,a['sha256'])
    if key in assets:continue
    assert rt.digest(a['path'])==a['sha256'];x=rt.load_audio(a['path']);t=time.perf_counter();y=preprocess(x,cond);assets[key]=a if cond=='raw' else rt.save_audio(folder/'audio'/cond/(a['sha256']+'.wav'),y);preprocessing_seconds+=time.perf_counter()-t;t=time.perf_counter();quality[key]=scorer(y);quality_seconds+=time.perf_counter()-t
   if i%10==0 or i==75:status(stage='defense_quality',condition=cond,completed=i+1,total=76)
 rt.atomic_json(folder/'post_defense_quality.json',{'records':[{'condition':c,'input_audio_sha256':h,'audio':assets[c,h],'quality':quality[c,h]} for c,h in assets]});del scorer;gc.collect();torch.cuda.empty_cache();meta={};probe={(r['target'],r['id'],r['variant']):r for r in [json.loads(s) for s in (folder/'profile_scores.jsonl').read_text().splitlines()]}
 for name in TARGETS:
  config=load_config(rt.ROOT/f'configs/targets/{name}.yaml');t=time.perf_counter();target,metadata=load_target(config,'cuda');target_loading_seconds+=time.perf_counter()-t;target.eval().requires_grad_(False);meta[name]=metadata;torch.cuda.reset_peak_memory_stats();predictions={}
  for cond in conditions:
   for i,r in enumerate(rows):
    for v,a in r['audio'].items():
     asset=assets[cond,a['sha256']];k=(name,r['id'],v)
     if cond=='raw' and k in probe:hyp=probe[k]['hypothesis']
     elif asset['sha256'] in predictions:hyp=predictions[asset['sha256']]
     else:
      x=rt.load_audio(asset['path']).cuda();hyp,timing=measured(lambda:target.transcribe(x));decode_costs.append({'target':name,'condition':cond,'id':r['id'],'variant':v,**timing});predictions[asset['sha256']]=hyp
     item={'id':r['id'],'speaker':r['speaker'],'target':name,'defense':cond,'variant':v,'reference':r['text'],'hypothesis':hyp,'input_audio_sha256':a['sha256'],'evaluated_audio_sha256':asset['sha256'],**error_counts(r['text'],hyp),'quality':quality[cond,a['sha256']]};scores.append(item);rt.atomic_json(folder/'records'/name/cond/f'{r["id"]}--{v}.json',item)
    if i%10==0 or i==75:status(stage='transfer',target=name,condition=cond,completed=i+1,total=76)
  peaks.append({'target':name,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});del target;gc.collect();torch.cuda.empty_cache()
 summaries={};comparisons={};speakers=sorted({r['speaker'] for r in rows});weights=np.random.default_rng(20261027).multinomial(len(speakers),[1/len(speakers)]*len(speakers),size=10000)
 for cond in conditions:
  summaries[cond]={}
  for target in TARGETS:
   summaries[cond][target]={}
   for v in VARIANTS:
    pool=[r for r in scores if r['defense']==cond and r['target']==target and r['variant']==v];assert len(pool)==76;summaries[cond][target][v]={'count':76,**aggregate_errors(pool),'quality':{k:sum(r['quality'][k] for r in pool)/76 for k in ['dnsmos','nisqa','utmos']}}
  arr=np.array([[sum(r['substitutions']+r['deletions']+r['insertions'] for r in scores if r['speaker']==sp and r['defense']==cond and r['variant']==v) for v in ['candidate','control']]+[sum(r['reference_units'] for r in scores if r['speaker']==sp and r['defense']==cond and r['variant']=='candidate')] for sp in speakers]);sample=weights@arr;delta=100*(arr[:,0].sum()-arr[:,1].sum())/arr[:,2].sum();boot=100*(sample[:,0]-sample[:,1])/sample[:,2];ci=np.quantile(boot,[.025,.975]).tolist();comparisons[cond]={'family_balanced_delta_wer_points':float(delta),'speaker_bootstrap95':ci,'preservation_screen':bool(delta>=0 and ci[0]>=-3)}
 rt.write_jsonl(folder/'decode_costs.jsonl',decode_costs);rt.write_jsonl(folder/'scores.jsonl',scores);assert rt.backend_manifest()==backend;result={'status':'complete' if conditions==CONDITIONS else 'partial_conditions_budgeted_before_predictions','targets':meta,'conditions_completed':conditions,'required_primary_conditions':CONDITIONS,'all_required_conditions_completed':conditions==CONDITIONS,'candidate_reselected':False,'summaries':summaries,'comparisons':comparisons,'preservation_status':'pass' if conditions==CONDITIONS and all(v['preservation_screen'] for v in comparisons.values()) else 'not_established','wall_seconds':time.perf_counter()-began,'resources':{'defense_preprocessing_and_save_seconds':preprocessing_seconds,'post_defense_quality_seconds':quality_seconds,'target_loading_seconds':target_loading_seconds,'decode_wall_seconds':sum(r['wall_seconds'] for r in decode_costs),'decode_gpu_stream_elapsed_seconds':sum(r['gpu_stream_elapsed_seconds'] for r in decode_costs),'cached_profile_predictions_cost_charged_in_profile':True},'peak_memory_by_target':peaks,'post_defense_quality_reported':True,'human_intelligibility_established':False,'untouched_confirmation_final_inference':False,'limits':'76 exposed development points; exact named preprocessors, not full WaveGuard detection or unavailable learned defenses. Any missing primary condition prevents preservation pass.'};rt.atomic_json(folder/'results.json',result)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['profile','evaluate']);a=p.parse_args();profile() if a.stage=='profile' else evaluate()
