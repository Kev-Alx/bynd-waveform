"""Matched full-cohort generation, fresh-feature profiling, and immutable receipts."""
import sys,pathlib,time,json,gc,hashlib,resource
P=pathlib.Path('/workspace/improvement');D=P/'execution_tradeoff';sys.path[:0]=[str(P),str(D),'/workspace','/workspace/scripts']
import torch,numpy as np
import execution_runtime as rt
from method import optimize,anchored_magnitude
from scaled_components import checkpoint_blocks

def check():
 plan=rt.read(D/'plan.json')
 for p,h in plan['bindings'].items():assert rt.digest(p)==h,f'Changed binding: {p}'
 return plan

def status(**x):rt.atomic_json(D/'progress.json',{'updated_at_utc':rt.stamp(),**x});print(json.dumps(x),flush=True)

def event_call(fn):
 torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();t=time.perf_counter();v=fn();b.record();torch.cuda.synchronize();return v,time.perf_counter()-t,a.elapsed_time(b)/1000

def generate():
 began=time.perf_counter();plan=check();old=rt.read(rt.RUN/'plan.json');rt.check(old);rows=rt.read_jsonl(P/'execution48/development-48.jsonl');hist={r['id']:r for r in rt.read_jsonl(P/'execution_scaled/fvpilot/generated.jsonl')};folder=D/'pilot'
 assert not folder.exists(),'Preserve previous generation';folder.mkdir();result={'status':'running','count':48,'source_only':True,'target_models_evaluated':[],'training_seconds':0,'teacher_data_generation':'same per-recording FV50 optimization; no learned training','records':[]};rt.atomic_json(folder/'generation-result.json',result)
 with rt.backend_mode('strict_math'):
  torch.set_num_threads(4);rt.seed_everything(2026)
  from r3_real_runner_v2 import load_encoder_only
  status(stage='feature_preprocessing',completed=0);t=time.perf_counter();native,encoder,meta=load_encoder_only(old);torch.cuda.synchronize();result['encoder_load_wall_seconds']=time.perf_counter()-t;torch.cuda.reset_peak_memory_stats();pp=[]
  for i,r in enumerate(rows):
   assert rt.digest(r['audio'])==r['audio_sha256'];x=rt.load_audio(r['audio']).cuda()
   def encode():
    with torch.no_grad():return encoder(x).detach()
   q,w,g=event_call(encode);asset=hist[r['id']]['diagnostics']['q'];assert rt.digest(asset['path'])==asset['sha256'];prior=torch.load(asset['path'],map_location='cuda',weights_only=True);diff=float((q-prior).double().norm()/prior.double().norm());maximum=float((q-prior).abs().max());assert diff<1e-5 and maximum<1e-4
   pp.append({'id':r['id'],'audio_seconds':r['duration'],'wall_seconds':w,'gpu_event_seconds':g,'seconds_per_audio_second':w/r['duration'],'exact_cached_features':torch.equal(q,prior),'relative_feature_difference':diff,'maximum_feature_difference':maximum});status(stage='feature_preprocessing',completed=i+1)
   del x,q,prior
  result['preprocessing']={'records':pp,'wall_seconds':sum(x['wall_seconds'] for x in pp),'gpu_event_seconds':sum(x['gpu_event_seconds'] for x in pp),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'encoder_metadata':meta,'cached_features_used_for_frozen_pair_comparison':True};del encoder,native;gc.collect();torch.cuda.empty_cache()
  t=time.perf_counter();vocoder,source,native,meta=rt.models(old);checkpoint_blocks([*source.model.encoder.blocks,*source.model.decoder.blocks]);torch.cuda.synchronize();result['source_decoder_load_wall_seconds']=time.perf_counter()-t;result['source_metadata']=meta;result['backend']=rt.backend_manifest()
  first=rows[0];q=torch.load(hist[first['id']]['diagnostics']['q']['path'],map_location='cuda',weights_only=True);x=rt.load_audio(first['audio']).cuda();t=time.perf_counter();optimize(q,x,first['text'],vocoder,source,once=True,steps=1);result['warmup_wall_seconds']=time.perf_counter()-t;del q,x
  generated=[]
  for i,r in enumerate(rows):
   uid=r['id'];start=time.perf_counter();q=torch.load(hist[uid]['diagnostics']['q']['path'],map_location='cuda',weights_only=True);x=rt.load_audio(r['audio']).cuda();runs={};ys={};ds={}
   for once in ([False,True] if i%2==0 else [True,False]):
    label='once' if once else 'twice';torch.cuda.empty_cache()
    y,d,receipt=optimize(q,x,r['text'],vocoder,source,once=once,progress=lambda step:status(stage='generation',recording=i+1,id=uid,path=label,step=step))
    runs[label]=receipt;ys[label]=y.cpu();ds[label]=d.cpu();del y,d
   assert torch.equal(ys['once'],ys['twice']) and torch.equal(ds['once'],ds['twice']),'Paired implementation changed output'
   frozen_asset=hist[uid]['audio']['single_50'];assert rt.digest(frozen_asset['path'])==frozen_asset['sha256'];frozen=rt.load_audio(frozen_asset['path']);frozen_equal=torch.equal(ys['once'],frozen)
   # All real-data exactness checks are retained; a mismatch stops this version.
   assert frozen_equal,'Changed frozen single50 output'
   raw=ys['once'].numpy();anchor_asset=hist[uid]['audio']['fv_reconstruction'];assert rt.digest(anchor_asset['path'])==anchor_asset['sha256'];anchor=rt.load_audio(anchor_asset['path']).numpy();assets={'raw50':rt.save_audio(folder/'audio/raw50'/f'{uid}.wav',ys['once'])};post={}
   for variant,fraction in [('stft_identity',0.),('anchor25',.25)]:
    t=time.perf_counter();restored=anchored_magnitude(raw,anchor,fraction);elapsed=time.perf_counter()-t;assets[variant]=rt.save_audio(folder/'audio'/variant/f'{uid}.wav',torch.from_numpy(restored));post[variant]={'wall_seconds':elapsed,'seconds_per_audio_second':elapsed/r['duration'],'maximum_change':float(np.max(np.abs(restored-raw)))}
   receipt={'id':uid,'audio_seconds':r['duration'],'paired_order':list(runs),'runs':runs,'paired_delta_equal':True,'paired_audio_equal':True,'frozen_audio_equal':frozen_equal,'postprocessing':post,'wall_seconds_including_both_paths':time.perf_counter()-start};rt.atomic_json(folder/'timings'/f'{uid}.json',receipt)
   record={'id':uid,'text':r['text'],'speaker':r['speaker'],'audio':assets,'diagnostics':{'timing':str(folder/'timings'/f'{uid}.json')},'count':48,'source_only':True};generated.append(record);rt.atomic_json(folder/'records'/f'{uid}.json',record);result['records'].append(receipt);rt.atomic_json(folder/'generation-result.json',result);status(stage='generation',completed=i+1,total=48)
   del q,x,ys,ds,frozen;gc.collect()
  rt.write_jsonl(folder/'generated.jsonl',generated);result.update(status='complete',wall_seconds=time.perf_counter()-began,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_at_utc=rt.stamp());rt.atomic_json(folder/'generation-result.json',result);check();rt.check(old);status(stage='generation',status='complete',completed=48)
if __name__=='__main__':generate()
