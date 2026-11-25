"""Fresh-audio inference; native-source queries appear only in validation control."""
import sys,pathlib,json,time,gc
D=pathlib.Path(__file__).resolve().parent;P=D.parent;sys.path[:0]=[str(D),str(P),'/workspace','/workspace/scripts']
import torch
from torch.nn import functional as F
import execution_runtime as rt
from feature_runtime import check,status,measured,load_policy,sources
from feature_policy import RADII

PILOT_VARIANTS=['initial_r050',*[f'{arm}_r{int(rho*1000):03d}' for arm in ['source','quality'] for rho in RADII],'quality_shuffled_r050']
def main(cohort):
 plan=check();old=rt.read(rt.RUN/'plan.json');validation=cohort=='validation';folder=D/cohort;folder.mkdir(exist_ok=False);manifest=P/('execution_refine/validation-76.jsonl' if validation else 'execution48/development-48.jsonl');rows=rt.read_jsonl(manifest);assert len(rows)==(76 if validation else 48);freeze=rt.read(D/'selection.json') if validation else None
 if validation:
  assert freeze['status']=='validate' and rt.digest(freeze['model']['path'])==freeze['model']['sha256']
  for path,h in freeze['bindings'].items():assert rt.digest(path)==h
 began=time.perf_counter();timings=[];records=[]
 with rt.backend_mode('strict_math'):
  torch.set_num_threads(4);rt.seed_everything(11)
  from r3_real_runner_v2 import load_encoder_only
  load_start=time.perf_counter();native_encoder,encoder,meta=load_encoder_only(old);vocoder,_=rt.load_frozen_vocoder(old['base_config']['features'],old['feature_artifacts'],'cuda');policies={}
  for arm in ([freeze['arm']] if validation else ['initial','source','quality']):
   path=D/'training/initial.pt' if arm=='initial' else D/'training'/arm/'final.pt';policies[arm],_=load_policy(path);policies[arm].eval().requires_grad_(False)
  source=None
  if validation:
   # Original quality-admissible control; exact unchanged .02 LR50-step method.
   _,source,native_source,source_meta=sources()
   from scaled_attack import optimize as original_optimize
  load_seconds=time.perf_counter()-load_start;torch.cuda.reset_peak_memory_stats()
  for i,row in enumerate(rows):
   uid=row['id'];assert rt.digest(row['audio'])==row['audio_sha256'];t=time.perf_counter();x=rt.load_audio(row['audio']).cuda();input_io=time.perf_counter()-t
   def encode():
    with torch.no_grad():return encoder(x).detach()
   q,encoding=measured(encode)
   with torch.no_grad():neutral,neutral_t=measured(lambda:vocoder(q).detach())
   assets={};clip_times=[]
   if validation:
    assets['clean']=rt.save_audio(folder/'audio/clean'/f'{uid}.wav',x);assets['reconstruction']=rt.save_audio(folder/'audio/reconstruction'/f'{uid}.wav',neutral);tasks=[('candidate',freeze['arm'],freeze['rho'])]
   else:tasks=[('initial_r050','initial',.05),*[(f'{arm}_r{int(rho*1000):03d}',arm,rho) for arm in ['source','quality'] for rho in RADII],('quality_shuffled_r050','quality',.05)]
   conditioning=None
   if not validation:
    nextrow=rows[(i+1)%len(rows)];other=rt.load_audio(nextrow['audio']).cuda()
    with torch.no_grad():oq=encoder(other);conditioning=F.interpolate(oq.T[None],size=q.shape[0],mode='linear',align_corners=False)[0].T.detach()
    del other,oq
   for variant,arm,rho in tasks:
    m=policies[arm];torch.cuda.reset_peak_memory_stats()
    def predict():
     with torch.no_grad():delta=m(q,rho,conditioning=conditioning if variant=='quality_shuffled_r050' else None);return vocoder(q+delta).detach(),delta.detach()
    (y,delta),elapsed=measured(predict);start=time.perf_counter();assets[variant]=rt.save_audio(folder/'audio'/variant/f'{uid}.wav',y);output_io=time.perf_counter()-start
    item={'id':uid,'variant':variant,'audio_seconds':row['duration'],'input_io_seconds':input_io,'encoding':encoding,'policy_and_decoder':elapsed,'output_io_seconds':output_io,'steady_generation_seconds':input_io+encoding['wall_seconds']+elapsed['wall_seconds']+output_io,'seconds_per_audio_second':(input_io+encoding['wall_seconds']+elapsed['wall_seconds']+output_io)/row['duration'],'actual_relative_radius':float(delta.norm()/q.norm()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'source_queries_at_inference':0,'per_recording_optimization_updates':0};timings.append(item);clip_times.append(item);del y,delta
   if validation:
    def save_control(step,audio,delta,receipt):
     assets['control']=rt.save_audio(folder/'audio/control'/f'{uid}.wav',audio);rt.atomic_json(folder/'control_traces'/f'{uid}.json',receipt)
    original_optimize(q,x,row['text'],vocoder,{'small':source},method='quality',uid=uid,steps=50,on_checkpoint=save_control)
   assert set(assets)==set(['clean','reconstruction','candidate','control'] if validation else PILOT_VARIANTS)
   record={'id':uid,'text':row['text'],'speaker':row['speaker'],'audio':assets,'count':len(rows),'source_only':True};rt.atomic_json(folder/'records'/f'{uid}.json',record);records.append(record);rt.atomic_json(folder/'timings'/f'{uid}.json',clip_times);status(stage='generation',cohort=cohort,completed=i+1,total=len(rows));del x,q,neutral,conditioning;gc.collect()
  rt.write_jsonl(folder/'generated.jsonl',records);rt.write_jsonl(folder/'latencies.jsonl',timings);result={'status':'complete','cohort':cohort,'count':len(rows),'wall_seconds':time.perf_counter()-began,'model_loading_wall_seconds':load_seconds,'encoder_metadata':meta,'policy_checkpoints':{k:str(D/'training/initial.pt' if k=='initial' else D/'training'/k/'final.pt') for k in policies},'target_models_evaluated':[],'inference_scope':'Input IO +fresh WavLM6 encoding +single policy/decoder +WAV output IO. Hash verification and shared diagnosis/reference/shuffle work are outside per-variant latency; no ASR queries except separately labelled validation control.','validation_control_source_calls':len(rows)*50 if validation else 0};rt.atomic_json(folder/'generation-result.json',result);status(stage='generation',cohort=cohort,status='complete',completed=len(rows))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('cohort',choices=['pilot','validation']);main(p.parse_args().cohort)
