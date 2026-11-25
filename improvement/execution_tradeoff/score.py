"""Fresh native source scoring with ASR and quality costs separately measured."""
import sys,pathlib,time,json,resource
P=pathlib.Path('/workspace/improvement');D=P/'execution_tradeoff';sys.path[:0]=[str(P),str(D),'/workspace','/workspace/scripts']
import torch
import execution_runtime as rt
from source_evaluate_refine import native_policy
from beyondwaveform.evaluation import QualityScorer,aggregate_errors,error_counts
from beyondwaveform.models.whisper import load_whisper
from pilot import check,event_call,status

def main():
 began=time.perf_counter();check();plan=rt.read(rt.RUN/'plan.json');rt.check(plan);folder=D/'pilot';assert not (folder/'source-summary.json').exists();records=rt.read_jsonl(folder/'generated.jsonl');cohort=rt.read_jsonl(P/'execution48/development-48.jsonl');assert len(records)==48 and {r['id'] for r in records}=={r['id'] for r in cohort};assert all(set(r['audio'])=={'raw50','stft_identity','anchor25'} for r in records)
 expected=rt.read(rt.BASE/'evaluation/summary.json')['evaluation_backend'];backend=native_policy(expected);t=time.perf_counter();native,meta=load_whisper(plan['base_config']['surrogate'],'cuda')
 import run_quality_research as historical
 historical.configure_scorers(plan['quality']);scorer=QualityScorer('/workspace/artifacts');torch.cuda.synchronize();load_seconds=time.perf_counter()-t;torch.cuda.reset_peak_memory_stats();scores=[];timings=[]
 for i,r in enumerate(records):
  for variant,asset in r['audio'].items():
   assert rt.digest(asset['path'])==asset['sha256'];audio=rt.load_audio(asset['path']);hyp,w,g=event_call(lambda:native.transcribe(audio.cuda()));q,qw,qg=event_call(lambda:scorer(audio));item={'id':r['id'],'variant':variant,'reference':r['text'],'hypothesis':hyp,'audio_sha256':asset['sha256'],**error_counts(r['text'],hyp),'quality':q};scores.append(item);timings.append({'id':r['id'],'variant':variant,'asr_wall_seconds':w,'asr_gpu_event_seconds':g,'quality_wall_seconds':qw,'quality_gpu_event_seconds':qg});rt.atomic_json(folder/'score-records'/f'{r["id"]}--{variant}.json',item)
  status(stage='evaluation',completed=i+1,total=48)
 summary={}
 for v in ['raw50','stft_identity','anchor25']:
  pool=[s for s in scores if s['variant']==v];summary[v]={'count':len(pool),**aggregate_errors(pool),'quality':{k:sum(s['quality'][k] for s in pool)/48 for k in ['dnsmos','nisqa','utmos']}}
 rt.write_jsonl(folder/'source-scores.jsonl',scores);rt.write_jsonl(folder/'evaluation-timings.jsonl',timings)
 result={'status':'complete','source_only':True,'target_models_evaluated':[],'source_metadata':meta,'scorer_manifest':plan['quality'],'evaluation_backend':backend,'summary':summary,'scores_sha256':rt.digest(folder/'source-scores.jsonl'),'resources':{'elapsed_wall_seconds':time.perf_counter()-began,'loading_wall_seconds':load_seconds,'asr_wall_seconds':sum(t['asr_wall_seconds'] for t in timings),'quality_wall_seconds':sum(t['quality_wall_seconds'] for t in timings),'asr_gpu_event_seconds':sum(t['asr_gpu_event_seconds'] for t in timings),'quality_gpu_event_seconds':sum(t['quality_gpu_event_seconds'] for t in timings),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},'baseline_policy_exactly_matched':rt.backend_manifest()==expected,'human_intelligibility_established':False};assert result['baseline_policy_exactly_matched'];rt.atomic_json(folder/'source-summary.json',result);check();print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
