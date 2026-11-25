"""Unchanged native source/MOS scoring; performance always full cohort."""
import sys,pathlib,json,time,gc
D=pathlib.Path(__file__).resolve().parent;P=D.parent;sys.path[:0]=[str(D),str(P),'/workspace','/workspace/scripts']
import torch
import execution_runtime as rt
from feature_runtime import check,status,measured
from feature_inference import PILOT_VARIANTS
from source_evaluate_refine import native_policy
from beyondwaveform.evaluation import QualityScorer,error_counts,aggregate_errors

def main(cohort):
 plan=check();old=rt.read(rt.RUN/'plan.json');folder=D/cohort;assert not (folder/'source-summary.json').exists();records=rt.read_jsonl(folder/'generated.jsonl');manifest=P/('execution48/development-48.jsonl' if cohort=='pilot' else 'execution_refine/validation-76.jsonl');allowed={r['id']:r for r in rt.read_jsonl(manifest)};assert len(records)==len(allowed)==(48 if cohort=='pilot' else 76);variants=PILOT_VARIANTS if cohort=='pilot' else ['clean','reconstruction','candidate','control'];assert all(r['text']==allowed[r['id']]['text'] and set(r['audio'])==set(variants) for r in records)
 expected=rt.read(rt.BASE/'evaluation/summary.json')['evaluation_backend'];native_policy(expected);began=time.perf_counter();source,meta=rt.load_whisper(old['base_config']['surrogate'],'cuda')
 import run_quality_research as historical
 historical.configure_scorers(old['quality']);scorer=QualityScorer('/workspace/artifacts');torch.cuda.synchronize();loading=time.perf_counter()-began;torch.cuda.reset_peak_memory_stats();scores=[];times=[]
 rt.atomic_json(folder/'evaluation-registration.json',{'backend':expected,'manifest_sha256':rt.digest(folder/'generated.jsonl'),'cohort_manifest_sha256':rt.digest(manifest),'source_metadata':meta,'scorer_manifest':old['quality'],'target_models_evaluated':[]})
 for i,r in enumerate(records):
  for variant,asset in r['audio'].items():
   assert rt.digest(asset['path'])==asset['sha256'];x=rt.load_audio(asset['path']);hyp,a=measured(lambda:source.transcribe(x.cuda()));quality,q=measured(lambda:scorer(x));item={'id':r['id'],'variant':variant,'reference':r['text'],'hypothesis':hyp,'audio_sha256':asset['sha256'],**error_counts(r['text'],hyp),'quality':quality};scores.append(item);t={'id':r['id'],'variant':variant,'asr':a,'quality':q};times.append(t);rt.atomic_json(folder/'score-records'/f'{r["id"]}--{variant}.json',item)
  rt.write_jsonl(folder/'scoring-costs-partial.jsonl',times);status(stage='evaluation',cohort=cohort,completed=i+1,total=len(records))
 summary={}
 for variant in variants:
  rr=[r for r in scores if r['variant']==variant];assert len(rr)==len(records);summary[variant]={'count':len(rr),**aggregate_errors(rr),'quality':{k:sum(r['quality'][k] for r in rr)/len(rr) for k in ['dnsmos','nisqa','utmos']}}
 rt.write_jsonl(folder/'source-scores.jsonl',scores);rt.write_jsonl(folder/'scoring-costs.jsonl',times);assert rt.backend_manifest()==expected;result={'status':'complete','source_only':True,'target_models_evaluated':[],'source_metadata':meta,'scorer_manifest':old['quality'],'evaluation_backend':expected,'summary':summary,'scores_sha256':rt.digest(folder/'source-scores.jsonl'),'resources':{'wall_seconds':time.perf_counter()-began,'loading_wall_seconds':loading,'asr_wall_seconds':sum(t['asr']['wall_seconds'] for t in times),'quality_wall_seconds':sum(t['quality']['wall_seconds'] for t in times),'asr_gpu_stream_elapsed_seconds':sum(t['asr']['gpu_stream_elapsed_seconds'] for t in times),'quality_gpu_stream_elapsed_seconds':sum(t['quality']['gpu_stream_elapsed_seconds'] for t in times),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'human_intelligibility_established':False};rt.atomic_json(folder/'source-summary.json',result);status(stage='evaluation',cohort=cohort,status='complete',completed=len(records));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('cohort',choices=['pilot','validation']);main(p.parse_args().cohort)
