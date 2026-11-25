"""Fixed full-cohort WER/MOS and declared efficiency gates; no target selection."""
import pathlib,json,sys,statistics
D=pathlib.Path(__file__).resolve().parent;P=D.parent;sys.path[:0]=[str(D),str(P),'/workspace','/workspace/scripts']
import execution_runtime as rt
from feature_runtime import check
ITERATIVE_SECONDS=16.773926311657608

def passes(s):return s['error_percent']>75 and all(s['quality'][k]>=v for k,v in [('dnsmos',3),('nisqa',3.75),('utmos',3.5)])
def efficiency(rows):
 assert len(rows) in (48,76) and len({r['id'] for r in rows})==len(rows)
 latency=statistics.mean(r['steady_generation_seconds'] for r in rows);peak=max(r['peak_allocated_bytes'] for r in rows);queries=all(r['source_queries_at_inference']==0 and r['per_recording_optimization_updates']==0 for r in rows)
 return {'pass':latency<=ITERATIVE_SECONDS/10 and peak<20*1024**3 and queries,'mean_seconds':latency,'maximum_mean_seconds':ITERATIVE_SECONDS/10,'peak_allocated_bytes':peak,'zero_source_queries_and_search':queries,'research_target_not_user_latency_limit':True}
def main(cohort):
 check();summary=rt.read(D/cohort/'source-summary.json')['summary'];timings=rt.read_jsonl(D/cohort/'latencies.jsonl')
 if cohort=='pilot':
  allowed={k:s for k,s in summary.items() if k.startswith(('source_r','quality_r'))};cost={k:efficiency([r for r in timings if r['variant']==k]) for k in summary};eligible=[k for k,s in allowed.items() if passes(s) and cost[k]['pass']]
  chosen=sorted(eligible,key=lambda v:(-allowed[v]['error_percent'],-allowed[v]['quality']['utmos'],cost[v]['mean_seconds'],v))[0] if eligible else None
  result={'status':'validate' if chosen else 'scientific_stop_no_joint_pass','candidate':chosen,'all_wer_quality_gates':{k:passes(s) for k,s in summary.items()},'efficiency_gates':cost,'all_gates':{k:passes(s) and cost[k]['pass'] for k,s in summary.items()},'candidate_reselected_on_targets':False,'bindings':{str(D/'plan.json'):rt.digest(D/'plan.json'),str(D/'pilot/source-summary.json'):rt.digest(D/'pilot/source-summary.json'),str(D/'pilot/generated.jsonl'):rt.digest(D/'pilot/generated.jsonl'),str(D/'pilot/latencies.jsonl'):rt.digest(D/'pilot/latencies.jsonl')}}
  if chosen:
   arm,r=chosen.split('_r');path=D/'training'/arm/'final.pt';result.update(arm=arm,rho=int(r)/1000,model={'path':str(path),'sha256':rt.digest(path)},inference_latency_mean_seconds=cost[chosen]['mean_seconds']);result['bindings'][str(path)]=rt.digest(path)
  path=D/'selection.json'
 else:
  freeze=rt.read(D/'selection.json');assert freeze['status']=='validate'
  for p,h in freeze['bindings'].items():assert rt.digest(p)==h
  cost=efficiency([r for r in timings if r['variant']=='candidate']);qualified=passes(summary['candidate']) and cost['pass'];result={'status':'evaluate_transfer' if qualified else 'scientific_stop_validation_joint_failure','candidate':freeze['candidate'],'source_validation_gate':passes(summary['candidate']),'efficiency_gate':cost,'control_source_summary':summary['control'],'candidate_reselected':False,'bindings':{str(D/'selection.json'):rt.digest(D/'selection.json'),str(D/'validation/source-summary.json'):rt.digest(D/'validation/source-summary.json'),str(D/'validation/generated.jsonl'):rt.digest(D/'validation/generated.jsonl'),str(D/'validation/latencies.jsonl'):rt.digest(D/'validation/latencies.jsonl')}};path=D/'transfer_freeze.json'
 if path.exists():raise FileExistsError('Do not overwrite a frozen selection')
 rt.atomic_json(path,result);path.chmod(0o444);print(json.dumps(result,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('cohort',choices=['pilot','validation']);main(p.parse_args().cohort)
