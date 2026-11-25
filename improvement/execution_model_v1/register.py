"""Freeze this isolated experiment, retaining earlier registrations verbatim."""
import pathlib,json,hashlib,datetime,sys,ast
D=pathlib.Path(__file__).resolve().parent;P=D.parent;ROOT=P.parent
sys.path[:0]=[str(D),str(P),str(ROOT),str(ROOT/'scripts')]
from feature_policy import FeaturePolicy

def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 path=D/'plan.json'
 if path.exists():raise FileExistsError('Already registered; preserve it')
 files=list(D.glob('*.py'))+[D/'check_browser.cjs',D/'user_steering.json',D/'proxy_calibration.json',D/'preexisting_baseline_drift.json',P/'23_model_directions_and_sources.md',P/'24_learned_feature_protocol.md',P/'model_direction_sources/consensus_receipts.json',P/'execution/plan.json',P/'execution_scaled/plan.json',P/'execution/data/train-10h.jsonl',P/'execution48/development-48.jsonl',P/'execution_refine/validation-76.jsonl']
 files+=list((ROOT/'beyondwaveform').rglob('*.py'))
 for name in ['execution_runtime.py','scaled_components.py','scaled_attack.py','scaled_transfer_evaluate.py','source_evaluate_refine.py','r3_real_runner_v2.py','build_audio_browser_data.py','execution_tradeoff/refresh_interface.py']:files.append(P/name)
 for name in ['probe_deterministic_execution.py','probe_feature_gradients.py','probe_numerical_repeatability.py','run_quality_research.py','run_logged.py']:files.append(ROOT/'scripts'/name)
 files+=list((ROOT/'configs/targets').glob('*.yaml'))
 old=json.loads((P/'execution/plan.json').read_text());files+=list(map(pathlib.Path,old['frozen_files']))
 bindings={str(p):digest(p) for p in sorted(set(files))}
 changes=json.loads((D/'preexisting_baseline_drift.json').read_text())['changed_files'];assert len(changes)==1 and changes[0]['path']=='/workspace/beyondwaveform/training.py'
 rows=[json.loads(s) for s in (P/'execution/data/train-10h.jsonl').read_text().splitlines()];pilot=[json.loads(s) for s in (P/'execution48/development-48.jsonl').read_text().splitlines()];validation=[json.loads(s) for s in (P/'execution_refine/validation-76.jsonl').read_text().splitlines()]
 assert len(rows)==2807 and len(pilot)==48 and len(validation)==76
 assert {r['id'] for r in rows}.isdisjoint(r['id'] for r in pilot+validation)
 assert {r['speaker'] for r in rows}.isdisjoint(r['speaker'] for r in pilot+validation)
 for f in D.glob('*.py'):ast.parse(f.read_text(),str(f))
 plan={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'registered_before_gpu_profile','authorization':'Latest user requests continuing experiments with an open method direction; learned models/larger data authorized; GPU work queued and detached, max13hours. Publication planning deferred.','maximum_overall_seconds':46800,'queue_counts_against_ceiling':True,'maximum_queue_seconds':14400,'teacher_optimized_targets':False,'training':{'count':len(rows),'audio_seconds':sum(r['duration'] for r in rows),'manifest':str(P/'execution/data/train-10h.jsonl'),'warmup_maximum_seconds':1200,'warmup_maximum_updates':300,'arms':['source','quality'],'maximum_seconds_each_arm':7200,'accumulation':4,'optimizer':'AdamW','learning_rate':.0002,'gradient_clip':5.,'seed':11,'common_update_limit':'min(6000,max(100,floor(7200/(4*profile_mean_microbatch_seconds*1.30))))'},'policy':FeaturePolicy().registration(),'quality_caps':json.loads((D/'proxy_calibration.json').read_text())['caps'],'joint_gate':{'source_wer_strict_gt':75,'dnsmos_min':3.,'nisqa_min':3.75,'utmos_min':3.5},'proposed_efficiency_screen':{'inference_speedup_min':10,'reference_seconds_per_recording':16.773926311657608,'peak_allocated_gib_strict_lt':20,'user_specified_latency_limit':False,'rationale':'Single-pass shared feature policy should remove repeated source/decoder backward work;10x is a falsifiable research target relative to measured strong FV50,20GiB leaves4GiB engineering headroom on24GiB. Training and preprocessing still charged separately.'},'pilot_count':48,'pilot_variants':8,'validation_count':76,'selection_before_validation_and_targets':True,'untouched_confirmation_final_inference':False,'external_messages':False,'server':'existing0.0.0.0:8888; inspect and preserve','preexisting_unrelated_baseline_drift':{r['path']:r for r in changes},'bindings':bindings}
 path.write_text(json.dumps(plan,indent=2)+'\n');path.chmod(0o444)
 print(json.dumps({'status':plan['status'],'bindings':len(bindings),'parameters':plan['policy']['parameters'],'training_audio_hours':plan['training']['audio_seconds']/3600,'sha256':digest(path)},indent=2))
if __name__=='__main__':main()
