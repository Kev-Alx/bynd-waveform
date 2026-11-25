import json,pathlib,sys,time,fcntl
P=pathlib.Path('/workspace/improvement');sys.path[:0]=[str(P),str(P/'execution_tradeoff'),'/workspace','/workspace/scripts']
import torch
import execution_runtime as rt
from scaled_components import checkpoint_blocks
from method import optimize
D=P/'execution_tradeoff'
lock=(P/'execution/continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=rt.read(rt.RUN/'plan.json');rt.check(plan)
records=rt.read_jsonl(P/'execution_scaled/fvpilot/generated.jsonl');cohort=rt.read_jsonl(P/'execution48/development-48.jsonl');byid={r['id']:r for r in records};sample=min(cohort,key=lambda r:r['duration'])
with rt.backend_mode('strict_math'):
 torch.set_num_threads(4);rt.seed_everything(2026);vocoder,source,native,meta=rt.models(plan);checkpoint_blocks([*source.model.encoder.blocks,*source.model.decoder.blocks])
 uid=sample['id'];qasset=byid[uid]['diagnostics']['q'];assert rt.digest(qasset['path'])==qasset['sha256'];q=torch.load(qasset['path'],map_location='cuda',weights_only=True);x=rt.load_audio(sample['audio']).cuda()
 # Warm up separately, then measure paired paths from the same zero edit.
 optimize(q,x,sample['text'],vocoder,source,once=True,steps=1)
 ya,da,a=optimize(q,x,sample['text'],vocoder,source,once=False,steps=5)
 yb,db,b=optimize(q,x,sample['text'],vocoder,source,once=True,steps=5)
 result={'status':'pass' if torch.equal(da,db) and torch.equal(ya,yb) else 'fail','id':uid,'audio_seconds':sample['duration'],'engineering_only':True,'performance_pilot':False,'delta_equal':torch.equal(da,db),'audio_equal':torch.equal(ya,yb),'maximum_delta_difference':float((da-db).abs().max()),'twice':a,'once':b,'speed_ratio':a['wall_seconds']/b['wall_seconds'],'backend':rt.backend_manifest()}
 rt.atomic_json(D/'engineering.json',result);print(json.dumps(result,indent=2));assert result['status']=='pass'
