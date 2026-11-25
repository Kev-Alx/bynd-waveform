import json, importlib, requests
from pathlib import Path
import torch
from beyondwaveform.io import digest, load_audio, read_jsonl
from beyondwaveform.evaluation import QualityScorer

torch.set_num_threads(4)
base=Path('/tmp/beyondwaveform-quality-audit')
run=Path('reproduction-runs/english-feature-test-999')
summary=json.loads((run/'surrogate-eval/summary.json').read_text())
record=read_jsonl(run/'generated.jsonl')[0]
stored=[json.loads(s) for s in (run/'surrogate-eval/utterances.jsonl').read_text().splitlines() if s.strip()]
expected=summary['quality_provenance']['artifact_hashes']
files=[('DNSMOS/DNSMOS/sig_bak_ovr.onnx','https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx'),('DNSMOS/DNSMOS/model_v8.onnx','https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx'),('NISQA/nisqa.tar','https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa.tar')]
verified={}
for name,url in files:
 path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  print('Fetching historical scorer for hash comparison:',name,flush=True)
  with requests.get(url,stream=True,timeout=(15,45)) as r:
   r.raise_for_status()
   with path.open('wb') as f:
    for chunk in r.iter_content(1024*1024):f.write(chunk)
 h=digest(path);want=next(v for k,v in expected.items() if k.endswith('/'+name))
 if h!=want: raise ValueError(f'Current download differs from historical checkpoint {name}: {h} != {want}')
 verified[name]=h
m=importlib.import_module('torchmetrics.functional.audio.dnsmos');m.DNSMOS_DIR=str(base/'DNSMOS')
n=importlib.import_module('torchmetrics.functional.audio.nisqa');n.NISQA_DIR=str(base/'NISQA')
scorer=QualityScorer('artifacts')
utmos_path=Path('artifacts/torchhub/checkpoints/utmos22_strong_step7459_v1.pt')
assert digest(utmos_path)==expected[str(utmos_path)]
verified[str(utmos_path)]=digest(utmos_path)
results={'utterance_id':record['id'],'verified_checkpoint_hashes':verified,'fixtures':{}}
for variant in ['clean','adversarial']:
 print('Scoring fixture:',variant,flush=True)
 asset=record['audio'][variant];assert digest(asset['path'])==asset['sha256']
 x=load_audio(asset['path'])
 actual=scorer(x)
 old=next(r['quality'] for r in stored if r['id']==record['id'] and r['variant']==variant)
 with torch.no_grad():
  dns=m.deep_noise_suppression_mean_opinion_score(x[None],16000,False,'cpu',1).flatten().tolist()
  nis=n.non_intrusive_speech_quality_assessment(x[None],16000).flatten().tolist()
 assert abs(actual['dnsmos']-dns[3])<1e-9
 assert abs(actual['nisqa']-nis[0])<1e-6
 results['fixtures'][variant]={'audio_sha256':asset['sha256'],'samples':x.numel(),'scores':actual,'historical_scores':old,'difference_from_historical':{k:actual[k]-old[k] for k in actual},'dnsmos_all_channels_p808_sig_bak_ovrl':dns,'nisqa_all_channels_mos_noi_dis_col_loud':nis}
Path('verification/strict-parity-audit/quality-probes.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
