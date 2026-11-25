"""CPU-only proxy budgets from the previously quality-admissible48 control."""
import pathlib,json,sys,time
D=pathlib.Path(__file__).resolve().parent;P=D.parent;sys.path[:0]=[str(D),str(P),'/workspace','/workspace/scripts']
import torch,soundfile as sf
from feature_policy import spectral_proxy,envelope_proxy

def main():
 torch.set_num_threads(2);start=time.perf_counter();rows=[json.loads(s) for s in (P/'execution_scaled/fvpilot/generated.jsonl').read_text().splitlines()];records=[]
 for r in rows:
  a,sa=sf.read(r['audio']['fv_quality50']['path'],dtype='float32');b,sb=sf.read(r['audio']['fv_reconstruction']['path'],dtype='float32');assert sa==sb==16000 and len(a)==len(b)
  with torch.no_grad():sp=float(spectral_proxy(torch.from_numpy(a),torch.from_numpy(b)));ep=float(envelope_proxy(torch.from_numpy(a),torch.from_numpy(b)))
  records.append({'id':r['id'],'spectral':sp,'envelope':ep})
 assert len(records)==48
 caps={k:sum(r[k] for r in records)/48 for k in ['spectral','envelope']};assert all(v>1e-8 for v in caps.values());result={'count':48,'caps':caps,'basis':'Mean MR-log-STFT and frame-RMS-envelope distance of fixed quality-admissible FV50 control to zero-edit HiFi-GAN reconstruction; proxy budgets are not MOS guarantees','quality_reference':{'wer':49.798792756539236,'dnsmos':3.1739,'nisqa':4.0698,'utmos':3.5789},'wall_seconds':time.perf_counter()-start,'records':records};(D/'proxy_calibration.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
