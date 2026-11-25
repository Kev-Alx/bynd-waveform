"""Replay frozen pilot evidence; no models, holdouts, or old files are modified."""
import argparse,csv,json,pathlib,statistics
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy.signal import stft
from scipy.stats import spearmanr
ROOT=pathlib.Path('/workspace/improvement'); OUT=ROOT/'execution_tradeoff'
def main():
 p=argparse.ArgumentParser();p.add_argument('--require-goal',action='store_true');p.add_argument('--signals',action='store_true');a=p.parse_args()
 allrows=list(csv.DictReader((ROOT/'all_results.csv').open()))
 result={'frozen_condition_rows':len(allrows),'joint_passes':sum(r['goalPass']=='True' for r in allrows),'timing_scope':'saved cumulative optimization loops only; excludes loading/calibration/preprocessing; not GPU-active seconds'}
 manifest=[json.loads(x) for x in (ROOT/'execution48/development-48.jsonl').read_text().splitlines()]; durations={r['id']:r['duration'] for r in manifest};assert len(durations)==48
 result['audio_seconds']=sum(durations.values());result['methods']={};per=[]
 for study,vs in [('fvpilot',['single_50','single_150','single_1500','ensemble_raw_50','ensemble_raw_200','ensemble_raw_500','ensemble_eot_50','ensemble_eot_200','ensemble_eot_500']),('melpilot',['mel_eot_50','mel_eot_200','mel_eot_500'])]:
  scores=[json.loads(x) for x in (ROOT/f'execution_scaled/{study}/source-scores.jsonl').read_text().splitlines()]
  for v in vs:
   rs=[r for r in scores if r['variant']==v];assert len(rs)==48 and sum(r['reference_units'] for r in rs)==994
   ts=[]
   for uid in durations:
    d=json.loads((ROOT/f'execution_scaled/{study}/traces/{v}/{uid}.json').read_text());t=d['wall_seconds_cumulative'];ts.append(t);per.append({'id':uid,'variant':v,'audio_seconds':durations[uid],'optimization_seconds':t,'seconds_per_audio_second':t/durations[uid]})
   result['methods'][v]={'count':48,'S':sum(r['substitutions'] for r in rs),'D':sum(r['deletions'] for r in rs),'I':sum(r['insertions'] for r in rs),'wer':100*sum(r['substitutions']+r['deletions']+r['insertions'] for r in rs)/994,'quality':{k:statistics.mean(r['quality'][k] for r in rs) for k in ['dnsmos','nisqa','utmos']},'optimization_seconds':sum(ts),'mean_seconds_per_recording':statistics.mean(ts),'p50_seconds':float(np.median(ts)),'p95_seconds':float(np.percentile(ts,95)),'seconds_per_audio_second':sum(ts)/result['audio_seconds'],'new_1000_recordings_hours':statistics.mean(ts)*1000/3600,'teacher_10_audio_hours_gpu_allocated_hours_loop_only':sum(ts)/result['audio_seconds']*10}
 if a.signals:
  generated=[json.loads(x) for x in (ROOT/'execution_scaled/fvpilot/generated.jsonl').read_text().splitlines()];signals=[]
  scores={(r['id'],r['variant']):r for r in [json.loads(x) for x in (ROOT/'execution_scaled/fvpilot/source-scores.jsonl').read_text().splitlines()]}
  for row in generated:
   clean,_=sf.read(row['audio']['clean']['path'],dtype='float32');_,_,zc=stft(clean,16000,nperseg=512,noverlap=352);mc=np.abs(zc)
   for v in ['fv_reconstruction','fv_quality50','single_50','single_1500','ensemble_eot_200','ensemble_eot_500']:
    y,_=sf.read(row['audio'][v]['path'],dtype='float32');_,_,zy=stft(y,16000,nperseg=512,noverlap=352);m=np.abs(zy);n=min(m.shape[1],mc.shape[1]);m=m[:,:n];c=mc[:,:n];frame_c=np.sqrt(np.mean(c*c,axis=0));frame_m=np.sqrt(np.mean(m*m,axis=0));quiet=frame_c<.1*max(frame_c)
    s=scores[row['id'],v];q=s['quality'];signals.append({'id':row['id'],'variant':v,'log_magnitude_distance':float(np.mean(np.abs(np.log(np.maximum(m,1e-5))-np.log(np.maximum(c,1e-5))))),'relative_magnitude_error':float(np.linalg.norm(m-c)/np.linalg.norm(c)),'energy_ratio':float(np.sum(y*y)/np.sum(clean*clean)),'envelope_relative_error':float(np.linalg.norm(frame_m-frame_c)/np.linalg.norm(frame_c)),'quiet_energy_ratio':float(np.sum(frame_m[quiet]**2)/max(np.sum(frame_c[quiet]**2),1e-12)),'utmos':q['utmos'],'S':s['substitutions'],'D':s['deletions'],'I':s['insertions']})
  (OUT/'signal_diagnostics.json').write_text(json.dumps(signals,indent=2)+'\n');result['signals']={}
  for v in sorted({r['variant'] for r in signals}):
   rr=[r for r in signals if r['variant']==v];result['signals'][v]={k:statistics.mean(r[k] for r in rr) for k in ['log_magnitude_distance','relative_magnitude_error','energy_ratio','envelope_relative_error','quiet_energy_ratio']};result['signals'][v]['utmos_vs_logmag_spearman']=float(spearmanr([r['utmos'] for r in rr],[r['log_magnitude_distance'] for r in rr]).statistic)
 (OUT/'saved_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
 with (OUT/'saved_latency_per_recording.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
 print(json.dumps(result,indent=2))
 if a.require_goal:assert result['joint_passes']>0,'Frozen evidence reproduces joint WER/quality failure: 0 of 67 pass'
if __name__=='__main__':main()
