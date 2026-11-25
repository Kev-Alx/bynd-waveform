"""CPU/stdlib audit of saved evidence only; never loads or executes a model.
Run: /opt/venv/bin/python /workspace/improvement/audit_saved_baselines.py
All outputs remain beside this file under the repository-root improvement/.
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib, json, math, statistics, subprocess, unicodedata
from datetime import datetime, timezone
ROOT=Path('/workspace'); OUT=ROOT/'improvement'
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
 return h.hexdigest()
def norm(s):
 s=unicodedata.normalize('NFKC',s).lower()
 return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s).split())
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  curr=[i]
  for j,y in enumerate(b,1): curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=curr
 return prev[-1]
file_hashes={}; issues=[]; waveform_checked=set(); source_checked=set()
def check_hash(p,expected):
 p=Path(p)
 if not p.is_absolute(): p=ROOT/p
 actual=sha(p); file_hashes[str(p)]=actual
 if actual!=expected: issues.append(f'Hash mismatch: {p}')
 return actual
specs=[
 ('historical-selected-999',ROOT/'reproduction-runs/english-feature-test-999','surrogate-eval',999),
 ('paper-budget-mean-999',ROOT/'reproduction-runs/paper-quality-research/paper-setting-test-999','evaluation',999),
 *[(n,ROOT/'reproduction-runs/paper-quality-stable-execution/pilot'/n,'evaluation',24) for n in ['strong-sum-lr-0.1','quality-sum-lr-0.02','timestamp-sum-lr-0.05']],
 *[(n,ROOT/'reproduction-runs/paper-quality-stable-ablations/pilot'/n,'evaluation',24) for n in ['without-tv','without-hf','without-perceptual']],
 *[(n,ROOT/'reproduction-runs/paper-quality-coordinate-research/pilot'/n,'evaluation',24) for n in ['time_energy','channel_std']]
]
results=[]
for name,run,ev,count in specs:
 print('Auditing',name,flush=True)
 meta=read(run/'run.json'); config=meta['config'].get('parent_config',meta['config']); gen=rows(run/'generated.jsonl'); summary=read(run/ev/'summary.json'); scores=rows(run/ev/'utterances.jsonl')
 check_hash(run/'generated.jsonl',meta['generated_sha256']); check_hash(run/ev/'utterances.jsonl',summary['utterances_sha256']); check_hash(meta['manifest'],meta['manifest_sha256'])
 if meta['status']!='complete' or len(gen)!=count: issues.append(f'Incomplete run {name}')
 ids=[r['id'] for r in gen]
 if len(set(ids))!=count or meta['ids']!=ids: issues.append(f'ID mismatch {name}')
 manifest={r['id']:r for r in rows(meta['manifest'])}; generated={r['id']:r for r in gen}
 if len(scores)!=3*count or len({(r['id'],r['variant']) for r in scores})!=3*count: issues.append(f'Score coverage {name}')
 for g in gen:
  m=manifest[g['id']]
  if g['text']!=m['text'] or g['input_audio_sha256']!=m['audio_sha256']: issues.append(f'Manifest reference {name}/{g["id"]}')
  if m['audio'] not in source_checked:
   check_hash(m['audio'],m['audio_sha256']); source_checked.add(m['audio'])
  for a in g['audio'].values():
   if a['path'] not in waveform_checked: check_hash(a['path'],a['sha256']);waveform_checked.add(a['path'])
  for asset_key in ['delta','trajectory_receipt']:
   asset=g.get(asset_key)
   if isinstance(asset,dict) and asset.get('path'):
    expected=asset.get('file_sha256',asset.get('sha256'))
    if expected: check_hash(asset['path'],expected)
  if len(g['trace'])!=50: issues.append(f'Step count {name}/{g["id"]}')
  if any(not math.isfinite(float(v)) for t in g['trace'] for v in t.values() if isinstance(v,(int,float))): issues.append(f'Nonfinite trace {name}')
  if any(t['radius_after_update']>config['attack']['rho']+1e-6 for t in g['trace']): issues.append(f'Radius trace {name}')
 aggregates={}
 for variant in ['clean','reconstruction','adversarial']:
  pool=[r for r in scores if r['variant']==variant]
  for s in pool:
   g=generated[s['id']]
   if s['reference']!=g['text'] or s['audio_sha256']!=g['audio'][variant]['sha256']: issues.append(f'Score/audio link {name}/{s["id"]}')
   rn,hn=norm(s['reference']),norm(s['hypothesis'])
   if rn!=s['reference_normalized'] or hn!=s['hypothesis_normalized']: issues.append(f'Normalization {name}')
   if len(rn.split())!=s['reference_units'] or distance(rn.split(),hn.split())!=sum(s[k] for k in ['substitutions','deletions','insertions']): issues.append(f'Edit distance {name}/{s["id"]}/{variant}')
  a={'count':len(pool),**{k:sum(s[k] for s in pool) for k in ['substitutions','deletions','insertions','reference_units']}}
  a['error_percent']=100*sum(a[k] for k in ['substitutions','deletions','insertions'])/a['reference_units']
  a['quality']={k:statistics.fmean(s['quality'][k] for s in pool) for k in ['dnsmos','nisqa','utmos']}
  published=summary['summary'][variant]
  if any(a[k]!=published[k] for k in ['count','substitutions','deletions','insertions','reference_units','error_percent']) or any(abs(a['quality'][k]-published['quality'][k])>1e-12 for k in a['quality']): issues.append(f'Aggregate differs {name}/{variant}')
  a['all_quality_above_2_5']=sum(all(s['quality'][k]>2.5 for k in s['quality']) for s in pool)
  a['quality_at_or_below_2_5']={k:sum(s['quality'][k]<=2.5 for s in pool) for k in a['quality']}
  aggregates[variant]=a
 file_hashes[str(run/'verification.json')]=sha(run/'verification.json')
 verification=read(run/'verification.json')
 if 'summary_sha256' in verification: check_hash(run/ev/'summary.json',verification['summary_sha256'])
 verification_receipts={k:verification[k] for k in ['status','count','scope','checked_projected_iterates','active_terms','disabled_terms_verified_zero','coordinate_kind','coordinate_weights_rebuilt','theta_delta_pairs_verified','original_physical_ball_verified','exact_control_score_rows'] if k in verification}
 if 'resynthesis' in verification:
  rs=verification['resynthesis']; verification_receipts['resynthesis_count']=len(rs); verification_receipts['reported_max_abs_error']=max((x.get('max_abs_error',0) for x in rs),default=None)
 radii=[g['trace'][-1]['radius_after_update'] for g in gen]
 results.append({'name':name,'run':str(run),'evaluation':str(run/ev),'count':count,'config':meta['config'],'roles':meta['roles'],'manifest':meta['manifest'],'manifest_sha256':meta['manifest_sha256'],'fingerprint':meta['fingerprint'],'aggregates':aggregates,'independent_edit_distance_rows':len(scores),'trace_updates_checked':sum(len(g['trace']) for g in gen),'final_radius_from_saved_trace':{'min':min(radii),'median':statistics.median(radii),'max':max(radii)},'generation_seconds_sum':meta.get('generation_seconds'),'saved_verification_receipts':verification_receipts})
manifest_paths={
 'pilot20': ROOT/'data/manifests/pilot.jsonl',
 'reconstruction12': ROOT/'data/manifests/reconstruction-pilot.jsonl',
 'validation100': ROOT/'data/manifests/english-validation.jsonl',
 'test999': ROOT/'data/manifests/english-test-999.jsonl',
 'reused_dev24': ROOT/'reproduction-runs/paper-quality-overnight/data/pilot-24.jsonl',
 'prospective48':ROOT/'reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl'
}
manifest_rows={k:rows(p) for k,p in manifest_paths.items()}; manifests={}
for k,rs in manifest_rows.items():
 manifests[k]={'path':str(manifest_paths[k]),'sha256':sha(manifest_paths[k]),'count':len(rs),'speakers':len({r['speaker'] for r in rs}),'roles':sorted({r['role'] for r in rs}),'splits':sorted({r['split'] for r in rs}),'duration_seconds_sum':sum(r['duration'] for r in rs),'duration_seconds_min':min(r['duration'] for r in rs),'duration_seconds_max':max(r['duration'] for r in rs),'sample_rates':sorted({r['sample_rate'] for r in rs}),'ids':[r['id'] for r in rs]}
 for r in rs:
  if r['audio'] not in source_checked: check_hash(r['audio'],r['audio_sha256']);source_checked.add(r['audio'])
overlaps=[]
for i,(a,aa) in enumerate(manifest_rows.items()):
 for b,bb in list(manifest_rows.items())[i+1:]:
  overlaps.append({'a':a,'b':b,'id_overlap':len({r['id'] for r in aa}&{r['id'] for r in bb}),'audio_sha256_overlap':len({r['audio_sha256'] for r in aa}&{r['audio_sha256'] for r in bb}),'speaker_overlap':len({r['speaker'] for r in aa}&{r['speaker'] for r in bb})})
# Inventory every extant per-utterance score file. Exposure is historical evidence,
# never a claim that an evaluation-only target may be used to tune a candidate.
exposure=[]
for p in sorted((ROOT/'reproduction-runs').rglob('utterances.jsonl')):
 try: rs=rows(p)
 except (UnicodeDecodeError,json.JSONDecodeError): continue
 ids={r.get('id') for r in rs}
 hits={k:len(ids&{r['id'] for r in mr}) for k,mr in manifest_rows.items()}
 if any(hits.values()): exposure.append({'path':str(p),'score_rows':len(rs),'cohort_id_counts':hits})
inputs=['paper/BeyondWaveform.md','BeyondWaveform.pdf','beyondwaveform/attacks.py','beyondwaveform/losses.py','beyondwaveform/pipeline.py','beyondwaveform/config.py','beyondwaveform/evaluation.py','beyondwaveform/data.py','beyondwaveform/models/feature.py','beyondwaveform/models/whisper.py','docs/STRICT_PARITY_AUDIT.md','docs/THREAD_SUMMARY_AND_PLAN.md','configs/english_selected.yaml','configs/english_selected.freeze.json']
for p in inputs: file_hashes[str(ROOT/p)]=sha(ROOT/p)
jobs={}
for k in ['english-feature-test-999','paper-quality-stable-execution','paper-quality-stable-ablations','paper-quality-coordinate-research']:
 p=ROOT/'reproduction-runs'/k/'job.json'; d=read(p); start=datetime.fromisoformat(d['started_at']); finish=datetime.fromisoformat(d['finished_at']);jobs[k]={'status':d['status'],'exit_code':d['exit_code'],'started_at':d['started_at'],'finished_at':d['finished_at'],'elapsed_wall_seconds':(finish-start).total_seconds()}
resource={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'nvidia_smi':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,'disk':subprocess.run(['df','-h','/workspace'],capture_output=True,text=True).stdout,'cpu_count':__import__('os').cpu_count(),'memory':Path('/proc/meminfo').read_text().splitlines()[:3]}
artifact_inventory={}
for sub in ['artifacts/checkpoints','artifacts/whisper','artifacts/hf-hub','artifacts/quality-research-scorers','data/LibriSpeech','data/data_aishell/wav']:
 p=ROOT/sub;artifact_inventory[sub]=[{'name':x.name,'bytes':x.stat().st_size if x.is_file() else None,'is_directory':x.is_dir()} for x in sorted(p.iterdir())]
report={'audit_type':'Saved evidence and CPU arithmetic only; no neural re-inference, training, attack generation, or source edits','created_at':datetime.now(timezone.utc).isoformat(),'git_head':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),'preexisting_working_tree_preserved':True,'runs':results,'manifests':manifests,'cohort_overlaps':overlaps,'score_file_exposure':exposure,'jobs':jobs,'resources':resource,'available_artifact_inventory':artifact_inventory,'checked_unique_source_audio_files':len(source_checked),'checked_unique_saved_waveforms':len(waveform_checked),'file_sha256':file_hashes,'issues':issues,'status':'pass' if not issues else 'issues_found'}
(OUT/'baseline_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'issues':issues,'runs':len(results),'score_rows':sum(x['independent_edit_distance_rows'] for x in results),'source_audio':len(source_checked),'saved_waveforms':len(waveform_checked),'scored_files':len(exposure),'prospective48_scored_files':sum(x['cohort_id_counts']['prospective48']>0 for x in exposure),'results':[{'name':r['name'],**r['aggregates']['adversarial']} for r in results]},indent=2),flush=True)
