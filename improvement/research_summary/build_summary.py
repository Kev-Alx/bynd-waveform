"""Reconcile saved research evidence and produce the requested Markdown summary.
No audio/model inference, experiment edits, new selection, or GPU work.
"""
from pathlib import Path
import json,csv,hashlib,math,statistics,collections,re,datetime
P=Path('/workspace/improvement');R=P/'research_summary';ROOT=P.parent
INPUTS={}
def read(path):
 p=Path(path);p=p if p.is_absolute() else P/p;INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def lines(path):
 p=Path(path);p=p if p.is_absolute() else P/p;INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def link(label,path):
 p=Path(path);p=p if p.is_absolute() else P/p;return f'[{label}]({p})'
def table(headers,rows):
 def cell(v):return str(v).replace('|',' / ').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(cell,row))+' |' for row in rows]])
def fmt(x,n=4):return f'{float(x):.{n}f}'
def quality(s):return all(s[k]>=v for k,v in [('dnsmos',3),('nisqa',3.75),('utmos',3.5)])
def missing(s):return ', '.join((['WER'] if float(s['wer'])<=75 else [])+[label for key,label,floor in [('dnsmos','DNSMOS',3),('nisqa','NISQA',3.75),('utmos','UTMOS',3.5)] if float(s[key])<floor]) or 'None'
def metrics(s):return [fmt(s['error_percent']),*[fmt(s['quality'][k]) for k in ['dnsmos','nisqa','utmos']]]
NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
rows=list(csv.DictReader((P/'all_results.csv').open()));INPUTS[str(P/'all_results.csv')]=hashlib.sha256((P/'all_results.csv').read_bytes()).hexdigest()
assert len(rows)==78 and sum(int(r['count']) for r in rows)==3192
summaries={};sources={};scored=0
for source in dict.fromkeys(r['sourceFile'] for r in rows):
 path=ROOT/source;doc=read(path);summaries[source]=doc
 score=path.parent/('utterances.jsonl' if path.parent.name=='evaluation' else 'source-scores.jsonl');rr=lines(score);sources[source]=str(score)
 for variant,s in doc['summary'].items():
  match=[r for r in rr if r['variant']==variant];assert len(match)==s['count'];assert len({r['id'] for r in match})==len(match)
  for k in ['substitutions','deletions','insertions','reference_units']:assert sum(r[k] for r in match)==s[k]
  assert math.isclose(100*sum(s[k] for k in ['substitutions','deletions','insertions'])/s['reference_units'],s['error_percent'],abs_tol=1e-10)
  for k in ['dnsmos','nisqa','utmos']:assert math.isclose(statistics.mean(r['quality'][k] for r in match),s['quality'][k],abs_tol=1e-10)
 for row in [r for r in rows if r['sourceFile']==source]:
  s=doc['summary'][row['variant']];assert int(row['count'])==s['count'];assert int(row['referenceWords'])==s['reference_units'];assert math.isclose(float(row['wer']),s['error_percent'],abs_tol=1e-10)
  for k in ['dnsmos','nisqa','utmos']:assert math.isclose(float(row[k]),s['quality'][k],abs_tol=1e-10)
  assert (row['qualityPass']=='True')==quality({k:float(row[k]) for k in ['dnsmos','nisqa','utmos']});scored+=s['count']
assert scored==3192 and all(r['goalPass']=='False' for r in rows)
base=read('baseline_evidence.json');model=read('execution_model_v1/results.json');state=read('execution_model_v1/exit_receipt.json');wrapper=read('execution_model_v1/controller_exit.json');timings=lines('execution_model_v1/pilot/latencies.jsonl');pilot=read('execution_model_v1/pilot/source-summary.json');profile=read('execution_model_v1/profile.json');trade=read('execution_tradeoff/results.json');diagnosis=read('execution_tradeoff/saved_diagnosis.json');ui=read('audio_browser_data_receipt.json');browser=read('execution_model_v1/browser_receipt.json')
assert len(timings)==384 and len({(r['id'],r['variant']) for r in timings})==384
model_audit=[]
for variant,s in pilot['summary'].items():
 rr=[r for r in timings if r['variant']==variant];assert len(rr)==48 and len({r['id'] for r in rr})==48
 speed=statistics.mean(r['steady_generation_seconds'] for r in rr);mem=max(r['peak_allocated_bytes'] for r in rr);q=quality(s['quality']);wer=s['error_percent']>75;eff=speed<=16.773926311657608/10 and mem<20*1024**3 and all(r['source_queries_at_inference']==r['per_recording_optimization_updates']==0 for r in rr)
 model_audit.append({'variant':variant,'count':48,'wer_pass':wer,'quality_pass':q,'efficiency_pass':eff,'joint_pass':wer and q and eff,'mean_generation_seconds':speed,'peak_allocated_bytes':mem})
assert not any(r['joint_pass'] for r in model_audit)
assert wrapper['exit_code']==1 and state['status']=='failed_execution' and state['delivery_status']=='complete'
assert not (P/'execution_model_v1/selection.json').exists()
(R/'model_gate_review.json').write_text(json.dumps({'reviewed_at_taipei':NOW,'scope':'Independent CPU review of complete saved scores/timings, not a repaired pipeline selection or new experiment','original_selection_error':'Unique-utterance manifest reader rejected valid repeated IDs across eight latency variants','per_variant':model_audit,'eligible_candidates':[],'original_exit_receipts_preserved':True},indent=2)+'\n')
# One row per complete saved condition, keeping controls and checkpoints visible.
groups=collections.defaultdict(list)
for row in rows:groups[row['study']].append(row)
order=['Historical original control · LR 0.02','Historical original control · LR 0.05','Historical original control · LR 0.1','Word-risk objective','Local feature mixtures','Learned waveform generator','Generator checkpoint · 500 updates','Generator checkpoint · 1000 updates','Generator checkpoint · 1500 updates','Waveform mechanism pilot','Waveform refinement','Quiet-region protection','Decoder qualification','Larger models and longer feature runs','BigVGAN mel-space pilot','Quality and efficiency pilot','Learned speech-feature model']
assert set(order)==set(groups)
changes={
 'clean':'Unmodified speech; control', 'reconstruction':'Frozen WavLM6 → HiFi-GAN, no edit; control','baseline_frozen':'Original FV50, LR0.1; attack-strength control','baseline_attenuated':'Original attack scaled to R2 residual norm; diagnostic','initial-generator':'Untrained waveform U-Net; control',
 'fv_quality':'Original FV50, LR0.02; quality control','fv50':'Original FV50, LR0.02; quality control','fv100':'Same original LR0.02 trajectory, 100 updates','full50':'Waveform sign-PGD50, STFT/content penalties','pgd_full':'Waveform sign-PGD50, STFT/content penalties','pgd_nll':'Same waveform PGD50 with recognition loss only','r3_initial':'Frozen untrained R3 waveform model','r3_trained':'Frozen trained R3 waveform model','weighted100':'50 more weighted waveform updates after full50','constrained100':'50 more projected/backtracked updates under distortion caps','envelope_nll50':'Waveform PGD50 with clean-local-energy budgets','envelope_full50':'Same local-energy budgets plus STFT/content penalties','attenuated_global_nll50':'Global NLL residual scaled to envelope residual RMS','global_full50':'Reused weighted waveform PGD50 control','global_nll50':'Reused NLL-only waveform PGD50 control','bigvgan_reconstruction':'Qualified 22.05kHz BigVGAN →16kHz; no edit','fv_quality50':'Reused original FV50, LR0.02; quality control','fv_reconstruction':'Frozen original feature decoder; no edit','raw50':'Original strong FV50 with retained decoder graph','stft_identity':'Raw FV50 through STFT/inverse-STFT only','anchor25':'Raw FV50, 25% log-magnitude blend to reconstruction',
 'initial_r050':'Untrained feature model, relative radius0.05','quality_shuffled_r050':'Quality model conditioned on next clip, radius0.05; diagnostic'
}
def change(study,v):
 if v in changes:return changes[v]
 if v=='adversarial':
  if study=='Word-risk objective':return 'N-best word risk + one deletion candidate; feature search'
  if study=='Local feature mixtures':return 'Convex local feature mixtures with cosine≥0.95'
  if 'Generator checkpoint' in study:return 'R3 waveform model at '+study.split('·')[-1].strip()
  if study=='Learned waveform generator':return 'Selected trained 1.59M waveform U-Net'
  return 'Original FV50, sum NLL; '+study.split('·')[-1].strip()
 if v.startswith('single_'):return 'Original small-only FV, LR0.1; '+v.split('_')[-1]+' cumulative updates'
 if v.startswith('ensemble_'):return 'Small/large-v2/Data2Vec FV; '+('sampled transforms' if '_eot_' in v else 'identity only')+'; '+v.split('_')[-1]+' updates'
 if v.startswith('mel_eot_'):return 'BigVGAN log-mel edits; three sources + transforms; '+v.split('_')[-1]+' updates'
 if v.startswith(('source_r','quality_r')):return ('Recognition-only' if v.startswith('source') else 'Violation-only quality correction')+' feature model; radius'+fmt(int(v.split('_r')[1])/1000,3)
 raise ValueError((study,v))
contexts={
 'Word-risk objective':'Feature search, 50 updates; same frozen encoder/decoder as original. No learned attack-model training.',
 'Local feature mixtures':'Feature search, 50 updates; neighboring frames within ±2, cosine floor0.95, global radius0.1. No model training.',
 'Learned waveform generator':'R3: 1.59M waveform U-Net trained on T1; one seed,1779updates/7116microbatches. New audio uses its frozen predicted residual.',
 'Waveform mechanism pilot':'Controlled per-recording waveform search versus the frozen R3 model and original FV. Waveform Linf/RMS≤0.003,50 sign updates; no new model training.',
 'Waveform refinement':'Continue the same waveform initializer to100updates; weighted loss versus hard proxy constraints. Original FV50/FV100 use LR0.02. No attack-model training.',
 'Quiet-region protection':'Per-recording waveform search with clean25ms/10ms frame-energy bounds; primary NLL-only, weighted ablation and RMS-matched control. No model training.',
 'Decoder qualification':'Reconstruction-only qualification before BigVGAN attacks; original controls retained. No newly trained decoder.',
 'Larger models and longer feature runs':'Frozen WavLM6/HiFi-GAN; original small-only LR0.1 or three-source ensemble. Ensemble weights0.5/0.25/0.25, per-recording clean gradient calibration. All rho0.1; full original TV/HF. No attack-model training.',
 'BigVGAN mel-space pilot':'Frozen BigVGAN log-mel decoder, three sources and transforms; LR0.01,relative radius0.1 plus log-mel box≤log2. Different norm units from FV, not a matched perceptual budget.',
 'Quality and efficiency pilot':'Matched decoder-graph implementation and one fixed25% spectral repair. The model weights stay frozen; no training. Exact raw FV50 WAV parity was demonstrated.',
 'Learned speech-feature model':'5.07M conditioned temporal feature policy trained directly on T10. Common300-update warmup, source3791updates, quality3680updates. No optimized attack-teacher targets; fresh WavLM encoding and one policy/decoder forward per new clip.'
}
md=['# Every completed research condition','',f'Compiled {NOW} (Asia/Taipei). Covers the earlier audited starting point and all completed improvement experiments through the latest model run.','',link('Short overview','26_research_summary.md')+' · '+link('Data, runtime and evidence','28_data_runtime_and_evidence.md'),'','## Read the tables','',
'WER is source Whisper-small corpus word error rate; higher is the attack objective. DNSMOS, NISQA and UTMOS are automatic speech-quality estimates; higher is better. The fixed joint requirement is **WER>75%, DNSMOS≥3, NISQA≥3.75, UTMOS≥3.5**. “Missing gate” names every unmet requirement. A quality-passing clean recording is a control, not an attack success.','',
'**Coverage:78 scored conditions /3192 scored recording-condition rows.** Reused controls and checkpoints remain visible; these are not78independent experiments or3192unique recordings. Main catalogue cohorts are D24(24recordings/475words) and D48(48recordings/994words). All later performance pilots use D48. Training pools T1/T10 and exposure limits are defined in the companion data file.','',
'## Earlier baseline audit: ten complete runs','',
'These runs predate the improvement experiments and were rechecked from saved evidence in the earlier chat. Three24-point original settings also appear in the78-condition catalogue below; do not add them as independent replications. The999-point runs are historical test-clean results, not a fresh test of an improved method. Paper values are separate claims under an unmatched protocol.']
legacy=[]
for r in base['runs']:
 c=r['config'].get('parent_config',r['config']);a=c['attack'];s=r['aggregates']['adversarial'];desc=f"rho{a['rho']};lambda{a['lambda_perc']};{c['surrogate']['nll_reduction']}NLL;LR{a['learning_rate']}"
 if r['name'].startswith('without-'):desc+='; '+r['name']
 if r['name'] in ['time_energy','channel_std']:desc+='; '+r['name']+' optimizer coordinates'
 legacy.append([link(r['name'],r['evaluation']),r['count'],desc,*metrics(s),fmt(r['aggregates']['clean']['error_percent']),fmt(r['aggregates']['reconstruction']['error_percent'])])
md+=['',table(['Audited run','N','Original setting / change','Attack WER%','DNSMOS','NISQA','UTMOS','Clean WER%','Reconstruction WER%'],legacy),'',
'Paper headline, for context only: WER75.4300%, DNSMOS2.9540, NISQA3.5260, UTMOS3.5900. The author subset/artifacts and several implementation choices were not recovered; those numbers cannot be treated as a matched local baseline. '+link('Original-method audit','01_original_method_and_baseline.md')+'.','',
'## Complete78-condition development catalogue']
for i,study in enumerate(order,1):
 rr=groups[study];n=int(rr[0]['count']);context=contexts.get(study,'Frozen original feature method; no attack-model training.' if study.startswith('Historical') else 'Saved R3 checkpoint from the same T1 training run; no independent retraining.')
 md+=['',f'### {i}. {study}','',f'**Data:** '+('D24:24recordings/475reference words.' if n==24 else 'D48:48recordings/994reference words.')+' '+context,'',link('Saved full-cohort scores',ROOT/rr[0]['sourceFile'])+'.','',table(['Condition','Change or control','WER%','DNSMOS','NISQA','UTMOS','Missing gate'],[[r['variant'],change(study,r['variant']),*[fmt(r[k]) for k in ['wer','dnsmos','nisqa','utmos']],missing(r)] for r in rr])]
md+=['','## Attempts without a completed performance table','',
 table(['Attempt / check','Data and execution','What happened','How to interpret it'],[
 ['Frozen-original replay','3duration-selected D24 recordings;150total updates','All saved losses,gradients,deltas,Adam states and final audio matched exactly','Execution equivalence on these controls; not a new attack result'],
 ['R1 canonical word-risk','3calibration clips;3generation probes;then intended D24','Full D24 stopped at first clip:all4beam hypotheses had correct normalized words,so every risk cost waszero','No completed24-point performance/MOS result;3-clip generation observations use another backend and are not a fair WER comparison'],
 ['R1 logging-only wrapper','Same failed attempt,scientific code unchanged','Preserved clearer failure receipts','Not another scientific method or scored run'],
 ['R2 geometry smoke','3duration-selected recordings','Execution checks completed before D24','Engineering check;not48-point performance evidence'],
 ['R3 cached-reference preflight','3duration-selected recordings','Strict clean features marginally exceeded the old cached absolute tolerance','Stopped before training;named strict-reference variant followed'],
 ['R3 strict-reference attempt','Gradient calibration before training','Upstream WavLM in-place operation broke autograd','Execution failure;no trained-model result'],
 ['R3 value-preserving autograd repair','CPU upstream regressions and real GPU preflight','Cloned saved convolution input;forward values preserved;then completed R3 training','Repair lineage for the one completed R3 experiment'],
 ['R3 saved failure replay','Existing24saved outputs;no new model calls','Nearly zero residual;all normalized transcripts unchanged;diagnostic exit1','A recorded negative attack-signal finding,not a crashed benchmark'],
 ['Latest model automatic selection','After all384pilot outputs were scored','Manifest reader rejected repeated recording IDs across8variant timing rows;selection exited1','Training/scoring remain complete;no frozen selection written;independent CPU review findszeroeligible candidates'],
 ['Conditional validation76 and new transfer/defenses','Registered only','No improved condition earned the WER/quality gate;latest selection also had the error above','Not run;absence must not be reported as a defense/transfer result'],
 ['Untouched confirmation48,final200,human andphysical tests','Reserved/planned only','No improvement-cycle evaluation executed','No generalization,human-content,OTA or full-defense claim']]),'',
 link('Latest independent gate review','research_summary/model_gate_review.json')+' · '+link('Canonical R1 stop','execution/r1-report.md')+' · '+link('R3 execution lineage','05_execution_report.md')+'.','',
'Rows and numerical gates were checked against saved summaries and per-recording edit/quality rows for this consolidation. No recognition,quality model or GPU experiment was rerun. Full precision remains in '+link('all_results.csv','all_results.csv')+' and the linked receipts.']
(P/'27_all_runs_comparison.md').write_text('\n'.join(md)+'\n')
# Freeze reviewed table-level data for the two narrative companions.
reviewed={'compiled_at_taipei':NOW,'catalogue_conditions':len(rows),'catalogue_score_rows':scored,'catalogue_rows':rows,'legacy_runs':base['runs'],'latest_model_gate_review':model_audit,'latest_model_status':{k:state[k] for k in ['status','queue_wait_seconds','active_wall_seconds','overall_wall_seconds','delivery_status']},'interface':{k:ui[k] for k in ['conditions','scored_recordings','unique_recordings','unique_audio_assets']}}
(R/'reviewed_results.json').write_text(json.dumps(reviewed,indent=2)+'\n')
(R/'source_manifest.json').write_text(json.dumps({'reviewed_at_taipei':NOW,'files_sha256':INPUTS},indent=2)+'\n')
print(json.dumps({'conditions':len(rows),'scored_rows_reconciled':scored,'legacy_runs':len(base['runs']),'all_fixed_joint_gates_failed':True,'latest_latency_rows':len(timings),'original_model_exit_code':wrapper['exit_code'],'output':str(P/'27_all_runs_comparison.md')},indent=2))
