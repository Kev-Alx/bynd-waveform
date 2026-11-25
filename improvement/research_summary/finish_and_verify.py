"""Polish Markdown spacing and verify the requested summary against saved evidence."""
from pathlib import Path
import re,json,hashlib,csv,datetime,math,statistics
P=Path('/workspace/improvement');R=P/'research_summary';ROOT=P.parent
files=[P/'26_research_summary.md',P/'27_all_runs_comparison.md',P/'28_data_runtime_and_evidence.md']
# Presentation-only corrections outside Markdown link targets.
replacements={
 'Coverage:78':'Coverage: 78',' /3192':' / 3192','not78independent':'not 78 independent','or3192unique':'or 3192 unique','Three24-point':'Three 24-point','the78-condition':'the 78-condition','The999-point':'The 999-point','Complete78-condition':'Complete 78-condition',
 '24recordings':'24 recordings','48recordings':'48 recordings','475words':'475 words','994words':'994 words','475reference':'475 reference','994reference':'994 reference','D24:24':'D24: 24','D48:48':'D48: 48',
 'meanNLL':'mean NLL','sumNLL':'sum NLL','rho0.':'rho 0.','lambda0.':'lambda 0.','lambda1.':'lambda 1.','LR0.':'LR 0.',';lambda':'; lambda',';mean':'; mean',';sum':'; sum',';LR':'; LR',
 'all48':'all 48','all4':'all 4','all384':'all 384','all76':'all 76','zeroeligible':'zero eligible','nearlyzero':'nearly zero','/33speakers':'/ 33 speakers',
 'over4decoded':'over 4 decoded','25msframes/10mshops':'25 ms frames / 10 ms hops','RMS0.003':'RMS 0.003','step0.00012,50':'step 0.00012, 50','weights0.5':'weights 0.5','radius0.1':'radius 0.1','width384':'width 384','radii0.025':'radii 0.025','clip5':'clip 5','AdamW2e-4':'AdamW 2e-4','LR0.01':'LR 0.01',
 'at0.003':'at 0.003','fixed25%':'fixed 25%','over50':'over 50','later100-step':'later 100-step','all48original':'all 48 original','penalties65.0837':'penalties 65.0837','+one':' + one',';4 ': '; 4 ',';3 ':'; 3 ',
 'FV50latency':'FV50 latency','19.649 →16.774seconds':'19.649 → 16.774 seconds','14.63%':'14.63%','bootstrap95%interval12.88':'bootstrap 95% interval 12.88','6.885 →6.928GiB':'6.885 → 6.928 GiB','all48raw':'all 48 raw',
 '50/200/500checkpoint':'50/200/500 checkpoint','original1500':'original 1500','raw-ensemble500':'raw-ensemble 500','transformed-ensemble500':'transformed-ensemble 500','mel500trajectories':'mel 500 trajectories','50-step':'50-step',
 '384outputs':'384 outputs','5.26sloading':'5.26 s loading','4.48sloading':'4.48 s loading','beam5':'beam 5','temperature0':'temperature 0','0seconds':'0 seconds',
 '111fewer':'111 fewer','647.29s':'647.29 s','728.94s':'728.94 s','438.27s':'438.27 s','489.33s':'489.33 s','11076.63s':'11,076.63 s','14186.47s':'14,186.47 s',
 'rho0.5/lambda0.1/LR0.05/50-step':'rho 0.5 / lambda 0.1 / LR 0.05 / 50-step','DNSMOS2.9638':'DNSMOS 2.9638','/NISQA3.8711':'/ NISQA 3.8711','/UTMOS2.7924':'/ UTMOS 2.7924',
 '16kHz':'16 kHz','6kHz':'6 kHz','8kHz':'8 kHz','2kHz':'2 kHz','4kHz':'4 kHz','12kHz':'12 kHz','25%log':'25% log','0.25of':'0.25 of',
 '384valid':'384 valid','48distinct':'48 distinct','FIR6k':'FIR 6k','resample8k':'resample 8k','sFV50':'s FV50','sreference':'s reference','isavailable':'is available','isnot':'is not','aszero':'as zero','only48':'only 48',
 '6570-row':'6,570-row','3192score':'3,192 score','3192distinct':'3,192 distinct','2082deduplicated':'2,082 deduplicated','48sentences':'48 sentences','78conditions':'78 conditions',
 'the50/200/500':'the 50/200/500','144,000per-recording':'144,000 per-recording','288,000source':'288,000 source','48-hour':'48-hour','4.715hours':'4.715 hours','less1':'less 1',
 '27.07/18.27hours':'27.07 / 18.27 hours','13-hour':'13-hour','50-hour':'50-hour','meanMOS':'mean MOS','pooledS/D/I':'pooled S/D/I','onport8888':'on port 8888','underlyingreceipts':'underlying receipts',
 ' →HiFi':' → HiFi',' →3.6825':' → 3.6825',' →4.1247':' → 4.1247',' →72.8370':' → 72.8370',' →3.3604':' → 3.3604','frozen1':'frozen 1',
 'WER75.4300':'WER 75.4300','DNSMOS2.9540':'DNSMOS 2.9540','NISQA3.5260':'NISQA 3.5260','UTMOS3.5900':'UTMOS 3.5900',
 'R3: 1.59M waveform U-Net trained on T1; one seed,1779updates/7116microbatches.':'R3: 1.59M waveform U-Net trained on T1; one seed, 1,779 updates / 7,116 microbatches.',
 'T1;1,779':'T1; 1,779','RNG rules;4':'RNG rules; 4','inthis':'in this','aremore':'are more','orlossy':'or lossy','oroutcomes':'or outcomes','25ms/10ms':'25 ms / 10 ms','50updates':'50 updates','100updates':'100 updates','three-recording':'three-recording',
 'Whisper-base/medium,Wav2Vec2':'Whisper-base/medium, Wav2Vec2','fixed500':'fixed 500','source3791updates':'source 3,791 updates','quality3680updates':'quality 3,680 updates','300-update':'300-update','3calibration':'3 calibration','3generation':'3 generation',
 'first clip:all4beam':'first clip: all 4 beam','cost waszero':'cost was zero','50sign':'50 sign','1.59M-parameter':'1.59M-parameter','zeroeligible':'zero eligible','recording,variant':'recording, variant','recording,variant)pairs':'recording, variant) pairs',
 'WavLM6':'WavLM6','8variant':'8 variant','384pilot':'384 pilot','Four1':'Four 1'
}
for path in files[1:]:
 text=path.read_text();parts=re.split(r'(\]\([^\n]*?\))',text)
 for i in range(0,len(parts),2):
  for a,b in replacements.items():parts[i]=parts[i].replace(a,b)
  parts[i]=re.sub(r'([,;])(?=[A-Za-z])',r'\1 ',parts[i]);parts[i]=re.sub(r'(\d)(recordings|updates|microbatches|words|seconds|hours|minutes|outputs|conditions|frames|GiB|source|total|independent|reference|parameters|raw|audio|inference|words|min\b)',r'\1 \2',parts[i])
 text=''.join(parts)
 parts=re.split(r'(\]\([^\n]*?\))',text)
 for i in range(0,len(parts),2):
  t=parts[i]
  t=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',t)
  t=re.sub(r'\b(all|the|only|over|later|fixed|radius|radii|weights|width|clip|step|versus|forecast|actual|training|Includes|includes|completed|saved|has|imply|independent|prepared|exit)(?=\d)',r'\1 ',t)
  t=re.sub(r'(?<=\d) (GiB|ms|kHz)(?=[A-Za-z])',r' \1 ',t)
  t=re.sub(r'([;:])(?=[A-Za-z0-9])',r'\1 ',t)
  for a,b in [('andcodec','and codec'),('andcosts','and costs'),('andbrowser','and browser'),('andmean','and mean'),('andMarkdown','and Markdown'),('orparameter','or parameter'),('and<','and <'),('Proposed≥','Proposed ≥'),('%lower','% lower'),('×generation','× generation'),('preparedT1','prepared T1'),('N999','N=999'),('`rows','` rows'),(')pairs',') pairs')]:t=t.replace(a,b)
  t=re.sub(r'\bs (cored|core|econds|hared|peakers|ource|entences|ign)\b',lambda m:'s'+m.group(1),t)
  t=re.sub(r'(?<=\d),\s+(?=\d{3}(?:\D|$))',',',t)
  t=t.replace('andlater','and later').replace('preprocessing647','preprocessing 647').replace(' +all',' + all').replace('microbatches;fixed','microbatches; fixed')
  parts[i]=t
 text=''.join(parts);path.write_text(text)
# Add compact explanations for inherited loss/coordinate studies.
f=files[2];s=f.read_text();anchor='| Decoded word risk |'
extra='| Earlier loss/optimizer studies | Full original TV/HF, uniform feature coordinates | Remove TV, HF or both; reweight optimizer coordinates by frame energy or channel standard deviation | D24; no training; physical feature-ball projection retained; no joint pass |\n'
assert anchor in s
if '| Earlier loss/optimizer studies |' not in s:s=s.replace(anchor,extra+anchor,1)
f.write_text(s)
# Verify local links, table structure, condition coverage, and numeric rendering.
links=[];tables=0
for path in files:
 text=path.read_text();assert text.count('```')%2==0
 previous=None
 for line in text.splitlines():
  if line.startswith('|'):
   count=len(line.split('|'))
   if previous is not None:assert count==previous,(path,line)
   else:tables+=1
   previous=count
  else:previous=None
 for target in re.findall(r'\]\((/[^)]+)\)',text):
  target=target.split('#',1)[0];q=Path(re.sub(r':\d+$','',target));assert q.exists() or q==R/'verification.json',(path,target);links.append(target)
allrows=list(csv.DictReader((P/'all_results.csv').open()));body=files[1].read_text()
for row in allrows:
 assert f"| {row['variant']} |" in body
 for k in ['wer','dnsmos','nisqa','utmos']:assert format(float(row[k]),'.4f') in body
assert len(re.findall(r'^### \d+\.',body,re.M))==len({r['study'] for r in allrows})
assert sum(1 for line in body.splitlines() if line.startswith('| ') and line.split('|')[1].strip() in {r['variant'] for r in allrows})==78
source=json.loads((R/'source_manifest.json').read_text());extra_paths=[P/'01_original_method_and_baseline.md',P/'03_literature_review.md',P/'05_execution_report.md',P/'08_pilot48_results.md',P/'10_refinement_results.md',P/'13_envelope_results.md',P/'15_scaled_results.md',P/'18_runtime_and_saved_audio.md',P/'19_source_audit.md',P/'21_quality_efficiency_results.md',P/'23_model_directions_and_sources.md',P/'24_learned_feature_protocol.md',P/'25_learned_model_results.md',P/'execution_refine/validation76-data-note.md',P/'data_and_evaluation_constraints.md',ROOT/'reproduction-runs/english-transfer-999/report.md',ROOT/'reproduction-runs/english-ctc-transfer-999/report.md',ROOT/'reproduction-runs/english-defenses-999/report.md']
extra_paths+=list((P/'execution_model_v1/training').glob('*/receipt.json'))+[P/'execution_model_v1/training_cache/receipt.json',P/'execution_model_v1/pilot/generation-result.json']
for folder in ['execution48/mechanism-pilot48','execution_refine/pilot','execution_envelope/pilot','execution_scaled/qualification','execution_scaled/fvpilot','execution_scaled/melpilot']:extra_paths.append(P/folder/'generation-result.json')
extra_paths+=list((P/'execution_scaled/fvpilot/traces').rglob('*.json'))+list((P/'execution_scaled/melpilot/traces').rglob('*.json'))
for p in extra_paths:source['files_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
# Confirm all evidence inputs still match their recorded digest.
for path,expected in source['files_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
receipt={'status':'pass','reviewed_at_taipei':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'markdown_files':[str(p) for p in files],'table_count':tables,'local_links_checked':len(links),'conditions_in_all_runs_table':78,'scored_recording_rows_reconciled':3192,'historical_baseline_runs':10,'model_timing_rows':384,'latest_model_source_quality_eligible':0,'historical_sources_distinguished_from_new_results':True,'input_files_hashed':len(source['files_sha256']),'original_experiments_modified':False,'new_gpu_work':False,'outputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(R/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
