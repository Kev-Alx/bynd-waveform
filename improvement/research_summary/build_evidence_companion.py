"""Render saved data/cost evidence as Markdown without running any model."""
import sys,pathlib,json,statistics,datetime,math,re
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D))
# Use the reviewed snapshot; preserve experiment files and their original receipts.
P=D.parent;ROOT=P.parent
read=lambda p:json.loads((P/p).read_text())
def lines(path):return [json.loads(s) for s in (P/path).read_text().splitlines() if s.strip()]
def link(label,path):
 p=pathlib.Path(path);p=p if p.is_absolute() else P/p;return f'[{label}]({p})'
def table(headers,rows):
 clean=lambda x:str(x).replace('|',' / ').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(clean,headers))+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(clean,r))+' |' for r in rows]])
f=lambda x,n=3:f'{x:.{n}f}'
base=read('baseline_evidence.json');model=read('execution_model_v1/results.json');state=read('execution_model_v1/exit_receipt.json');trade=read('execution_tradeoff/results.json');profile=read('execution_model_v1/profile.json')
md=['# Data, computation and evidence limits','',link('Overview','26_research_summary.md')+' · '+link('Every scored condition','27_all_runs_comparison.md'),'','## 1. Data used and data kept untouched','',
'All improvement experiments use English LibriSpeech speech. They do not establish Mandarin, physical-playback or human-listening performance. Dataset rows below are pools, not independent experiments. Training audio was acquired and prepared during the earlier research; the opening baseline audit predates that acquisition.']
data=[]
for name,label,purpose in [('reused_dev24','D24: original development','Early R1/R2/R3 and original controls; repeatedly exposed'),('test999','Historical test999','Earlier baseline/transfer/defense results; already exposed, not a fresh test'),('validation100','Historical validation100','Earlier development; 24 recordings later added to D48'),('pilot20','Legacy pilot20','Earlier exploration; two recordings overlap validation100'),('reconstruction12','Legacy reconstruction12','Earlier encoder/decoder checks'),('prospective48','Untouched confirmation48','Reserved; no improvement-cycle inference; speakers overlap development')]:
 a=base['manifests'][name];data.append([link(label,a['path']),a['count'],a['speakers'],f(a['duration_seconds_sum']),purpose])
for path,label,purpose in [('execution48/development-48.jsonl','D48: main pilot','24 old D24 +24 added historical-validation recordings; every later performance pilot'),('execution_refine/validation-76.jsonl','Exposed validation76','Remaining validation100 recordings; prepared, never evaluated for these methods'),('execution/data/train-1h.jsonl','T1: train-clean-100','286 recordings; R3 waveform-model training'),('execution/data/train-10h.jsonl','T10: train-clean-100','2,807 recordings; new feature-model training; T1 is nested inside'),('execution/data/final-reserve-200.jsonl','Untouched final200: test-other','Prospectively frozen harder-speech reserve; no improvement-cycle inference')]:
 rr=lines(path);data.append([link(label,path),len(rr),len({r['speaker'] for r in rr}),f(sum(r['duration'] for r in rr)),purpose])
md+=['',table(['Pool / manifest','Recordings','Speakers','Audio seconds','Use and exposure'],data),'',
'D24 has **475 reference words**; D48 has **994**; historical test999 has **20,310**. New-model timing uses the same D48 total of **360.010 audio seconds**. T1/T10 contain about **1.003/10.004 hours** of speech, not 1/10 hours of training compute.','',
 table(['Boundary','What the evidence permits'],[
 ['D48 versus validation76','No recording/audio-hash/normalized-text overlap; 26 shared speakers. Validation76 is exposed development, not a fresh or speaker-disjoint test.'],
 ['Training versus evaluation','Prepared training excludes the historical/final speakers and duplicate audio/text; the new model registration checked training IDs and speakers against D48/validation76.'],
 ['Confirmation48','Unscored new utterances; all 28 speakers appear in historical development. It cannot establish unseen-speaker generalization.'],
 ['Final200','200 test-other recordings /33speakers, selected to exclude historical/training speakers and duplicate audio/text. Its distribution is harder than test-clean.'],
 ['Metadata versus outcomes','Pool sizes/overlap use saved manifests and receipts. This consolidation did not open protected audio or infer protected outcomes.'],
 ['Repeated development','All candidate development scores are exposed. Repeated tuning and one-seed model training limit generalization/variance claims.']]),'',
 link('Data and evaluation rules','data_and_evaluation_constraints.md')+' · '+link('Validation76 construction','execution_refine/validation76-data-note.md')+'.','',
'## 2. Original method versus changed methods','',
 table(['Method family','Original behavior','Change tested','Training data / new-recording cost'],[
 ['Original FV','WavLM6 →per-recording feature edit →HiFi-GAN; frozen Whisper-small gradients','Quality/strong learning rates and longer trajectories','No attack-model training; repeat optimization for every new recording'],
 ['Decoded word risk','Teacher-forced transcript NLL','Expected bounded word error over4decoded paths; append a deletion candidate in one declared repair','No training; beam refresh/path scoring adds search cost'],
 ['Local feature geometry','Free feature tensor within a global ball','Convex neighboring-frame mixtures, ±2frames/cosine≥0.95','No training; useful feature directions became scarce'],
 ['Waveform U-Net','Per-recording feature optimization','1,590,833 trainable parameters predicting a waveform residual; STFT+WavLM content losses','T1;1,779updates/7,116microbatches; fixed waveform Linf/RMS0.003'],
 ['Waveform PGD / continuation','Feature-space Adam with original TV/HF','Waveform sign updates, step0.00012,50updates; full penalties versus NLL-only; later100-step/capped variants','No attack-model training; weighted penalties65.0837(STFT)/186.6123(content)'],
 ['Clean-energy budgets','Uniform global waveform bound','25msframes/10mshops; perturbation RMS≤clean-frame RMS×10^(-30/20),capped at0.003','No training; protects declared frame energy, not human words'],
 ['Larger-source ensemble','Whisper-small only','Whisper-small/large-v2/Data2Vec weights0.5/0.25/0.25; clean-gradient norm calibration','No attack-model training;3source gradients/update; optional identity/FIR6k/resample8k draws'],
 ['BigVGAN log-mel','Feature-conditioned HiFi-GAN','Separately qualified frozen BigVGAN; mel LR0.01,relative radius0.1,log-mel box≤log2','No newly trained vocoder; representation/budget units differ from original FV'],
 ['Graph reuse / spectrum repair','Two decoder renders/update in the expanded implementation','Retain one decoder graph; then test a separate fixed25%log-magnitude blend','No learned model; new recordings still pay the FV50search'],
 ['Direct-trained feature policy','Individually optimized feature residual','5,070,464parameters; width384,eight temporal blocks,global context; fixed radii0.025/0.05/0.1','T10; no optimized attack-teacher targets; fresh encoder +one policy/HiFi-GAN forward at inference'],
 ['Feature-model quality arm','Source-only policy gradient','Activate normalized spectral/envelope violation correction above fixed proxy caps; project conflicting gradient component','Same warmup/data/RNG rules;4microbatches/update; optimizer AdamW2e-4,clip5; completed counts differ']]),'',
'Fixed quality-proxy caps for the feature policy: spectral **0.5987145876**, envelope **0.06727352284**, calibrated from all48original quality-control recordings against neutral reconstruction. These are proxy budgets, not MOS guarantees. Training transformations are differentiable approximations; they are not exact evaluation LPF/WaveGuard implementations.','',
'## 3. Measured per-recording generation cost','',
'Use timing scopes carefully. Historical rows below measure optimizer regions, excluding loading, feature preparation and scoring. The graph-reuse comparison used identical resident models and cached features in one session. Learned-model rows include input/output I/O and fresh WavLM encoding, but exclude one-time loading, hashing and shared diagnostic/reference work. These are timings of different-quality outputs; they are not an equivalent-success speed contest.','']
# Every expanded trajectory checkpoint with preserved per-recording elapsed timers.
costrows=[]
for phase in ['fvpilot','melpilot']:
 folder=P/'execution_scaled'/phase/'traces';summary=read(f'execution_scaled/{phase}/source-summary.json')['summary']
 for v in sorted(folder.iterdir(),key=lambda z:(z.name.rsplit('_',1)[0],int(z.name.rsplit('_',1)[1]))):
  rr=[json.loads(q.read_text()) for q in v.glob('*.json')];assert len(rr)==48;seconds=[r['wall_seconds_cumulative'] for r in rr];s=summary[v.name]
  costrows.append([v.name,'Cumulative search',f(statistics.mean(seconds)),f(sum(seconds)/360.0099375),f(sum(seconds)/3600),f(s['error_percent'],2),f(s['quality']['utmos'],3)])
for name,v in trade['loops'].items():costrows.append(['Matched FV50: '+('twice-render' if name=='twice' else 'graph reuse'),'Search, same session',f(v['latency_seconds']['mean']),f(v['seconds_per_audio_second']),f(v['wall_seconds']/3600),'83.90','3.198'])
for v in model['runtime']:costrows.append([v['variant'],'Fresh encoding +generation +I/O',f(v['latencySeconds']),f(v['secondsPerAudioSecond'],4),f(v['latencySeconds']*48/3600,5),f(v['wer'],2),f(v['utmos'],3)])
md+=[table(['Condition','Timing scope','Mean sec/clip','Sec/audio sec','48-clip hours','WER%','UTMOS'],costrows),'',
'Expanded checkpoints are cumulative snapshots of four underlying trajectories. **Do not add the50/200/500checkpoint times** as separate experiments. The independent original1500, raw-ensemble500, transformed-ensemble500 and mel500trajectories account for the expanded optimization workload. Learned per-condition48-clip totals are attributed inference sums; the actual eight-condition process shares encoding/diagnostics and has its own wall time below.','',
'**Exact-output implementation result:** graph reuse reduced measured FV50latency **19.649 →16.774seconds/recording**, a **14.63%** reduction (paired recording-bootstrap95%interval12.88–16.19%). Peak allocated memory increased **6.885 →6.928GiB**. All48raw WAVs matched the frozen outputs exactly. The fixed spectral repair then added **0.563seconds total**, but changed WER **83.9034 →72.8370%** and UTMOS **3.1981 →3.3604**. Neither output meets the joint requirement.','',
'## 4. Experiment-level generation, scoring and memory','',
'“Generation” includes whatever setup/diagnostics its runner timed; it is not necessarily the optimizer-loop sum above. Evaluation includes both recognition and CPU quality scoring. Memory uses binaryGiB; “—” means unavailable in this table, not zero. No new attack-model training occurred in the non-learned rows.','']
phase_rows=[]
for folder,label in [('execution48/mechanism-pilot48','Waveform mechanism48'),('execution_refine/pilot','Waveform refinement48'),('execution_envelope/pilot','Quiet-region48'),('execution_scaled/qualification','BigVGAN reconstruction qualification48'),('execution_scaled/fvpilot','Expanded FV,all checkpoints'),('execution_scaled/melpilot','BigVGAN mel,all checkpoints')]:
 gen=read(folder+'/generation-result.json');sc=read(folder+'/source-summary.json');res=sc['resources'];evaltime=res.get('elapsed_wall_seconds',res.get('wall_seconds'));phase_rows.append([label,f(gen['wall_seconds']/60,2),f(evaltime/60,2),f(gen['peak_allocated_bytes']/1024**3),f(res['peak_allocated_bytes']/1024**3),'Saved runner wall timers'])
phase_rows.append(['Graph/repair paired pilot',f(trade['costs']['measured_generation_process_gpu_allocation_wall_seconds']/60,2),f(trade['costs']['formal_evaluation']['elapsed_wall_seconds']/60,2),'6.928',f(trade['costs']['formal_evaluation']['peak_allocated_bytes']/1024**3),'42.40min whole paired cycle;two generation paths'])
mg=read('execution_model_v1/pilot/generation-result.json');ms=read('execution_model_v1/pilot/source-summary.json')['resources'];phase_rows.append(['Learned feature model,eight conditions',f(mg['wall_seconds']/60,2),f(ms['wall_seconds']/60,2),'2.113',f(ms['peak_allocated_bytes']/1024**3),'Training/preprocessing excluded here;listed next'])
md+=[table(['Experiment','Generation min','Source/MOS min','Peak gen GiB','Peak eval GiB','Scope'],phase_rows),'',
'The older R3 training region was **3541.89seconds**; its entire detached training/checkpoint-evaluation process was **3661.90seconds**. The initial development/R3 scheduled subtotal was **1.563hours**; with the first48-point mechanism generation/evaluation it was **2.128hours**. The early ledger explicitly omits some startup/failed-log timings, so it is not an exact grand total. '+link('Early compute ledger','execution_refine/compute-ledger.md')+'.','',
'The expanded queue finished normally in **45h51m**, under its then-declared48-hour ceiling. It performed **144,000per-recording optimization updates /288,000source forward-backward pairs**, not large-model training. Its feature and mel generation took **27.07/18.27hours**. Adding model diversity/steps did not resolve quality. The later13-hour instruction applied to the new learned-model queue; it does not describe the earlier expanded run.','',
'## 5. Latest learned model: actual phase accounting','']
phase=[]
for name,path,desc in [('Preprocessing','execution_model_v1/training_cache/receipt.json','2,807 clean feature/neutral-reference caches;not attack teachers'),('Common warmup','execution_model_v1/training/warmup/receipt.json','300updates /1,200microbatches'),('Source-only arm','execution_model_v1/training/source/receipt.json','3,791updates /15,164microbatches;update cap'),('Quality-corrected arm','execution_model_v1/training/quality/receipt.json','3,680updates /14,720microbatches;wall cap')]:
 a=read(path);stream=a.get('gpu_stream_elapsed_seconds',a.get('microbatch_gpu_stream_elapsed_seconds'));phase.append([name,f(a['wall_seconds'],2),f(stream,2),f(a['peak_allocated_bytes']/1024**3),f(a['peak_reserved_bytes']/1024**3),desc])
phase.extend([['Pilot generation,384outputs',f(mg['wall_seconds'],2),'Not separately totaled','2.113','3.096','5.26sloading;actualprocess shares encoding/diagnostics'],['Recognition inside pilot evaluation',f(ms['asr_wall_seconds'],2),f(ms['asr_gpu_stream_elapsed_seconds'],2),'—','—','Native beam5,temperature0'],['Quality inside pilot evaluation',f(ms['quality_wall_seconds'],2),f(ms['quality_gpu_stream_elapsed_seconds'],2),'—','—','CPU DNSMOS/NISQA/UTMOS;stream events include host gaps'],['Full pilot evaluation',f(ms['wall_seconds'],2),'Nested times above',f(ms['peak_allocated_bytes']/1024**3),f(ms['peak_reserved_bytes']/1024**3),'Includes4.48sloading and overhead;do not add nested rows again']])
md+=[table(['Phase','Wall seconds','CUDA-stream seconds','Allocated GiB','Reserved GiB','Coverage'],phase),'',
 table(['Overall accounting','Measured result'],[
 ['Optimized teacher-target generation','0seconds;this method trained directly through the source'],
 ['Preprocessing +all training arms',f((model['costs']['preprocessing']['wall_seconds']+sum(model['costs']['train_'+arm]['wall_seconds'] for arm in ['warmup','source','quality']))/3600)+'hours;includes comparison-arm cost'],
 ['Queue wait',f(state['queue_wait_seconds']/3600)+'hours;not active GPU allocation'],
 ['GPU-phase allocation wall',f(state['active_wall_seconds']/3600)+'hours;includes loading/I/O/CPU intervals while this queue owned its GPU phase'],
 ['Whole detached queue',f(state['overall_wall_seconds']/3600)+'hours including wait and delivery;13-hour maximum'],
 ['Execution/delivery','Training and scoring complete;selection exit1;report/build/browser delivery complete'],
 ['Exact kernel-active GPU time /energy','Not measured. Allocation wall and CUDA-stream elapsed intervals are not these quantities.']]),'',
'**Selection error:** the timing JSONL has384valid `(recording,variant)`rows but only48distinct recording IDs. A reader meant for unique-ID utterance manifests rejected it. We preserved the failed exit and independently checked all saved WER/MOS/timing rows in '+link('model_gate_review.json','research_summary/model_gate_review.json')+'. Zero candidates qualify, so fixing that reader alone would not change the scientific conclusion.','',
'**Comparison limit:** both arms start from the same warmup and use the same data/radius/augmentation schedule, but the quality arm completed111fewer updates. At radius0.1,quality correction raises UTMOS **2.7783 →3.6825**, while WER falls **8.3501 →4.1247%**. This is a measured tradeoff in two time-bounded runs, not a clean equal-update estimate or proof of the failure mechanism. The untrained and shuffled controls also have near-clean WER.','',
'## 6. Forecasts and efficiency targets','',
 table(['Forecast /target','Basis and value','What it does not establish'],[
 ['Graph-reuse target','Proposed≥10%lower matched loop time and<20GiBallocated;met','It was a research target,not a user latency limit;quality remained failed'],
 ['Learned-model target','Proposed≥10×generation reduction versus16.774sFV50,and<20GiBallocated;measured output-generation timings/memory meet it','The model fails attack WER;this is not a faster effective attack'],
 ['1,000new FV50+repair recordings','4.715hours using measured generation/preprocessing/repair/I/O assumptions','Conditional same-distribution forecast;includes per-recording search,excludes formal evaluation'],
 ['Teacher corpus,preparedT1/T10','Length-aware earlier forecast1.578/15.577hours of original-search target generation','Not performed for this new model;count-only/duration-only sensitivities differed'],
 ['1,000new learned quality_r050 outputs',f(model['forecast']['new1000_inference_hours'],4)+'inference hours;'+f(model['forecast']['lifetime1000_selected_hours'],3)+'hours including warmup+quality-arm training/preprocessing','Illustrative failed-model output cost;no model was selected;excludes full research/evaluation overhead'],
 ['10,000new learned quality_r050 outputs',f(model['forecast']['new10000_inference_hours'],4)+'inference hours;'+f(model['forecast']['lifetime10000_selected_hours'],3)+'hours including that training/preprocessing','Same hardware/audio-duration distribution;not completed larger-audio inference'],
 ['Arithmetic break-even',f(model['forecast']['selected_training_break_even_recordings'],1)+'recordings versus16.774sreference,using displayed quality model','Both reference/model fail the joint requirements in different ways;not a valid deployment crossover']]),'',
'The learned pipeline first measured **0.365seconds/training microbatch** on short/median/long inputs and forecast costs before admitting the long stages. The profile forecast preprocessing647.29s;actual728.94s. Warmup forecast438.27s;actual489.33s. Forecast two-arm training11076.63s before margins;actual14186.47s. Margins/time caps mattered;the quality arm stopped short of the common update limit. No convergence or success benefit follows from the larger-data arithmetic below.','',
 table(['Hypothetical training data','Preprocessing forecast,hours','One full training pass compute forecast,hours','Evidence status'],[[f(r['audio_hours'],0)+'audio hours',f(r['preprocessing_hours']),f(r['one_full_training_pass_compute_hours']),'Three-input engineering-profile extrapolation;50/100-hour corpora not run'] for r in profile['larger_data_forecasts']]),'',
'These profile-based larger-data forecasts exclude training-loading/optimizer overhead and assume the same duration distribution;actual repeated training demonstrates that the profile estimate was optimistic. Preserve that error rather than presenting the forecasts as measured corpus throughput. '+link('Profile/admission receipts','execution_model_v1/profile.json')+' · '+link('Previous prepared-data forecast','execution_tradeoff/prepared_data_forecast.json')+'.','',
'## 7. Transfer and defense results: historical versus new','',
'Historical evidence below belongs to the earlier **999-recording, rho0.5/lambda0.1/LR0.05/50-step** attack. Its source quality was **DNSMOS2.9638 /NISQA3.8711 /UTMOS2.7924**, failing current requirements. These saved results were carried into the research audit;they were not produced by the new methods and do not establish their transfer.','',
 table(['Historical target,N999','Clean WER%','Reconstruction WER%','Attack WER%'],[
 ['Whisper-tiny','7.6514','7.5234','67.8976'],['Whisper-base','5.5687','5.6573','68.7100'],['Whisper-medium','3.0182','3.0773','62.4274'],['Whisper-large-v2','4.1408','3.4318','65.5391'],['Wav2Vec2-large CTC','1.9301','1.9941','44.5446'],['HuBERT-large CTC','2.1664','2.2550','48.2078']]),'',
 link('Historical Whisper transfer',ROOT/'reproduction-runs/english-transfer-999/report.md')+' · '+link('Historical CTC transfer',ROOT/'reproduction-runs/english-ctc-transfer-999/report.md')+'.','',
 table(['Historical preprocessing,N999','Clean WER%','Reconstruction WER%','Attack WER%','Limit'],[
 ['Raw','3.963565','3.727228','74.106352','Source recognizer only'],['LPF6kHz','3.835549','4.076809','73.540128','Independent named low-pass implementation'],['WaveGuard down/up8kHz','4.081733','4.165436','71.733136','Resampling/PCM16;not full detector/AUC'],['WaveGuard down/up2kHz','32.373215','34.800591','101.600197','Strongly damages clean speech;high attack WER needs that context']]),'',
 link('Historical defense evidence',ROOT/'reproduction-runs/english-defenses-999/report.md')+'. Post-defense MOS was not measured in that historical table.','',
 table(['New-method endpoint','Actual status'],[
 ['Validation76','Prepared protocol;no improvement method advanced to it'],['New four-target transfer','Not run:Whisper-base/medium,Wav2Vec2-large,HuBERT-large were reserved for gated evaluation'],['New exact defenses','Not run:raw,LPF6k,WaveGuard8k,withheldLPF4k/WaveGuard12k'],['CE-AT,DT/DE-AT,DMW,TI','Matched defended weights/recipes unavailable'],['AudioPure/PVP sentence setting','Original comparable sentence-ASR configurations/artifacts not recovered'],['Human intelligibility /physical playback','Not measured'],['Novelty /publication','Literature informed experiments;publication planning deferred;no contribution or acceptance claim']]),'',
'## 8. Literature and other deliverables','',
 table(['Work','What it contributed','Saved source'],[
 ['Original-paper/source audit','Verified supplied paper identity,independent implementation contracts and missing author details',link('Original audit','01_original_method_and_baseline.md')],
 ['Earlier primary-literature review','Compared latent/diffusion,semantic/prosodic,transfer/psychoacoustic andcodec precedents;motivated R1/R2/R3',link('Literature review','03_literature_review.md')],
 ['Quality/runtime source audit','Duplicate decoding,NLL/WER divergence,prior R3 collapse andcosts motivated the paired test',link('Source audit','19_source_audit.md')],
 ['Consensus+author-code review','Separated conditioned FAPG/SpeechAdvGAN from universal UAPG/AudioShield;DDSP andcodec ideas remained alternatives',link('Model directions','23_model_directions_and_sources.md')],
 ['Publication-positioning note','An intermediate document;later user direction explicitly deferred publication planning',link('Historical note','22_publication_positioning.md')],
 ['Audio comparison','78conditions,3192score rows,48sentences,2082deduplicated original-WAV assets;listen/results/runtime/model views','http://localhost:8888/'],
 ['Interface verification','Latest model report/build andbrowser checks completed;existing server retained',link('Browser receipt','execution_model_v1/browser_receipt.json')],
 ['Reproducibility','Frozen plans,raw scores/WAVs,hashes,detached logs/heartbeats/exit receipts;no original experiment overwritten',link('Experiment handoff','HANDOFF.md')]]),'',
'Playback uses the saved16kHz float WAVs without loudness normalization,trimming orlossy conversion. Identical PCM files can share one served asset;3192score rows therefore do not imply3192distinct WAVs. The interface was checked as responding onport8888during this summary;no restart or UI changes were needed.','',
'## 9. How this consolidation was checked','',
 table(['Check','Result'],[
 ['Current development catalogue','78conditions;3192per-recording score rows;counts,unique IDs per condition,pooledS/D/I,reference totals,WER andmeanMOS reconciled'],['Historical starting-point evidence','Ten complete runs retained from the earlier independent6570-row audit;the consolidation reused that evidence rather than rerunning models'],['Latest timing table','384unique(recording,variant)pairs;48complete recordings for each variant'],['Latest gate arithmetic','No variant passes the fixed WER/MOS gate;all source results preserved despite selection error'],['Protected data','Only frozen metadata/receipts used;no confirmation/final audio oroutcomes opened'],['Cost scope','Nested timers/trajectory checkpoints explicitly identified;missing timing/memory isnot treated aszero'],['New work inthis summary','CPU evidence reconciliation andMarkdown writing only;no GPU run,model scoring orparameter change']]),'',
 link('Reviewed data snapshot','research_summary/reviewed_results.json')+' · '+link('Input hashes','research_summary/source_manifest.json')+' · '+link('Consolidation verification','research_summary/verification.json')+'.','',
'No exact all-history kernel-active GPU total isavailable:early timer coverage is incomplete andlater wall/stream timers measure different scopes. Per-phase measured costs above aremore defensible than a fabricated grand total.']
# Improve prose spacing without altering source filenames, hashes, IDs or Markdown links.
text='\n'.join(md)+'\n'
def clean_prose(text):
 parts=re.split(r'(\]\([^\n]*?\))',text)
 for i in range(0,len(parts),2):
  parts[i]=re.sub(r'(\d)(recordings|records|frames|words|conditions|parameters|updates|microbatches|seconds|minutes|hours|outputs|variants|source|total|step|search|audio|reference|raw|clean|saved|independent|point|log|speech|frame|probe|sloading|clipped|per-recording|GiB|NLL|FV|min\b|sFV|sreference|sloading|sclip|s\b)',r'\1 \2',parts[i])
  parts[i]=re.sub(r'([,;])(?=[A-Za-z])',r'\1 ',parts[i]);parts[i]=parts[i].replace('→per','→ per').replace(' →16',' → 16').replace('corpus throughput','corpus throughput')
 return ''.join(parts)
text=clean_prose(text)
(P/'28_data_runtime_and_evidence.md').write_text(text)
print(json.dumps({'data_pools':len(data),'generation_latency_conditions':len(costrows),'phase_cost_rows':len(phase_rows),'output':str(P/'28_data_runtime_and_evidence.md')},indent=2))
