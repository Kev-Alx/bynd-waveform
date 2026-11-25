import json, statistics, hashlib, importlib.metadata
from pathlib import Path
import torch, jiwer
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
from beyondwaveform.evaluation import error_counts, aggregate_errors, normalize_text
from beyondwaveform.io import digest

torch.set_num_threads(4)
out=Path('verification/strict-parity-audit/results.json')
run=Path('reproduction-runs/english-feature-test-999')
summary=json.loads((run/'surrogate-eval/summary.json').read_text())
rows=[json.loads(s) for s in (run/'surrogate-eval/utterances.jsonl').read_text().splitlines()]
assert digest(run/'surrogate-eval/utterances.jsonl') == summary['utterances_sha256']
normalizer=EnglishTextNormalizer()
results={'paper_sha256':digest('BeyondWaveform.pdf'),'baseline_commit':'d7f70e8a717db7feba35b69dacabade7de196b25','existing_tests':'59 passed', 'summary_sha256':digest(run/'surrogate-eval/summary.json'),'wer_checks':{}}
for variant in ['clean','reconstruction','adversarial']:
 pool=[r for r in rows if r['variant']==variant]
 counts=[error_counts(r['reference'],r['hypothesis'],'en') for r in pool]
 totals=aggregate_errors(counts)
 for k,v in totals.items(): assert abs(v-summary['summary'][variant][k])<1e-10
 refs=[normalizer(r['reference']) for r in pool]
 hyps=[normalizer(r['hypothesis']) for r in pool]
 alternative=jiwer.process_words(refs,hyps)
 results['wer_checks'][variant]={'count':len(pool),'repository':totals,'whisper_english_normalizer':{'error_percent':100*alternative.wer,'substitutions':alternative.substitutions,'deletions':alternative.deletions,'insertions':alternative.insertions,'reference_units':alternative.hits+alternative.substitutions+alternative.deletions},'macro_utterance_wer_percent':statistics.mean(100*(r['substitutions']+r['deletions']+r['insertions'])/r['reference_units'] for r in counts),'quality_mean_recomputed':{k:statistics.mean(r['quality'][k] for r in pool) for k in ['dnsmos','nisqa','utmos']}}
 for k,v in results['wer_checks'][variant]['quality_mean_recomputed'].items(): assert abs(v-summary['summary'][variant]['quality'][k])<1e-10
examples=[]
for ref,hyp in [("DON'T",'do not'),('TWENTY ONE','21'),('MISTER SMITH','Mr. Smith')]:
 examples.append({'reference':ref,'hypothesis':hyp,'repository_normalized':[normalize_text(ref,'en'),normalize_text(hyp,'en')],'repository_wer_percent':aggregate_errors([error_counts(ref,hyp)])['error_percent'],'whisper_normalized':[normalizer(ref),normalizer(hyp)],'whisper_wer_percent':100*jiwer.wer(normalizer(ref),normalizer(hyp))})
results['normalization_examples']=examples
tokenizer=get_tokenizer(True,language='en',task='transcribe')
refs=[r['reference'] for r in rows if r['variant']=='clean']
raw=[len(tokenizer.encode(t.strip())) for t in refs]
lower=[len(tokenizer.encode(t.strip().lower())) for t in refs]
results['teacher_forcing_token_counts']={'raw_mean':statistics.mean(raw),'raw_median':statistics.median(raw),'raw_min':min(raw),'raw_max':max(raw),'lowercase_mean':statistics.mean(lower),'lowercase_median':statistics.median(lower),'all_refs_uppercase':all(t==t.upper() for t in refs)}
results['packages']={name:importlib.metadata.version(name) for name in ['torch','torchaudio','openai-whisper','torchmetrics','librosa','onnxruntime','jiwer','numpy']}
generated=[json.loads(s) for s in (run/'generated.jsonl').read_text().splitlines()]
radii=[r['trace'][-1]['radius_after_update'] for r in generated]
results['recorded_final_radii']={'count':len(radii),'min':min(radii),'median':statistics.median(radii),'max':max(radii),'exceed_paper_radius_0_1':sum(v>0.1 for v in radii),'saved_delta_count':len(list(run.glob('*.delta.pt'))),'evidence':'recorded traces; not independently recomputed from delta tensors'}
results['short_audio_count_below_dnsmos_9_01_seconds']={v:sum(r['audio'][v]['samples']<9.01*16000 for r in generated) for v in ['clean','adversarial']}
results['audit_date']='2026-10-15 Asia/Taipei'
results['scope']='Static source audit, 59 existing tests, saved-score recomputation, one real pretrained pilot gradient probe, two quality fixtures. No full ASR rerun, new attack sweep, or model changes.'
results['source_sha256']={str(f):digest(f) for f in sorted(Path('beyondwaveform').rglob('*.py'))}
out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
