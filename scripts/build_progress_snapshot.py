"""Create the reviewed snapshot for the HTML reproduction progress report.

Reads existing results only; does not generate audio, score models, or tune attacks.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import hashlib
import json
import math
import sys

ROOT = Path('/workspace')
RUNS = ROOT / 'reproduction-runs'
DEST = RUNS / 'progress-report-reviewed.json'
receipts = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = ROOT / relative
    data = path.read_bytes()
    receipts.append({'file': relative, 'sha256': hashlib.sha256(data).hexdigest()})
    return json.loads(data)


def audit_eval(directory):
    rel = directory.relative_to(ROOT).as_posix()
    summary = read(rel + '/summary.json')
    path = directory / 'utterances.jsonl'
    assert digest(path) == summary['utterances_sha256'], rel
    grouped = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row['variant']].append(row)
    for variant, expected in summary['summary'].items():
        rows = grouped[variant]
        assert len(rows) == expected['count'] == 999
        assert len({r['id'] for r in rows}) == 999
        for key in ('substitutions', 'deletions', 'insertions', 'reference_units'):
            assert sum(r[key] for r in rows) == expected[key]
        assert expected['reference_units'] == 20310
        wer = 100 * sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in rows) / expected['reference_units']
        assert math.isclose(wer, expected['error_percent'], abs_tol=1e-10)
        for key, value in expected.get('quality', {}).items():
            assert math.isclose(sum(r['quality'][key] for r in rows) / len(rows), value, abs_tol=1e-10)
    receipts.append({'file': rel + '/utterances.jsonl', 'sha256': summary['utterances_sha256']})
    return summary


def source(label, files, definition, ids, caveats=None):
    return {'label': label, 'provider': 'Local frozen research artifacts', 'files': files,
            'filters': ['English LibriSpeech test-clean: 999 fixed utterances; 20,310 reference words per variant'] if 'validation' not in label.lower() else ['English dev-clean: 100 fixed validation utterances'],
            'metricDefinitions': [{'label': 'Metric scope', 'definition': definition, 'componentIds': ids}],
            'caveats': caveats or [],
            'evidenceFlow': [{'title': 'Read saved evidence', 'detail': 'Existing JSON summaries, frozen plans, verification receipts and supplied paper were read. Models and attacks were not rerun for this report.'},
                             {'title': 'Reconcile aggregates', 'detail': 'For completed test evaluations, saved JSONL digests, unique IDs, counts, edit totals, corpus WER and mean MOS were recomputed and checked against summaries.'}]}


now = datetime.now(timezone(timedelta(hours=8)))
core = audit_eval(RUNS / 'english-feature-test-999/surrogate-eval')
verify = read('reproduction-runs/english-feature-test-999/verification.json')
assert verify['status'] == 'passed'
assert digest(RUNS / 'english-feature-test-999/surrogate-eval/summary.json') == verify['summary_sha256']
comparison = read('reproduction-runs/english-feature-test-999/paper-comparison.json')
queries = {}
controls = []
for variant, label in [('clean', 'Original clean'), ('reconstruction', 'Vocoder reconstruction'), ('adversarial', 'Feature attack')]:
    s = core['summary'][variant]
    controls.append({'variant': label, 'count': s['count'], 'words': s['reference_units'], 'wer': s['error_percent'], **s['quality']})
queries['core'] = {'rows': controls, 'source': source('Verified English core test', ['reproduction-runs/english-feature-test-999/surrogate-eval/summary.json', 'reproduction-runs/english-feature-test-999/verification.json'], 'WER (%) = 100 × summed substitutions, deletions and insertions / summed reference words. MOS values are arithmetic utterance means; higher WER is stronger attack, higher MOS is better quality.', ['summary', 'controls', 'core-chart', 'methods'])}

paper_rows = []
for metric, p in [('WER (%)', comparison['comparisons']['adversarial']['wer']), *[(k.upper(), v) for k,v in comparison['comparisons']['adversarial']['quality'].items()]]:
    paper_rows.append({'metric': metric, 'observed': p['observed'], 'paper': p['paper'], 'gap': p.get('gap_percentage_points', p.get('gap_mos')), 'unit': 'pp' if metric.startswith('WER') else 'MOS'})
queries['paper'] = {'rows': paper_rows, 'source': source('Paper Table 1 and Table 4 comparison', ['BeyondWaveform.pdf', 'paper/BeyondWaveform.md', 'reproduction-runs/english-feature-test-999/paper-comparison.json'], 'Observed minus paper; WER gaps are percentage points and MOS gaps are score units. Point differences across independent protocols do not establish equivalence.', ['summary', 'paper-comparison', 'quality-analysis', 'methods'], ['The selected feature configuration is rho=0.5/lambda=0.1; the paper reports rho=0.1/lambda=1.0. Author manifests, encoder/vocoder revisions, decoder and scorer recipes are unavailable.'])}

transfer = []
for suite in ['english-transfer-999', 'english-ctc-transfer-999']:
    results = read(f'reproduction-runs/{suite}/results.json')
    receipt = read(f'reproduction-runs/{suite}/verification.json')
    assert results['status'] == 'complete' and receipt['status'] == 'passed'
    assert digest(RUNS / suite / 'results.json') == receipt['results_sha256']
    for target in results['targets']:
        model = target['model']
        evaluated = audit_eval(RUNS / suite / model)
        assert evaluated['summary'] == target['summary']
        assert digest(RUNS / suite / model / 'summary.json') == target['summary_sha256']
        anchor = {'large-v2': 52.96, 'wav2vec2': 42.12, 'hubert': 39.8}.get(model)
        label = {'tiny':'Whisper tiny', 'base':'Whisper base', 'medium':'Whisper medium', 'large-v2':'Whisper large-v2', 'wav2vec2':'Wav2Vec2', 'hubert':'HuBERT'}[model]
        s = target['summary']
        transfer.append({'model':label, 'family':'Whisper' if model in ['tiny','base','medium','large-v2'] else 'CTC', 'clean':s['clean']['error_percent'], 'reconstruction':s['reconstruction']['error_percent'], 'attack':s['adversarial']['error_percent'], 'paper':anchor, 'gap':None if anchor is None else s['adversarial']['error_percent']-anchor})
queries['transfer'] = {'rows': transfer, 'source': source('Verified raw transfer tests', ['reproduction-runs/english-transfer-999/results.json', 'reproduction-runs/english-transfer-999/verification.json', 'reproduction-runs/english-ctc-transfer-999/results.json', 'reproduction-runs/english-ctc-transfer-999/verification.json', 'paper/BeyondWaveform.md'], 'All targets reuse the same 999 feature-attack WAVs and matched clean/reconstruction controls. Raw corpus WER (%); no target feedback or retuning. Paper anchors are exact prose values for large, Wav2Vec2 and HuBERT; other cells are unavailable, not zero.', ['summary', 'transfer-analysis', 'transfer-chart', 'transfer-table', 'methods'], ['large-v2 is an independent revision; CTC uses pinned independent checkpoints and greedy decoding without a language model. Defenses are not evaluated.'])}

ab = read('reproduction-runs/english-ablations-999/results.json')
ab_verify = read('reproduction-runs/english-ablations-999/verification.json')
bootstrap = read('reproduction-runs/english-ablations-999/paired-bootstrap.json')
assert ab['status'] == 'complete' and ab_verify['status'] == 'passed'
assert digest(RUNS / 'english-ablations-999/results.json') == ab_verify['results_sha256'] == bootstrap['results_sha256']
labels = {'full':'Full loss', 'without-attack':'Without attack', 'without-perceptual':'Without perceptual', 'without-tv':'Without temporal penalty', 'without-hf':'Without frequency penalty'}
ablations = []
for e in ab['experiments']:
    if e['name'] != 'full':
        measured = audit_eval(Path(e['generation_run']) / 'surrogate-eval')
        assert measured['summary'] == e['summary']
    s=e['summary']['adversarial']; b=bootstrap['conditions'][e['name']]['wer']
    ablations.append({'condition':labels[e['name']], 'wer':s['error_percent'], **s['quality'], 'gap':b['difference_from_full'], 'low':b['paired_difference_ci95'][0], 'high':b['paired_difference_ci95'][1]})
queries['ablations'] = {'rows': ablations, 'source': source('Verified loss removal test', ['reproduction-runs/english-ablations-999/results.json', 'reproduction-runs/english-ablations-999/verification.json', 'reproduction-runs/english-ablations-999/paired-bootstrap.json', 'paper/BeyondWaveform.md'], 'One prescribed loss removal at a time, same fixed 999 test IDs/configuration/controls. WER gaps versus full loss; percentile 95% intervals from 5,000 paired utterance resamples, seed 2026, with corpus word denominators recomputed.', ['summary', 'ablation-analysis', 'ablation-table', 'ablation-intervals', 'next-steps', 'methods'], [bootstrap['note'], 'Not an exact Figure 8 recreation; paper full-loss DNSMOS conflicts with Table 4. Do not choose a new configuration using these test results.'])}

wave = read('reproduction-runs/english-waveform-validation/results.json')
progress = read('reproduction-runs/english-waveform-validation/progress.json')
job = read('reproduction-runs/english-waveform-validation/job.json')
method_counts = Counter(c['method'] for c in wave['candidates'])
wave_rows=[]
for method,total in [('pgd',15),('mi-fgsm',30),('vmi-fgsm',90)]:
    selected=RUNS / 'english-waveform-validation/selected' / (method+'.selection.json')
    row={'method':method.upper(), 'completed':method_counts[method], 'planned':total, 'status':'Selected on validation' if selected.exists() else 'Search running', 'wer':None, 'dnsmos':None, 'nisqa':None, 'utmos':None}
    if selected.exists():
        sel=read(selected.relative_to(ROOT).as_posix())
        eligible=[c for c in sel['candidates'] if c['eligible']]
        s=sel['selected']
        assert s['error_percent'] == max(c['error_percent'] for c in eligible)
        assert all(v>2.5 for v in s['quality'].values())
        row.update({'wer':s['error_percent'], **s['quality']})
    wave_rows.append(row)
alive=all(Path('/proc',str(job[k])).exists() for k in ['supervisor_pid','child_pid'])
queries['waveform']={'rows':wave_rows, 'source':source('Ongoing waveform validation snapshot', ['reproduction-runs/english-waveform-validation/results.json', 'reproduction-runs/english-waveform-validation/progress.json', 'reproduction-runs/english-waveform-validation/job.json'], '100 validation clips per setting; PGD 15, MI-FGSM 30, VMI-FGSM 90. Selected WER and MOS are validation observations only, and must not be compared directly with paper test anchors. Cumulative counts reflect this saved snapshot.', ['completion', 'waveform-analysis', 'waveform-table', 'next-steps'], ['The search continues independently. This HTML is a static snapshot and will not refresh automatically. Final 999-clip baseline tests have not been performed.'])}

selected=read('reproduction-runs/validation-grid/selection.json')['selected']
default=read('reproduction-runs/validation-grid/rho-0.1_lambda-1.0/evaluation/summary.json')['summary']['adversarial']
queries['selection']={'rows':[{'setting':'Selected rho=0.5, lambda=0.1', 'wer':selected['error_percent'], **selected['quality']}, {'setting':'Paper rho=0.1, lambda=1.0 (independent implementation)', 'wer':default['error_percent'], **default['quality']}], 'source':source('Feature validation selection', ['reproduction-runs/validation-grid/selection.json','reproduction-runs/validation-grid/verification.json','reproduction-runs/validation-grid/rho-0.1_lambda-1.0/evaluation/summary.json','configs/english_selected.freeze.json'], '16 configurations on 100 dev-clean validation clips. Strict mean MOS >2.5 for each scorer, then maximum corpus WER. The winner is frozen before testing; paper-default validation is not a paper test comparison.', ['completion','selection-table','methods'])}

status_rows=[
 {'part':'Core implementation and real model/scorer loading','status':'Complete','evidence':'WavLM-Large → feature perturbation → HiFi-GAN; Whisper-small objective; DNSMOS/NISQA/UTMOS; audited frozen artifacts.'},
 {'part':'Data preparation and split audit','status':'Complete','evidence':'12 disjoint pilot clips, 100 validation clips, 999 English test clips; both corpora downloaded. Mandarin preparation remains.'},
 {'part':'Optimizer pilot and feature validation','status':'Complete','evidence':'3 learning rates on pilot; 16 rho/lambda settings; selected configuration frozen before test.'},
 {'part':'English feature attack and controls','status':'Complete · verified','evidence':'999 clips × clean/reconstruction/attack; all quality metrics; 49,950 attack updates checked.'},
 {'part':'Whisper transfer','status':'Complete · verified','evidence':'tiny/base/medium/large-v2; 11,988 score rows with matched controls.'},
 {'part':'CTC transfer','status':'Complete · verified','evidence':'Wav2Vec2/HuBERT; 5,994 score rows; pinned artifacts and greedy decoding.'},
 {'part':'Loss removal ablations','status':'Complete · verified','evidence':'5 conditions, 14,985 score rows; paired bootstrap intervals; four new generation runs.'},
 {'part':'Waveform baseline validation','status':'Running','evidence':f"{len(wave['candidates'])}/135 saved completed settings; PGD/MI-FGSM selections frozen; VMI-FGSM in progress."},
 {'part':'Final waveform baseline test and transfer','status':'Pending','evidence':'No held-out 999-clip baseline measurements yet.'},
 {'part':'Mandarin reproduction','status':'Pending','evidence':'AISHELL-1 downloaded; speaker-disjoint preparation, surrogate fine-tuning and held-out CER still needed.'},
 {'part':'Defenses and remaining published baselines','status':'Not measured','evidence':'Author recipes/artifacts unresolved for defense suite, Muting Whisper and SlothSpeech; LPF example is not paper replication.'},
 {'part':'Human listening, acoustics and physical OTA','status':'Not measured','evidence':'Human study and new playback recordings required; paper acoustic sample/curves have not been numerically recreated.'},
]
queries['completion']={'rows':status_rows, 'source':{'label':'Project implementation, handoff and saved status receipts','files':['README.md','docs/REPRODUCTION.md','docs/HANDOFF.md','verification/architecture.json'], 'evidenceFlow':[{'title':'Inspect project state','detail':'Read implementation/handoff and original completion/verification receipts. Current waveform parent processes were present. No new experiments launched.'}]}}

snapshot={'title':'English attack strength is close; the full paper is not yet reproduced', 'surface':'report', 'generatedAt':now.isoformat(), 'status':'reviewed', 'report':{'asOf':now.date().isoformat()}, 'filters':[], 'queries':queries,
 'reportMeta':{'preparedAt':now.strftime('%Y-%m-%d %H:%M Asia/Taipei'), 'waveUpdatedAt':progress['updated_at'], 'waveCompleted':len(wave['candidates']), 'waveExpected':135, 'waveMethod':progress['method'], 'wavePhase':progress['phase'], 'waveSettingIndex':progress['setting_index'], 'waveProcessAlive':alive, 'meanQualityGate':verify['quality_diagnostics']['adversarial']['mean_gate_passed'], 'clipQualityPass':747, 'clipQualityTotal':999, 'dnsmosFail':58,'nisqaFail':15,'utmosFail':209,'fullWerCi':bootstrap['conditions']['full']['wer']['ci95'], 'generationSeconds':verify['attack_diagnostics']['generation_seconds']['mean']},
 'sourceReceipts':receipts}
DEST.write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps({'snapshot':str(DEST),'queries':len(queries),'completed_test_evaluations_reconciled':11,'wave_completed':len(wave['candidates']),'processes_present':alive,'prepared_at':now.isoformat()}))
