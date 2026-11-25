"""Build a reviewed, sample-preserving listening catalogue from saved receipts.

No models run here. Each published WAV retains the original bytes. Identical
decoded samples may share a WAV; the receipt records both file hashes.
"""
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/workspace/improvement')
APP = ROOT / 'audio_browser'
ASSETS = APP / 'src/content/assets/audio'
STATUS = ROOT / 'audio_browser_data_status.json'

STUDIES = {
    'execution/r1-deletion-development24': ('r1', 'Word-risk objective'),
    'execution/r2-canonical-zero-uniform-dev24': ('r2', 'Local feature mixtures'),
    'execution/r3-waveform-autograd-repair': ('r3', 'Learned waveform generator'),
    'execution48/mechanism-pilot48': ('mechanism', 'Waveform mechanism pilot'),
    'execution_refine/pilot': ('refinement', 'Waveform refinement'),
    'execution_envelope/pilot': ('envelope', 'Quiet-region protection'),
    'execution_scaled/qualification': ('decoder', 'Decoder qualification'),
    'execution_scaled/fvpilot': ('scaled-fv', 'Larger models and longer feature runs'),
    'execution_scaled/melpilot': ('scaled-mel', 'BigVGAN mel-space pilot'),
}


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def status(**values):
    p = STATUS.with_suffix('.tmp')
    p.write_text(json.dumps({'updated_at_utc': now(), **values}, indent=2))
    p.replace(STATUS)


def describe(study, variant):
    common = {
        'clean': ('Clean speech', 'The original speech, without an optimized perturbation.'),
        'reconstruction': ('HiFi-GAN reconstruction', 'Speech encoded and decoded by the original frozen WavLM / HiFi-GAN pipeline, with no attack.'),
        'fv_reconstruction': ('HiFi-GAN reconstruction', 'The original feature decoder with no optimized edit.'),
        'bigvgan_reconstruction': ('BigVGAN reconstruction', 'The larger 112.4M-parameter BigVGAN decoder, without an attack. This checks the decoder itself.'),
        'fv_quality': ('Original feature method · 50 steps · LR 0.02', 'The quality-admissible original feature-space control; Whisper-small supplies the gradient.'),
        'fv_quality50': ('Original feature method · 50 steps · LR 0.02', 'The quality-admissible original feature-space control; Whisper-small supplies the gradient.'),
        'fv50': ('Original feature method · 50 steps · LR 0.02', 'The quality-admissible original feature-space control; Whisper-small supplies the gradient.'),
        'fv100': ('Original feature method · 100 steps', 'The original quality-control optimizer continued for twice as many updates.'),
        'pgd_full': ('Waveform PGD · weighted loss · 50 steps', 'Optimize a small additive waveform edit, balancing recognition loss with spectral and speech-feature penalties.'),
        'full50': ('Waveform PGD · weighted loss · 50 steps', 'The 50-step waveform initializer, with spectral and speech-feature penalties.'),
        'global_full50': ('Waveform PGD · weighted loss · 50 steps', 'The global waveform-bound weighted-loss control, reused from the mechanism pilot.'),
        'pgd_nll': ('Waveform PGD · recognition loss only', 'Remove the quality penalties to isolate how much recognition strength they cost.'),
        'global_nll50': ('Waveform PGD · recognition loss only', 'The global waveform-bound recognition-only control, reused from the mechanism pilot.'),
        'r3_initial': ('Generator · untrained', 'The untrained waveform U-Net control. A shared model predicts an edit for each input.'),
        'initial-generator': ('Generator · untrained', 'The untrained waveform U-Net control, seed 11.'),
        'r3_trained': ('Generator · trained', 'The 1.59M-parameter waveform U-Net after one seed and about one hour of training. It learned an almost-zero edit.'),
        'weighted100': ('Waveform PGD · weighted continuation · 100 steps', 'Continue the weighted waveform objective for another 50 steps, for 100 total.'),
        'constrained100': ('Waveform PGD · hard constraints · 100 steps', 'Continue from weighted PGD50 while refusing updates that exceed the initial spectral or content distortion.'),
        'envelope_nll50': ('Quiet-region budget · recognition loss only', 'Allow less perturbation in quieter 25 ms speech windows; 50 waveform updates.'),
        'envelope_full50': ('Quiet-region budget · weighted loss', 'The same quiet-region budget with spectral and content penalties restored.'),
        'attenuated_global_nll50': ('Uniformly attenuated waveform control', 'Scale a global recognition-only attack to match the quiet-region candidate\'s residual RMS.'),
        'baseline_frozen': ('Original strong control · 50 steps · LR 0.1', 'The historical stronger original feature setting; it failed the UTMOS floor.'),
        'baseline_attenuated': ('Local-mixture amplitude control', 'The saved amplitude-control condition for the local feature-mixture experiment.'),
    }
    if variant in common:
        return common[variant]
    if variant.startswith('single_'):
        n = variant.split('_')[-1]
        return (f'Original feature method · {n} steps · LR 0.1', 'A stronger, longer original feature-space attack against Whisper-small. These are known-method controls, not a new architecture.')
    if variant.startswith('ensemble_raw_'):
        n = variant.split('_')[-1]
        return (f'Three-model feature ensemble · {n} steps', 'Combine Whisper-small (50%), Whisper-large-v2 (25%), and Data2Vec CTC (25%) gradients; keep the original feature decoder and penalties.')
    if variant.startswith('ensemble_eot_'):
        n = variant.split('_')[-1]
        return (f'Three-model feature ensemble + transforms · {n} steps', 'Optimize against the three source models while drawing identity, low-pass filtering, or resampling during optimization. Transfer and defense effectiveness remain untested.')
    if variant.startswith('mel_eot_'):
        n = variant.split('_')[-1]
        return (f'BigVGAN mel ensemble + transforms · {n} steps', 'Optimize bounded mel-spectrogram edits through BigVGAN using all three source models and sampled audio transformations.')
    if variant == 'adversarial':
        if study.startswith('r1'):
            return ('Word-risk with deletion candidate', 'Replace token loss with risk over candidate transcripts; add one deletion candidate because the native candidates often contain no word errors.')
        if study.startswith('r2'):
            return ('Local feature mixtures', 'Restrict each edited speech frame to mixtures of similar clean frames. The restricted search barely attacked the recognizer.')
        if study.startswith('r3'):
            return ('Generator · trained' + (' · ' + study.split('-')[-1] + ' updates' if 'checkpoint' in study else ''), 'A shared 1.59M-parameter waveform U-Net, trained on one hour of disjoint speech; this saved checkpoint emitted almost no perturbation.')
        if study.startswith('original-'):
            lr = study.split('-')[-1]
            return (f'Original feature method · 50 steps · LR {lr}', 'Historical original-implementation control, scored on the 24-recording development cohort.')
    return (variant.replace('_', ' '), 'Saved experimental condition; see the linked source receipt for its exact configuration.')


def main():
    started = time.monotonic()
    status(status='running', phase='collecting saved scores', completed=0)
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs, clips, scores, assets, provenance = [], {}, [], {}, []
    sources = []
    for p in sorted(ROOT.rglob('source-summary.json')):
        if APP in p.parents:
            continue
        rel = str(p.parent.relative_to(ROOT))
        if rel in STUDIES:
            study, label = STUDIES[rel]
        elif '/checkpoints/update-' in rel:
            n = rel.split('update-')[1].split('-')[0]
            study, label = f'r3-checkpoint-{int(n)}', f'Generator checkpoint · {int(n)} updates'
        else:
            raise ValueError(f'Unclassified source summary: {p}')
        generated = p.parent / 'generated.jsonl'
        if not generated.exists():
            generated = p.parent.parent / 'generated.jsonl'
        sources.append((p, generated, p.parent / 'source-scores.jsonl', study, label))
    legacy = Path('/workspace/reproduction-runs/paper-quality-stable-execution/pilot')
    for dirname, lr in [('quality-sum-lr-0.02', '0.02'), ('timestamp-sum-lr-0.05', '0.05'), ('strong-sum-lr-0.1', '0.1')]:
        p = legacy / dirname
        sources.append((p / 'evaluation/summary.json', p / 'generated.jsonl', p / 'evaluation/utterances.jsonl', f'original-{lr}', f'Historical original control · LR {lr}'))

    for si, (summary_path, generated_path, score_path, study, label) in enumerate(sources):
        summary = read(summary_path)['summary']
        generated = {r['id']: r for r in lines(generated_path)}
        saved_scores = lines(score_path)
        provenance.append({'study': study, 'summary_file': str(summary_path), 'summary_sha256': sha(summary_path), 'generated_file': str(generated_path), 'generated_sha256': sha(generated_path), 'score_file': str(score_path), 'score_sha256': sha(score_path)})
        for variant, s in summary.items():
            run_id = study + '--' + variant
            name, description = describe(study, variant)
            q = s['quality']
            failures = [k.upper() for k, floor in [('dnsmos', 3), ('nisqa', 3.75), ('utmos', 3.5)] if q[k] < floor]
            run = {'id': run_id, 'study': label, 'studyId': study, 'variant': variant, 'name': name, 'description': description,
                   'count': s['count'], 'referenceWords': s['reference_units'], 'wer': s['error_percent'],
                   **q, 'qualityPass': not failures, 'goalPass': not failures and s['error_percent'] > 75,
                   'failedMetrics': ', '.join(failures), 'substitutions': s['substitutions'], 'deletions': s['deletions'], 'insertions': s['insertions'],
                   'sourceFile': str(summary_path.relative_to(Path('/workspace')))}
            matches = [v for v in saved_scores if v['variant'] == variant]
            assert len(matches) == s['count'], (run_id, len(matches), s['count'])
            for key in ['substitutions', 'deletions', 'insertions', 'reference_units']:
                assert sum(v[key] for v in matches) == s[key], (run_id, key)
            assert math.isclose(100 * sum(s[k] for k in ['substitutions', 'deletions', 'insertions']) / s['reference_units'], s['error_percent'], abs_tol=1e-10)
            for key in ['dnsmos', 'nisqa', 'utmos']:
                assert math.isclose(sum(v['quality'][key] for v in matches) / len(matches), q[key], abs_tol=1e-7), (run_id, key)
            for record in matches:
                g = generated[record['id']]
                audio = g['audio'][variant]
                source = Path(audio['path'])
                source_hash = sha(source)
                assert source_hash == audio['sha256'] == record['audio_sha256'], (run_id, record['id'], 'file hash')
                pcm, sr = sf.read(source, dtype='float32', always_2d=True)
                assert np.isfinite(pcm).all()
                pcm_key = hashlib.sha256(str(sr).encode() + str(pcm.shape).encode() + pcm.tobytes()).hexdigest()
                if pcm_key not in assets:
                    filename = pcm_key + '.wav'
                    dest = ASSETS / filename
                    if not dest.exists():
                        shutil.copyfile(source, dest)
                    assert sha(dest) == source_hash
                    bins = [float(np.max(np.abs(v))) for v in np.array_split(pcm[:, 0], 96) if len(v)]
                    assets[pcm_key] = {'file': 'audio/' + filename, 'duration': len(pcm) / sr, 'waveform': bins,
                                       'samples': len(pcm), 'sampleRate': sr, 'sha256': source_hash}
                a = assets[pcm_key]
                uid = record['id']
                clip = {'id': uid, 'reference': record['reference'], 'speaker': str(g.get('speaker', uid.split('-')[0])), 'duration': len(pcm) / sr}
                if uid not in clips or variant == 'clean':
                    clips[uid] = clip
                else:
                    assert clips[uid]['reference'] == record['reference']
                scores.append({'runId': run_id, 'clipId': uid, 'reference': record['reference'], 'hypothesis': record['hypothesis'],
                               'wer': 100 * sum(record[k] for k in ['substitutions', 'deletions', 'insertions']) / record['reference_units'],
                               **record['quality'], 'substitutions': record['substitutions'], 'deletions': record['deletions'], 'insertions': record['insertions'],
                               'referenceWords': record['reference_units'], 'audio': a['file'], 'duration': a['duration'], 'waveform': a['waveform'],
                               'originalSha256': source_hash, 'servedSha256': a['sha256'], 'pcmSha256': pcm_key})
            runs.append(run)
        status(status='running', phase='verifying and collecting audio', completed=si + 1, total=len(sources), runs=len(runs), recordings=len(scores), unique_audio=len(assets))
        print(f'Collected {label}: {len(runs)} conditions, {len(scores)} scored recordings, {len(assets)} WAV assets', flush=True)

    snapshot = read(APP / 'src/data.json')
    snapshot.update(title='Speech experiment audio comparison', generatedAt=now(), buildStatus='creating', status='observed', filters=[])
    methods = [{'language': 'python', 'code': 'build_audio_browser_data.py: read saved summary, generation and per-recording score receipts; verify SHA-256, exact S/D/I/reference totals, word-weighted corpus WER, and arithmetic mean quality. Copy original WAV bytes; deduplicate only identical float32 samples at the same sample rate and shape.'}]
    source = {'label': 'Completed development experiment receipts', 'files': sorted({r['sourceFile'] for r in runs}),
              'filters': ['Development recordings only; 24-point and 48-point cohorts remain separate.', 'Repeated controls are retained per study and are not independent replications.'],
              'evidenceFlow': [{'title': 'Read saved scoring receipts', 'detail': 'Each run is joined by recording ID and condition to generated.jsonl and source-scores.jsonl (legacy controls use evaluation/utterances.jsonl).'},
                               {'title': 'Reconcile metrics and audio', 'detail': 'Check counts, S/D/I, denominator, mean quality, source-audio SHA-256 and sample identity. No ASR or quality model is run again.'}],
              'metricDefinitions': [{'label': 'WER (%)', 'definition': '100 × (substitutions + deletions + insertions) / reference words, pooled over the whole condition. Can exceed 100% because insertions count. Higher means more source-ASR errors, not proven human intelligibility.'},
                                    {'label': 'Quality floors', 'definition': 'The condition mean must satisfy DNSMOS OVRL ≥ 3, NISQA MOS ≥ 3.75, and UTMOS ≥ 3.5. These are automatic proxies, not human ratings.'},
                                    {'label': 'Joint goal', 'definition': 'Corpus WER strictly above 75% and all three mean-quality floors. Per-recording scores do not determine this cohort gate.'}],
              'caveats': ['Whisper-small is the common source scorer. No held-out transfer, defense, confirmation48, final200, or human evaluation has been completed in these research cycles.']}
    snapshot['queries'] = {'runs': {'rows': runs, 'source': source, 'methods': methods},
                           'recordings': {'rows': scores, 'source': {**source, 'label': 'Per-recording transcripts, quality and original audio'}, 'methods': methods},
                           'clips': {'rows': sorted(clips.values(), key=lambda r: r['id']), 'source': {**source, 'label': 'Development sentence catalogue'}, 'methods': methods}}
    (APP / 'src/data.json').write_text(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')))
    fields = ['study', 'name', 'variant', 'count', 'referenceWords', 'wer', 'dnsmos', 'nisqa', 'utmos', 'qualityPass', 'goalPass', 'failedMetrics', 'substitutions', 'deletions', 'insertions', 'sourceFile']
    for p in [ROOT / 'all_results.csv', APP / 'src/content/assets/all_results.csv']:
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');w.writeheader();w.writerows(runs)
    receipt = {'status': 'complete', 'generated_at_utc': now(), 'conditions': len(runs), 'scored_recordings': len(scores), 'unique_recordings': len(clips),
               'unique_audio_assets': len(assets), 'audio_bytes': sum(p.stat().st_size for p in ASSETS.glob('*.wav')), 'seconds': time.monotonic() - started,
               'metric_validation': 'All condition counts, pooled S/D/I/reference words and arithmetic mean quality reconciled.',
               'audio_policy': 'Original WAV bytes, with no normalization, resampling, trimming or lossy encoding. PCM-identical files share an asset.',
               'snapshot_sha256': sha(APP / 'src/data.json'), 'provenance': provenance}
    (ROOT / 'audio_browser_data_receipt.json').write_text(json.dumps(receipt, indent=2))
    # Human-readable complete tables keep populations and studies explicit.
    md = ['# Complete recognition and audio-quality results', '', 'Every row below comes from saved per-recording scores, reconciled against the saved summary. WER is pooled by reference words; quality scores are arithmetic recording means. Higher WER means more machine errors. Higher quality is better. WER can exceed 100% due to insertions.', '', 'Quality floors: DNSMOS ≥ 3, NISQA ≥ 3.75, UTMOS ≥ 3.5. The joint research goal additionally requires WER > 75%. No condition met that joint goal. Repeated controls are retained under their actual study; they are not extra independent observations.', '']
    for n in [48, 24]:
        md += [f'## {n}-recording development cohort', '']
        for study in dict.fromkeys(r['study'] for r in runs if r['count'] == n):
            md += [f'### {study}', '', '| Condition | N | WER % | DNSMOS | NISQA | UTMOS | Quality floors |', '|---|---:|---:|---:|---:|---:|---|']
            for r in runs:
                if r['study'] == study and r['count'] == n:
                    result = 'pass' if r['qualityPass'] else 'fail: ' + r['failedMetrics']
                    md.append(f"| {r['name']} | {n} | {r['wer']:.4f} | {r['dnsmos']:.4f} | {r['nisqa']:.4f} | {r['utmos']:.4f} | {result} |")
            md += ['']
    md += ['## Coverage and unavailable results', '', 'Includes every completed source-summary.json under improvement plus the three historical 24-point original controls (LR 0.02, 0.05 and 0.1), including saved generator checkpoints. This is the complete development research catalogue, not an inventory of unrelated legacy reproduction runs. Canonical word-risk failures and generation-only smoke probes have no complete WER/MOS result and are not assigned synthetic scores.', '', 'The 76-recording validation, unseen-model transfer, defense comparisons, untouched confirmation/final cohorts and a blinded human study did not run. The frozen source/quality advancement gates selected stop. Shared development speakers and repeated tuning prevent treating these tables as final generalization evidence.', '', 'Raw rows: [CSV](all_results.csv). Reproducer: [build_audio_browser_data.py](build_audio_browser_data.py). Original experiment logs and audio remain preserved.']
    (ROOT / '16_all_results.md').write_text('\n'.join(md) + '\n')
    status(**{k: v for k, v in receipt.items() if k != 'provenance'})
    print(json.dumps({k: v for k, v in receipt.items() if k != 'provenance'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        status(status='failed', error=repr(e));raise
