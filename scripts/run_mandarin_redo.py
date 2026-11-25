"""Corrected, resumable AISHELL development pilot and saved-WAV CER scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beyondwaveform.evaluation import aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything, write_jsonl
from beyondwaveform.models.whisper import load_whisper

OUT = ROOT / 'reproduction-runs/mandarin-cer-corrected'
VARIANTS = ['clean', 'reconstruction', 'adversarial']
TAIPEI = timezone(timedelta(hours=8))


def now():
    return datetime.now(timezone.utc).isoformat()


def corrected_target(config):
    # Match supervised_loss's <|notimestamps|> prefix explicitly. Preserve all
    # other settings, including beam size, temperature and checkpoint.
    target = json.loads(json.dumps(config['surrogate']))
    target.setdefault('decoding', {})['without_timestamps'] = True
    return target


def reusable_clean_score(cache_folder, record, target_metadata):
    """Reuse only an identical audio/model/decoder/reference observation."""
    path = cache_folder/'records'/f"{record['id']}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved['target'] != target_metadata:
        return None
    for row in saved['rows']:
        if (row['variant']=='clean' and row['id']==record['id']
                and row['reference']==record['text']
                and row['audio_sha256']==record['audio']['clean']['sha256']):
            return row
    return None


def freeze(path, value, manifest=False):
    if path.exists():
        prior = read_jsonl(path) if manifest else json.loads(path.read_text())
        if prior != value:
            raise ValueError(f'Frozen protocol differs: {path}')
    else:
        (write_jsonl if manifest else atomic_json)(path, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'reproduction-runs/mandarin-independent/manifests'
    reserve = read_jsonl(base / 'attack-reserve.jsonl')
    exposed = {r['id'] for r in read_jsonl(base / 'attack-validation-100.jsonl')}
    available = [r for r in reserve if r['id'] not in exposed]
    rng = random.Random(2026)
    pilot = []
    for speaker in sorted({r['speaker'] for r in available}):
        rows = sorted((r for r in available if r['speaker'] == speaker), key=lambda r: (r['duration'], r['id']))
        for i in range(12):
            pilot.append(rng.choice(rows[i*len(rows)//12:(i+1)*len(rows)//12]))
    assert len(pilot) == 48 and len({r['id'] for r in pilot}) == 48
    config = json.loads((ROOT / 'reproduction-runs/mandarin-cer-evaluation/config.json').read_text())
    config['surrogate'] = corrected_target(config)
    configs = {'selected': config, 'paper': json.loads(json.dumps(config))}
    configs['paper']['attack'].update(rho=0.1, lambda_perc=1.0)
    prior = []
    sources = []
    for name in ['mandarin-cer-evaluation', 'mandarin-cer-next-3000']:
        folder = ROOT / 'reproduction-runs' / name
        verified = json.loads((folder / 'verification.json').read_text())
        assert verified['status'] == 'passed'
        assert digest(folder / 'results.json') == verified['results_sha256']
        assert digest(folder / 'utterances.jsonl') == verified['utterances_sha256']
        prior += read_jsonl(folder / 'test-subset.jsonl')
        for p in sorted((folder / 'test').glob('chunk-*/run/run.json')):
            meta = json.loads(p.read_text())
            generated = p.parent / 'generated.jsonl'
            assert meta['status'] == 'complete' and digest(generated) == meta['generated_sha256']
            sources.append({'run': str(p.parent), 'run_sha256': digest(p),
                            'generated_sha256': digest(generated)})
    assert len(prior) == len({r['id'] for r in prior}) == 5000
    assert len({r['audio_sha256'] for r in prior}) == 5000
    assert not {r['speaker'] for r in pilot} & {r['speaker'] for r in prior}
    train_ids = {r['id'] for r in read_jsonl(base / 'finetune.jsonl')}
    assert not train_ids & {r['id'] for r in pilot + prior}
    freeze(OUT / 'pilot-48.jsonl', pilot, True)
    freeze(OUT / 'test-5000.jsonl', prior, True)
    for name, cfg in configs.items():
        freeze(OUT / f'{name}-config.json', cfg)
    checkpoint = config['surrogate']['checkpoint']
    original_plan = json.loads((ROOT / 'reproduction-runs/mandarin-cer-evaluation/plan.json').read_text())
    assert digest(checkpoint) == original_plan['checkpoint_sha256']
    plan = {'status': 'frozen', 'pilot_count': 48, 'test_count': 5000,
            'scope': 'Corrected no-timestamp decoding; fresh 48-clip development pilot before 5,000 official AISHELL test recordings',
            'pilot_selection': 'Seed 2026; 12 duration-stratified random clips per each of four reserved development speakers; excludes the exposed 100-clip validation pool; no score-based sampling',
            'pilot_manifest_sha256': digest(OUT / 'pilot-48.jsonl'),
            'test_manifest_sha256': digest(OUT / 'test-5000.jsonl'),
            'checkpoint_sha256': digest(checkpoint),
            'config_sha256': {k: digest(OUT / f'{k}-config.json') for k in configs},
            'saved_test_generation': sources,
            'paper': {'clean_cer': 6.29, 'attack_cer': 72.25, 'url': 'https://arxiv.org/html/2606.05678v1#S5'},
            'comparison_tolerance_pp': {'clean': 2.0, 'adversarial': 5.0},
            'protocol': 'Keep original references and normalization; one fixed checkpoint and explicit without_timestamps=True for all three audio variants. Selected English attack settings retained as initial candidate. Paper rho/lambda is a separately labeled development diagnostic. No test-based tuning, text trimming, repetition removal, or score-based sample exclusions.'}
    freeze(OUT / 'plan.json', plan)
    extension_path = OUT/'additional-candidates.json'
    if extension_path.exists():
        extension = json.loads(extension_path.read_text())
        assert extension['base_plan_sha256'] == digest(OUT/'plan.json')
        plan = json.loads(json.dumps(plan))
        plan['candidate_extension_sha256'] = digest(extension_path)
        for name, value in extension['candidates'].items():
            assert name not in plan['config_sha256']
            path = OUT/f'{name}-config.json'
            assert digest(path) == value['config_sha256']
            plan['config_sha256'][name] = value['config_sha256']
    return plan


class Scorer:
    def __init__(self, plan, stage, candidate):
        self.plan, self.stage, self.candidate = plan, stage, candidate
        self.folder = OUT / stage / candidate
        self.folder.mkdir(parents=True, exist_ok=True)
        self.total = 5000 if stage.startswith('test') else 48
        self.started = time.monotonic()
        self.processed = 0
        self.elapsed_scoring = 0
        self.reused_clean_rows = 0
        self.pilot_eval_seconds = 12
        pilot_folder = OUT/'pilot'/candidate
        if (pilot_folder/'results.json').exists():
            prior = json.loads((pilot_folder/'results.json').read_text())
            timings = [json.loads(p.read_text()) for p in (pilot_folder/'records').glob('*.json')]
            if prior['status']=='complete' and prior['count']==48 and len(timings)==48:
                self.pilot_eval_seconds = sum(r['seconds'] for r in timings)/len(timings)
        self.state = {'status': 'running', 'phase': f'{stage}-{candidate}', 'started_at': now(),
                      'scope': plan['scope'], 'total_utterances': self.total, 'generated_utterances': 0}

    def progress(self, done=0, phase=None, status='running', **extra):
        rate = self.elapsed_scoring/self.processed if self.processed else self.pilot_eval_seconds
        remaining = extra.pop('remaining_override', (self.total-done) * rate)
        eta_basis = extra.pop('eta_basis', f'Observed corrected-decoder scoring time; starting estimate {self.pilot_eval_seconds:.2f} seconds per recording from the completed 48-clip pilot when available')
        self.state.update(status=status, phase=phase or f'{self.stage}-{self.candidate}-scoring',
                          updated_at=now(), evaluated_utterances=done,
                          clip_progress={'completed': done, 'total': self.total, 'percent': round(100*done/self.total, 1)},
                          estimated_remaining_seconds=round(remaining),
                          estimated_remaining_range_seconds=[round(remaining*.7), round(remaining*1.8)],
                          estimated_completion_taipei=(datetime.now(TAIPEI)+timedelta(seconds=remaining)).isoformat(timespec='seconds'),
                          eta_basis=eta_basis, **extra)
        atomic_json(OUT / 'progress.json', self.state)
        text = f"# Corrected AISHELL CER\n\nUpdated: {datetime.now(TAIPEI).isoformat(timespec='seconds')} (Asia/Taipei)\n\nStatus: **{status}**; phase: **{self.state['phase']}**.\n\nRecordings: **{done}/{self.total}**. Estimated remaining: **{remaining/3600:.2f} hours**; completion: **{self.state['estimated_completion_taipei']}**.\n\n{self.state['eta_basis']}\n\nPilot: 48 reserved development recordings. Test: the existing 5,000 official test recordings. Corrected decoder is applied to all audio variants.\n"
        (OUT / 'JOB_STATUS.md').write_text(text)
        sys.path.insert(0, str(ROOT / 'scripts'))
        from update_job_status import update
        update(OUT.parent, 'paper-quality-coordinate-research')

    def generate(self, config_path, manifest, folder):
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(ROOT / 'scripts/generate.py'), '--config', str(config_path),
                   '--manifest', str(manifest), '--output', str(folder), '--device', 'cuda']
        with (self.folder / 'generation.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while True:
                generated = [json.loads(p.read_text()) for p in (folder / 'records').glob('*.json')]
                count = len(generated)
                seconds = sum(r['generation_seconds'] for r in generated)/count if count else 20
                self.progress(0, phase=f'{self.stage}-{self.candidate}-generation', generated_utterances=count,
                              remaining_override=(self.total-count)*seconds+self.total*self.pilot_eval_seconds,
                              eta_basis=f'Observed generation time ({seconds:.1f} seconds/clip) plus completed 48-clip scoring timing when available ({self.pilot_eval_seconds:.2f} seconds/clip)')
                try:
                    code = child.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code:
                raise RuntimeError(f'Generation exited {code}; inspect {self.folder}/generation.log')
        return read_jsonl(folder / 'generated.jsonl')

    def score(self):
        import torch
        torch.set_num_threads(8)
        seed_everything(2026)
        cfg_path = OUT / f'{self.candidate}-config.json'
        assert digest(cfg_path) == self.plan['config_sha256'][self.candidate]
        config = json.loads(cfg_path.read_text())
        source_paths = [ROOT/'beyondwaveform'/name for name in
                        ['attacks.py','losses.py','pipeline.py','evaluation.py','io.py','config.py',
                         'models/whisper.py','models/feature.py','models/targets.py']]
        source_hashes = {str(p):digest(p) for p in source_paths}
        freeze(self.folder/'source-freeze.json',source_hashes)
        variants = ['clean'] if self.stage.endswith('clean') else VARIANTS
        manifest = OUT / ('test-5000.jsonl' if self.stage.startswith('test') else 'pilot-48.jsonl')
        expected = read_jsonl(manifest)
        if self.stage == 'pilot-clean':
            records = [{'id': r['id'], 'text': r['text'], 'audio': {'clean': {'path': r['audio'], 'sha256': r['audio_sha256']}}} for r in expected]
        elif self.stage == 'test-clean' or (self.stage == 'test' and self.candidate == 'selected'):
            records = []
            for source in self.plan['saved_test_generation']:
                folder = Path(source['run'])
                assert digest(folder/'run.json') == source['run_sha256']
                assert digest(folder/'generated.jsonl') == source['generated_sha256']
                records += read_jsonl(folder/'generated.jsonl')
        else:
            records = self.generate(cfg_path, manifest, self.folder / 'generation')
        assert len(records) == len(expected) == self.total
        assert all(digest(path)==sha for path,sha in source_hashes.items()), 'Scoring/generation code changed during the stage'
        self.state['generated_utterances'] = 0 if self.stage == 'pilot-clean' else len(records)
        by_id = {r['id']: r for r in expected}
        assert {r['id'] for r in records} == set(by_id)
        target, metadata = load_whisper(config['surrogate'], 'cuda')
        assert metadata['sha256'] == self.plan['checkpoint_sha256']
        assert metadata['decoding']['without_timestamps'] is True
        rows = []
        self.progress()
        for i, record in enumerate(records):
            if i%50==0:
                assert all(digest(path)==sha for path,sha in source_hashes.items()), 'Frozen scoring source changed'
            assert record['text'] == by_id[record['id']]['text']
            cached = self.folder / 'records' / f"{record['id']}.json"
            if cached.exists():
                saved = json.loads(cached.read_text())
                assert saved['config_sha256'] == digest(cfg_path)
                assert saved['target'] == metadata
                scored = saved['rows']
                self.reused_clean_rows += int(saved.get('reused_clean',False))
                assert [r['variant'] for r in scored] == variants
                for r in scored:
                    assert r['reference'] == record['text']
                    assert r['audio_sha256'] == record['audio'][r['variant']]['sha256']
                    assert digest(record['audio'][r['variant']]['path']) == r['audio_sha256']
            else:
                started = time.monotonic()
                scored = []
                reused_clean = False
                for variant in variants:
                    asset = record['audio'][variant]
                    assert digest(asset['path']) == asset['sha256']
                    clean = None
                    if self.stage == 'test' and variant == 'clean':
                        for candidate in ['paper','selected','paper-lr025']:
                            clean = reusable_clean_score(OUT/'test-clean'/candidate,record,metadata)
                            if clean is not None:
                                break
                    if clean is not None:
                        scored.append(clean)
                        reused_clean = True
                        self.reused_clean_rows += 1
                        continue
                    hypothesis = target.transcribe(load_audio(asset['path']).to('cuda'))
                    scored.append({'id': record['id'], 'variant': variant, 'audio_sha256': asset['sha256'],
                                   'reference': record['text'], 'hypothesis': hypothesis,
                                   **error_counts(record['text'], hypothesis, 'zh')})
                seconds = time.monotonic()-started
                atomic_json(cached, {'config_sha256': digest(cfg_path), 'target': metadata, 'rows': scored,
                                     'seconds': seconds,'reused_clean':reused_clean})
                self.processed += 1
                self.elapsed_scoring += seconds
            rows.extend(scored)
            summary = {v: {'count': i+1, **aggregate_errors([r for r in rows if r['variant'] == v])} for v in variants}
            result = {'status': 'complete' if i+1 == self.total else 'partial', 'metric': 'CER',
                      'stage': self.stage, 'candidate': self.candidate, 'count': i+1, 'target': metadata,
                      'config_sha256': digest(cfg_path), 'plan_sha256': digest(OUT/'plan.json'), 'summary': summary}
            result['reused_clean_rows'] = self.reused_clean_rows
            if self.candidate not in ['selected','paper']:
                result['candidate_extension_sha256'] = self.plan['candidate_extension_sha256']
            atomic_json(self.folder / 'results.json', result)
            atomic_json(OUT / 'results.json', result)
            self.progress(i+1)
        assert len({(r['id'],r['variant']) for r in rows}) == self.total*len(variants)
        for r in rows:
            counts = error_counts(r['reference'], r['hypothesis'], 'zh')
            assert all(r[k] == v for k,v in counts.items())
        write_jsonl(self.folder / 'utterances.jsonl', rows)
        atomic_json(self.folder / 'verification.json', {'status': 'passed', 'count': self.total,
                    'pairs': len(rows), 'results_sha256': digest(self.folder/'results.json'),
                    'utterances_sha256': digest(self.folder/'utterances.jsonl'), 'checked_at': now()})
        report = ['# Corrected AISHELL CER', '', f'{self.stage}; {self.candidate}; {self.total} recordings.', '',
                  '| Audio | CER (%) | Paper CER (%) |', '|---|---:|---:|']
        for v, values in summary.items():
            p = self.plan['paper'].get('clean_cer' if v=='clean' else 'attack_cer' if v=='adversarial' else '')
            report.append(f"| {v} | {values['error_percent']:.6f} | {p if p is not None else '—'} |")
        report += ['', self.plan['protocol'], '', 'The development pilot is diagnostic; final comparison requires the complete frozen test evaluation.']
        attack=config['attack']
        report += ['', f"Attack settings: rho={attack['rho']}, lambda={attack['lambda_perc']}, {attack['steps']} updates, Adam learning rate={attack['learning_rate']}. Decoder: forced Mandarin, beam size 5, temperature 0, without_timestamps=True.",
                   f'Clean observations reused with identical audio hash, checkpoint, decoder and reference: {self.reused_clean_rows}.']
        if self.candidate=='paper-lr025':
            report += ['', 'Learning rate 0.025 is an independently calibrated development choice for a parameter omitted by the paper. Numerical agreement does not establish recovery of the author recipe. All attempted pilot configurations remain available under pilot/.']
        if self.stage=='test':
            report += ['', 'This table covers the complete frozen 5,000-record test manifest. The author exact subset and fine-tuned checkpoint are not established as matching this implementation.']
        for path in [self.folder/'report.md', OUT/'report.md']:
            path.write_text('\n'.join(report)+'\n')
        self.progress(self.total, status='complete', phase=f'{self.stage}-{self.candidate}-complete')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot-clean', 'pilot', 'test-clean', 'test'], default='pilot-clean')
    parser.add_argument('--candidate', choices=['selected', 'paper', 'paper-lr025'], default='selected')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    if args.prepare_only:
        print('Frozen fresh 48-clip development pilot and 5,000-record test protocol.'); return
    if args.stage != 'pilot-clean':
        baseline = json.loads((OUT/'pilot-clean/selected/results.json').read_text())
        assert baseline['status'] == 'complete' and baseline['count'] == 48, '48-record clean pilot must complete first'
        assert baseline['summary']['clean']['error_percent'] <= 8.29, 'Clean development baseline needs diagnosis before attacks'
    if args.stage == 'test':
        pilot = json.loads((OUT/'pilot'/args.candidate/'results.json').read_text())
        assert pilot['status'] == 'complete' and pilot['count'] == 48, 'Complete the same candidate on the 48-record pilot before test evaluation'
    with (OUT/'evaluation.lock').open('a+') as lock, (OUT.parent/'auxiliary-gpu.lock').open('a+') as gpu:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Scorer(plan, args.stage, args.candidate)
        try:
            runner.score()
        except BaseException as error:
            runner.progress(status='failed', phase=f'{args.stage}-{args.candidate}-failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
