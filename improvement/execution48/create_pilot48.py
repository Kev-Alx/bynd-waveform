"""Freeze outcome-independent 48-utterance development pilot; CPU metadata only.

This preserves the former development24 in place and adds 24 utterances from
the historically exposed english-validation100 pool. No model is imported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

import soundfile as sf

ROOT = Path('/workspace')
OUT = ROOT / 'improvement/execution48'
SEED = 20261022
PATHS = {
    'previous_development24': ROOT / 'improvement/execution/development-24.jsonl',
    'historically_exposed_validation100': ROOT / 'data/manifests/english-validation.jsonl',
    'sealed_confirmation48': ROOT / 'reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl',
    'sealed_final200': ROOT / 'improvement/execution/data/final-reserve-200.jsonl',
    'train1h': ROOT / 'improvement/execution/data/train-1h.jsonl',
    'train10h': ROOT / 'improvement/execution/data/train-10h.jsonl',
    'known_layer6_cache_audit': ROOT / 'reproduction-runs/feature-delta-diagnostics/parent-audits/pilot.json',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def norm(text):
    # Same English normalization as beyondwaveform.evaluation.normalize_text.
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in text).split())


def keys(items):
    return {
        'id': {r['id'] for r in items},
        'speaker': {str(r['speaker']) for r in items},
        'audio_sha256': {r['audio_sha256'] for r in items},
        'normalized_transcript': {norm(r['text']) for r in items},
    }


def audio_info(row):
    info = sf.info(row['audio'])
    actual_duration = info.frames / info.samplerate
    return {
        'sample_rate': info.samplerate, 'channels': info.channels,
        'samples': info.frames, 'duration_seconds': actual_duration,
        'format': info.format, 'subtype': info.subtype,
        'metadata_matches_manifest': info.samplerate == row['sample_rate']
            and info.frames == row['samples']
            and abs(actual_duration - row['duration']) < 1e-8,
        'admissible': info.samplerate == 16000 and info.channels == 1
            and 0.045 <= actual_duration <= 30,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {k: rows(p) for k, p in PATHS.items() if k != 'known_layer6_cache_audit'}
    old = groups['previous_development24']
    pool = groups['historically_exposed_validation100']
    assert len(old) == 24 and len({r['id'] for r in old}) == 24
    assert len(pool) == 100 and len({r['id'] for r in pool}) == 100
    protected = {k: keys(groups[k]) for k in (
        'sealed_confirmation48', 'sealed_final200', 'train1h', 'train10h')}
    old_keys = keys(old)
    validation = {r['id']: audio_info(r) for r in old + pool}
    excluded = []
    eligible = []
    for row in pool:
        reasons = []
        if not validation[row['id']]['metadata_matches_manifest']:
            raise ValueError(f'Changed metadata: {row["id"]}')
        if not validation[row['id']]['admissible']:
            reasons.append('native_audio_metadata_not_admissible')
        row_keys = keys([row])
        if any(row_keys[k] & old_keys[k] for k in ('id', 'audio_sha256', 'normalized_transcript')):
            reasons.append('duplicates_previous_development24')
        for name, banned in protected.items():
            fields = ('id', 'audio_sha256', 'normalized_transcript')
            if name != 'sealed_confirmation48':
                fields += ('speaker',)
            if any(row_keys[k] & banned[k] for k in fields):
                reasons.append(name + '_overlap')
        if reasons:
            excluded.append({'id': row['id'], 'reasons': reasons})
        else:
            eligible.append(row)
    eligible.sort(key=lambda r: (hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest(), r['id']))
    additions, seen_text, seen_hash = [], set(old_keys['normalized_transcript']), set(old_keys['audio_sha256'])
    for row in eligible:
        if norm(row['text']) in seen_text or row['audio_sha256'] in seen_hash:
            continue
        additions.append(row)
        seen_text.add(norm(row['text']))
        seen_hash.add(row['audio_sha256'])
        if len(additions) == 24:
            break
    if len(additions) != 24:
        raise ValueError('Not enough eligible additions; never relax exclusion rules automatically')
    selected = [dict(r, role='pilot') for r in old + additions]
    selected_keys = keys(selected)
    assert len(selected_keys['id']) == 48
    for name, banned in protected.items():
        required = ('id', 'audio_sha256', 'normalized_transcript')
        if name != 'sealed_confirmation48':
            required += ('speaker',)
        for field in required:
            assert not selected_keys[field] & banned[field], (name, field)
    audio_checks = []
    for row in selected:
        meta = validation[row['id']]
        assert meta['metadata_matches_manifest'] and meta['admissible'], row['id']
        actual = sha(row['audio'])
        assert actual == row['audio_sha256'], f'Changed source audio: {row["id"]}'
        audio_checks.append({'id': row['id'], 'path': row['audio'], 'audio_sha256': actual, **meta})
    cache_audit = json.loads(PATHS['known_layer6_cache_audit'].read_text())
    cache_index = {r['id']: r for r in cache_audit['items']}
    cache_checks = []
    for row in selected:
        cached = cache_index.get(row['id'])
        if cached:
            actual = sha(cached['q_path'])
            assert actual == cached['q_sha256']
            cache_checks.append({'id': row['id'], 'status': 'verified_historical_layer6_cache',
                'q_path': cached['q_path'], 'q_sha256': actual})
        else:
            cache_checks.append({'id': row['id'], 'status': 'not_in_known_audited_layer6_cache',
                'action': 'fresh feature extraction required before feature-space experiments'})
    manifest = OUT / 'development-48.jsonl'
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError('Existing frozen pilot differs; refusing overwrite')
    manifest.write_text(content)
    manifest.chmod(0o444)
    receipt = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_metadata_and_audio_verified', 'seed': SEED,
        'selection': 'Preserve previous24 order; append first24 eligible sorted by SHA256("20261022:" + id), then id. Transcript/hash deduplication in that order.',
        'count': 48, 'previous_count': 24, 'additional_count': 24,
        'speakers': len(selected_keys['speaker']),
        'duration_seconds': sum(r['duration'] for r in selected),
        'minimum_seconds': min(r['duration'] for r in selected),
        'maximum_seconds': max(r['duration'] for r in selected),
        'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'input_files': {name: {'path': str(p), 'sha256': sha(p)} for name, p in PATHS.items()},
        'cpu_only': True, 'models_loaded': False, 'model_outcomes_used_for_selection': False,
        'audio_transformations': 'none; no resampling, truncation, or amplitude normalization',
        'admissibility': {'sample_rate': 16000, 'channels': 1, 'minimum_seconds': 0.045, 'maximum_seconds': 30},
        'historical_validation_pool_count': len(pool), 'eligible_addition_count': len(eligible),
        'metadata_rejections': excluded, 'old_ids_in_order': [r['id'] for r in old],
        'new_ids_in_order': [r['id'] for r in additions],
        'exposure': 'Development data already used in historical study; not fresh confirmation or a final-test estimate. Selection never reads per-clip model outcomes.',
        'protected_overlap': {name: {field: len(selected_keys[field] & banned[field])
            for field in selected_keys} for name, banned in protected.items()},
        'confirmation_speaker_overlap_policy': 'Confirmation utterances/audio/text excluded; shared dev-clean speakers allowed and reported. Final200 and training speakers excluded.',
        'near_duplicate_limit': 'ID, exact compressed-audio SHA256 and normalized-transcript checks; no claim of perceptual deduplication.',
        'audio_verification': audio_checks, 'known_layer6_query_cache': cache_checks,
        'known_layer6_query_cache_available': sum(c['status'] == 'verified_historical_layer6_cache' for c in cache_checks),
    }
    receipt_path = OUT / 'development-48.receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.chmod(0o444)
    note = f'''# Development pilot of 48 utterances

Frozen manifest: `{manifest}` (SHA256 `{receipt['manifest_sha256']}`).

The pilot preserves the previous 24 utterances in their original order and adds 24 from the historically exposed English validation100 pool. Additions were ordered by SHA256 of `20261022:ID` after metadata and overlap checks. No source recognition, quality, or attack outcomes were used for selection. This remains development data, with no fresh-confirmation claim.

All 48 files passed source SHA256 and native 16 kHz mono metadata checks. Whole utterances span {receipt['minimum_seconds']:.3f}–{receipt['maximum_seconds']:.3f} seconds, totaling {receipt['duration_seconds']:.3f} seconds across {receipt['speakers']} speakers. No audio was resampled, truncated, or amplitude-normalized.

There is zero ID, exact compressed-audio hash, or normalized-transcript overlap with sealed confirmation48, sealed final200, or training manifests. There is also zero speaker overlap with final200 and training. Confirmation uses the same development split: {receipt['protected_overlap']['sealed_confirmation48']['speaker']} speakers overlap; this pilot does not establish speaker-independent generalization.

The known historical layer6 query-cache audit provides verified files for {receipt['known_layer6_query_cache_available']} utterances (the original 24). The 24 additions are absent from that audit and require fresh extraction before feature-space experiments. The receipt records paths and hashes; no tensors or models were loaded in this selection step.

The script, all source manifests, every selected audio file, and known cached query files are hash-bound in `development-48.receipt.json`. Older plans, results, confirmation48, and final200 were preserved.
'''
    (OUT / 'pilot48-selection.md').write_text(note)
    print(json.dumps({k: receipt[k] for k in ('status', 'count', 'speakers', 'duration_seconds', 'manifest_sha256', 'known_layer6_query_cache_available', 'protected_overlap')}, indent=2))


if __name__ == '__main__':
    main()
