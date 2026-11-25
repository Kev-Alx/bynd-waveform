"""Freeze the remaining historical validation utterances; no model inference.

Reads protected manifests as metadata only. It never opens protected audio or
prediction/quality outcomes. The frozen development48 manifest is preserved.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

import soundfile as sf

ROOT = Path('/workspace')
OUT = ROOT / 'improvement/execution_refine'
PATHS = {
    'historically_exposed_validation100': ROOT / 'data/manifests/english-validation.jsonl',
    'development48': ROOT / 'improvement/execution48/development-48.jsonl',
    'sealed_confirmation48': ROOT / 'reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl',
    'sealed_final200': ROOT / 'improvement/execution/data/final-reserve-200.jsonl',
    'train1h': ROOT / 'improvement/execution/data/train-1h.jsonl',
    'train10h': ROOT / 'improvement/execution/data/train-10h.jsonl',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def norm(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c
                           for c in text).split())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keys(items):
    return {
        'id': {r['id'] for r in items},
        'speaker': {str(r['speaker']) for r in items},
        'audio_sha256': {r['audio_sha256'] for r in items},
        'normalized_transcript': {norm(r['text']) for r in items},
    }


def frozen_write(path, content):
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Existing frozen artifact differs: {path}')
    else:
        path.write_text(content)
    path.chmod(0o444)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / 'validation-76.receipt.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        assert sha(OUT / 'validation-76.jsonl') == old['manifest_sha256']
        assert sha(__file__) == old['script_sha256']
        for item in old['input_files'].values():
            assert sha(item['path']) == item['sha256']
        for item in old['audio_verification']:
            assert sha(item['path']) == item['audio_sha256']
        print(json.dumps({'status': 'existing_frozen_artifacts_reverified', 'count': old['count']}))
        return
    groups = {name: rows(path) for name, path in PATHS.items()}
    pool, pilot = groups['historically_exposed_validation100'], groups['development48']
    assert len(pool) == 100 and len(keys(pool)['id']) == 100
    assert len(pilot) == 48 and len(keys(pilot)['id']) == 48
    pool_ids, pilot_ids = keys(pool)['id'], keys(pilot)['id']
    assert len(pool_ids & pilot_ids) == 24
    selected = [dict(row, role='validation') for row in pool if row['id'] not in pilot_ids]
    assert len(selected) == 76
    selected_keys = keys(selected)
    assert all(len(selected_keys[k]) == 76 for k in ('id', 'audio_sha256', 'normalized_transcript'))
    overlaps = {}
    for name in ('development48', 'sealed_confirmation48', 'sealed_final200', 'train1h', 'train10h'):
        banned = keys(groups[name])
        overlaps[name] = {field: len(selected_keys[field] & banned[field]) for field in selected_keys}
        fields = ('id', 'audio_sha256', 'normalized_transcript')
        if name in ('sealed_final200', 'train1h', 'train10h'):
            fields += ('speaker',)
        for field in fields:
            assert overlaps[name][field] == 0, (name, field)
    checks = []
    for row in selected:
        info = sf.info(row['audio'])
        duration = info.frames / info.samplerate
        assert info.samplerate == row['sample_rate'] == 16000
        assert info.channels == 1 and info.frames == row['samples']
        assert abs(duration - row['duration']) < 1e-8 and 0.045 <= duration <= 30
        actual = sha(row['audio'])
        assert actual == row['audio_sha256'], row['id']
        checks.append({'id': row['id'], 'path': row['audio'], 'audio_sha256': actual,
                       'sample_rate': info.samplerate, 'channels': info.channels,
                       'samples': info.frames, 'duration_seconds': duration,
                       'format': info.format, 'subtype': info.subtype,
                       'metadata_matches_manifest': True, 'admissible': True})
    manifest = OUT / 'validation-76.jsonl'
    frozen_write(manifest, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    receipt = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_metadata_and_audio_verified',
        'count': 76, 'speakers': len(selected_keys['speaker']),
        'duration_seconds': sum(r['duration'] for r in selected),
        'minimum_seconds': min(r['duration'] for r in selected),
        'maximum_seconds': max(r['duration'] for r in selected),
        'selection': 'Original english-validation100 order, removing the24 IDs present in frozen development48; retain all76 remaining utterances.',
        'manifest': str(manifest), 'manifest_sha256': sha(manifest),
        'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'input_files': {name: {'path': str(path), 'sha256': sha(path)} for name, path in PATHS.items()},
        'cpu_only': True, 'models_loaded': False, 'model_outcomes_used_for_selection': False,
        'protected_audio_opened': False, 'protected_outcomes_read': False,
        'audio_transformations': 'none; whole native16k mono audio, unchanged reference text',
        'exposure': 'Historically exposed validation pool; utterances unseen by the current newly developed method, not a fresh final-test estimate. Existing baseline outcomes may already be known.',
        'candidate_freeze_requirement': 'Freeze method/code/hyperparameters and gate result on development48 before running this validation; use identical76 input utterances/references for every candidate and baseline control.',
        'protected_overlap': overlaps,
        'speaker_policy': 'Development48 and confirmation48 share dev-clean speakers; those overlaps are disclosed, not claimed as speaker-independent validation. Final200 and training speaker overlaps are forbidden.',
        'near_duplicate_limit': 'Exact ID/compressed-audio hash/normalized-transcript checks only; no perceptual near-duplicate guarantee.',
        'ids_in_order': [r['id'] for r in selected], 'audio_verification': checks,
    }
    frozen_write(receipt_path, json.dumps(receipt, indent=2) + '\n')
    note = f'''# Validation of 76 remaining historical utterances

The [manifest](/workspace/improvement/execution_refine/validation-76.jsonl) contains the76 remaining utterances of the historical English validation100, after removing the24 added to development48. Original order, audio paths, reference text and metadata are preserved. SHA256: `{receipt['manifest_sha256']}`. Selection uses no recognition, attack or quality outcomes.

All76 selected audio hashes and native16kHz mono metadata match their original manifests. Whole utterances span {receipt['minimum_seconds']:.3f}–{receipt['maximum_seconds']:.3f} seconds, totaling {receipt['duration_seconds']:.3f} seconds across {receipt['speakers']} speakers. Nothing was resampled, cropped or normalized.

There is no ID, compressed-audio hash or normalized-transcript overlap with development48, confirmation48, final200 or training. Final200 and training also share no speakers. Shared dev-clean speakers remain: {overlaps['development48']['speaker']} with development48 and {overlaps['sealed_confirmation48']['speaker']} with confirmation48. This is utterance-disjoint validation, with historical exposure and shared speakers; it cannot support fresh-test or speaker-independent claims. Protected manifests were read only as metadata; protected audio and outcomes were not opened.

Freeze the candidate method, code, hyperparameters and development48 gate decision before validating. Evaluate all candidate and baseline conditions on these same76 inputs, references and fixed native source/MOS policy. Any further method changes after seeing these outcomes make this pool development data. Future pilots keep the user's requested48 utterances. Confirmation48 and final200 remain untouched.

The [receipt](/workspace/improvement/execution_refine/validation-76.receipt.json) binds the script, all source manifests and every selected audio file. Rerunning the script verifies the frozen artifacts and refuses a changed selection.
'''
    (OUT / 'validation76-data-note.md').write_text(note)
    print(json.dumps({k: receipt[k] for k in ('status', 'count', 'speakers', 'duration_seconds', 'manifest_sha256', 'protected_overlap')}, indent=2))


if __name__ == '__main__':
    main()
