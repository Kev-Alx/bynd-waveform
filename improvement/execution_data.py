"""Acquire official corpora and freeze outcome-independent research cohorts.

Every write stays under /workspace/improvement. No ASR/quality model is loaded.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path('/workspace')
OUT = ROOT / 'improvement'
sys.path.insert(0, str(ROOT))
from beyondwaveform.assets import download, download_ranged, safe_extract
from beyondwaveform.data import prepare_manifest
from beyondwaveform.evaluation import normalize_text
from beyondwaveform.io import atomic_json, digest, read_jsonl, write_jsonl


def acquire(split):
    base = 'https://www.openslr.org/resources/12/'
    data = OUT / 'data'
    downloads = data / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    md5file = download(base + 'md5sum.txt', downloads / 'official-md5sum.txt')
    checks = {s.split()[1].lstrip('*'): s.split()[0] for s in md5file.read_text().splitlines() if s.strip()}
    filename = split + '.tar.gz'
    archive = downloads / filename
    if not archive.exists():
        download_ranged(base + filename, archive, workers=8)
    actual = digest(archive, 'md5')
    if actual != checks[filename]:
        raise ValueError(f'Official archive checksum mismatch: {split}')
    marker = downloads / (filename + '.extracted.json')
    if not marker.exists():
        safe_extract(archive, data)
        atomic_json(marker, {'url': base + filename, 'md5': actual, 'sha256': digest(archive),
            'license': 'CC BY 4.0', 'publisher': 'https://www.openslr.org/12/'})
    index = data / (split + '-index.jsonl')
    if not index.exists():
        prepare_manifest('librispeech', data, split, 'test' if split.startswith('test') else 'train', index)
    return read_jsonl(index)


def history():
    baseline = json.loads((OUT / 'baseline_evidence.json').read_text())
    all_rows = {}
    for meta in baseline['manifests'].values():
        for row in read_jsonl(meta['path']):
            all_rows[row['id']] = row
    return list(all_rows.values())


def key(row, seed):
    return hashlib.sha256(f'{seed}:{row["id"]}'.encode()).hexdigest()


def balanced(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['speaker']].append(row)
    speakers = sorted(groups)
    random.Random(seed).shuffle(speakers)
    for speaker in speakers:
        groups[speaker].sort(key=lambda r: key(r, seed))
    ordered = []
    while any(groups.values()):
        for speaker in speakers:
            if groups[speaker]:
                ordered.append(groups[speaker].pop(0))
    return ordered


def freeze_final(rows, old):
    path = OUT / 'execution' / 'data' / 'final-reserve-200.jsonl'
    if path.exists():
        return read_jsonl(path)
    old_speakers = {r['speaker'] for r in old}
    old_hashes = {r['audio_sha256'] for r in old}
    old_texts = {normalize_text(r['text'], 'en') for r in old}
    eligible = [r for r in rows if r['speaker'] not in old_speakers
        and r['audio_sha256'] not in old_hashes and normalize_text(r['text'], 'en') not in old_texts
        and 1 <= r['duration'] <= 30]
    selected, seen_texts = [], set()
    for row in balanced(eligible, 20261021):
        text = normalize_text(row['text'], 'en')
        if text in seen_texts:
            continue
        selected.append(row)
        seen_texts.add(text)
        if len(selected) == 200:
            break
    if len(selected) != 200 or len({r['speaker'] for r in selected}) < 30:
        raise ValueError('Cannot meet the preregistered final reserve size/speaker constraints')
    write_jsonl(path, selected)
    path.chmod(0o444)
    atomic_json(path.with_suffix('.receipt.json'), {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_unscored', 'manifest_sha256': digest(path),
        'count': len(selected), 'speakers': len({r['speaker'] for r in selected}),
        'seed': 20261021, 'corpus': 'LibriSpeech test-other',
        'role': 'untouched final reserve; harder-speech shift, not a matched test-clean pool',
        'selection': 'speaker round robin; seeded within-speaker SHA256 ordering; no model outcomes',
        'excluded_speaker_overlap': 0, 'excluded_source_hash_overlap': 0,
        'excluded_normalized_transcript_overlap': 0,
        'historical_manifest_union_count': len(old),
        'near_duplicate_limit': 'Speaker/source and exact compressed-audio/text checks; no claim of exhaustive perceptual duplicate detection.',
        'pretraining_exposure': 'Unseen by this attack study; pretrained-model exposure is not certified absent.',
        'license': 'CC BY 4.0', 'publisher': 'https://www.openslr.org/12/'})
    return selected


def freeze_train(rows, old, final):
    folder = OUT / 'execution' / 'data'
    excluded = old + final
    speakers = {r['speaker'] for r in excluded}
    hashes = {r['audio_sha256'] for r in excluded}
    texts = {normalize_text(r['text'], 'en') for r in excluded}
    allowed = [r for r in rows if r['speaker'] not in speakers and r['audio_sha256'] not in hashes
        and normalize_text(r['text'], 'en') not in texts and 1 <= r['duration'] <= 30]
    ordered = balanced(allowed, 11)
    selected, seen, seconds = [], set(), 0.0
    for row in ordered:
        text = normalize_text(row['text'], 'en')
        if text in seen:
            continue
        selected.append(row)
        seen.add(text)
        seconds += row['duration']
        for hours in (1, 10):
            path = folder / f'train-{hours}h.jsonl'
            if seconds >= hours * 3600 and not path.exists():
                write_jsonl(path, selected)
                path.chmod(0o444)
                atomic_json(path.with_suffix('.receipt.json'), {'status': 'frozen_unscored',
                    'manifest_sha256': digest(path), 'count': len(selected), 'seconds': seconds,
                    'speakers': len({r['speaker'] for r in selected}), 'seed': 11,
                    'selection': 'nested speaker-balanced whole utterances; fixed duration-first stopping; no outcomes',
                    'license': 'CC BY 4.0', 'publisher': 'https://www.openslr.org/12/',
                    'excluded_speaker_audio_transcript_overlap': 0})
        if seconds >= 36000:
            break
    if seconds < 36000:
        raise ValueError('Not enough eligible data for nested 1h/10h pools')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    old = history()
    final = freeze_final(acquire('test-other'), old)
    if args.train:
        freeze_train(acquire('train-clean-100'), old, final)
    print(json.dumps({'final_reserve_count': len(final), 'training_prepared': args.train}), flush=True)


if __name__ == '__main__':
    main()
