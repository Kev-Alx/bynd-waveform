"""Artifact containment, provenance and reproducible development cohorts."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def configure_runtime():
    paths = {'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'HF_HOME': 'cache/huggingface',
             'HUGGINGFACE_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_HOME': 'cache/torch',
             'NUMBA_CACHE_DIR': 'cache/numba', 'MPLCONFIGDIR': 'cache/matplotlib',
             'TORCHINDUCTOR_CACHE_DIR': 'cache/torchinductor', 'PIP_CACHE_DIR': 'cache/pip'}
    for key, relative in paths.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['SB_DISABLE_QUIRKS'] = 'allow_tf32'


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Output outside artifact root: {path}')
    return path


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    tmp.replace(path)


def seeded_key(uid, seed=20261022):
    return hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest()


def balanced_screen(rows, count, seed=20261022):
    """Round robin speakers; selection uses only metadata, never ASR outcomes."""
    groups = {}
    for row in rows:
        groups.setdefault(str(row['speaker']), []).append(row)
    for values in groups.values():
        values.sort(key=lambda r: seeded_key(r['id'], seed))
    speakers = sorted(groups, key=lambda s: seeded_key(s, seed))
    chosen = []
    while len(chosen) < min(count, len(rows)):
        for speaker in speakers:
            if groups[speaker] and len(chosen) < count:
                chosen.append(groups[speaker].pop(0))
    return chosen


def verify_audio_asset(asset):
    if digest(asset['path']) != asset['sha256']:
        raise ValueError(f'Audio hash mismatch: {asset["path"]}')


def initialize_torch(seed=11):
    import random
    import numpy as np
    import torch
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
