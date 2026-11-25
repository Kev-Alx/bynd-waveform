"""Acquire pinned public research assets only inside improvement/."""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime, timezone

RUN = Path('/workspace/improvement/execution_scaled')
os.environ['HF_HOME'] = str(RUN / 'hf-home')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import snapshot_download


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    configs = [('facebook/data2vec-audio-base-960h', '32331f3123e703528918aa688a9a38232d58c872',
        'data2vec', ['*.json', 'README.md', 'pytorch_model.bin']),
        ('nvidia/bigvgan_v2_24khz_100band_256x', 'c329ede9e9bbc100ddf5c91e2330a61921262370',
        'bigvgan', ['*.py', 'alias_free_activation/torch/*.py', 'config.json', 'README.md', 'LICENSE', 'bigvgan_generator.pt'])]
    records = []
    for repo, revision, name, patterns in configs:
        folder = RUN / 'models' / name
        print(f'Acquiring {repo}@{revision}', flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(folder),
            cache_dir=str(RUN / 'hf-cache'), allow_patterns=patterns, token=False, max_workers=2)
        files = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(folder.rglob('*'))
            if p.is_file() and '.cache' not in p.parts}
        records.append({'repo': repo, 'revision': revision, 'directory': str(folder), 'files': files})
        print(f'Completed {repo}: {sum(v["bytes"] for v in files.values())} bytes', flush=True)
    path = RUN / 'assets.json'
    if path.exists(): raise FileExistsError('Preserve original asset receipt')
    path.write_text(json.dumps({'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'public_no_token': True, 'models': records}, indent=2) + '\n')


if __name__ == '__main__': main()
