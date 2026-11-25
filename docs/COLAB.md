# Run BeyondWaveform in Google Colab with Google Drive

Open `notebooks/beyondwaveform_colab.ipynb` using Colab's **File > Upload notebook**, and save a copy in Drive. Alternatively, paste the cells below into a new notebook in order. Select **Runtime > Change runtime type > Python 3 > GPU** before starting. Start with the one-clip smoke test; GPU memory requirements and speed depend on the assigned hardware and clip length.

Code, model artifacts, corpus downloads, manifests, caches, and results are stored under `MyDrive/BeyondWaveform`. Python and installed packages stay on the temporary Colab runtime and are rebuilt after a reset. This avoids tying the project to Colab's changing Python and PyTorch versions. The environment follows the repository's Docker setup: Python 3.11 and PyTorch/TorchAudio 2.4.1 with CUDA 12.1 wheels.

The commands expect the BeyondWaveform source layout shown in this workspace. The GitHub clone may differ; Cell 2 checks for the required scripts before proceeding. Local uncommitted files and existing results are not transferred automatically. For a private repository, upload a copy of the source folder to `MyDrive/BeyondWaveform/bynd-waveform` before running Cell 2, or authenticate Git separately. Keep tokens out of notebook cells and repository URLs. Existing manifests generated under `/workspace` contain absolute paths: regenerate them on Colab rather than reusing them directly. Keep the Drive folder path stable when resuming runs.

## Cell 1 — Mount Drive and set persistent locations

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os, subprocess, sys, json, shutil

ROOT = Path('/content/drive/MyDrive/BeyondWaveform')
PROJECT = ROOT / 'bynd-waveform'
for folder in ('cache/huggingface', 'cache/torch', 'cache/xdg'):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

os.environ.update({
    'HF_HOME': str(ROOT / 'cache/huggingface'),
    'TORCH_HOME': str(ROOT / 'cache/torch'),
    'XDG_CACHE_HOME': str(ROOT / 'cache/xdg'),
    'UV_CACHE_DIR': '/content/bw-uv-cache',
    'UV_PYTHON_INSTALL_DIR': '/content/bw-python',
    'NUMBA_CACHE_DIR': '/content/bw-numba-cache',
    'PYTHONUNBUFFERED': '1',
    'MPLBACKEND': 'Agg',
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
})

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

print('Persistent files:', ROOT)
```

## Cell 2 — Get the code onto Drive

This cell clones the correct repository into `MyDrive/BeyondWaveform/bynd-waveform`. It overrides the previous `PROJECT` variable, so Cell 1 from an older notebook is also supported. An existing Git clone is reused after checking its remote; it is not pulled or overwritten. The earlier `speech-gan` folder is left untouched. No packaging file is required in this cell.

```python
from pathlib import Path
import os
import subprocess

REPO = 'https://github.com/Kev-Alx/bynd-waveform.git'
ROOT = Path('/content/drive/MyDrive/BeyondWaveform')
PROJECT = ROOT / 'bynd-waveform'
ROOT.mkdir(parents=True, exist_ok=True)

if not (PROJECT / '.git').exists():
    subprocess.run(['git', 'clone', REPO, str(PROJECT)], check=True)
else:
    origin = subprocess.check_output(
        ['git', '-C', str(PROJECT), 'remote', 'get-url', 'origin'],
        text=True,
    ).strip()
    if origin.rstrip('/').removesuffix('.git') not in (
        REPO.removesuffix('.git'), 'git@github.com:Kev-Alx/bynd-waveform'
    ):
        raise RuntimeError(f'{PROJECT} belongs to a different repository: {origin}')
    print('Using existing clone:', PROJECT)

os.chdir(PROJECT)
print('Working directory:', PROJECT)
print('Files:', sorted(p.name for p in PROJECT.iterdir()))
```

## Cell 3 — Install the separate Python environment

Run again after a runtime reset. Downloads can take several minutes. Use this environment's `PY` executable for project commands; the notebook kernel keeps Colab's Python for Drive mounting and playback.

```python
import os
import sys
import shutil
import subprocess
from pathlib import Path

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

run('apt-get', 'update', '-qq')
run('apt-get', 'install', '-y', '-qq',
    'ffmpeg', 'libsndfile1', 'libgomp1', 'build-essential', 'pkg-config')
run(sys.executable, '-m', 'pip', 'install', '-q', 'uv')
UV = shutil.which('uv')
assert UV, 'uv installation did not put its executable on PATH.'
PY = '/content/bw-venv/bin/python'
if not Path(PY).exists():
    run(UV, 'venv', '--python', '3.11', '--seed', '/content/bw-venv')

run(PY, '-m', 'pip', 'install',
    'pip==24.3.1', 'setuptools==75.6.0', 'wheel==0.45.1')
run(PY, '-m', 'pip', 'install',
    'torch==2.4.1', 'torchaudio==2.4.1',
    '--index-url', 'https://download.pytorch.org/whl/cu121')

constraints = Path('/content/bw-constraints.txt')
constraints.write_text('\n'.join([
    'torch==2.4.1+cu121', 'torchaudio==2.4.1+cu121',
    'numpy==1.26.4', 'numba==0.60.0', 'llvmlite==0.43.0',
    'huggingface-hub==0.26.5', 'tokenizers==0.20.3',
    'fsspec==2024.9.0',
]) + '\n')
os.environ['PIP_CONSTRAINT'] = str(constraints)
run(PY, '-m', 'pip', 'install',
    'numpy==1.26.4', 'scipy==1.14.1', 'numba==0.60.0',
    'llvmlite==0.43.0', 'huggingface-hub==0.26.5',
    'tokenizers==0.20.3', 'safetensors==0.4.5', 'tiktoken==0.8.0',
    'librosa==0.10.2.post1', 'soundfile==0.12.1',
    'onnxruntime==1.20.1', 'matplotlib==3.9.3')
# Install the source scripts' dependencies even without packaging metadata.
run(PY, '-m', 'pip', 'install', '--no-build-isolation',
    'openai-whisper==20240930', 'transformers==4.46.3',
    'jiwer==3.0.5', 'torchmetrics==1.6.1',
    'PyYAML==6.0.2', 'requests==2.32.3', 'tqdm==4.67.1')
if (PROJECT / 'pyproject.toml').is_file():
    run(PY, '-m', 'pip', 'install', '--no-build-isolation', '-e', PROJECT)
else:
    print('Running scripts directly from source; pyproject.toml is not required.')
run(PY, '-m', 'pip', 'check')
run(PY, '-c', '''
import torch, torchaudio, whisper
from transformers import WavLMModel
assert torch.__version__ == '2.4.1+cu121'
assert torchaudio.__version__ == '2.4.1+cu121'
assert torch.cuda.is_available(), 'Select a GPU runtime, then rerun setup.'
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
''')
(ROOT / 'environment.freeze.txt').write_text(
    subprocess.check_output([PY, '-m', 'pip', 'freeze'], text=True)
)
```

## Cell 4 — Download models and a small corpus split

This downloads the WavLM encoder, HiFi-GAN vocoder, Whisper-small, and LibriSpeech dev-clean. It downloads the full dev-clean archive, even though the trial uses only three clips. Allow several GB of free Drive storage; archives and extracted data are both retained. Subsequent runs verify and reuse existing downloads. Extracting and indexing many files on mounted Drive may be slow.

```python
run(PY, 'scripts/download_models.py',
    '--components', 'feature', 'whisper', '--whisper', 'small')
run(PY, 'scripts/download_data.py',
    '--corpus', 'librispeech', '--splits', 'dev-clean')
```

## Cell 5 — Prepare three short pilot clips

This is a setup trial, not a validation or final test result. Three clips are selected from recordings up to eight seconds long. The manifest is rebuilt with the correct Drive paths.

```python
PILOT = 'data/manifests/colab-pilot-3.jsonl'
run(PY, 'scripts/prepare_data.py',
    '--corpus', 'librispeech', '--split', 'dev-clean',
    '--role', 'pilot', '--count', '3', '--max-seconds', '8',
    '--output', PILOT)
```

## Cell 6 — Run the one-clip, two-step smoke test

The existing smoke script generates one clip and evaluates the clean, reconstruction, and adversarial audio using Whisper-small.

```python
SMOKE = 'runs/colab-smoke'
run(PY, 'scripts/real_smoke.py',
    '--manifest', PILOT, '--config', 'configs/english.yaml',
    '--output', SMOKE, '--device', 'cuda')
```

## Cell 7 — Inspect scores and listen

```python
from IPython.display import Audio, display

result = json.loads((PROJECT / SMOKE / 'evaluation/summary.json').read_text())
print(json.dumps(result['summary'], indent=2))
for variant in ('clean', 'reconstruction', 'adversarial'):
    wav = next((PROJECT / SMOKE / 'audio' / variant).glob('*.wav'))
    print(variant, wav.name)
    display(Audio(filename=str(wav), normalize=False))
```

## Cell 8 — Optional: run 50 attack steps on all three clips

Change `RUN_LONGER_DEMO` to `True` after the smoke test works. This uses the repository's pilot-chosen learning rate of 0.05. It remains a pilot demonstration, not the full reproduction. Generation and evaluation run in separate processes to release GPU memory between them.

```python
RUN_LONGER_DEMO = False
if RUN_LONGER_DEMO:
    DEMO = 'runs/colab-pilot-50step'
    run(PY, 'scripts/generate.py',
        '--config', 'configs/english_lr005.yaml', '--manifest', PILOT,
        '--output', DEMO, '--device', 'cuda')
    run(PY, 'scripts/evaluate.py',
        '--run', DEMO, '--target', 'surrogate',
        '--output', f'{DEMO}/evaluation', '--device', 'cuda')
```

## Saved files and resuming

The main persistent locations are:

```text
MyDrive/BeyondWaveform/
  bynd-waveform/
    artifacts/                 model weights and pinned upstream sources
    data/                      archives, extracted audio, and manifests
    runs/colab-smoke/           one-clip smoke audio and evaluation
    runs/colab-pilot-50step/    optional three-clip demonstration
  cache/                       Hugging Face, Torch, and XDG caches
  environment.freeze.txt       installed package versions
```

Save the notebook itself into this folder using Colab. After a runtime reset, rerun Cells 1–3, then the download/preparation cells and the run you want to resume. Existing model/data downloads are reused. Generation verifies and reuses completed clips; an interrupted clip is recomputed. Evaluation restarts. Keep the config, manifest, model files, output directory, and Drive paths unchanged for a resumed run; use a new output directory for a different experiment.

If CUDA runs out of memory, use shorter pilot clips and a new manifest/output path, or select a GPU with more memory. Reducing the attack step count reduces time, but does not necessarily reduce peak memory.

For full reproduction, follow `docs/REPRODUCTION.md`: create disjoint pilot/validation/test manifests, audit them, perform validation-only selection with all three quality metrics, then evaluate the held-out test set. The optional quality scorers also use a `~/.torchmetrics` cache; link that directory to a Drive folder before enabling quality scoring if you want those downloads persisted too.

Colab runtime termination still stops computation; a Drive mount or a detached process does not keep the runtime alive. Mounted Drive can be slower and subject to file-operation quotas. This Drive-first layout favors persistence; larger runs may benefit from staging inputs on the VM while keeping results on Drive.

References: [Google Colab external data notebook](https://colab.research.google.com/notebooks/io.ipynb), [Colab FAQ: persistence, limits, and Drive performance](https://research.google.com/colaboratory/faq.html), [official PyTorch previous-version installation commands](https://docs.pytorch.org/get-started/previous-versions/), [uv Python environments](https://docs.astral.sh/uv/pip/environments/).

These cells were checked against the repository's CLI and Docker dependency pins. The notebook has not been executed in a hosted Colab runtime.
