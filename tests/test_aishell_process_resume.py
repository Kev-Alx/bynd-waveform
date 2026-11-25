import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from beyondwaveform.io import atomic_json, digest, save_audio, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL, signal.SIGALRM])
@pytest.mark.parametrize("worker_script", ["run_aishell_epoch3.py", "run_aishell_epoch3_full_run.py"])
def test_killed_worker_reuses_completed_audio_and_transcriptions(tmp_path, stop_signal, worker_script):
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint, no GPU model used")
    rows = []
    for index in range(3):
        audio = tmp_path / f"input-{index}.wav"
        asset = save_audio(audio, torch.sin(torch.arange(1024) * (.02 + index * .001)) * .1)
        rows.append({"id": f"clip-{index}", "audio": str(audio), "audio_sha256": asset["sha256"],
                     "text": "你好", "language": "zh", "role": "test"})
    write_jsonl(output / "manifest.jsonl", rows)
    config = {"seed": 2026, "surrogate": {"checkpoint": str(checkpoint), "language": "zh"}, "features": {},
              "attack": {"steps": 2, "rho": .1, "lambda_perc": .1, "learning_rate": .05}, "spectral": {}}
    atomic_json(output / "config.json", config)
    tokenizer = ROOT / "artifacts/python/jieba-0.42.1"
    plan = {"count": 3, "checkpoint_sha256": digest(checkpoint), "source_hashes": {},
            "input_hashes": {str(path): digest(path) for path in [output / "manifest.jsonl", output / "config.json", checkpoint]},
            "tokenizer_path": str(tokenizer), "tokenizer_dictionary_sha256": digest(tokenizer / "jieba/dict.txt"),
            "initial_generation_seconds": .6, "initial_score_seconds": .001}
    atomic_json(output / "plan.json", plan)
    harness = tmp_path / "harness.py"
    harness.write_text('''
import importlib.util,json,pathlib,signal,sys,time
import torch
from torch import nn
spec=importlib.util.spec_from_file_location("test_worker",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output=pathlib.Path(sys.argv[2]);controls=getattr(module,"frozen",module)
controls.ROOT=output/"isolated-root"
for name in ["mandarin-cer-full-dev","mandarin-cer-corrected"]:
 (controls.ROOT/"reproduction-runs"/name).mkdir(parents=True,exist_ok=True)
plan=json.loads((output/"plan.json").read_text())
class ASR(nn.Module):
 def nll(self,audio,text):return (audio+.2).square().mean()
 def transcribe(self,audio):return "你好"
class Encoder(nn.Module):
 def forward(self,audio):return audio[:1024].reshape(32,32)
class Vocoder(nn.Module):
 def forward(self,features):return torch.tanh(features.flatten()*.9)
def load_asr(*a):
 if hasattr(module,"FullRunWorker"):assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
 return ASR(),{"sha256":plan["checkpoint_sha256"]}
controls.load_whisper=load_asr
controls.load_feature_models=lambda *a:(Encoder(),Vocoder(),{"fixture":True})
original=controls.feature_attack
def slowed(*a,**kw):
 time.sleep(.6)
 return original(*a,**kw)
controls.feature_attack=slowed
if hasattr(module,"FullRunWorker"):module.FullRunWorker(output,device="cpu").run()
else:module.Worker(output,time.time()+60,device="cpu").run()
''')

    def launch():
        return subprocess.Popen([sys.executable, str(harness), str(ROOT / "scripts" / worker_script), str(output)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                start_new_session=True)

    first = launch()
    try:
        limit = time.monotonic() + 30
        while len(list((output / "scores").glob("*.json"))) < 3:
            assert first.poll() is None, first.stderr.read().decode()
            assert time.monotonic() < limit, "First record did not finish"
            time.sleep(.02)
        os.kill(first.pid, stop_signal)
        first.wait(timeout=10)
        records = [json.loads(path.read_text()) for path in (output / "generation/records").glob("*.json")]
        assert records
        committed = list((output / "generation/records").glob("*.json")) + list((output / "scores").glob("*.json"))
        committed += [Path(asset["path"]) for row in records for asset in row["audio"].values()]
        before = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in committed}
        if stop_signal != signal.SIGKILL:
            job = json.loads((output / "job.json").read_text())
            assert job["status"] == "stopped"
            if stop_signal == signal.SIGALRM:
                assert job["stop_reason"] == ("session time limit" if worker_script.endswith("full_run.py") else "five-hour session limit")
        second = launch()
        try:
            _, error = second.communicate(timeout=30)
            assert second.returncode == 0, error.decode()
        finally:
            if second.poll() is None:
                second.kill();second.wait()
        job = json.loads((output / "job.json").read_text())
        assert job["status"] == "complete" and job["reused_generated"] >= len(records)
        assert job["reused_transcriptions"] >= 3
        if worker_script.endswith("full_run.py"):
            assert job["deadline_taipei"] is None and job["runtime_limit"] == "none"
            progress = json.loads((output / "progress.json").read_text())
            assert progress["session_deadline_taipei"] is None
            status = (output / "JOB_STATUS.md").read_text()
            assert "No runtime limit" in status and "another bounded session" not in status
        for path, expected in before.items():
            assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == expected
        results = json.loads((output / "results.json").read_text())
        assert all(summary["count"] == 3 for summary in results["summary"].values())
        assert json.loads((output / "independent-audit.json").read_text())["character_distance_checks"] == 9
    finally:
        if first.poll() is None:
            first.kill();first.wait()
