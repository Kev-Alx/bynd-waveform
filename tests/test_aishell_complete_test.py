import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from beyondwaveform.io import atomic_json, digest, save_audio, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from manage_aishell_epoch3 import remaining_official_test
from run_aishell_epoch3_complete_test import CompleteTestWorker
import run_aishell_epoch3 as frozen


def test_remaining_manifest_is_exact_complement_without_changed_references():
    official = [{"id": str(i), "audio_sha256": hashlib.sha256(str(i).encode()).hexdigest(), "text": "你好"}
                for i in range(7176)]
    completed = official[::2] + official[1::2][:1412]
    assert len(completed) == 5000
    remaining = remaining_official_test(completed, official)
    assert len(remaining) == 2176
    assert {r["id"] for r in completed}.isdisjoint(r["id"] for r in remaining)
    assert {r["id"] for r in completed + remaining} == {r["id"] for r in official}
    changed = copy.deepcopy(completed)
    changed[0]["text"] = "altered reference"
    with pytest.raises(AssertionError, match="references differ"):
        remaining_official_test(changed, official)
    duplicate = official[:-1] + [official[0]]
    with pytest.raises(AssertionError, match="Duplicate official test IDs"):
        remaining_official_test(completed, duplicate)


def test_detachable_worker_writes_final_combined_md_and_audit(tmp_path, monkeypatch):
    parent, output = tmp_path / "parent", tmp_path / "remaining"
    parent.mkdir();output.mkdir()
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"fixture without GPU models")
    rows = []
    for i in range(4):
        audio = tmp_path / f"source-{i}.wav"
        asset = save_audio(audio, torch.sin(torch.arange(1024) * (.02 + i * .001)) * .1)
        rows.append({"id": f"clip-{i}", "audio": str(audio), "audio_sha256": asset["sha256"], "text": "你好", "language": "zh"})
    config = {"surrogate": {"checkpoint": str(checkpoint), "language": "zh"}, "features": {}, "spectral": {},
              "attack": {"steps": 2, "rho": .1, "lambda_perc": .1, "learning_rate": .05}}
    for folder in [parent, output]:
        atomic_json(folder / "config.json", config)
    write_jsonl(parent / "manifest.jsonl", rows[:1])
    write_jsonl(output / "manifest.jsonl", rows[1:])
    official = tmp_path / "official.jsonl"
    write_jsonl(official, rows)
    atomic_json(parent / "plan.json", {"checkpoint_sha256": digest(checkpoint)})
    summary = {variant: {"count": 1,
                        "cer": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_units": 2, "error_percent": 50.},
                        "wer_jieba": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_units": 1, "error_percent": 100.}}
               for variant in frozen.VARIANTS}
    atomic_json(parent / "results.json", {"status": "complete", "summary": summary})
    atomic_json(parent / "independent-audit.json", {"status": "passed", "character_distance_checks": 3, "word_distance_checks": 3})
    tokenizer = ROOT / "artifacts/python/jieba-0.42.1"
    atomic_json(output / "plan.json", {
        "count": 3, "official_test_count": 4, "full_test_manifest": str(official),
        "checkpoint_sha256": digest(checkpoint), "source_hashes": {},
        "input_hashes": {str(p): digest(p) for p in [checkpoint, output / "config.json", output / "manifest.jsonl", official]},
        "tokenizer_path": str(tokenizer), "tokenizer_dictionary_sha256": digest(tokenizer / "jieba/dict.txt"),
        "initial_generation_seconds": .01, "initial_score_seconds": .01,
        "combined_parent": {"output": str(parent), "count": 1, "plan_sha256": digest(parent / "plan.json"),
                            "results_sha256": digest(parent / "results.json")}})
    isolated = tmp_path / "isolated-root"
    for name in ["mandarin-cer-full-dev", "mandarin-cer-corrected"]:
        (isolated / "reproduction-runs" / name).mkdir(parents=True)
    monkeypatch.setattr(frozen, "ROOT", isolated)
    import run_aishell_epoch3_complete_test as complete
    monkeypatch.setattr(complete, "ROOT", isolated)

    class ASR(nn.Module):
        def nll(self, audio, text):
            return (audio + .2).square().mean()

        def transcribe(self, audio):
            return "你好"

    class Encoder(nn.Module):
        def forward(self, audio):
            return audio[:1024].reshape(32, 32)

    class Vocoder(nn.Module):
        def forward(self, features):
            return torch.tanh(features.flatten() * .9)

    monkeypatch.setattr(frozen, "load_whisper", lambda *a: (ASR(), {"sha256": digest(checkpoint)}))
    monkeypatch.setattr(frozen, "load_feature_models", lambda *a: (Encoder(), Vocoder(), {}))
    prior_bytes = (parent / "results.json").read_bytes()
    worker = CompleteTestWorker(output, device="cpu")
    worker.run()
    result = json.loads((output / "combined-results.json").read_text())
    assert result["status"] == "complete" and result["count"] == 4
    for value in result["summary"].values():
        assert value["count"] == 4
        assert value["cer"]["error_percent"] == 12.5
        assert value["wer_jieba"]["error_percent"] == 25.
    report = (output / "report.md").read_text()
    assert "**complete**" in report and "12.5000%" in report and "25.0000%" in report
    audit = json.loads((output / "combined-audit.json").read_text())
    assert audit["status"] == "passed" and audit["count"] == 4
    assert audit["character_distance_checks"] == audit["word_distance_checks"] == 12
    assert (parent / "results.json").read_bytes() == prior_bytes
