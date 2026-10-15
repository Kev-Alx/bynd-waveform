import json
import pytest
import torch
from torch import nn
from beyondwaveform.io import save_audio, write_jsonl
from beyondwaveform.pipeline import generate, evaluate


class FakeASR(nn.Module):
    def nll(self, x, text):
        return (x + 0.2).square().mean()

    def transcribe(self, x):
        return "hello world"


class FakeEncoder(nn.Module):
    def forward(self, x):
        return x[:1024].reshape(32, 32)


class FakeVocoder(nn.Module):
    def forward(self, z):
        return torch.tanh(z.flatten() * 0.9)


def test_generate_resume_export_evaluate_and_config_collision(tmp_path, monkeypatch):
    import beyondwaveform.pipeline as pipeline
    monkeypatch.setattr(pipeline, "load_whisper", lambda *a: (FakeASR(), {"sha256": "same"}))
    monkeypatch.setattr(pipeline, "load_feature_models", lambda *a: (FakeEncoder(), FakeVocoder(), {"source": "fixture"}))
    monkeypatch.setattr(pipeline, "load_target", lambda *a: (FakeASR(), {"sha256": "same"}))
    path = tmp_path / "source.wav"
    source = save_audio(path, torch.randn(1024)*0.1)
    manifest = tmp_path / "pilot.jsonl"
    write_jsonl(manifest, [{"id": "one", "audio": str(path), "audio_sha256": source["sha256"],
                            "text": "hello world", "language": "en", "role": "pilot"}])
    config = {"surrogate": {"language": "en"}, "features": {}, "attack": {"steps": 2}}
    run = tmp_path / "run"
    first = generate(config, manifest, run, "cpu")
    resumed = generate(config, manifest, run, "cpu")
    assert first["generated_sha256"] == resumed["generated_sha256"]
    results = evaluate(run, {"kind": "whisper"}, tmp_path / "eval", "cpu")
    assert results["evaluation_scope"] == "surrogate"
    assert set(results["summary"]) == {"clean", "reconstruction", "adversarial"}
    assert all(v["error_percent"] == 0 for v in results["summary"].values())
    config["attack"]["rho"] = 0.5
    with pytest.raises(ValueError, match="different config"):
        generate(config, manifest, run, "cpu")

