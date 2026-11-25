import json
import os
import signal
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from beyondwaveform.io import atomic_json, digest, save_audio, write_jsonl
from beyondwaveform.training import finetune
from beyondwaveform.training_resume import resume_finetune


@pytest.fixture
def source(tmp_path, monkeypatch):
    import beyondwaveform.training as original
    import beyondwaveform.training_resume as continuation

    control = {"on_loss": lambda: None}

    class TinySurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1, bias=False)
            self.model.dims = SimpleNamespace(fixture=1)
            nn.init.constant_(self.model.weight, .8)

        def supervised_loss(self, audio, text):
            control["on_loss"]()
            return (self.model.weight[0, 0] - audio.mean()).square()

    def load(config, *args, **kwargs):
        model = TinySurrogate()
        if config.get("checkpoint"):
            model.model.load_state_dict(torch.load(config["checkpoint"], weights_only=True)["model_state_dict"])
        return model, {"fixture": True}

    monkeypatch.setattr(original, "load_whisper", load)
    monkeypatch.setattr(continuation, "load_whisper", load)
    manifests = []
    for group, values in [("finetune", [.1, .2, .3, .4]), ("validation", [.5, .6])]:
        rows = []
        for i, value in enumerate(values):
            audio = tmp_path / f"{group}-{i}.wav"
            asset = save_audio(audio, torch.full((1024,), value))
            rows.append({"id": f"{group}-{i}", "audio": str(audio), "audio_sha256": asset["sha256"],
                         "text": "你好", "corpus": "aishell", "split": "dev", "role": group, "language": "zh"})
        path = tmp_path / f"{group}.jsonl"
        write_jsonl(path, rows)
        manifests.append(path)
    config = {"model": {"language": "zh"}, "epochs": 2, "learning_rate": .05,
              "gradient_accumulation": 2, "save_epoch_checkpoints": True}
    before = tmp_path / "epoch-2"
    finetune(config, *manifests, before, device="cpu")
    config = {**config, "epochs": 3}
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    model = before / "epoch-002.pt"
    optimizer = before / "optimizer.pt"
    resume = {"checkpoint": {"path": str(model), "sha256": digest(model)},
              "optimizer": str(optimizer), "optimizer_sha256": digest(optimizer),
              "completed_epochs": 2, "next_epoch": 3, "history": json.loads((before / "history.json").read_text())}
    for field, path in zip(["training_config", "train_manifest", "validation_manifest"], [config_path, *manifests]):
        resume.update({field: str(path), field + "_sha256": digest(path)})
    resume_path = tmp_path / "resume.json"
    atomic_json(resume_path, resume)
    return resume_path, config, manifests, control


def assert_same_training(first, second):
    a = torch.load(first / "continuation.pt", weights_only=False)
    b = torch.load(second / "continuation.pt", weights_only=False)
    torch.testing.assert_close(a["model_state_dict"]["weight"], b["model_state_dict"]["weight"], rtol=0, atol=0)
    for parameter, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][parameter][key], rtol=0, atol=0)
    assert a["history"] == b["history"]


def test_epoch_3_restores_adamw_and_matches_uninterrupted_training(source, tmp_path):
    resume, config, manifests, _ = source
    output = tmp_path / "continued"
    result = resume_finetune(resume, output, device="cpu")
    assert result["status"] == "complete" and result["completed_epochs"] == 3
    reference = tmp_path / "uninterrupted"
    finetune(config, *manifests, reference, device="cpu")
    actual = torch.load(output / "continuation.pt", weights_only=False)
    expected = torch.load(reference / "last.pt", weights_only=True)
    torch.testing.assert_close(actual["model_state_dict"]["weight"], expected["model_state_dict"]["weight"], rtol=0, atol=0)
    saved = torch.load(reference / "optimizer.pt", weights_only=False)
    assert actual["optimizer"]["state"][0]["step"] == saved["optimizer"]["state"][0]["step"]


@pytest.mark.parametrize("stop_kind", ["runtime", "signal"])
def test_stop_finishes_accumulation_and_saved_state_resumes_exactly(source, tmp_path, stop_kind):
    resume, _, _, control = source
    baseline = tmp_path / "baseline"
    resume_finetune(resume, baseline, device="cpu")
    elapsed = [0.]

    def loss_hook():
        elapsed[0] += 1
        if stop_kind == "signal" and elapsed[0] == 1:
            os.kill(os.getpid(), signal.SIGUSR1)

    control["on_loss"] = loss_hook
    partial = tmp_path / "partial"
    result = resume_finetune(resume, partial, max_seconds=2.5 if stop_kind == "runtime" else 100,
                             save_reserve_seconds=1, device="cpu", clock=lambda: elapsed[0])
    assert result["status"] == "stopped"
    assert result["cursor"]["train_position"] == 2
    assert result["completed_epochs"] == 2
    assert result["stop_reason"] == ("runtime-limit-saving-reserve" if stop_kind == "runtime" else f"signal-{signal.SIGUSR1}")
    receipt = json.loads((partial / "resume-state.json").read_text())
    assert digest(receipt["bundle"]) == receipt["bundle_sha256"] and receipt["rng_state_saved"]
    control["on_loss"] = lambda: None
    completed = tmp_path / "resumed-again"
    result = resume_finetune(partial / "resume-state.json", completed, device="cpu")
    assert result["status"] == "complete"
    assert_same_training(baseline, completed)


def test_runtime_stop_during_validation_resumes_without_retraining(source, tmp_path):
    resume, _, _, control = source
    baseline = tmp_path / "baseline"
    resume_finetune(resume, baseline, device="cpu")
    elapsed = [0.]

    def advance():
        elapsed[0] += 1

    control["on_loss"] = advance
    partial = tmp_path / "partial-validation"
    result = resume_finetune(resume, partial, max_seconds=5.5, save_reserve_seconds=1,
                             device="cpu", clock=lambda: elapsed[0])
    assert result["status"] == "stopped"
    assert result["cursor"]["train_position"] == 4
    assert result["cursor"]["validation_position"] == 1
    control["on_loss"] = lambda: None
    completed = tmp_path / "completed-validation"
    resume_finetune(partial / "resume-state.json", completed, device="cpu")
    assert_same_training(baseline, completed)
