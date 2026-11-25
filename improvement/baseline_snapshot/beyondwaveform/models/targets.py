from __future__ import annotations
import importlib
import torch
from torch import nn
from .whisper import load_whisper


def load_extension(spec):
    """Explicit user-supplied baseline/defense adapters, never guessed recipes."""
    module, symbol = spec["factory"].split(":")
    return getattr(importlib.import_module(module), symbol)(**spec.get("kwargs", {}))


class CTCASR(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        from transformers import AutoModelForCTC, AutoProcessor
        if not config.get("revision"):
            raise ValueError("CTC checkpoint revision must be explicitly pinned")
        kwargs = {"revision": config["revision"], "trust_remote_code": False}
        self.processor = AutoProcessor.from_pretrained(config["model_id"], **kwargs)
        self.model = AutoModelForCTC.from_pretrained(config["model_id"], **kwargs).to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio):
        batch = self.processor(audio.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        ids = self.model(**batch).logits.argmax(-1)
        return self.processor.batch_decode(ids)[0]


def load_target(config, device):
    kind = config["kind"]
    if kind == "whisper":
        return load_whisper(config, device)
    if kind == "ctc":
        return CTCASR(config, device), config
    if kind == "extension":
        return load_extension(config), config
    raise ValueError(f"Unknown target type {kind}")
