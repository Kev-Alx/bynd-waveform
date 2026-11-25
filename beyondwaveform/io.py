"""Stable manifests, provenance, and explicit audio policy."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate utterance IDs: {path}")
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    for uid in ids:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", uid) or uid in {".", ".."}:
            raise ValueError(f"Unsafe utterance ID: {uid!r}")
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_audio(path, sample_rate=16000, max_seconds=30.0):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.mean(axis=1).copy())
    if rate != sample_rate:
        x = torchaudio.functional.resample(x, rate, sample_rate)
    if x.numel() < 720 or not torch.isfinite(x).all():
        raise ValueError(f"Audio is nonfinite or too short for two WavLM frames: {path}")
    if max_seconds is not None and x.numel() > round(max_seconds * sample_rate):
        raise ValueError(f"Audio exceeds {max_seconds}s; no silent truncation: {path}")
    if x.abs().max() > 1.00001:
        raise ValueError(f"Input exceeds [-1,1]; no implicit amplitude normalization: {path}")
    if x.square().sum() == 0:
        raise ValueError(f"Silent input: {path}")
    return x


def save_audio(path, audio, sample_rate=16000):
    """IEEE float WAV avoids quantization and implicit clipping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = audio.detach().cpu().float().reshape(-1)
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Cannot export empty or nonfinite audio")
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, x.numpy(), sample_rate, subtype="FLOAT")
    tmp.replace(path)
    return {"path": str(path.resolve()), "sha256": digest(path), "samples": x.numel(),
            "sample_rate": sample_rate, "peak": x.abs().max().item(), "format": "WAV/FLOAT"}


def environment():
    packages = {}
    for name in ["torch", "torchaudio", "numpy", "openai-whisper", "transformers", "torchmetrics", "jiwer"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
