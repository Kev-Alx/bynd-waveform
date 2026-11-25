"""Corpus error rates, optional quality models, and explicit defense adapters."""
from __future__ import annotations
import math
import re
import unicodedata
from pathlib import Path
import jiwer
import torch
import torchaudio
from tqdm import tqdm
from .assets import UTMOS_REVISION
from .io import digest
from .models.targets import load_extension


def normalize_text(text, language):
    """Engineering choice: NFKC, lowercase, punctuation-to-space; no number rewrite."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    text = " ".join(text.split())
    return "".join(text.split()) if language == "zh" else text


def error_counts(reference, hypothesis, language="en"):
    reference, hypothesis = normalize_text(reference, language), normalize_text(hypothesis, language)
    if not reference:
        raise ValueError("Empty reference after normalization is not scoreable")
    result = jiwer.process_characters(reference, hypothesis) if language == "zh" else jiwer.process_words(reference, hypothesis)
    return {"substitutions": result.substitutions, "deletions": result.deletions,
            "insertions": result.insertions, "reference_units": result.hits + result.substitutions + result.deletions,
            "reference_normalized": reference, "hypothesis_normalized": hypothesis}


def aggregate_errors(records):
    counts = {k: sum(row[k] for row in records) for k in ["substitutions", "deletions", "insertions", "reference_units"]}
    if counts["reference_units"] <= 0:
        raise ValueError("No reference units")
    counts["error_percent"] = 100 * (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["reference_units"]
    return counts


class QualityScorer:
    def __init__(self, artifact_root="artifacts", metrics=("dnsmos", "nisqa", "utmos")):
        self.metrics = list(metrics)
        if not set(self.metrics) <= {"dnsmos", "nisqa", "utmos"}:
            raise ValueError("Unknown quality metric")
        self.utmos = None
        if "utmos" in self.metrics:
            root = Path(artifact_root)
            source = root / "sources" / f"SpeechMOS-{UTMOS_REVISION}"
            if not source.exists():
                raise FileNotFoundError("Run download_models.py --components utmos")
            torch.hub.set_dir(str((root / "torchhub").resolve()))
            self.utmos = torch.hub.load(str(source), "utmos22_strong", source="local", pretrained=True).cpu().eval()

    @torch.no_grad()
    def __call__(self, audio, sample_rate=16000):
        x = audio.detach().cpu().float()[None]
        scores = {}
        if "dnsmos" in self.metrics:
            from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
            values = deep_noise_suppression_mean_opinion_score(x, sample_rate, personalized=False, device="cpu", num_threads=1)
            scores["dnsmos"] = values.reshape(-1, 4)[0, 3].item()  # OVRL, not P.808/SIG/BAK
        if "nisqa" in self.metrics:
            from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
            values = non_intrusive_speech_quality_assessment(x, sample_rate)
            scores["nisqa"] = values.reshape(-1, 5)[0, 0].item()  # overall MOS
        if "utmos" in self.metrics:
            scores["utmos"] = self.utmos(x, sample_rate).mean().item()
        if not all(math.isfinite(value) for value in scores.values()):
            raise FloatingPointError("Nonfinite quality score")
        return scores


def quality_provenance(artifact_root):
    root = Path(artifact_root)
    candidates = list((root / "torchhub" / "checkpoints").glob("*"))
    # TorchMetrics 1.6 uses these cache locations.
    for folder in [Path.home() / ".torchmetrics" / "DNSMOS", Path.home() / ".torchmetrics" / "NISQA"]:
        candidates.extend(folder.rglob("*"))
    return {"dnsmos_channel": "OVRL", "dnsmos_personalized": False, "nisqa_channel": "MOS",
            "utmos_revision": UTMOS_REVISION, "aggregation": "arithmetic utterance mean",
            "artifact_hashes": {str(p): digest(p) for p in tqdm(candidates, desc="Hashing quality models", unit="files") if p.is_file()}}


def defend(audio, config):
    name = config.get("kind", "none")
    if name == "none":
        return audio
    if name == "lpf":
        if "cutoff_hz" not in config:
            raise ValueError("LPF cutoff is unspecified by the paper; set it explicitly")
        cutoff = float(config["cutoff_hz"])
        if not 0 < cutoff < 8000:
            raise ValueError("Invalid LPF cutoff")
        # Explicit independent two-pole biquad; paper filter is unspecified.
        return torchaudio.functional.lowpass_biquad(audio, 16000, cutoff, Q=config.get("q", 0.70710678))
    raise ValueError(f"Defense {name!r} has no recovered recipe. Supply an extension target implementing transcribe(audio).")
