"""Native Whisper differentiable frontend and audited lexical teacher forcing."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
import whisper
from whisper.tokenizer import get_tokenizer
from ..losses import lexical_nll
from ..io import digest


def nll_transcript(text, policy="original"):
    """Explicit attack-target orthography; recognition references stay intact."""
    text = text.strip()
    if policy == "original":
        return text
    if policy == "lower":
        return text.lower()
    if policy == "sentence":
        return text.lower().capitalize()
    raise ValueError(f"Unknown NLL transcript policy: {policy}")


def teacher_forcing_prefix(tokenizer, timestamp_policy="no_timestamps"):
    if timestamp_policy == "no_timestamps":
        return list(tokenizer.sot_sequence_including_notimestamps)
    if timestamp_policy == "initial_timestamp":
        return [*tokenizer.sot_sequence, tokenizer.timestamp_begin]
    raise ValueError(f"Unknown NLL timestamp policy: {timestamp_policy}")


def teacher_forcing_tokens(tokenizer, text, device, timestamp_policy="no_timestamps"):
    prefix = teacher_forcing_prefix(tokenizer, timestamp_policy)
    lexical = tokenizer.encode(text.strip())
    if not lexical or any(token >= tokenizer.eot for token in lexical):
        raise ValueError("Transcript must contain ordinary lexical tokens only")
    sequence = torch.tensor(prefix + lexical + [tokenizer.eot], device=device, dtype=torch.long)
    inputs, labels = sequence[:-1], sequence[1:]
    # Prefix is conditioning only. EOT and all other specials are unscored.
    mask = labels < tokenizer.eot
    return inputs[None], labels[None], mask[None]


class WhisperASR(nn.Module):
    def __init__(self, model, language="en", reduction="mean", decoding=None, nll_transcript_policy="original",
                 nll_timestamp_policy="no_timestamps"):
        super().__init__()
        self.model = model
        self.language = language
        self.reduction = reduction
        nll_transcript("validation", nll_transcript_policy)
        self.nll_transcript_policy = nll_transcript_policy
        self.tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                       language=language, task="transcribe")
        teacher_forcing_prefix(self.tokenizer, nll_timestamp_policy)
        self.nll_timestamp_policy = nll_timestamp_policy
        self.decoding = decoding or {"beam_size": 5, "temperature": 0.0}

    def mel(self, audio):
        if audio.ndim != 1 or audio.numel() > whisper.audio.N_SAMPLES:
            raise ValueError("Whisper path accepts one utterance <=30 s; truncation is forbidden")
        audio = whisper.pad_or_trim(audio, whisper.audio.N_SAMPLES)
        return whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels)[None]

    def nll(self, audio, text):
        target = nll_transcript(text, self.nll_transcript_policy)
        inputs, labels, mask = teacher_forcing_tokens(self.tokenizer, target, audio.device, self.nll_timestamp_policy)
        if inputs.shape[-1] > self.model.dims.n_text_ctx:
            raise ValueError("Transcript exceeds Whisper context")
        return lexical_nll(self.model(self.mel(audio), inputs), labels, mask, self.reduction)

    def supervised_loss(self, audio, text):
        """Independent fine-tuning recipe: lexical tokens AND EOT, no prefix loss."""
        inputs, labels, mask = teacher_forcing_tokens(self.tokenizer, text, audio.device)
        if inputs.shape[-1] > self.model.dims.n_text_ctx:
            raise ValueError("Transcript exceeds Whisper context")
        mask = mask | (labels == self.tokenizer.eot)
        return lexical_nll(self.model(self.mel(audio), inputs), labels, mask, "mean")

    @torch.no_grad()
    def transcribe(self, audio):
        return whisper.transcribe(self.model, audio, language=self.language, task="transcribe",
                                  fp16=False, condition_on_previous_text=False, **self.decoding)["text"].strip()


def load_whisper(config, device, trainable=False):
    path = Path(config["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"Missing local Whisper checkpoint {path}; run download_models.py")
    model = whisper.load_model(str(path), device=device)
    adapter = WhisperASR(model, config.get("language", "en"), config.get("nll_reduction", "mean"),
                         config.get("decoding"), config.get("nll_transcript_policy", "original"),
                         config.get("nll_timestamp_policy", "no_timestamps"))
    adapter.requires_grad_(trainable)
    adapter.train(trainable)
    metadata = {"checkpoint": str(path.resolve()), "sha256": digest(path),
                     "dimensions": vars(model.dims), "language": adapter.language,
                     "nll_reduction": adapter.reduction, "decoding": adapter.decoding}
    if "nll_transcript_policy" in config:
        metadata["nll_transcript_policy"] = adapter.nll_transcript_policy
    if "nll_timestamp_policy" in config:
        metadata["nll_timestamp_policy"] = adapter.nll_timestamp_policy
    return adapter, metadata
