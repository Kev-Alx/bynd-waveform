"""Native Whisper differentiable frontend and audited lexical teacher forcing."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
import whisper
from whisper.tokenizer import get_tokenizer
from ..losses import lexical_nll
from ..io import digest


def teacher_forcing_tokens(tokenizer, text, device):
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    lexical = tokenizer.encode(text.strip())
    if not lexical or any(token >= tokenizer.eot for token in lexical):
        raise ValueError("Transcript must contain ordinary lexical tokens only")
    sequence = torch.tensor(prefix + lexical + [tokenizer.eot], device=device, dtype=torch.long)
    inputs, labels = sequence[:-1], sequence[1:]
    # Prefix is conditioning only. EOT and all other specials are unscored.
    mask = labels < tokenizer.eot
    return inputs[None], labels[None], mask[None]


class WhisperASR(nn.Module):
    def __init__(self, model, language="en", reduction="mean", decoding=None):
        super().__init__()
        self.model = model
        self.language = language
        self.reduction = reduction
        self.tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                       language=language, task="transcribe")
        self.decoding = decoding or {"beam_size": 5, "temperature": 0.0}

    def mel(self, audio):
        if audio.ndim != 1 or audio.numel() > whisper.audio.N_SAMPLES:
            raise ValueError("Whisper path accepts one utterance <=30 s; truncation is forbidden")
        audio = whisper.pad_or_trim(audio, whisper.audio.N_SAMPLES)
        return whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels)[None]

    def nll(self, audio, text):
        inputs, labels, mask = teacher_forcing_tokens(self.tokenizer, text, audio.device)
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
    adapter = WhisperASR(model, config.get("language", "en"), config.get("nll_reduction", "mean"), config.get("decoding"))
    adapter.requires_grad_(trainable)
    adapter.train(trainable)
    return adapter, {"checkpoint": str(path.resolve()), "sha256": digest(path),
                     "dimensions": vars(model.dims), "language": adapter.language,
                     "nll_reduction": adapter.reduction, "decoding": adapter.decoding}

