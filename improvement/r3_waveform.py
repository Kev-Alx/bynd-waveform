"""Registered R3 waveform generator and source-only training primitives.

This module neither downloads data nor loads/evaluates ASR targets. Audio is one
native 16 kHz utterance; all reference text is passed through unchanged. Import
from this directory without modifying the independently audited baseline code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SAMPLE_RATE = 16_000
AMPLITUDE_LADDER = (0.001, 0.003, 0.01)
TRAINING_SEEDS = (11, 29, 47)
FFT_SIZES = (256, 512, 1024)
MAX_PARAMETERS = 3_000_000


class _DeterministicLinearUpsample(torch.autograd.Function):
    """Native linear forward with its native CPU derivative in strict CUDA.

    PyTorch2.4 rejects CUDA linear-interpolation backward under deterministic
    algorithms. The CPU derivative uses the identical operator/coordinate
    convention; transfers and CPU work count in measured training time.
    """
    @staticmethod
    def forward(context, hidden: Tensor, size: int) -> Tensor:
        context.input_shape = list(hidden.shape)
        context.output_size = [size]
        context.device = hidden.device
        return F.interpolate(hidden, size=size, mode="linear", align_corners=False)

    @staticmethod
    def backward(context, gradient: Tensor):
        cpu_gradient = torch.ops.aten.upsample_linear1d_backward.default(
            gradient.detach().cpu(), context.output_size, context.input_shape, False, None)
        return cpu_gradient.to(device=context.device), None


def linear_upsample(hidden: Tensor, size: int, *, deterministic_backward: bool | None = None) -> Tensor:
    if deterministic_backward is None:
        deterministic_backward = hidden.is_cuda and torch.are_deterministic_algorithms_enabled()
    if deterministic_backward:
        return _DeterministicLinearUpsample.apply(hidden, size)
    return F.interpolate(hidden, size=size, mode="linear", align_corners=False)


def _audio(audio: Tensor) -> None:
    if audio.ndim != 1 or audio.numel() == 0 or not audio.is_floating_point():
        raise ValueError("One nonempty floating-point mono waveform required")
    if not bool(torch.isfinite(audio).all()):
        raise ValueError("Audio must be finite")


def project_waveform_residual(
    original: Tensor, residual: Tensor, linf: float, rms_limit: float = 0.003
) -> Tensor:
    """Differentiably enforce Linf, RMS, and valid full-scale audio constraints.

    Projection is conservative rather than an exact Euclidean projection onto
    the intersection. RMS scaling and the final box clamp can only reduce each
    residual magnitude because the original audio is already in [-1, 1].
    """
    _audio(original)
    if residual.shape != original.shape or not bool(torch.isfinite(residual).all()):
        raise ValueError("Residual must be finite and match the original waveform")
    if not math.isfinite(linf) or not 0 < linf <= 1:
        raise ValueError("Linf bound must be finite and in (0,1]")
    if not math.isfinite(rms_limit) or not 0 < rms_limit <= 0.003:
        raise ValueError("Registered full-scale residual RMS is in (0,.003]")
    if bool((original.abs() > 1).any()):
        raise ValueError("Original audio must lie within [-1,1]; no normalization")
    bounded = residual.clamp(min=-linf, max=linf)
    # Avoid sqrt'(0)=inf: clamp before sqrt, including the all-zero control.
    rms = bounded.square().mean().clamp_min(torch.finfo(bounded.dtype).tiny).sqrt()
    scale = torch.clamp(rms_limit / rms, max=1.0)
    return (original + bounded * scale).clamp(-1.0, 1.0) - original


class WaveformResidualGenerator(nn.Module):
    """Six-level, kernel-15 U-Net with a bounded one-channel residual head.

    Encoder channels: 16/32/64/96/128/128. Each decoder stage upsamples
    linearly to the corresponding encoder length, concatenates that skip,
    then applies a kernel-15 Conv1d and GELU. The last stage upsamples to the
    padded input length and uses 16 channels; a kernel-1 tanh head emits mono.
    All Conv1d biases are enabled. Constant right padding to a multiple of 64
    is removed exactly; original speech is never resampled or time-warped.
    """

    def __init__(self, linf: float = 0.003, rms_limit: float = 0.003):
        super().__init__()
        if linf not in AMPLITUDE_LADDER:
            raise ValueError(f"Use a registered amplitude point: {AMPLITUDE_LADDER}")
        if not 0 < rms_limit <= 0.003:
            raise ValueError("RMS limit must be in (0,.003]")
        self.linf, self.rms_limit = float(linf), float(rms_limit)
        channels = (1, 16, 32, 64, 96, 128, 128)
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(a, b, 15, stride=2, padding=7), nn.GELU())
            for a, b in zip(channels[:-1], channels[1:])
        ])
        # Six decoder stages mirror six downsamplings; five have skip inputs.
        decoder_shapes = ((256, 128), (224, 96), (160, 64), (96, 32), (48, 16), (16, 16))
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(a, b, 15, padding=7), nn.GELU())
            for a, b in decoder_shapes
        ])
        self.head = nn.Conv1d(16, 1, 1)
        if self.parameter_count > MAX_PARAMETERS:
            raise ValueError("Architecture exceeds registered three-million-parameter ceiling")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def residual(self, audio: Tensor) -> Tensor:
        _audio(audio)
        length = audio.numel()
        padded = F.pad(audio[None, None], (0, (-length) % 64))
        skips = []
        hidden = padded
        for block in self.encoder:
            hidden = block(hidden)
            skips.append(hidden)
        for block, skip in zip(self.decoder[:-1], reversed(skips[:-1])):
            hidden = linear_upsample(hidden, size=skip.shape[-1])
            hidden = block(torch.cat((hidden, skip), dim=1))
        hidden = linear_upsample(hidden, size=padded.shape[-1])
        hidden = self.decoder[-1](hidden)
        raw = self.linf * torch.tanh(self.head(hidden)[0, 0, :length])
        return project_waveform_residual(audio, raw, self.linf, self.rms_limit)

    def forward(self, audio: Tensor) -> Tensor:
        return audio + self.residual(audio)

    def registration(self) -> dict:
        return {
            "architecture": "six-level Conv1d linear-upsample U-Net",
            "encoder_channels": [16, 32, 64, 96, 128, 128],
            "decoder_channels": [128, 96, 64, 32, 16, 16],
            "kernel": 15, "encoder_stride": 2, "head_kernel": 1,
            "skip": "concatenate corresponding encoder output, except final stage",
            "activation": "GELU; tanh residual head", "bias": True,
            "linear_upsample_backward": "native CPU derivative with device copies under strict CUDA; native otherwise",
            "batch_normalization": False, "dropout": False,
            "padding": "constant-zero right to multiple of 64; exact original-length crop",
            "sample_rate": SAMPLE_RATE, "linf": self.linf,
            "rms_limit": self.rms_limit, "parameters": self.parameter_count,
            "parameter_ceiling": MAX_PARAMETERS,
        }


def multiresolution_stft_loss(adversarial: Tensor, original: Tensor) -> Tensor:
    """Exact registered periodic-Hann log-magnitude L1, averaged over scales.

    Centered reflection uses concatenated slices, numerically identical to
    native reflect padding but permitting the strict CUDA backward path.
    """
    _audio(adversarial)
    _audio(original)
    if adversarial.shape != original.shape:
        raise ValueError("STFT reference duration must match exactly")
    if original.numel() <= max(FFT_SIZES) // 2:
        raise ValueError("Reflection-centered 1024-point STFT requires >512 samples")
    terms = []
    for size in FFT_SIZES:
        window = torch.hann_window(size, periodic=True, device=adversarial.device,
                                   dtype=adversarial.dtype)
        settings = dict(n_fft=size, hop_length=size // 4, win_length=size,
                        window=window, center=False, return_complex=True)
        def reflected_stft(audio):
            pad = size // 2
            reflected = torch.cat((audio[1:pad + 1].flip(0), audio,
                                   audio[-pad - 1:-1].flip(0)))
            return torch.stft(reflected, **settings)
        adv_stft = reflected_stft(adversarial)
        with torch.no_grad():
            clean_stft = reflected_stft(original.detach())
        terms.append(((adv_stft.abs() + 1e-5).log()
                      - (clean_stft.abs() + 1e-5).log()).abs().mean())
    return torch.stack(terms).mean()


def relative_feature_loss(adversarial_features: Tensor, clean_features: Tensor) -> Tensor:
    """Squared Frobenius layer-6 distance with a detached clean denominator."""
    if adversarial_features.shape != clean_features.shape:
        raise ValueError("Native WavLM layer-6 frame alignment changed")
    reference = clean_features.detach()
    return (adversarial_features - reference).square().sum() / (reference.square().sum() + 1e-8)


class DifferentiableWavLM6(nn.Module):
    """Frozen layer-6 WavLM preserving audio gradients through its conv stack.

    The pinned upstream checkpoint may set feature_grad_mult=0, making its
    extract_features() enter no_grad. Execute the same inference path directly
    and omit only GradMultiply/no_grad, which do not alter forward values.
    The underlying baseline module and its config remain unmodified.
    """

    def __init__(self, model: nn.Module, normalize: bool = False, layer: int = 6):
        super().__init__()
        if layer != 6:
            raise ValueError("Registered R3 content anchor is WavLM layer 6")
        self.model = model.eval().requires_grad_(False)
        self.normalize = bool(normalize)
        self.layer = layer
        super().train(False)

    @classmethod
    def from_encoder(cls, encoder: nn.Module) -> "DifferentiableWavLM6":
        return cls(encoder.model, normalize=encoder.normalize, layer=encoder.layer)

    def train(self, mode: bool = True):
        # Frozen anchor must stay in evaluation mode even when its owner trains.
        return super().train(False)

    def forward(self, audio: Tensor) -> Tensor:
        _audio(audio)
        if self.normalize:
            audio = F.layer_norm(audio, audio.shape)
        hidden = self.model.feature_extractor(audio[None]).transpose(1, 2)
        hidden = self.model.layer_norm(hidden)
        if self.model.post_extract_proj is not None:
            hidden = self.model.post_extract_proj(hidden)
        hidden = self.model.dropout_input(hidden)
        hidden, _ = self.model.encoder(hidden, padding_mask=None, layer=self.layer - 1)
        return hidden[0]


@dataclass(frozen=True)
class PenaltyWeights:
    stft: float
    content: float

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0 for value in (self.stft, self.content)):
            raise ValueError("Calibrated penalty weights must be finite and positive")


def _gradient_norm(loss: Tensor, audio: Tensor, *, keep: bool = False) -> float:
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError("Calibration loss must be a finite scalar")
    gradient, = torch.autograd.grad(loss, audio, retain_graph=keep)
    norm = float(torch.linalg.vector_norm(gradient.detach()))
    if not math.isfinite(norm):
        raise ValueError("Nonfinite calibration gradient")
    return norm


def calibrate_penalty_weights(
    clips: Sequence[Mapping], source_nll: Callable[[Tensor, str], Tensor],
    feature_encoder: nn.Module, *, linf: float, rms_limit: float = 0.003, seed: int = 11,
) -> tuple[PenaltyWeights, dict]:
    """Calibrate once on exactly three previously registered development clips.

    Each clip has ``id``, one-dimensional ``audio`` tensor and original ``text``.
    The caller binds source_nll to the frozen Whisper-small lexical SUM NLL;
    no ASR transcription, target query, or transcript modification occurs here.
    """
    if seed != 11 or len(clips) != 3 or len({clip["id"] for clip in clips}) != 3:
        raise ValueError("Calibration requires three distinct frozen dev clips and seed 11")
    if linf not in AMPLITUDE_LADDER:
        raise ValueError("Calibration must bind a registered Linf point")
    random_generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for clip in clips:
        original = clip["audio"].detach()
        _audio(original)
        signs = torch.randint(0, 2, original.shape, generator=random_generator, dtype=torch.int64)
        residual = (2 * signs - 1).to(device=original.device, dtype=original.dtype) * 1e-4
        probe = (original + project_waveform_residual(original, residual, linf, rms_limit)).detach().requires_grad_()
        nll = source_nll(probe, clip["text"])
        nll_norm = _gradient_norm(nll, probe)
        stft_norm = _gradient_norm(multiresolution_stft_loss(probe, original), probe)
        with torch.no_grad():
            clean_features = feature_encoder(original).detach()
        content_norm = _gradient_norm(relative_feature_loss(feature_encoder(probe), clean_features), probe)
        if nll_norm <= 0 or stft_norm <= 1e-12 or content_norm <= 1e-12:
            raise ValueError(f"Degenerate R3 calibration on {clip['id']}; do not inflate weights")
        ratios = (nll_norm / stft_norm, nll_norm / content_norm)
        if not all(math.isfinite(value) and value > 0 for value in ratios):
            raise ValueError("Nonfinite R3 gradient ratio; calibration failed")
        rows.append({"id": clip["id"], "samples": original.numel(),
                     "source_sum_nll_gradient_norm": nll_norm,
                     "stft_gradient_norm": stft_norm, "content_gradient_norm": content_norm,
                     "stft_ratio": ratios[0], "content_ratio": ratios[1],
                     "probe_linf": float((probe.detach() - original).abs().max()),
                     "probe_rms": float((probe.detach() - original).square().mean().sqrt())})
    weights = PenaltyWeights(*(sorted(row[key] for row in rows)[1]
                              for key in ("stft_ratio", "content_ratio")))
    for row in rows:
        row["weighted_stft_gradient_norm"] = weights.stft * row["stft_gradient_norm"]
        row["weighted_content_gradient_norm"] = weights.content * row["content_gradient_norm"]
    return weights, {"status": "completed", "seed": seed, "probe_amplitude": 1e-4,
                     "probe_distribution": "seeded CPU torch Rademacher, in clip order",
                     "linf": linf, "rms_limit": rms_limit, "weights": asdict(weights),
                     "selection": "median of three source_sum_NLL/penalty audio-gradient norms",
                     "clips": rows}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 11
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    accumulation_steps: int = 4
    gradient_clip: float = 1.0
    checkpoint_interval: int = 500
    max_updates: int = 12_000
    max_gpu_hours: float = 8.0

    def __post_init__(self):
        if self.seed not in TRAINING_SEEDS:
            raise ValueError(f"Registered training seeds: {TRAINING_SEEDS}")
        fixed = (self.learning_rate == 1e-4 and self.weight_decay == 0
                 and self.accumulation_steps == 4 and self.gradient_clip == 1
                 and self.checkpoint_interval == 500)
        if not fixed or not 0 < self.max_updates <= 12_000 or not 0 < self.max_gpu_hours <= 8:
            raise ValueError("Training config changes the registered R3 optimizer/budget")


def make_generator(seed: int = 11, linf: float = 0.003) -> WaveformResidualGenerator:
    """Seed initialization before construction, restoring unrelated RNG state."""
    if seed not in TRAINING_SEEDS:
        raise ValueError("Use registered seed 11, 29, or 47")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return WaveformResidualGenerator(linf=linf)


class SourceOnlyTrainer:
    """Batch-one accumulation with sequential loss backward passes.

    Recomputing deterministic generator output separately for each loss keeps
    source-ASR and WavLM activation graphs out of memory simultaneously. All
    passes count toward measured training cost. The caller owns source-only
    development scoring/checkpoint selection and the persisted data manifest.
    No final-target module is accepted or called.
    """

    def __init__(self, generator: WaveformResidualGenerator,
                 source_nll: Callable[[Tensor, str], Tensor], feature_encoder: nn.Module,
                 weights: PenaltyWeights, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.generator, self.source_nll = generator, source_nll
        self.feature_encoder = feature_encoder.eval().requires_grad_(False)
        self.weights = weights
        self.optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-4,
                                           betas=(0.9, 0.999), weight_decay=0)
        self.optimizer.zero_grad(set_to_none=True)
        self.microbatches = self.updates = 0
        self.started = time.monotonic()
        self.best_checkpoint_key: tuple[float, float, int] | None = None

    @property
    def checkpoint_due(self) -> bool:
        return self.updates > 0 and self.updates % self.config.checkpoint_interval == 0

    @property
    def budget_exhausted(self) -> bool:
        # Conservative wall-time gate; CUDA event accounting belongs to runtime.
        return self.updates >= self.config.max_updates or (
            time.monotonic() - self.started >= self.config.max_gpu_hours * 3600)

    def microbatch(self, original: Tensor, original_text: str,
                   clean_features: Tensor | None = None) -> dict:
        if self.budget_exhausted:
            raise RuntimeError("Registered R3 update/time cap reached")
        _audio(original)
        if original.numel() < 720 or original.numel() > 30 * SAMPLE_RATE:
            raise ValueError("Use untruncated registered utterances of 720..480000 samples")
        if not original_text.strip():
            raise ValueError("Original reference transcript is required")
        self.generator.train()
        denominator = self.config.accumulation_steps
        nll = self.source_nll(self.generator(original), original_text)
        if nll.ndim != 0 or not bool(torch.isfinite(nll)):
            raise ValueError("Source lexical sum NLL is nonfinite or not scalar")
        (-nll / denominator).backward()
        stft = multiresolution_stft_loss(self.generator(original), original)
        (self.weights.stft * stft / denominator).backward()
        if clean_features is None:
            with torch.no_grad():
                clean_features = self.feature_encoder(original).detach()
        content = relative_feature_loss(self.feature_encoder(self.generator(original)), clean_features)
        (self.weights.content * content / denominator).backward()
        self.microbatches += 1
        gradient_norm = None
        stepped = self.microbatches % denominator == 0
        if stepped:
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.gradient_clip, error_if_nonfinite=True))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
        return {"source_sum_nll": float(nll.detach()), "stft_loss": float(stft.detach()),
                "content_loss": float(content.detach()), "weights": asdict(self.weights),
                "optimizer_step": stepped, "updates": self.updates,
                "microbatches": self.microbatches, "preclip_gradient_norm": gradient_norm,
                "source_backward_passes": 1, "content_backward_passes": 1,
                "generator_forward_passes": 3, "batch_size": 1}

    def consider_checkpoint(self, *, source_wer: float, log_stft: float,
                            admissible: bool) -> bool:
        """Highest admissible source WER, then lower STFT, then earliest update."""
        if not self.checkpoint_due:
            raise ValueError("Source-only checkpoint selection occurs every 500 updates")
        if not admissible:
            return False
        if not math.isfinite(source_wer) or not math.isfinite(log_stft):
            raise ValueError("Checkpoint metrics must be finite")
        key = (-source_wer, log_stft, self.updates)
        if self.best_checkpoint_key is None or key < self.best_checkpoint_key:
            self.best_checkpoint_key = key
            return True
        return False

    def save_checkpoint(self, destination: str | Path, *, manifest_sha256: str,
                        calibration: Mapping) -> Path:
        """Persist only a completed source-only update inside improvement/."""
        path = Path(destination).resolve()
        root = Path(__file__).resolve().parent
        if not path.is_relative_to(root):
            raise ValueError("All new R3 artifacts must stay within improvement/")
        if self.microbatches % self.config.accumulation_steps:
            raise ValueError("Save between optimizer updates, not mid accumulation")
        if len(manifest_sha256) != 64:
            raise ValueError("Bind the frozen T1 manifest SHA256")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": self.generator.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "config": asdict(self.config),
                    "architecture": self.generator.registration(), "weights": asdict(self.weights),
                    "updates": self.updates, "microbatches": self.microbatches,
                    "manifest_sha256": manifest_sha256, "calibration": dict(calibration),
                    "best_checkpoint_key": self.best_checkpoint_key,
                    "source_only": True}, path)
        return path
