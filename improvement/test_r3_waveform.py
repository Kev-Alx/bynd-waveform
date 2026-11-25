"""CPU-only mathematical and training-contract checks for the R3 implementation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from r3_waveform import (
    AMPLITUDE_LADDER, DifferentiableWavLM6, PenaltyWeights, SourceOnlyTrainer,
    TrainingConfig, WaveformResidualGenerator, calibrate_penalty_weights,
    linear_upsample, make_generator, multiresolution_stft_loss, project_waveform_residual,
    relative_feature_loss,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 720, 16_000, 16_031])
def test_exact_original_duration_and_registered_architecture(length):
    model = make_generator()
    audio = torch.linspace(-1.0, 1.0, length)
    with torch.no_grad():
        adversarial = model(audio)
    assert adversarial.shape == audio.shape
    assert torch.isfinite(adversarial).all()
    assert adversarial.abs().max() <= 1
    assert len(model.encoder) == len(model.decoder) == 6
    assert model.parameter_count == 1_590_833
    assert model.parameter_count <= 3_000_000
    assert [block[0].out_channels for block in model.encoder] == [16, 32, 64, 96, 128, 128]
    assert all(block[0].kernel_size == (15,) and block[0].stride == (2,)
               for block in model.encoder)
    assert not any(isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
                   for module in model.modules())


@pytest.mark.parametrize("linf", AMPLITUDE_LADDER)
def test_residual_bounds_and_audio_boundary_projection(linf):
    audio = torch.linspace(-1, 1, 2001)
    residual = torch.full_like(audio, 0.25)
    projected = project_waveform_residual(audio, residual, linf)
    assert projected.abs().max() <= linf + 1e-7
    assert projected.square().mean().sqrt() <= 0.003 + 1e-7
    assert (audio + projected).abs().max() <= 1
    assert projected[-1] == 0  # No positive perturbation beyond original peak.
    model = make_generator(linf=linf)
    actual = model(audio) - audio
    assert actual.abs().max() <= linf + 1e-7
    assert actual.square().mean().sqrt() <= 0.003 + 1e-7


def test_projection_zero_is_finite_differentiable_control():
    audio = torch.zeros(720)
    residual = torch.zeros(720, requires_grad=True)
    projected = project_waveform_residual(audio, residual, 0.003)
    assert torch.equal(projected, residual)
    gradient, = torch.autograd.grad(projected.sum(), residual)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_generator_gradient_reaches_original_audio_and_parameters():
    model = make_generator()
    audio = torch.linspace(-0.5, 0.5, 800, requires_grad=True)
    model(audio).square().mean().backward()
    assert torch.isfinite(audio.grad).all() and audio.grad.abs().sum() > 0
    assert model.encoder[0][0].weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0


def test_stft_matches_exact_formula_and_has_nonzero_audio_gradient():
    original = 0.1 * torch.sin(torch.arange(800) * 0.05)
    adversarial = (original + torch.linspace(-1e-4, 1e-4, 800)).requires_grad_()
    loss = multiresolution_stft_loss(adversarial, original)
    expected = []
    for size in (256, 512, 1024):
        window = torch.hann_window(size, periodic=True)
        arguments = dict(n_fft=size, hop_length=size // 4, win_length=size,
                         window=window, center=True, pad_mode="reflect", return_complex=True)
        expected.append(((torch.stft(adversarial, **arguments).abs() + 1e-5).log()
                         - (torch.stft(original, **arguments).abs() + 1e-5).log()).abs().mean())
    assert torch.equal(loss, torch.stack(expected).mean())
    gradient, = torch.autograd.grad(loss, adversarial)
    native_gradient, = torch.autograd.grad(torch.stack(expected).mean(), adversarial)
    # Reflection-gradient accumulation order can differ by float32 roundoff;
    # compare whole-gradient relative error instead of near-zero coordinates.
    assert (gradient - native_gradient).norm() / native_gradient.norm() < 1e-6
    assert loss > 0 and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert multiresolution_stft_loss(original, original) == 0


def test_content_exact_relative_squared_distance_and_detached_clean_features():
    clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    adversarial = (clean.detach() + 0.1).requires_grad_()
    loss = relative_feature_loss(adversarial, clean)
    assert torch.allclose(loss, torch.tensor(0.04 / 30.0))
    loss.backward()
    assert adversarial.grad.abs().sum() > 0
    assert clean.grad is None
    with pytest.raises(ValueError, match="alignment"):
        relative_feature_loss(adversarial[:, :1], clean)


@pytest.mark.parametrize("size", [2, 7, 16, 31])
def test_strict_linear_upsample_native_forward_and_cpu_derivative(size):
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).requires_grad_()
    actual = linear_upsample(hidden, size, deterministic_backward=True)
    native = nn.functional.interpolate(hidden, size=size, mode="linear", align_corners=False)
    weight = torch.arange(actual.numel(), dtype=torch.float32).reshape(actual.shape) / 10
    gradient, = torch.autograd.grad((actual * weight).sum(), hidden)
    native_gradient, = torch.autograd.grad((native * weight).sum(), hidden)
    assert torch.equal(actual, native)
    assert torch.equal(gradient, native_gradient)


class FakeEncoder(nn.Module):
    def forward(self, value, padding_mask=None, layer=None):
        assert padding_mask is None and layer == 5
        return value * 1.1, []


class FakeWavLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Conv1d(1, 4, 5, stride=2, bias=False)
        self.layer_norm = nn.LayerNorm(4)
        self.post_extract_proj = nn.Linear(4, 4)
        self.dropout_input = nn.Dropout(0.4)
        self.encoder = FakeEncoder()
        self.feature_grad_mult = 0

    def extract_features(self, source, output_layer, ret_layer_results=False):
        assert output_layer == 6 and not ret_layer_results
        with torch.no_grad():
            value = self.feature_extractor(source)
        value = self.dropout_input(self.post_extract_proj(self.layer_norm(value.transpose(1, 2))))
        return self.encoder(value, padding_mask=None, layer=5)[0], None


class FakeFeatureExtractor(nn.Module):
    """Pinned WavLM conv extractor itself inserts the mono channel dimension."""
    def __init__(self, convolution):
        super().__init__()
        self.convolution = convolution

    def forward(self, audio):
        return self.convolution(audio[:, None])


@pytest.mark.parametrize("normalize", [False, True])
def test_frozen_wavlm_adapter_forward_parity_and_audio_gradient(normalize):
    model = FakeWavLM().eval()
    # Use the actual wrapper interface: extract_features expects raw [B,T].
    convolution = model.feature_extractor
    model.feature_extractor = FakeFeatureExtractor(convolution)
    audio = torch.linspace(-0.3, 0.3, 800, requires_grad=True)
    reference_audio = nn.functional.layer_norm(audio, audio.shape) if normalize else audio
    with torch.no_grad():
        expected = model.extract_features(reference_audio[None], output_layer=6,
                                          ret_layer_results=False)[0][0]
    wrapper = SimpleNamespace(model=model, normalize=normalize, layer=6)
    adapter = DifferentiableWavLM6.from_encoder(wrapper)
    adapter.train()
    actual = adapter(audio)
    assert torch.equal(actual, expected)
    assert not adapter.training and not model.training
    assert model.feature_grad_mult == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    gradient, = torch.autograd.grad(actual.square().sum(), audio)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


class ToyContent(nn.Module):
    def forward(self, audio):
        return torch.stack((audio[::2], audio[1::2]), dim=-1)


def calibration_clips():
    return [{"id": f"dev-{index}", "audio": 0.1 * torch.sin(torch.arange(800) * 0.05 + index),
             "text": f"ORIGINAL TRANSCRIPT {index}"} for index in range(3)]


def test_calibration_three_fixed_probes_deterministic_ratios_and_nonzero_losses():
    seen = []
    def source_nll(audio, text):
        seen.append(text)
        return audio.square().sum()
    weights, report = calibrate_penalty_weights(calibration_clips(), source_nll,
                                                ToyContent(), linf=0.003)
    repeated, repeated_report = calibrate_penalty_weights(calibration_clips(), source_nll,
                                                         ToyContent(), linf=0.003)
    assert weights == repeated and report == repeated_report
    assert seen[:3] == [clip["text"] for clip in calibration_clips()]
    assert weights.stft == sorted(row["stft_ratio"] for row in report["clips"])[1]
    assert weights.content == sorted(row["content_ratio"] for row in report["clips"])[1]
    assert all(row["stft_gradient_norm"] > 1e-12 and row["content_gradient_norm"] > 1e-12
               for row in report["clips"])


def test_calibration_rejects_degenerate_penalty_without_inflating_weight():
    class ZeroContent(nn.Module):
        def forward(self, audio):
            return audio[:, None] * 0
    with pytest.raises(ValueError, match="Degenerate"):
        calibrate_penalty_weights(calibration_clips(), lambda audio, text: audio.square().sum(),
                                  ZeroContent(), linf=0.003)


def test_source_only_accumulation_clipping_and_unchanged_reference_text():
    model = make_generator()
    seen = []
    def source_nll(audio, text):
        seen.append(text)
        return audio.square().sum()
    trainer = SourceOnlyTrainer(model, source_nll, ToyContent(), PenaltyWeights(0.1, 1.0))
    audio = calibration_clips()[0]["audio"]
    before = model.head.weight.detach().clone()
    logs = [trainer.microbatch(audio, "The ORIGINAL Text.") for _ in range(4)]
    assert seen == ["The ORIGINAL Text."] * 4
    assert [log["optimizer_step"] for log in logs] == [False, False, False, True]
    assert trainer.updates == 1 and trainer.microbatches == 4
    assert logs[-1]["preclip_gradient_norm"] > 0
    assert not torch.equal(before, model.head.weight)
    assert all(log["generator_forward_passes"] == 3 for log in logs)
    assert trainer.optimizer.defaults["lr"] == 1e-4
    assert trainer.optimizer.defaults["betas"] == (0.9, 0.999)
    assert trainer.optimizer.defaults["weight_decay"] == 0


def test_checkpoint_selection_uses_source_wer_then_stft_then_earliest():
    trainer = SourceOnlyTrainer(make_generator(), lambda audio, text: audio.sum(),
                                ToyContent(), PenaltyWeights(1, 1))
    trainer.updates = 500
    assert not trainer.consider_checkpoint(source_wer=50, log_stft=0.1, admissible=False)
    assert trainer.consider_checkpoint(source_wer=20, log_stft=0.1, admissible=True)
    trainer.updates = 1000
    assert not trainer.consider_checkpoint(source_wer=20, log_stft=0.1, admissible=True)
    assert trainer.consider_checkpoint(source_wer=20, log_stft=0.05, admissible=True)
    assert trainer.consider_checkpoint(source_wer=21, log_stft=0.2, admissible=True)
    with pytest.raises(ValueError, match="improvement"):
        trainer.save_checkpoint("/tmp/r3-not-allowed.pt", manifest_sha256="a" * 64, calibration={})


def test_seed_initialization_reproducible_and_config_caps():
    first, second, other = make_generator(11), make_generator(11), make_generator(29)
    assert torch.equal(first.head.weight, second.head.weight)
    assert not torch.equal(first.head.weight, other.head.weight)
    with pytest.raises(ValueError):
        TrainingConfig(max_updates=12_001)
    with pytest.raises(ValueError):
        TrainingConfig(max_gpu_hours=8.1)
    with pytest.raises(ValueError):
        TrainingConfig(seed=12)


@pytest.mark.parametrize("checkpoint_mode", [False, True])
def test_native_source_subprocess_cli_and_environment_isolation(checkpoint_mode):
    from r3_real_runner import source_evaluation_command
    parent = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TASK_SENTINEL": "preserved",
              "PYTHONDONTWRITEBYTECODE": "0"}
    command, environment = source_evaluation_command("r3-cpu-test", "/workspace/improvement/records.jsonl",
        "/workspace/improvement/output", checkpoint_mode=checkpoint_mode, parent_environment=parent)
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment
    assert environment["TASK_SENTINEL"] == "preserved"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert parent["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert command[1] == "/workspace/improvement/source_evaluate.py"
    assert command[command.index("--name") + 1] == "r3-cpu-test"
    assert command[command.index("--records") + 1] == "/workspace/improvement/records.jsonl"
    assert command[command.index("--output-dir") + 1] == "/workspace/improvement/output"
    assert "--checkpoint-mode" not in command
    assert {argument for argument in command if argument.startswith("--")} == {
        "--name", "--records", "--output-dir"}


@pytest.mark.parametrize("name,records,output", [
    ("r3-bad/name", "/workspace/improvement/a.jsonl", "/workspace/improvement/out"),
    ("other-family", "/workspace/improvement/a.jsonl", "/workspace/improvement/out"),
    ("r3-valid", "/tmp/a.jsonl", "/workspace/improvement/out"),
    ("r3-valid", "/workspace/improvement/a.jsonl", "/tmp/out"),
])
def test_native_source_subprocess_rejects_unsafe_name_or_artifact_paths(name, records, output):
    from r3_real_runner import GateFailure, source_evaluation_command
    with pytest.raises(GateFailure):
        source_evaluation_command(name, records, output, parent_environment={})
