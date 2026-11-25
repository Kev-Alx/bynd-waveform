"""CPU regressions on actual pinned upstream WavLM, not a toy content stub."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
import types

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r3_waveform import DifferentiableWavLM6 as OriginalAdapter
from r3_encoder_repair import DifferentiableWavLM6Repair

UPSTREAM = Path('/workspace/artifacts/sources/knn-vc-c616845c4e309e24d5927f15adbdf277a3d65358')


def actual_upstream_model(layer_norm_first=False):
    sys.path.insert(0, str(UPSTREAM))
    try:
        module = importlib.import_module('wavlm.WavLM')
    finally:
        sys.path.pop(0)
    config = module.WavLMConfig(dict(encoder_layers=6, encoder_embed_dim=16,
        encoder_ffn_embed_dim=32, encoder_attention_heads=2,
        conv_feature_layers='[(8,5,2),(8,3,2)]', conv_pos=8, conv_pos_groups=2,
        feature_grad_mult=0.0, dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, dropout_input=0.0, encoder_layerdrop=0.0,
        layer_norm_first=layer_norm_first))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(351)
        return module.WavLM(config).eval().requires_grad_(False)


def fixed_audio():
    return torch.randn(512, generator=torch.Generator().manual_seed(92)) * .1


def projected_loss(features):
    projection = torch.randn(features.shape, generator=torch.Generator().manual_seed(93))
    return (features * projection).sum()


@contextmanager
def out_of_place_positional_addition(encoder):
    """Independent reference: change only the offending upstream += to +.

    The actual module's other globals/operations remain the same. No upstream
    source file is edited. This avoids finite-difference noise from upstream
    LayerNorm/GELU's deliberate float32 casts even on a double model.
    """
    original = encoder.extract_features
    source = textwrap.dedent(inspect.getsource(type(encoder).extract_features))
    assert source.count('x += x_conv') == 1
    source = source.replace('x += x_conv', 'x = x + x_conv')
    namespace = dict(vars(importlib.import_module(type(encoder).__module__)))
    exec(compile(source, '<independent-out-of-place-WavLM-reference>', 'exec'), namespace)
    encoder.extract_features = types.MethodType(namespace['extract_features'], encoder)
    try:
        yield
    finally:
        encoder.extract_features = original


def test_actual_upstream_original_adapter_reproduces_saved_view_failure():
    model = actual_upstream_model()
    audio = fixed_audio().requires_grad_()
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        torch.autograd.grad(projected_loss(OriginalAdapter(model)(audio)), audio)


@pytest.mark.parametrize('layer_norm_first', [False, True])
def test_repair_preserves_native_values_and_has_finite_audio_gradient(layer_norm_first):
    model = actual_upstream_model(layer_norm_first)
    audio = fixed_audio()
    hooks_before = dict(model.encoder.pos_conv._forward_pre_hooks)
    with torch.no_grad():
        native = model.extract_features(audio[None], output_layer=6,
                                       ret_layer_results=False)[0][0]
    live = audio.clone().requires_grad_()
    repaired = DifferentiableWavLM6Repair(model)(live)
    assert torch.equal(native, repaired.detach())
    assert model.encoder.pos_conv._forward_pre_hooks == hooks_before
    gradient, = torch.autograd.grad(projected_loss(repaired), live)
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
    assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
    assert model.encoder.pos_conv._forward_pre_hooks == hooks_before


def test_repair_gradient_matches_actual_out_of_place_upstream_reference():
    model = actual_upstream_model()
    first_audio = fixed_audio().requires_grad_()
    repaired = DifferentiableWavLM6Repair(model)(first_audio)
    first_gradient, = torch.autograd.grad(projected_loss(repaired), first_audio)
    second_audio = fixed_audio().requires_grad_()
    with out_of_place_positional_addition(model.encoder):
        reference = OriginalAdapter(model)(second_audio)
        reference_gradient, = torch.autograd.grad(projected_loss(reference), second_audio)
    assert torch.equal(repaired, reference)
    torch.testing.assert_close(first_gradient, reference_gradient, rtol=1e-6, atol=1e-7)


def test_repair_removes_temporary_hook_when_encoder_forward_raises(monkeypatch):
    model = actual_upstream_model()
    adapter = DifferentiableWavLM6Repair(model)
    hooks_before = dict(model.encoder.pos_conv._forward_pre_hooks)
    def fail(_self, _audio):
        raise RuntimeError('synthetic forward failure after hook installation')
    monkeypatch.setattr(OriginalAdapter, 'forward', fail)
    with pytest.raises(RuntimeError, match='synthetic forward failure'):
        adapter(fixed_audio().requires_grad_())
    assert model.encoder.pos_conv._forward_pre_hooks == hooks_before
