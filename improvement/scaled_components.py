"""Larger public sources, differentiable transforms, and an independent decoder."""
from functools import wraps
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import torchaudio.functional as AF
import execution_runtime as rt

RUN = rt.OUT / 'execution_scaled'


def checkpoint_blocks(blocks):
    """Value-preserving non-reentrant checkpointing of frozen input gradients."""
    for block in blocks:
        original = block.forward
        @wraps(original)
        def forward(*args, _original=original, **kwargs):
            tensors = [a for a in args if isinstance(a, torch.Tensor)] + [a for a in kwargs.values() if isinstance(a, torch.Tensor)]
            if torch.is_grad_enabled() and any(a.requires_grad for a in tensors):
                return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
            return _original(*args, **kwargs)
        block.forward = forward


def transform(audio, name):
    if name == 'identity': return audio
    if name == 'resample8k':
        return AF.resample(AF.resample(audio, 16000, 8000), 8000, 16000)[:len(audio)]
    if name == 'fir6k':
        from scipy.signal import firwin
        kernel = torch.as_tensor(firwin(129, 6000, fs=16000, window='hann'), dtype=audio.dtype, device=audio.device)
        padded = torch.cat((audio[1:65].flip(0), audio, audio[-65:-1].flip(0)))
        return F.conv1d(padded[None, None], kernel[None, None]).flatten()
    raise ValueError('Unregistered training transform')


class Data2VecSource(nn.Module):
    def __init__(self, directory, device='cuda'):
        super().__init__()
        from transformers import AutoModelForCTC, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCTC.from_pretrained(str(directory), local_files_only=True,
            trust_remote_code=False, attn_implementation='eager').to(device).eval().requires_grad_(False)
        self.eval().requires_grad_(False)

    def nll(self, audio, text):
        # Match the pinned feature extractor's utterance normalization. The
        # original reference remains unchanged in all evaluation records.
        normalized = (audio - audio.mean()) / torch.sqrt(audio.var(unbiased=False) + 1e-7)
        labels = self.tokenizer(text.upper().replace('\u2019', "'"), return_tensors='pt').input_ids[0]
        if labels.numel() == 0 or bool((labels == self.tokenizer.unk_token_id).any()):
            raise ValueError('CTC loss reference contains unknown symbols; do not silently edit it')
        logits = self.model(normalized[None], attention_mask=torch.ones(1, len(audio), dtype=torch.long, device=audio.device)).logits
        # CPU CTC has a deterministic real derivative, unlike some CUDA CTC
        # backward paths. Transfers/time count in the source budget.
        log_probs = logits.log_softmax(-1).transpose(0, 1).cpu()
        loss = F.ctc_loss(log_probs, labels, torch.tensor([log_probs.shape[0]]),
            torch.tensor([len(labels)]), blank=self.tokenizer.pad_token_id, reduction='sum', zero_infinity=False)
        return loss


def load_sources(plan):
    models, metadata = {}, {}
    for name, key in [('small', 'small.pt'), ('large', 'large-v2.pt')]:
        config = {**plan['base_config']['surrogate'], 'checkpoint': str(rt.ROOT / 'artifacts/whisper' / key)}
        native, meta = rt.load_whisper(config, 'cuda')
        model = rt.equivalent_surrogate(rt.freeze(native))
        checkpoint_blocks([*model.model.encoder.blocks, *model.model.decoder.blocks])
        models[name], metadata[name] = model, {**meta, 'activation_checkpointing': 'non-reentrant encoder/decoder residual blocks'}
    models['ctc'] = Data2VecSource(RUN / 'models/data2vec')
    metadata['ctc'] = {'repo': 'facebook/data2vec-audio-base-960h',
        'revision': '32331f3123e703528918aa688a9a38232d58c872', 'loss': 'CPU summed CTC',
        'reference_policy': 'uppercase, typographic apostrophe to ASCII; unknown labels fail',
        'normalization': 'utterance mean and variance+1e-7', 'parameters': sum(p.numel() for p in models['ctc'].parameters())}
    return models, metadata


def input_gradient(model, audio, text, augmentation):
    live = audio.detach().requires_grad_(True)
    loss = model.nll(transform(live, augmentation), text)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite source loss')
    gradient, = torch.autograd.grad(loss, live)
    if not torch.isfinite(gradient).all() or gradient.double().norm() == 0:
        raise ValueError('Invalid source waveform gradient')
    return gradient.detach(), float(loss.detach())


def calibrate_sources(models, audio, text):
    norms, losses = {}, {}
    for name, model in models.items():
        gradient, loss = input_gradient(model, audio, text, 'identity')
        norms[name] = float(gradient.double().norm()); losses[name] = loss
    scales = {name: norms['small'] / value for name, value in norms.items()}
    return scales, {'gradient_norms': norms, 'nlls': losses, 'fixed_scales': scales,
        'rule': 'Once per utterance at its unmodified reconstruction; match each audio-gradient norm to small'}


def ensemble_gradient(models, audio, text, scales, augmentation, single=False):
    weights = {'small': 1.} if single else {'small': .5, 'large': .25, 'ctc': .25}
    combined = torch.zeros_like(audio); rows = {}
    for name, weight in weights.items():
        gradient, value = input_gradient(models[name], audio, text, augmentation)
        combined.add_(gradient, alpha=weight * scales[name])
        rows[name] = {'nll': value, 'gradient_norm': float(gradient.double().norm()),
            'fixed_scale': scales[name], 'weight': weight, 'source_backward_calls': 1}
    if not torch.isfinite(combined).all() or combined.double().norm() == 0:
        raise ValueError('Invalid combined source gradient')
    return combined, rows


def replicate_concat(x, left, right):
    return torch.cat((x[..., :1].expand(*x.shape[:-1], left), x,
        x[..., -1:].expand(*x.shape[:-1], right)), dim=-1)


def stable_bigvgan_padding(model):
    # Frozen operator adaptation: identical replicated endpoint values without
    # nondeterministic CUDA replication-pad backward. Source remains preserved.
    for module in model.modules():
        if module.__class__.__name__ == 'UpSample1d':
            def up(x, m=module):
                channels = x.shape[1]
                x = replicate_concat(x, m.pad, m.pad)
                x = m.ratio * F.conv_transpose1d(x, m.filter.expand(channels, -1, -1), stride=m.stride, groups=channels)
                return x[..., m.pad_left:-m.pad_right]
            module.forward = up
        elif module.__class__.__name__ == 'LowPassFilter1d':
            if module.padding_mode != 'replicate': raise ValueError('Unexpected upstream padding contract')
            def low(x, m=module):
                channels = x.shape[1]
                if m.padding: x = replicate_concat(x, m.pad_left, m.pad_right)
                return F.conv1d(x, m.filter.expand(channels, -1, -1), stride=m.stride, groups=channels)
            module.forward = low


class BigVGAN16(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        source = RUN / 'models/bigvgan'; sys.path.insert(0, str(source))
        module = importlib.import_module('bigvgan'); env = importlib.import_module('env')
        if Path(module.__file__).resolve().parent != source: raise ValueError('Wrong BigVGAN implementation')
        config = rt.read(source / 'config.json')
        self.model = module.BigVGAN(env.AttrDict(config), use_cuda_kernel=False)
        self.model.load_state_dict(torch.load(source / 'bigvgan_generator.pt', map_location='cpu', weights_only=True)['generator'], strict=True)
        self.model.remove_weight_norm(); stable_bigvgan_padding(self.model)
        checkpoint_blocks(self.model.resblocks)
        self.model.to(device).eval().requires_grad_(False)
        self.h = self.model.h
        self.mel_function = importlib.import_module('meldataset').get_mel_spectrogram
        self.original_samples, self.native_samples = None, None
        self.last_clipped_samples = 0

    @torch.no_grad()
    def encode(self, audio):
        self.original_samples = len(audio)
        native = AF.resample(audio, 16000, 24000)
        self.native_samples = len(native)
        native = F.pad(native, (0, (-len(native)) % 256))
        return self.mel_function(native[None], self.h)[0].transpose(0, 1).contiguous()

    def forward(self, features):
        if self.original_samples is None: raise ValueError('Set original duration via encode first')
        native = self.model(features.transpose(0, 1)[None])[0, 0, :self.native_samples]
        result = AF.resample(native, 24000, 16000)[:self.original_samples]
        if result.numel() != self.original_samples: raise ValueError('Decoder changed registered duration')
        self.last_clipped_samples = int((result.abs() > 1).sum().detach())
        return result.clamp(-1, 1)
