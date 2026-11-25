"""Native Whisper encoder attention adapters with exact zero-update identity."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8.):
        super().__init__()
        if rank < 1:
            raise ValueError('Positive rank required')
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self,x):
        # Match Whisper Linear's input-dtype behavior under its native attention.
        base = F.linear(x,self.base.weight.to(x.dtype),
                        None if self.base.bias is None else self.base.bias.to(x.dtype))
        return base + F.linear(F.linear(x,self.a.to(x.dtype)),self.b.to(x.dtype)) * (self.alpha/self.rank)


def attach_encoder_lora(whisper_model, rank=4, alpha=8.):
    whisper_model.requires_grad_(False)
    names = []
    for i, block in enumerate(whisper_model.encoder.blocks):
        for name in ('query','value'):
            module = getattr(block.attn,name)
            if isinstance(module,LoRALinear):
                raise ValueError('Adapters already attached')
            setattr(block.attn,name,LoRALinear(module,rank,alpha))
            names.append(f'encoder.blocks.{i}.attn.{name}')
    return names


def adapter_state(model):
    return {name: parameter.detach().cpu().clone() for name,parameter in model.named_parameters()
            if name.endswith(('.a','.b'))}


def load_adapter_state(model,state):
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if name.endswith(('.a','.b'))}
    if set(state) != expected:
        raise ValueError('Adapter key mismatch')
    with torch.no_grad():
        for name,value in state.items():
            if parameters[name].shape != value.shape:
                raise ValueError(f'Adapter shape mismatch: {name}')
            parameters[name].copy_(value)
