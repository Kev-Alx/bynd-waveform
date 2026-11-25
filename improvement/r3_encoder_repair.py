"""Value-preserving autograd repair for pinned upstream WavLM positional conv.

The original encoder adds its positional convolution output into the same input
whose transposed view Conv1d saves for backward. Clone that positional-conv
input only in a differentiable call. Forward values/order, weights, layer,
normalization and every loss remain unchanged; the frozen implementation and
the observed failed preflight are preserved separately.
"""
from __future__ import annotations

from r3_waveform import DifferentiableWavLM6 as FrozenDifferentiableWavLM6


def _independent_saved_input(_module, inputs):
    value, *remaining = inputs
    if value.requires_grad:
        return (value.clone(), *remaining)
    return None


class DifferentiableWavLM6Repair(FrozenDifferentiableWavLM6):
    def forward(self, audio):
        handle = self.model.encoder.pos_conv.register_forward_pre_hook(_independent_saved_input)
        try:
            return super().forward(audio)
        finally:
            handle.remove()


# The runner's frozen import interface is retained in its separately named copy.
DifferentiableWavLM6 = DifferentiableWavLM6Repair
