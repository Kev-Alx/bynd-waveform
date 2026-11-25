"""CPU-only actual-upstream reproduction of the WavLM positional-conv alias."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
from pathlib import Path
import sys
import torch

OUT = Path('/workspace/improvement')
SOURCE = Path('/workspace/artifacts/sources/knn-vc-c616845c4e309e24d5927f15adbdf277a3d65358')
sys.path[:0] = [str(OUT), str(SOURCE)]
from wavlm.WavLM import WavLM, WavLMConfig
from r3_waveform import DifferentiableWavLM6

torch.set_num_threads(1)
torch.manual_seed(351)
config = WavLMConfig(dict(encoder_layers=6, encoder_embed_dim=16,
    encoder_ffn_embed_dim=32, encoder_attention_heads=2,
    conv_feature_layers='[(8,5,2),(8,3,2)]', conv_pos=8, conv_pos_groups=2,
    feature_grad_mult=0.0, dropout=0.0, attention_dropout=0.0,
    activation_dropout=0.0, dropout_input=0.0, encoder_layerdrop=0.0))
model = WavLM(config).eval().requires_grad_(False).double()
adapter = DifferentiableWavLM6(model)
audio = torch.randn(512, dtype=torch.float64) * .1
with torch.no_grad():
    native = adapter(audio)
projection = torch.randn_like(native)
report = {'cpu_only': True, 'model': 'actual pinned upstream WavLM, tiny six-layer config',
    'config': config.__dict__, 'failure_reproduced': False}
live = audio.clone().requires_grad_()
try:
    output = adapter(live)
    torch.autograd.grad((output * projection).sum(), live)
except RuntimeError as error:
    report['failure_reproduced'] = 'modified by an inplace operation' in str(error)
    report['original_failure'] = str(error)

observed = []
def clone_saved_input(_module, inputs):
    observed.append(inputs[0].requires_grad)
    return (inputs[0].clone(), *inputs[1:]) if inputs[0].requires_grad else inputs

handle = model.encoder.pos_conv.register_forward_pre_hook(clone_saved_input)
try:
    live = audio.clone().requires_grad_()
    repaired = adapter(live)
    gradient, = torch.autograd.grad((repaired * projection).sum(), live)
    direction = torch.randn_like(audio)
    direction /= direction.norm()
    epsilon = 1e-5
    with torch.no_grad():
        upper = (adapter(audio + epsilon * direction) * projection).sum()
        lower = (adapter(audio - epsilon * direction) * projection).sum()
    finite_difference = float((upper - lower) / (2 * epsilon))
    derivative = float(gradient @ direction)
    report.update(forward_bitwise_equal=torch.equal(native, repaired.detach()),
        max_forward_difference=float((native - repaired.detach()).abs().max()),
        gradient_finite=bool(torch.isfinite(gradient).all()), gradient_norm=float(gradient.norm()),
        finite_difference=finite_difference, analytic_directional_derivative=derivative,
        derivative_relative_difference=abs(finite_difference - derivative)/max(abs(derivative),1e-10),
        hook_requires_grad_values=observed)
finally:
    handle.remove()
report['cuda_initialised'] = torch.cuda.is_initialized()
report['status'] = ('pass' if report['failure_reproduced'] and report['forward_bitwise_equal']
    and report['gradient_finite'] and report['gradient_norm'] > 0
    and report['derivative_relative_difference'] < 1e-3
    and not report['cuda_initialised'] else 'failed')
(OUT/'execution/r3-autograd-probe.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
