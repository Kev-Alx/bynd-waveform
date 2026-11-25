"""Validate an equivalent reflection execution path without changing production.

No decoding, MOS inference, loss-weight tuning or validation/test expansion.
Run in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from beyondwaveform.attacks import freeze, project_relative_ball_, relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.deterministic_frontend import high_frequency_fraction, whisper_mel
from beyondwaveform.io import atomic_json, digest, load_audio, read_jsonl, seed_everything
from beyondwaveform.losses import CleanReference, temporal_variation, high_frequency_fraction as native_hf
from beyondwaveform.models.whisper import WhisperASR, load_whisper
from probe_feature_gradients import load_frozen_vocoder
from probe_numerical_repeatability import backend_manifest, backend_mode, compare_gradients, tensor_value_sha256
from parallel_jobs import now
import run_quality_research as protocol


class ExplicitReflectionWhisper(WhisperASR):
    """Same model/targets and native recognition; explicit padding for NLL only."""
    def mel(self, audio):
        return whisper_mel(audio, self.model.dims.n_mels)


def equivalent_surrogate(native):
    return freeze(ExplicitReflectionWhisper(native.model, native.language, native.reduction,
        native.decoding, native.nll_transcript_policy, native.nll_timestamp_policy))


class ExplicitReflectionReference(CleanReference):
    """Retain exact original-clean denominators; change generated HF padding."""
    def terms(self, z, audio, temporal=True, high_frequency=True):
        zero = z.new_zeros(())
        tv = temporal_variation(z)/(self.tv+self.eps) if temporal else zero
        hf = high_frequency_fraction(audio, self.spectral)/(self.hf+self.eps) if high_frequency else zero
        return tv, hf


def objective(q, delta, text, vocoder, surrogate, attack, reference):
    audio = vocoder(q+delta)
    nll = surrogate.nll(audio, text)
    tv, hf = reference.terms(q+delta, audio)
    loss = -nll+attack.lambda_perc*(tv+attack.alpha_hf*hf)
    protocol.require(torch.isfinite(loss).item(), 'Nonfinite full objective')
    return loss, torch.stack([loss.detach(), nll.detach(), tv.detach(), hf.detach()])


def derivative(q, x, delta, text, vocoder, surrogate, attack, spectral):
    live = delta.detach().clone().requires_grad_()
    reference = ExplicitReflectionReference(q, x, spectral, attack.eps)
    loss, values = objective(q, live, text, vocoder, surrogate, attack, reference)
    gradient, = torch.autograd.grad(loss, live)
    protocol.require(torch.isfinite(gradient).all().item(), 'Nonfinite derivative')
    return values.cpu(), gradient.detach().cpu()


def save_tensor(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = value.detach().cpu().contiguous()
    torch.save(cpu, path)
    return {'path':str(path.resolve()), 'file_sha256':digest(path), 'value_sha256':tensor_value_sha256(cpu)}


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return {'tensor_sha256':tensor_value_sha256(value)}
    if isinstance(value, dict):
        return {str(k):tensor_tree(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(v) for v in value]
    return value


@torch.no_grad()
def forward_conformance(q, x, delta, text, vocoder, native, stable, attack, spectral):
    audio = vocoder(q+delta)
    native_reference = CleanReference(q, x, spectral, attack.eps)
    stable_reference = ExplicitReflectionReference(q, x, spectral, attack.eps)
    old_loss, old_values = objective(q, delta, text, vocoder, native, attack, native_reference)
    new_loss, new_values = objective(q, delta, text, vocoder, stable, attack, stable_reference)
    values = {'original_audio_hf':(native_hf(x, spectral), high_frequency_fraction(x, spectral)),
        'generated_hf':(native_hf(audio, spectral), high_frequency_fraction(audio, spectral)),
        'generated_mel':(native.mel(audio), stable.mel(audio)), 'full_objective_scalars':(old_values,new_values)}
    rows = {name: {'bitwise_equal':tensor_value_sha256(a)==tensor_value_sha256(b),
        'max_abs_difference':(a-b).abs().max().item(), 'native_value_sha256':tensor_value_sha256(a),
        'explicit_value_sha256':tensor_value_sha256(b)} for name,(a,b) in values.items()}
    return {'checks':rows, 'all_bitwise_equal':all(r['bitwise_equal'] for r in rows.values())}


def fixed_probe(q, x, delta, text, vocoder, native, stable, attack, spectral, seed, directory, old_state):
    conformance = forward_conformance(q, x, delta, text, vocoder, native, stable, attack, spectral)
    repeats, gradients = [], []
    for index in range(3):
        seed_everything(seed)
        values, gradient = derivative(q,x,delta,text,vocoder,stable,attack,spectral)
        artifact = save_tensor(gradient, directory / f'repeat-{index}.pt')
        repeats.append({'repeat':index, 'loss_values':values.tolist(),
            'loss_value_sha256':tensor_value_sha256(values), 'gradient':artifact})
        gradients.append(gradient)
    pairs = []
    for left,right in itertools.combinations(range(3),2):
        comparison = compare_gradients(gradients[left], gradients[right])
        equal = comparison['raw_value_sha256_equal'] and repeats[left]['loss_value_sha256'] == repeats[right]['loss_value_sha256']
        pairs.append({'left':left,'right':right,'comparison':comparison,'bitwise_equal':equal})
    old_artifact = old_state['repeats'][0]['gradient_artifact']
    protocol.require(digest(old_artifact['path']) == old_artifact['file_sha256'], 'Changed native math comparison gradient')
    old_gradient = torch.load(old_artifact['path'], map_location='cpu', weights_only=True)
    numerical = compare_gradients(old_gradient, gradients[0])
    # Predeclared float32 equivalence diagnostic across execution policies;
    # native math is itself nondeterministic and is not a bitwise reference.
    agrees = numerical['relative_l2_difference_to_left'] <= 1e-4 and numerical['max_abs_difference'] <= 1e-5
    return {'forward_conformance':conformance, 'repeats':repeats, 'pairs':pairs,
        'native_math_gradient_comparison':numerical, 'native_math_gradient_within_tolerance':agrees,
        'native_math_gradient_artifact':old_artifact,
        'passed':conformance['all_bitwise_equal'] and all(p['bitwise_equal'] for p in pairs) and agrees}


def trajectory(q, x, text, vocoder, surrogate, attack, spectral, directory):
    q, x = q.detach(), x.detach()
    delta = torch.zeros_like(q, requires_grad=True)
    reference = ExplicitReflectionReference(q, x, spectral, attack.eps)
    optimizer = torch.optim.Adam([delta], lr=attack.learning_rate)
    trace = []
    for step in range(1, 51):
        optimizer.zero_grad(set_to_none=True)
        loss, values = objective(q, delta, text, vocoder, surrogate, attack, reference)
        gradient, = torch.autograd.grad(loss, delta)
        protocol.require(torch.isfinite(gradient).all().item(), 'Nonfinite trajectory gradient')
        delta.grad = gradient
        optimizer.step()
        project_relative_ball_(delta, q, attack.rho, attack.eps)
        radius = relative_radius(delta,q,attack.eps).item()
        protocol.require(torch.isfinite(delta).all().item() and radius <= .100001, 'Invalid projected iterate')
        artifact = save_tensor(delta, directory / f'delta-{step:02d}.pt')
        trace.append({'step':step,'loss_values':values.tolist(), 'loss_value_sha256':tensor_value_sha256(values),
            'gradient_value_sha256':tensor_value_sha256(gradient), 'radius':radius,
            'optimizer_value_receipt':tensor_tree(optimizer.state_dict()), 'delta':artifact})
        if step in (1,10,25,50):
            print(f'Strict trajectory {directory.name}: update {step}/50, radius={radius:.8f}',flush=True)
    with torch.no_grad():
        audio = vocoder(q+delta)
    audio_artifact = save_tensor(audio, directory / 'final-audio.pt')
    state_path = directory / 'final-optimizer.pt'
    torch.save(optimizer.state_dict(), state_path)
    protocol.require(not any(p.grad is not None for m in (vocoder,surrogate) for p in m.parameters()),'Accumulated model gradients')
    return {'trace':trace, 'actual_optimization_updates':50, 'final_audio':audio_artifact,
        'final_optimizer':{'path':str(state_path.resolve()),'file_sha256':digest(state_path),
            'value_receipt':tensor_tree(optimizer.state_dict())}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('fixed','trajectory'),required=True)
    args=parser.parse_args()
    import run_deterministic_execution as driver
    plan=protocol.read(args.plan)
    driver.check(plan)
    protocol.require(not args.output.exists(), 'Refusing to overwrite completed evidence')
    directory=args.output.parent/args.output.stem
    protocol.require(not directory.exists(), 'Refusing to reuse partial tensor evidence')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Set cuBLAS workspace before fresh process start')
    parent=protocol.read(plan['parent_plan'])
    entry=parent['parents']['pilot']
    config=entry['config']
    attack,spectral=AttackConfig(**config['attack']),SpectralConfig(**config['spectral'])
    protocol.require(attack.steps==50 and attack.rho==.1 and attack.lambda_perc==1 and attack.alpha_hf==1 and
        attack.learning_rate==.1 and attack.optimizer=='adam' and attack.initialization=='zero' and
        attack.lr_schedule=='constant' and attack.attack and attack.temporal and attack.high_frequency and
        spectral.cutoff_hz==6000 and spectral.sample_rate==16000,'Changed fixed attack protocol')
    run=Path(entry['run'])
    metadata=protocol.read(run/'run.json')
    originals=read_jsonl(parent['pools']['pilot']['manifest'])[:3]
    cache=protocol.read(Path(plan['parent_plan']).parent/'parent-audits/pilot.json')
    cached={r['id']:r for r in cache['items']}
    records={r['id']:r for r in read_jsonl(run/'generated.jsonl')}
    native_math=protocol.read(plan['native_math_report'])
    old={r['id']:r for r in native_math['items']}
    items=[]
    with backend_mode('strict_math'):
        seed_everything(config['seed'])
        flags=backend_manifest()
        protocol.require(flags['deterministic_algorithms'] and not flags['deterministic_warn_only'] and
            flags['math_sdp_enabled'] and not any(flags[k] for k in ('flash_sdp_enabled','efficient_sdp_enabled','cudnn_sdp_enabled')),
            'Strict math controls not active')
        vocoder,vocoder_metadata=load_frozen_vocoder(config['features'],metadata['features'],'cuda')
        native,surrogate_metadata=load_whisper(config['surrogate'],'cuda')
        freeze(native)
        protocol.require(surrogate_metadata==metadata['surrogate'],'Changed surrogate provenance')
        stable=equivalent_surrogate(native)
        for original in originals:
            uid=original['id']; seed=records[uid]['seed']
            seed_everything(seed)
            q=torch.load(cached[uid]['q_path'],map_location='cuda',weights_only=True).detach()
            x=load_audio(original['audio']).cuda().detach()
            if args.stage=='fixed':
                delta=torch.load(run/(uid+'.delta.pt'),map_location='cuda',weights_only=True).detach()
                states={}
                for name,value in (('zero_delta',torch.zeros_like(q)),('saved_parent_delta',delta)):
                    states[name]=fixed_probe(q,x,value,original['text'],vocoder,native,stable,attack,spectral,seed,
                        directory/uid/name,old[uid]['states'][name])
                    print(f'Strict explicit reflection: {uid} {name}, passed={states[name]["passed"]}',flush=True)
                item={'id':uid,'seed':seed,'states':states,'passed':all(s['passed'] for s in states.values())}
            else:
                value=trajectory(q,x,original['text'],vocoder,stable,attack,spectral,directory/uid)
                item={'id':uid,'seed':seed,**value}
            items.append(item)
    driver.check(plan)
    atomic_json(args.output,{'status':'complete','created_at':now(),'plan':str(args.plan.resolve()),
        'plan_sha256':digest(args.plan),'stage':args.stage,'runtime':protocol.runtime(),'backend':flags,
        'vocoder':vocoder_metadata,'surrogate':surrogate_metadata,'items':items,
        'passed':all(i['passed'] for i in items) if args.stage=='fixed' else None,
        'scope':plan['scope']})


if __name__=='__main__':
    main()
