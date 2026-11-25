"""Long speech-generative optimization with fixed public-source/transform access."""
import hashlib
import time
import numpy as np
import torch
import execution_runtime as rt
from scaled_components import calibrate_sources, ensemble_gradient


def optimize(q, original, text, renderer, models, *, method, uid, steps=500, on_step=None, on_checkpoint=None):
    if method not in ('single', 'quality', 'ensemble_raw', 'ensemble_eot', 'mel_eot'):
        raise ValueError('Unknown fixed scaled method')
    allowed=(50,) if method=='quality' else ((50,150,200,500,600,1500) if method=='single' else (50,200,500))
    if steps not in allowed: raise ValueError('Unregistered optimization endpoint')
    q=q.detach(); delta=torch.zeros_like(q,requires_grad=True)
    optimizer=torch.optim.Adam([delta],lr=.01 if method=='mel_eot' else (.02 if method=='quality' else .1))
    reference=rt.ExplicitReflectionReference(q,original,rt.SpectralConfig(),1e-8)
    with torch.no_grad(): reconstruction=renderer(q)
    if method in ('single','quality'): scales, calibration={'small':1.},None
    else: scales,calibration=calibrate_sources(models,reconstruction,text)
    seed=int.from_bytes(hashlib.sha256(('20261022:'+uid).encode()).digest()[:8],'big')
    schedule=np.random.default_rng(seed).integers(0,3,size=500).tolist()
    names=('identity','fir6k','resample8k')
    traces=[]; start=time.perf_counter(); counts={'source_backward_calls':0,'source_forward_calls':0,
        'calibration_source_backward_calls':0 if method in ('single','quality') else 3,'vocoder_forward_calls':1,
        'vocoder_backward_calls':0,'optimizer_steps':0}
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(): audio=renderer(q+delta)
        counts['vocoder_forward_calls']+=1
        augmentation=names[schedule[step-1]] if method in ('ensemble_eot','mel_eot') else 'identity'
        g,components=ensemble_gradient(models,audio,text,scales,augmentation,single=method in ('single','quality'))
        counts['source_backward_calls']+=len(components); counts['source_forward_calls']+=len(components)
        rendered=renderer(q+delta); counts['vocoder_forward_calls']+=1
        if step==1 and not torch.equal(rendered.detach(),audio): raise ValueError('VJP rendering changed forward values')
        tv,hf=reference.terms(q+delta,rendered)
        # Exact chain rule for the frozen weighted source losses, without
        # retaining their activation graphs while differentiating the decoder.
        linear_source=(rendered*g).sum()
        derivative,=torch.autograd.grad(-linear_source+tv+hf,delta)
        counts['vocoder_backward_calls']+=1
        if not torch.isfinite(derivative).all() or derivative.double().norm()==0:
            raise ValueError('Invalid generative-space derivative')
        delta.grad=derivative; optimizer.step()
        with torch.no_grad():
            if method=='mel_eot': delta.clamp_(-np.log(2),np.log(2))
            rt.project_relative_ball_(delta,q,.1)
        counts['optimizer_steps']+=1
        radius=float(rt.relative_radius(delta,q))
        if not torch.isfinite(delta).all() or radius>.100001: raise ValueError('Invalid latent projection')
        nll=sum(v['weight']*v['fixed_scale']*v['nll'] for v in components.values())
        row={'step':step,'loss_evaluation_iterate':step-1,'method':method,'transform':augmentation,
            'components':components,'weighted_source_nll':nll,'tv':float(tv.detach()),'hf':float(hf.detach()),
            'loss':-nll+float((tv+hf).detach()),'relative_radius':radius,
            'feature_gradient_norm':float(derivative.double().norm()),'counts':dict(counts)}
        traces.append(row)
        if on_step and (step in (1,10,50,100,150,200) or step%100==0): on_step(dict(row))
        if step in allowed:
            with torch.no_grad(): final=renderer(q+delta).detach()
            counts['vocoder_forward_calls']+=1
            if not torch.isfinite(final).all() or final.abs().max()>1: raise ValueError('Invalid saved generative audio')
            receipt={'status':'completed','method':method,'selected_iterate':step,'trace':list(traces),
                'counts':dict(counts),'calibration':calibration,'transform_schedule_seed':seed,
                'schedule_sha256':hashlib.sha256(bytes(schedule)).hexdigest(),
                'selection':'Fixed50/200/500 checkpoints; full-cohort source/quality selection later',
                'relative_radius':radius,'wall_seconds_cumulative':time.perf_counter()-start,
                'representation':'BigVGAN log-mel' if method=='mel_eot' else 'WavLM6 features',
                'decoder_post_resample_clipped_samples':getattr(renderer,'last_clipped_samples',0),
                'log_mel_box_maximum':float(delta.detach().abs().max()) if method=='mel_eot' else None}
            if on_checkpoint: on_checkpoint(step,final,delta.detach(),receipt)
            del final
        del audio,rendered,g,derivative,tv,hf,linear_source
    return {'counts':counts,'wall_seconds':time.perf_counter()-start}
