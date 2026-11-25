"""Input-only inner optimizers; do not freeze or mutate trainable defenders."""
from __future__ import annotations

import torch
from beyondwaveform.attacks import project_relative_ball_,relative_radius
from beyondwaveform.config import SpectralConfig
from beyondwaveform.losses import CleanReference


def waveform_pgd(x, loss_fn, steps=5, epsilon=.002, step_size=None, random_start=False):
    if steps < 1 or epsilon <= 0:
        raise ValueError('Positive steps/epsilon required')
    step_size = step_size if step_size is not None else 2*epsilon/steps
    current=x.detach().clone()
    if random_start:current=(current+torch.empty_like(x).uniform_(-epsilon,epsilon)).clamp(-1,1)
    trace=[]
    for i in range(steps):
        current=current.detach().requires_grad_()
        loss=loss_fn(current)
        grad,=torch.autograd.grad(loss,current)
        if not torch.isfinite(loss) or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite waveform attack')
        current=x+(current.detach()+step_size*grad.sign()-x).clamp(-epsilon,epsilon)
        current=current.clamp(-1,1).detach()
        trace.append({'step':i+1,'nll_before_update':float(loss.detach()),
                      'linf_after_update':float((current-x).abs().max())})
    return current,trace


def feature_pgd(q,x,vocoder,loss_fn,steps=5,rho=.1,learning_rate=.05,lambda_perc=1.,
                initial=None,random_start=False,observer=None,checkpoints=()):
    if steps < 1 or rho <= 0:
        raise ValueError('Positive steps/rho required')
    q=q.detach()
    reference=CleanReference(q,x.detach(),SpectralConfig(),1e-8)
    delta=torch.zeros_like(q) if initial is None else initial.detach().clone()
    if random_start and initial is None:
        delta.normal_();delta.mul_(.1*rho*(q.norm()+1e-8)/delta.norm())
    project_relative_ball_(delta,q,rho)
    delta.requires_grad_()
    optimizer=torch.optim.Adam([delta],lr=learning_rate)
    trace=[]
    for i in range(steps):
        optimizer.zero_grad(set_to_none=True)
        audio=vocoder(q+delta)
        nll=loss_fn(audio)
        tv,hf=reference.terms(q+delta,audio)
        objective=-nll+lambda_perc*(tv+hf)
        grad,=torch.autograd.grad(objective,delta)
        if not torch.isfinite(objective) or not torch.isfinite(grad).all():
            raise FloatingPointError('Nonfinite feature attack')
        delta.grad=grad
        optimizer.step();project_relative_ball_(delta,q,rho)
        trace.append({'step':i+1,'nll_before_update':float(nll.detach()),
                      'objective_before_update':float(objective.detach()),
                      'tv_ratio':float(tv.detach()),'hf_ratio':float(hf.detach()),
                      'relative_radius':float(relative_radius(delta,q))})
        if observer is not None and i+1 in checkpoints:
            with torch.no_grad():observed_audio=vocoder(q+delta).detach()
            observer(i+1,observed_audio,delta.detach(),list(trace))
    with torch.no_grad():audio=vocoder(q+delta).detach()
    return audio,delta.detach(),trace
