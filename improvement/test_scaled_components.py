import pytest
import torch
from scaled_components import checkpoint_blocks, replicate_concat, transform, calibrate_sources, ensemble_gradient


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_checkpoint_has_same_forward_and_actual_input_derivative():
    block = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.GELU(), torch.nn.Linear(7, 4)).double().eval().requires_grad_(False)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    output = block(x); gradient, = torch.autograd.grad(output.square().sum(), x)
    checkpoint_blocks([block])
    replay = block(x); other, = torch.autograd.grad(replay.square().sum(), x)
    assert torch.equal(output, replay) and torch.equal(gradient, other)


def test_replicate_replacement_matches_padding_values_and_derivative():
    x = torch.randn(2, 3, 14, dtype=torch.float64, requires_grad=True)
    reference = torch.nn.functional.pad(x, (3, 5), mode='replicate')
    result = replicate_concat(x, 3, 5)
    assert torch.equal(result, reference)
    g, = torch.autograd.grad(result.square().sum(), x)
    h, = torch.autograd.grad(reference.square().sum(), x)
    assert torch.allclose(g, h, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('name', ['identity', 'fir6k', 'resample8k'])
def test_transforms_preserve_length_and_have_real_finite_derivatives(name):
    x = torch.randn(16001, requires_grad=True)
    y = transform(x, name)
    assert y.shape == x.shape and torch.isfinite(y).all()
    g, = torch.autograd.grad(y.square().mean(), x)
    assert g.shape == x.shape and torch.isfinite(g).all() and g.norm() > 0


def test_ensemble_is_derivative_of_frozen_weighted_losses():
    class Source:
        def __init__(self, scale): self.scale = scale
        def nll(self, x, text): return self.scale * (x + .3).square().sum()
    models = {'small': Source(1), 'large': Source(8), 'ctc': Source(.2)}
    x = torch.randn(900)
    scales, receipt = calibrate_sources(models, x, 'text')
    combined, rows = ensemble_gradient(models, x, 'text', scales, 'identity')
    expected = 2 * (x + .3)
    assert torch.allclose(combined, expected, atol=2e-6)
    assert scales['large'] == pytest.approx(.125) and scales['ctc'] == pytest.approx(5.)
    assert sum(r['source_backward_calls'] for r in rows.values()) == 3


def test_training_transform_registry_rejects_unknown_defense():
    with pytest.raises(ValueError): transform(torch.zeros(800), 'AudioPure')


def test_real_long_optimizer_receipt_and_independent_corruption_checks():
    import importlib.util
    from pathlib import Path
    from scaled_attack import optimize
    spec=importlib.util.spec_from_file_location('scaled_saved_audit',Path(__file__).parent/'execution_scaled/audit_scaled.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    class Source:
        def __init__(self,scale):self.scale=scale
        def nll(self,x,text):return self.scale*(x+.2).square().sum()
    models={'small':Source(1.),'large':Source(8.),'ctc':Source(.2)}
    q=torch.randn(64,16,dtype=torch.float64,generator=torch.Generator().manual_seed(3))*.1
    render=lambda z:torch.tanh(z.flatten())*.5
    saved=[]
    optimize(q,render(q).detach(),'REFERENCE',render,models,method='ensemble_raw',uid='toy',steps=50,
        on_checkpoint=lambda step,a,d,r:saved.append((a,d,r)))
    a,d,r=saved[0]
    assert d.norm()<=(q.norm()+1e-8)*.1+1e-12 and torch.isfinite(a).all()
    audit.base.ISSUES.clear();audit.audit_receipt(r,'toy/ensemble_raw_50')
    assert audit.base.ISSUES==[]
    import copy
    for mutate,phrase in [
        (lambda v:v['counts'].__setitem__('source_backward_calls',50),'source call budget'),
        (lambda v:v['trace'][3]['components']['large'].__setitem__('fixed_scale',99.),'changed source scale'),
        (lambda v:v['trace'][0].__setitem__('transform','resample8k'),'changed transform selection'),
        (lambda v:v['trace'].pop(),'partial trajectory')]:
        damaged=copy.deepcopy(r);mutate(damaged);audit.base.ISSUES.clear()
        audit.audit_receipt(damaged,'toy/ensemble_raw_50')
        assert any(phrase in issue for issue in audit.base.ISSUES)


def test_vocoder_vjp_recovers_direct_multi_model_derivative():
    from scaled_components import ensemble_gradient
    class Source:
        def __init__(self,scale):self.scale=scale
        def nll(self,x,text):return self.scale*(x+.2).square().sum()
    models={'small':Source(1.),'large':Source(8.),'ctc':Source(.2)}
    scales={'small':1.,'large':.125,'ctc':5.}
    q=torch.randn(30,dtype=torch.float64,requires_grad=True)
    audio=torch.tanh(q*2)
    total=sum(w*scales[k]*models[k].nll(audio,'text') for k,w in [('small',.5),('large',.25),('ctc',.25)])
    direct,=torch.autograd.grad(total,q)
    g,_=ensemble_gradient(models,audio.detach(),'text',scales,'identity')
    replay=torch.tanh(q*2)
    vjp,=torch.autograd.grad(replay,q,grad_outputs=g)
    assert torch.allclose(vjp,direct,atol=1e-14,rtol=1e-14)
