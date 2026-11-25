"""Check flag-aware objectives against independent full/ASR-only trajectories."""
import copy
from pathlib import Path

import pytest
import torch

from beyondwaveform.attacks import project_relative_ball_
from beyondwaveform.config import AttackConfig,SpectralConfig
from scripts.probe_deterministic_execution import ExplicitReflectionReference,trajectory
from scripts.run_deterministic_execution import compare_trajectories
from scripts.run_stable_ablation_research import ablation_objective,active_terms,cases,collect,verify_loss_receipt


def fixture():
    class Vocoder(torch.nn.Module):
        def forward(self,q):return (q.flatten().repeat_interleave(4)*.7).tanh()
    class Surrogate(torch.nn.Module):
        def nll(self,audio,text):
            assert text=='FIXED REFERENCE'
            return (audio+.2).square().mean()*1.3
    generator=torch.Generator().manual_seed(27)
    q=torch.randn(4,8,dtype=torch.float64,generator=generator)
    x=torch.randn(128,dtype=torch.float64,generator=generator)
    return q,x,Vocoder(),Surrogate(),SpectralConfig(n_fft=8,hop_length=2,win_length=8)


def test_only_the_predeclared_terms_change_and_original_configuration_is_preserved():
    base={'attack':vars(AttackConfig(learning_rate=.1)),'surrogate':{'nll_reduction':'sum'}}
    before=copy.deepcopy(base);rows=cases(base)
    assert [active_terms(AttackConfig(**r['config']['attack'])) for r in rows]==[
        {'asr':True,'tv':False,'hf':True},{'asr':True,'tv':True,'hf':False},{'asr':True,'tv':False,'hf':False}]
    for row in rows:
        config=copy.deepcopy(row['config'])
        config['attack'].update(temporal=True,high_frequency=True)
        assert config==base
    assert base==before


@pytest.mark.parametrize('tv,hf',[(True,True),(False,True),(True,False),(False,False)])
def test_component_removal_matches_independently_differentiated_expression(tv,hf):
    q,x,vocoder,surrogate,spectral=fixture()
    attack=AttackConfig(temporal=tv,high_frequency=hf,learning_rate=.1)
    reference=ExplicitReflectionReference(q,x,spectral,attack.eps)
    delta=torch.full_like(q,.003,requires_grad=True)
    loss,values=ablation_objective(q,delta,'FIXED REFERENCE',vocoder,surrogate,attack,reference)
    actual,=torch.autograd.grad(loss,delta)
    other=delta.detach().clone().requires_grad_()
    audio=vocoder(q+other)
    nll=surrogate.nll(audio,'FIXED REFERENCE')
    temporal,frequency=reference.terms(q+other,audio)
    expected=-nll
    if tv:expected=expected+temporal
    if hf:expected=expected+frequency
    gradient,=torch.autograd.grad(expected,other)
    torch.testing.assert_close(actual,gradient,rtol=2e-13,atol=1e-13)
    assert (tv or values[2].item()==0) and (hf or values[3].item()==0)
    assert not q.requires_grad and not x.requires_grad and delta.grad is None


def test_new_full_collector_reproduces_every_saved_update_of_existing_collector(tmp_path):
    q,x,vocoder,surrogate,spectral=fixture();attack=AttackConfig(learning_rate=.1)
    old=trajectory(q,x,'FIXED REFERENCE',vocoder,surrogate,attack,spectral,tmp_path/'old')
    new=collect(q,x,'FIXED REFERENCE',vocoder,surrogate,attack,spectral,tmp_path/'new')
    result=compare_trajectories({'items':[{'id':'fixed','seed':27,**old}]},{'items':[{'id':'fixed','seed':27,**new}]})
    assert result['passed'] and len(result['items'][0]['updates'])==50


def test_no_perceptual_trajectory_matches_pure_asr_adam_and_projection(tmp_path):
    q,x,vocoder,surrogate,spectral=fixture()
    attack=AttackConfig(temporal=False,high_frequency=False,learning_rate=.1)
    actual=collect(q,x,'FIXED REFERENCE',vocoder,surrogate,attack,spectral,tmp_path/'asr')
    delta=torch.zeros_like(q,requires_grad=True)
    optimizer=torch.optim.Adam([delta],lr=.1)
    for row in actual['trace']:
        optimizer.zero_grad(set_to_none=True)
        gradient,=torch.autograd.grad(-surrogate.nll(vocoder(q+delta),'FIXED REFERENCE'),delta)
        delta.grad=gradient;optimizer.step();project_relative_ball_(delta,q,.1,1e-8)
        saved=torch.load(row['delta']['path'],map_location='cpu',weights_only=True)
        torch.testing.assert_close(delta,saved,rtol=0,atol=0)
        assert row['loss_values'][2:]==[0.,0.]
    assert actual['actual_optimization_updates']==50


def test_unsupported_optimizer_or_budget_cannot_silently_run(tmp_path):
    q,x,vocoder,surrogate,spectral=fixture()
    for config in (AttackConfig(steps=51),AttackConfig(optimizer='sgd'),AttackConfig(initialization='normal')):
        with pytest.raises(ValueError,match='fixed fifty-update Adam protocol'):
            collect(q,x,'FIXED REFERENCE',vocoder,surrogate,config,spectral,tmp_path/'unused')
    assert not (tmp_path/'unused').exists()


def test_disabled_term_proof_binds_exported_scalars_to_actual_trajectory_receipt(tmp_path):
    q,x,vocoder,surrogate,spectral=fixture()
    attack=AttackConfig(temporal=False,high_frequency=False,learning_rate=.1)
    saved=collect(q,x,'FIXED REFERENCE',vocoder,surrogate,attack,spectral,tmp_path/'receipts')
    record={'active_terms':active_terms(attack),'trace':[{'step':r['step'],'loss_total':r['loss_values'][0],
        'nll':r['loss_values'][1],'tv_ratio':0.,'hf_ratio':0.,'radius_after_update':r['radius']} for r in saved['trace']]}
    verify_loss_receipt(record,saved,active_terms(attack))
    tampered=copy.deepcopy(saved);tampered['trace'][3]['loss_values'][2]=.2
    with pytest.raises(ValueError,match='Exported trace differs'):
        verify_loss_receipt(record,tampered,active_terms(attack))
    tampered=copy.deepcopy(record);tampered['active_terms']['hf']=True
    with pytest.raises(ValueError,match='Changed enabled terms'):
        verify_loss_receipt(tampered,saved,active_terms(attack))
