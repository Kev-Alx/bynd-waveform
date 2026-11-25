"""Integration checks for the isolated numerical execution experiment."""
import copy
from types import SimpleNamespace

import pytest
import torch

from beyondwaveform.attacks import freeze
from beyondwaveform.config import AttackConfig,SpectralConfig
from beyondwaveform.losses import CleanReference
from beyondwaveform.models.whisper import WhisperASR
from scripts.probe_deterministic_execution import (
    ExplicitReflectionReference,derivative,equivalent_surrogate,objective,save_tensor,trajectory,
)
from scripts.run_deterministic_execution import compare_trajectories,verify_tensor


def fixture():
    class Vocoder(torch.nn.Module):
        def forward(self,q):
            return (q.flatten().repeat_interleave(4)*.7).tanh()
    class Surrogate(torch.nn.Module):
        def nll(self,audio,text):
            assert text=='FIXED REFERENCE'
            return (audio+.2).square().mean()*1.3
    generator=torch.Generator().manual_seed(17)
    q=torch.randn(4,8,dtype=torch.float64,generator=generator)
    x=torch.randn(128,dtype=torch.float64,generator=generator)
    delta=torch.randn(4,8,dtype=torch.float64,generator=generator)*.01
    return q,x,delta,Vocoder(),Surrogate(),AttackConfig(learning_rate=.1),SpectralConfig(n_fft=8,hop_length=2,win_length=8)


def test_explicit_objective_preserves_native_denominators_and_full_expression():
    q,x,delta,vocoder,surrogate,attack,spectral=fixture()
    native=CleanReference(q,x,spectral,attack.eps)
    explicit=ExplicitReflectionReference(q,x,spectral,attack.eps)
    assert torch.equal(native.tv,explicit.tv) and torch.equal(native.hf,explicit.hf)
    live=delta.clone().requires_grad_()
    expected,old_values=objective(q,live,'FIXED REFERENCE',vocoder,surrogate,attack,native)
    old_gradient,=torch.autograd.grad(expected,live)
    values,gradient=derivative(q,x,delta,'FIXED REFERENCE',vocoder,surrogate,attack,spectral)
    torch.testing.assert_close(values,old_values,rtol=0,atol=0)
    torch.testing.assert_close(gradient,old_gradient,rtol=1e-12,atol=1e-12)
    assert not q.requires_grad and not x.requires_grad and delta.grad is None


def test_wrapper_retains_targets_model_reduction_and_native_decoding():
    class Model(torch.nn.Module):
        is_multilingual=False
        num_languages=99
        dims=SimpleNamespace(n_mels=80)
    native=freeze(WhisperASR(Model(),language='en',reduction='sum',decoding={'beam_size':5,'temperature':0.0},
        nll_transcript_policy='sentence',nll_timestamp_policy='initial_timestamp'))
    explicit=equivalent_surrogate(native)
    assert explicit.model is native.model
    for key in ('language','reduction','decoding','nll_transcript_policy','nll_timestamp_policy'):
        assert getattr(explicit,key)==getattr(native,key)
    assert explicit.transcribe.__func__ is native.transcribe.__func__
    audio=torch.linspace(-.1,.1,720)
    torch.testing.assert_close(explicit.mel(audio),native.mel(audio),rtol=0,atol=0)
    assert not explicit.training


def test_saved_tensor_integrity_detects_tampering(tmp_path):
    artifact=save_tensor(torch.tensor([1.,-0.,3.]),tmp_path/'value.pt')
    assert torch.equal(verify_tensor(artifact),torch.tensor([1.,-0.,3.]))
    torch.save(torch.tensor([2.,0.,3.]),artifact['path'])
    with pytest.raises(ValueError,match='Changed saved tensor'):
        verify_tensor(artifact)


def test_two_full_fifty_step_trajectories_compare_all_iterates_and_detect_mismatch(tmp_path):
    q,x,delta,vocoder,surrogate,attack,spectral=fixture()
    results=[]
    for repeat in range(2):
        value=trajectory(q,x,'FIXED REFERENCE',vocoder,surrogate,attack,spectral,tmp_path/str(repeat))
        results.append({'items':[{'id':'fixed','seed':17,**value}]})
    comparison=compare_trajectories(*results)
    assert comparison['passed'] and len(comparison['items'][0]['updates'])==50
    changed=copy.deepcopy(results[1])
    changed['items'][0]['trace'][9]['loss_values'][0]+=.01
    comparison=compare_trajectories(results[0],changed)
    assert not comparison['passed']
    assert not comparison['items'][0]['updates'][9]['trace_and_moments_bitwise_equal']
    with pytest.raises(ValueError,match='Missing trajectory evidence'):
        compare_trajectories({'items':[]},{'items':[]})
