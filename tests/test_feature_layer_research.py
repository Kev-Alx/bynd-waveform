import copy
import pytest
import torch

from beyondwaveform.models.feature import WavLMEncoder
from scripts.run_feature_layer_research import layer_contract,reconstruction_gate,effective


def baseline():
    return {'reconstruction':{'count':24,'error_percent':3.5789473684210527,
        'quality':{'dnsmos':3.331731573490439,'nisqa':4.212428420782089,'utmos':4.166711787382762}}}


@pytest.mark.parametrize('layer',[5,7])
def test_experimental_contract_retains_weights_and_discloses_training_layer_without_mutation(layer):
    base={'layer':6,'normalize':False,'vocoder_sha256':'frozen-generator','encoder_sha256':'frozen-encoder',
        'status':'matching upstream contract','vocoder_config':{'hubert_dim':1024}}
    before=copy.deepcopy(base);result=layer_contract(base,layer)
    assert base==before and result['layer']==layer and result['vocoder_training_layer']==6
    assert result['vocoder_sha256']==base['vocoder_sha256'] and result['encoder_sha256']==base['encoder_sha256']
    assert 'out-of-training-contract' in result['status']
    result['vocoder_config']['hubert_dim']=3
    assert base==before


@pytest.mark.parametrize('layer',[1,4,6,8,24])
def test_unplanned_layer_search_is_rejected(layer):
    with pytest.raises(ValueError,match='two predeclared adjacent layers'):layer_contract({'layer':6},layer)


def test_separate_wrappers_share_frozen_weights_without_changing_original_layer_or_input():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.7))
        def extract_features(self,x,output_layer,ret_layer_results):
            assert not ret_layer_results
            return (x[:,:,None].repeat(1,1,4)*self.scale+output_layer,None)
    model=Model();original=WavLMEncoder(model,layer=6,normalize=False)
    probes=[WavLMEncoder(model,layer=layer,normalize=False) for layer in (5,7)]
    x=torch.linspace(-.1,.1,8);before=x.clone()
    with torch.no_grad():base=original(x);left,right=[p(x) for p in probes]
    torch.testing.assert_close(left,base-1,rtol=0,atol=5e-7)
    torch.testing.assert_close(right,base+1,rtol=0,atol=5e-7)
    assert original.layer==6 and probes[0].model is probes[1].model is original.model
    assert not model.training and not model.scale.requires_grad and model.scale.grad is None
    assert torch.equal(x,before)


def test_reconstruction_gate_accepts_boundary_and_rejects_either_intelligibility_or_quality_regression():
    base=baseline();measured=copy.deepcopy(base)
    measured['reconstruction']['error_percent']+=2
    for metric in measured['reconstruction']['quality']:measured['reconstruction']['quality'][metric]-=.1
    assert reconstruction_gate(measured,base)
    changed=copy.deepcopy(measured);changed['reconstruction']['error_percent']+=.0001
    assert not reconstruction_gate(changed,base)
    for metric in ('dnsmos','nisqa','utmos'):
        changed=copy.deepcopy(measured);changed['reconstruction']['quality'][metric]-=.0001
        assert not reconstruction_gate(changed,base)
        changed=copy.deepcopy(measured);changed['reconstruction']['quality'][metric]=float('inf')
        assert not reconstruction_gate(changed,base)


def test_effective_cache_and_contract_are_per_layer_and_do_not_mutate_parent(tmp_path):
    plan={'feature_artifacts':{'layer':6,'vocoder_sha256':'original'},'cache_audit':'original-cache'}
    before=copy.deepcopy(plan)
    left=effective(tmp_path,plan,{'name':'layer-5','layer':5})
    right=effective(tmp_path,plan,{'name':'layer-7','layer':7})
    assert plan==before and left['cache_audit']!=right['cache_audit']
    assert left['feature_artifacts']['layer']==5 and right['feature_artifacts']['layer']==7
    assert all(p['feature_artifacts']['vocoder_training_layer']==6 for p in (left,right))
