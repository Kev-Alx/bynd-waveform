import copy
import pytest
import torch
from beyondwaveform.io import atomic_json,digest
from scripts.run_feature_layer_research_v2 import stage_command,layer_contract,verify_encoder_receipt
from scripts.probe_numerical_repeatability import tensor_value_sha256


@pytest.mark.parametrize('stage',['extract','reextract','reconstruction-evaluate','evaluate'])
def test_original_encoder_and_native_scoring_unset_cublas_workspace(stage):
    command=stage_command(['python','driver.py'],stage,'layer-5')
    assert command[:3]==['/usr/bin/env','-u','CUBLAS_WORKSPACE_CONFIG']
    assert command[3:5]==['python','driver.py']
    assert command[-4:]==['--stage',stage,'--case','layer-5']


@pytest.mark.parametrize('stage',['gate','generate','verify'])
def test_vocoder_verification_and_attack_use_strict_fresh_process_workspace(stage):
    command=stage_command(['python','driver.py'],stage,'layer-7')
    assert command[:2]==['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    assert '-u' not in command[:3]


def test_independent_encoder_receipt_binds_all_feature_values_and_backend(tmp_path):
    ids=[f'clip-{n}' for n in range(24)]
    expected_backend={'policy':'original-auto','cublas_workspace_config':None}
    plan={'feature_artifacts':{'layer':6,'vocoder_sha256':'fixed'},'pilot':{'ids':ids,'sha256':'manifest'},
        'encoder_backend':expected_backend}
    case={'name':'layer-5','layer':5}
    items=[]
    for uid in ids:
        path=tmp_path/'cache/layer-5/q'/(uid+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        value=torch.tensor([[.2,.3]]);torch.save(value,path)
        items.append({'id':uid,'q_path':str(path),'q_sha256':digest(path),'q_value_sha256':tensor_value_sha256(value)})
    cache_path=tmp_path/'cache/layer-5/audit.json'
    atomic_json(cache_path,{'features':layer_contract(plan['feature_artifacts'],5),'items':items})
    atomic_json(tmp_path/'extraction.json',{'cache_audits':{'layer-5':{'sha256':digest(cache_path)}}})
    receipt={'status':'complete','count':24,'cache_audit_sha256':digest(cache_path),'input_manifest_sha256':'manifest',
        'feature_value_hashes':{r['id']:r['q_value_sha256'] for r in items},'backend':expected_backend}
    receipt_path=tmp_path/'encoder-verification/layer-5.json';atomic_json(receipt_path,receipt)
    verify_encoder_receipt(tmp_path,plan,case)
    bad=copy.deepcopy(receipt);bad['feature_value_hashes'][ids[0]]='changed';atomic_json(receipt_path,bad)
    with pytest.raises(ValueError,match='Changed independent encoder receipt'):verify_encoder_receipt(tmp_path,plan,case)
    bad=copy.deepcopy(receipt);bad['backend']['cublas_workspace_config']=':4096:8';atomic_json(receipt_path,bad)
    with pytest.raises(ValueError,match='Changed independent encoder receipt'):verify_encoder_receipt(tmp_path,plan,case)
