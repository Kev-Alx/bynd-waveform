"""Prevent scope/config/cache drift in the predeclared performance batch."""
import copy

import pytest

from scripts.run_stable_quality_research import cases,identity,joint_target,validate_case
from beyondwaveform.pipeline import config_fingerprint


def anchors():
    base={'seed':2026,'method':'feature','save_delta':True,
        'features':{'artifact_root':'artifacts','layer':6,'normalize':False,'vocoder_variant':'regular'},
        'surrogate':{'checkpoint':'small.pt','language':'en','nll_reduction':'sum','decoding':{'beam_size':5,'temperature':0.0}},
        'attack':{'steps':50,'rho':.1,'lambda_perc':1,'alpha_hf':1,'eps':1e-8,'optimizer':'adam',
            'learning_rate':.1,'initialization':'zero','attack':True,'temporal':True,'high_frequency':True},
        'spectral':{'sample_rate':16000,'cutoff_hz':6000}}
    timestamp=copy.deepcopy(base)
    timestamp['attack']['learning_rate']=.05
    timestamp['surrogate']['nll_timestamp_policy']='initial_timestamp'
    return {'families':[{'config':base},{'config':timestamp}]}


def test_existing_anchors_are_preserved_and_quality_branch_does_not_mutate_source():
    previous=anchors();before=copy.deepcopy(previous)
    rows=cases(previous)
    for row in rows:validate_case(row)
    assert rows[0]['config']==previous['families'][0]['config']
    assert rows[2]['config']==previous['families'][1]['config']
    rows[1]['config']['attack']['rho']=.2
    assert previous==before and rows[0]['config']['attack']['rho']==.1


@pytest.mark.parametrize('field,value',[('steps',51),('rho',.5),('lambda_perc',.1),('alpha_hf',10),
    ('learning_rate',.05),('optimizer','sgd'),('initialization','normal'),('lr_schedule','cosine'),('temporal',False)])
def test_predeclared_protocol_rejects_unplanned_attack_changes(field,value):
    case=cases(anchors())[0];case['config']['attack'][field]=value
    with pytest.raises(ValueError,match='Changed predeclared performance anchor'):validate_case(case)


def test_execution_policy_is_part_of_generation_cache_identity():
    plan={'pilot':{'sha256':'fixed-manifest','ids':['fixed']},
        'execution':{'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True}}
    case=cases(anchors())[0]
    before=config_fingerprint(identity(plan,case))
    changed=copy.deepcopy(plan);changed['execution']['explicit_reflection']=False
    assert config_fingerprint(identity(changed,case))!=before
    changed=copy.deepcopy(plan);changed['pilot']['ids']=['another']
    assert config_fingerprint(identity(changed,case))!=before


def test_joint_descriptive_target_requires_wer_and_all_three_quality_channels():
    row={'error_percent':75.43,'quality':{'dnsmos':2.954,'nisqa':3.526,'utmos':3.590}}
    assert joint_target(row)
    for metric in ('dnsmos','nisqa','utmos'):
        changed=copy.deepcopy(row);changed['quality'][metric]-=.001
        assert not joint_target(changed)
    row['error_percent']-=.01
    assert not joint_target(row)
