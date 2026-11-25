"""One frozen held-out comparison, including exact named preprocessing paths."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torchaudio.functional as AF
import execution_runtime as rt
import scaled_scope as scope
import source_evaluate_refine as native_policy
from beyondwaveform.config import load_config
from beyondwaveform.evaluation import error_counts,aggregate_errors,QualityScorer
from beyondwaveform.models.targets import load_target

CONDITIONS=('raw','lpf6000','waveguard8000','lpf4000','waveguard12000','waveguard2000')
MAIN=CONDITIONS[:5]


def preprocess(audio,condition):
    if condition=='raw':return audio
    if condition.startswith('lpf'):
        return AF.lowpass_biquad(audio,16000,float(condition[3:]),Q=.70710678)
    if condition.startswith('waveguard'):
        sys.path.insert(0,str(rt.ROOT/'artifacts/defense-runtime'))
        from beyondwaveform.defenses import waveguard_down_up
        return waveguard_down_up(audio,int(condition[len('waveguard'):]))
    raise ValueError('Unknown registered evaluation defense')


def main():
    plan=scope.check();freeze=scope.validation_freeze();decision=rt.read(scope.RUN/'transfer-freeze.json')
    if decision['status']!='evaluate_transfer':raise ValueError('Unqualified validation cannot open targets')
    for path,digest in decision['bindings'].items():
        if rt.digest(path)!=digest:raise ValueError('Changed source validation after freeze')
    if decision['candidate']!=freeze['candidate'] or decision['control']!=freeze['control']:
        raise ValueError('Transfer reselection is prohibited')
    folder=scope.RUN/'transfer'
    if (folder/'results.json').exists() or (folder/'records').exists():raise FileExistsError('Preserve earlier target evidence')
    folder.mkdir(parents=True,exist_ok=True)
    records=rt.read_jsonl(scope.RUN/'validation/generated.jsonl');info=plan['cohorts']['validation']
    allowed=rt.read_jsonl(info['manifest'])
    if len(records)!=76 or [r['id'] for r in records]!=[r['id'] for r in allowed]:raise ValueError('Incomplete validation cohort')
    expected=rt.read(rt.BASE/'evaluation/summary.json')['evaluation_backend']
    native_policy.native_policy(expected)
    rt.atomic_json(folder/'registration.json',{'created_at_utc':rt.stamp(),'source_candidate_frozen':True,
        'candidate':freeze['candidate'],'control':freeze['control'],'target_configs':plan['target_configs'],
        'conditions':list(CONDITIONS),'known_training_approximations':['fir6k','torchaudio-resample8k'],
        'withheld_parameters':['lpf4000','waveguard12000'],'diagnostic_only':['waveguard2000'],
        'backend':expected,'plan_sha256':rt.digest(scope.RUN/'plan.json'),
        'input_manifest_sha256':rt.digest(scope.RUN/'validation/generated.jsonl')})
    import run_quality_research as quality_protocol
    quality_protocol.configure_scorers(rt.read(rt.RUN/'plan.json')['quality'])
    scorer=QualityScorer(str(rt.ROOT/'artifacts'))
    began=time.perf_counter();assets={};post_quality={};scores=[];target_meta={}
    # Cache transformations and quality once; target identities do not affect
    # their waveform or perceptual-score computation.
    for condition in CONDITIONS:
        for index,record in enumerate(records,1):
            for variant,original in record['audio'].items():
                if rt.digest(original['path'])!=original['sha256']:raise ValueError('Changed validation audio')
                key=(condition,original['sha256'])
                if key not in assets:
                    audio=rt.load_audio(original['path'])
                    transformed=preprocess(audio,condition)
                    path=folder/'audio'/condition/(original['sha256']+'.wav')
                    assets[key]=rt.save_audio(path,transformed)
                    post_quality[key]=scorer(transformed)
            rt.atomic_json(folder/'progress.json',{'status':'running','phase':'preprocessing_quality',
                'condition':condition,'completed':index,'count':76,'updated_at_utc':rt.stamp()})
        print('Prepared defense and post-defense quality',condition,flush=True)
    del scorer
    rt.atomic_json(folder/'post-defense-quality.json',{'records':[{'condition':k[0],'input_audio_sha256':k[1],
        'audio':assets[k],'quality':post_quality[k]} for k in assets]})
    for target_name,config_path in plan['target_configs'].items():
        config=load_config(config_path);target,metadata=load_target(config,'cuda');target.eval().requires_grad_(False)
        target_meta[target_name]=metadata;predictions={}
        for condition in CONDITIONS:
            for index,record in enumerate(records,1):
                for variant,original in record['audio'].items():
                    key=(condition,original['sha256']);asset=assets[key]
                    prediction_key=asset['sha256']
                    if prediction_key not in predictions:
                        waveform=rt.load_audio(asset['path']).cuda()
                        predictions[prediction_key]=target.transcribe(waveform)
                    hypothesis=predictions[prediction_key]
                    item={'id':record['id'],'speaker':record['speaker'],'target':target_name,'defense':condition,
                        'variant':variant,'reference':record['text'],'hypothesis':hypothesis,
                        'input_audio_sha256':original['sha256'],'evaluated_audio_sha256':asset['sha256'],
                        **error_counts(record['text'],hypothesis),'quality':post_quality[key]}
                    scores.append(item)
                    rt.atomic_json(folder/'records'/target_name/condition/f'{record["id"]}--{variant}.json',item)
                if index%10==0 or index==76:
                    rt.atomic_json(folder/'progress.json',{'status':'running','phase':'target_evaluation','target':target_name,
                        'condition':condition,'completed':index,'count':76,'updated_at_utc':rt.stamp()})
            print('Held-out evaluation completed',target_name,condition,flush=True)
        del target;torch.cuda.empty_cache()
    summaries={}
    for target in plan['target_configs']:
        summaries[target]={}
        for condition in CONDITIONS:
            summaries[target][condition]={}
            for variant in info['variants']:
                pool=[r for r in scores if r['target']==target and r['defense']==condition and r['variant']==variant]
                if len(pool)!=76 or len({r['id'] for r in pool})!=76:raise ValueError('Incomplete target/condition/variant denominator')
                summaries[target][condition][variant]={'count':76,**aggregate_errors(pool),
                    'quality':{k:sum(r['quality'][k] for r in pool)/76 for k in ('utmos','dnsmos','nisqa')},
                    'empty_output_rate':sum(not r['hypothesis_normalized'] for r in pool)/76,
                    'capped_corpus_error_percent':100*sum(min(r['reference_units'],r['substitutions']+r['deletions']+r['insertions']) for r in pool)/sum(r['reference_units'] for r in pool)}
    speakers=sorted({r['speaker'] for r in records});weights=np.random.default_rng(20261022).multinomial(len(speakers),[1/len(speakers)]*len(speakers),size=10000)
    comparisons={}
    for condition in CONDITIONS:
        aggregate=np.asarray([[sum(r['substitutions']+r['deletions']+r['insertions'] for r in scores
            if r['speaker']==speaker and r['defense']==condition and r['variant']==variant)
            for variant in ('candidate','control')]+[sum(r['reference_units'] for r in scores
            if r['speaker']==speaker and r['defense']==condition and r['variant']=='candidate')]
            for speaker in speakers])
        sample=weights@aggregate;deltas=100*(sample[:,0]-sample[:,1])/sample[:,2]
        point=100*(aggregate[:,0].sum()-aggregate[:,1].sum())/aggregate[:,2].sum()
        interval=np.quantile(deltas,[.025,.975]).tolist()
        comparisons[condition]={'family_balanced_delta_wer_points':float(point),'speaker_bootstrap95':interval,
            'replicates':10000,'point_noninferior':bool(point>=0),'lower_within_minus3':bool(interval[0]>=-3),
            'preservation_screen':bool(point>=0 and interval[0]>=-3),'diagnostic_only':condition not in MAIN}
    measured=rt.backend_manifest()
    if measured!=expected:raise ValueError('Evaluation backend changed during target scoring')
    rt.write_jsonl(folder/'scores.jsonl',scores)
    value={'status':'complete','preservation_status':'pass' if all(comparisons[k]['preservation_screen'] for k in MAIN) else 'not_established',
        'raw_transfer_superiority_signal':bool(comparisons['raw']['family_balanced_delta_wer_points']>=3 and comparisons['raw']['speaker_bootstrap95'][0]>0),
        'count':76,'targets':target_meta,'summaries':summaries,'comparisons':comparisons,'candidate_reselected':False,
        'backend':measured,'source_goal_achieved_on_validation':True,'scores_sha256':rt.digest(folder/'scores.jsonl'),
        'wall_seconds':time.perf_counter()-began,'human_intelligibility_established':False,
        'paper_training_defenses_measured':False,'full_waveguard_detection_measured':False,
        'scope':'One frozen exposed-validation comparison, not untouched final200 or proof against unavailable defenses'}
    rt.atomic_json(folder/'results.json',value)
    rt.atomic_json(folder/'progress.json',{'status':'complete','updated_at_utc':rt.stamp()})
    print(json.dumps({'status':value['status'],'preservation_status':value['preservation_status'],'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
