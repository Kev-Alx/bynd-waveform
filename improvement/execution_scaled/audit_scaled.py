"""Independent CPU checks of expanded cohorts, budgets, losses and source scores."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from collections import Counter

ROOT=Path('/workspace/improvement');RUN=ROOT/'execution_scaled'
spec=importlib.util.spec_from_file_location('saved_score_audit',ROOT/'execution_refine/audit_refinement.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.CYCLE=RUN


def audit_receipt(receipt,label):
    method=receipt['method'];n=receipt['selected_iterate'];single=method in ('single','quality')
    allowed=(50,) if method=='quality' else ((50,150,200,500,600,1500) if method=='single' else (50,200,500))
    base.verify(method in ('single','quality','ensemble_raw','ensemble_eot','mel_eot') and n in allowed,label+': unregistered method/iterate')
    trace=receipt['trace'];calls=receipt['counts'];base.verify(receipt['status']=='completed' and len(trace)==n,label+': partial trajectory')
    base.verify([r['step'] for r in trace]==list(range(1,n+1)),label+': noncontiguous updates')
    seed=int.from_bytes(hashlib.sha256(('20261022:'+label.split('/')[0]).encode()).digest()[:8],'big')
    schedule=np.random.default_rng(seed).integers(0,3,size=500).tolist()
    base.verify(receipt['transform_schedule_seed']==seed and receipt['schedule_sha256']==hashlib.sha256(bytes(schedule)).hexdigest(),label+': changed seeded transform schedule')
    if not single:
        calibration=receipt['calibration'];norms=calibration['gradient_norms'];scales=calibration['fixed_scales']
        base.verify(set(norms)==set(scales)=={'small','large','ctc'},label+': calibration source set changed')
        for name,value in norms.items():
            base.verify(math.isfinite(value) and value>0,label+': invalid clean calibration gradient')
            base.close(scales[name],norms['small']/value,label+': scale not fixed by initial gradient',atol=1e-12)
    else:
        scales={'small':1.};base.verify(receipt['calibration'] is None,label+': single source used ensemble calibration')
    for row in trace:
        tag=f'{label}/{row["step"]}';components=row['components']
        base.verify(row['method']==method and row['loss_evaluation_iterate']==row['step']-1,tag+': wrong iterate/objective')
        expected=('identity','fir6k','resample8k')[schedule[row['step']-1]] if method in ('ensemble_eot','mel_eot') else 'identity'
        base.verify(row['transform']==expected,tag+': changed transform selection')
        weights={'small':1.} if single else {'small':.5,'large':.25,'ctc':.25}
        base.verify(set(components)==set(weights),tag+': queried different source set')
        total=0
        for name,value in components.items():
            base.verify(value['source_backward_calls']==1 and math.isfinite(value['nll'])
                and math.isfinite(value['gradient_norm']) and value['gradient_norm']>0,tag+': invalid source component')
            base.close(value['fixed_scale'],scales[name],tag+': changed source scale',atol=0,rtol=0)
            base.close(value['weight'],weights[name],tag+': changed source weight',atol=0,rtol=0)
            total+=weights[name]*scales[name]*value['nll']
        base.close(row['weighted_source_nll'],total,tag+': source loss arithmetic differs',atol=1e-9)
        base.close(row['loss'],-total+row['tv']+row['hf'],tag+': full objective arithmetic differs',atol=2e-5)
        base.verify(math.isfinite(row['feature_gradient_norm']) and row['feature_gradient_norm']>0
            and 0<=row['relative_radius']<=.100001,tag+': invalid gradient/radius')
        base.verify(row['counts']['source_backward_calls']==row['step']*len(weights)
            and row['counts']['optimizer_steps']==row['step'],tag+': wrong actual source/step counters')
    expected_calls=n*(1 if single else 3)
    base.verify(calls['source_backward_calls']==calls['source_forward_calls']==expected_calls,label+': wrong source call budget')
    base.verify(calls['calibration_source_backward_calls']==(0 if single else 3),label+': hidden/missing calibration work')
    base.verify(calls['vocoder_backward_calls']==calls['optimizer_steps']==n,label+': wrong VJP/update count')
    base.verify(calls['vocoder_forward_calls']==1+2*n+sum(k<=n for k in allowed),label+': hidden/missing synthesis forwards')
    return {'method':method,'steps':n,'source_backward_calls':expected_calls,'calibration_calls':calls['calibration_source_backward_calls']}


def audit(phase):
    base.ISSUES.clear();base.HASHES.clear()
    plan=base.read(RUN/'plan.json');info=plan['cohorts'][phase];folder=RUN/phase
    base.VARIANTS=tuple(info['variants'])
    for path,digest in plan['bindings'].items():base.bind(path,digest)
    previous=base.read(ROOT/'execution/plan.json')
    for path,digest in previous['frozen_files'].items():base.bind(path,digest)
    base.verify(plan['source_only'] and not plan['target_models_evaluated'], 'Generation scope changed')
    base.bind(info['manifest'],info['sha256']);cohort=base.rows(info['manifest'])
    count=info['count'];base.verify(len(cohort)==len({r['id'] for r in cohort})==count,'Incomplete cohort')
    generation=base.read(folder/'generation-result.json')
    base.verify(generation['status']=='completed' and generation['actual_count']==generation['count']==count
        and generation['cohort']==phase and not generation['failures'],'Incomplete/failed generation')
    base.verify(generation['source_only'] and generation['target_models_evaluated']==[],'Unregistered generation queries')
    base.verify(generation['plan_sha256']==base.sha(RUN/'plan.json'),'Wrong generation registration')
    expected_backend=base.read(ROOT/'execution_scaled/probe-result.json')['backend']
    base.verify(generation['backend']==expected_backend,'Strict generation backend changed')
    if phase=='validation':
        freeze=base.read(RUN/'validation-freeze.json')
        base.verify(freeze['status']=='validate','Missing candidate freeze')
        for path,digest in freeze['bindings'].items():base.bind(path,digest)
        base.verify(datetime.fromisoformat(freeze['created_at_utc'])<datetime.fromisoformat(generation['started_at_utc']),'Validation precedes selection')
        base.bind(RUN/'validation-feature-cache.json',generation['validation_feature_cache_sha256'])
        cached={r['id']:r for r in base.read(RUN/'validation-feature-cache.json')['records']}
        base.verify(set(cached)=={r['id'] for r in cohort},'Incomplete validation feature-cache identity')
    base.bind(folder/'generated.jsonl',generation['generated_sha256']);records=base.rows(folder/'generated.jsonl')
    base.verify([r['id'] for r in records]==[r['id'] for r in cohort],'Generated cohort/order differs')
    checks=[]
    for record,row in zip(records,cohort):
        uid=row['id'];base.verify(base.read(folder/'records'/f'{uid}.json')==record,uid+': individual record differs')
        base.verify(record['text']==row['text'] and record['speaker']==row['speaker']
            and record['input_audio_sha256']==row['audio_sha256'],uid+': input/reference changed')
        base.verify(record['role']==phase and record['count']==count and record['source_only']
            and record['plan_sha256']==base.sha(RUN/'plan.json'),uid+': record scope changed')
        base.verify(set(record['audio'])==set(info['variants']),uid+': variant coverage differs')
        base.bind(row['audio'],row['audio_sha256']);original,rate=sf.read(row['audio'],dtype='float32')
        base.verify(rate==16000 and original.shape==(row['samples'],),uid+': source metadata differs')
        audio={v:base.waveform(a,uid+'/'+v) for v,a in record['audio'].items()}
        base.verify(np.array_equal(audio['clean'],original),uid+': changed clean waveform')
        q=base.tensor(record['diagnostics']['q'],uid+'/q')
        if phase=='validation':
            base.verify(record['diagnostics']['q']==cached[uid]['q'] and cached[uid]['audio_sha256']==row['audio_sha256'],uid+': mismatched cached features')
        for variant,diag in record['diagnostics'].items():
            if 'receipt' not in diag:continue
            base.bind(diag['receipt'],diag['sha256']);receipt=base.read(diag['receipt'])
            check=audit_receipt(receipt,uid+'/'+variant);checks.append({'id':uid,'variant':variant,**check})
            delta=base.tensor(diag['delta'],uid+'/'+variant+'/delta')
            mel=receipt['method']=='mel_eot'
            reference=base.tensor(record['diagnostics']['mel'],uid+'/mel') if mel else q
            base.verify(delta.shape==reference.shape,uid+'/'+variant+': wrong latent shape')
            radius=float(delta.double().norm()/reference.double().norm())
            base.verify(radius<=.100001,uid+'/'+variant+': actual latent radius violation')
            base.close(radius,receipt['relative_radius'],uid+'/'+variant+': final radius differs',atol=5e-7)
            expected_length=len(original) if mel else q.shape[0]*320
            base.verify(len(audio[variant])==expected_length,uid+'/'+variant+': decoder duration contract changed')
            if mel:base.verify(float(delta.abs().max())<=np.log(2)+1e-6,uid+'/'+variant+': log-mel box violated')
            base.verify(diag['method']==receipt['method'] and diag['step']==receipt['selected_iterate'],uid+'/'+variant+': endpoint metadata differs')
        if phase=='qualification':
            base.verify(len(audio['bigvgan_reconstruction'])==len(original),uid+': BigVGAN changed duration')
    scores=base.audit_scores(folder,cohort,records,generation,plan,previous)
    value={'status':'fail' if base.ISSUES else 'pass','phase':phase,'count':count,'source_only':True,
        'target_models_evaluated':[],'issues':base.ISSUES,'hashes_checked':len(base.HASHES),
        'trajectory_checks':checks,'independent_score_recount':scores,
        'scope':'Saved artifacts/counters/latent constraints/source counts. Neural gradient correctness supplied by separate real fit probe; no human/transfer claim.'}
    (folder/'audit.json').write_text(__import__('json').dumps(value,indent=2)+'\n')
    print(__import__('json').dumps({k:value[k] for k in ('status','phase','count','issues')},indent=2))
    return bool(base.ISSUES)


def audit_transfer():
    base.ISSUES.clear();base.HASHES.clear()
    plan=base.read(RUN/'plan.json');folder=RUN/'transfer';result=base.read(folder/'results.json')
    for path,digest in plan['bindings'].items():base.bind(path,digest)
    freeze=base.read(RUN/'transfer-freeze.json')
    base.verify(freeze['status']=='evaluate_transfer','Targets opened without passing frozen source validation')
    for path,digest in freeze['bindings'].items():base.bind(path,digest)
    registration=base.read(folder/'registration.json')
    base.verify(datetime.fromisoformat(freeze['created_at_utc'])<datetime.fromisoformat(registration['created_at_utc']),
        'Target registration did not follow source candidate freeze')
    base.verify(registration['plan_sha256']==base.sha(RUN/'plan.json') and registration['target_configs']==plan['target_configs'],
        'Target protocol identity changed')
    base.bind(RUN/'validation/generated.jsonl',registration['input_manifest_sha256'])
    base.bind(folder/'scores.jsonl',result['scores_sha256']);scores=base.rows(folder/'scores.jsonl')
    records=base.rows(RUN/'validation/generated.jsonl');allowed={r['id']:r for r in records}
    variants=plan['cohorts']['validation']['variants'];conditions=registration['conditions'];targets=list(plan['target_configs'])
    expected={(uid,t,d,v) for uid in allowed for t in targets for d in conditions for v in variants}
    base.verify(len(scores)==len(expected) and Counter((r['id'],r['target'],r['defense'],r['variant']) for r in scores)==Counter(expected),
        'Incomplete/duplicate target-condition-variant scores')
    cached=base.read(folder/'post-defense-quality.json')['records']
    cache={(r['condition'],r['input_audio_sha256']):r for r in cached}
    for item in cached:base.waveform(item['audio'],'transformed/'+item['input_audio_sha256'])
    for r in scores:
        uid=r['id'];record=allowed[uid];variant=r['variant']
        item=cache[r['defense'],record['audio'][variant]['sha256']]
        base.verify(r['reference']==record['text'] and r['speaker']==record['speaker'],uid+': target reference/speaker differs')
        base.verify(r['input_audio_sha256']==record['audio'][variant]['sha256'] and r['evaluated_audio_sha256']==item['audio']['sha256'],
            uid+': wrong target waveform scored')
        base.verify(r['quality']==item['quality'],uid+': target post-defense quality differs')
        reference,hypothesis=base.normalize(r['reference']),base.normalize(r['hypothesis'])
        base.verify(r['reference_normalized']==reference and r['hypothesis_normalized']==hypothesis,uid+': target normalization differs')
        distance=base.word_distance(reference.split(),hypothesis.split())
        base.verify(distance==r['substitutions']+r['deletions']+r['insertions'] and len(reference.split())==r['reference_units'],
            uid+': independent target word recount differs')
        base.verify(base.read(folder/'records'/r['target']/r['defense']/f'{uid}--{variant}.json')==r,uid+': target per-score receipt differs')
    for target in targets:
        for defense in conditions:
            for variant in variants:
                pool=[r for r in scores if r['target']==target and r['defense']==defense and r['variant']==variant]
                summary=result['summaries'][target][defense][variant]
                base.verify(summary['count']==len(pool)==76,'Target summary denominator differs')
                totals={k:sum(r[k] for r in pool) for k in ('substitutions','deletions','insertions','reference_units')}
                for key,value in totals.items():base.verify(summary[key]==value,'Target edit totals differ')
                wer=100*sum(totals[k] for k in ('substitutions','deletions','insertions'))/totals['reference_units']
                base.close(summary['error_percent'],wer,'Target WER aggregate differs',atol=1e-12)
                for key in ('utmos','dnsmos','nisqa'):
                    base.close(summary['quality'][key],sum(r['quality'][key] for r in pool)/76,'Target MOS aggregate differs',atol=1e-12)
                base.close(summary['empty_output_rate'],sum(not r['hypothesis_normalized'] for r in pool)/76,'Target empty-rate differs',atol=0)
                capped=100*sum(min(r['reference_units'],r['substitutions']+r['deletions']+r['insertions']) for r in pool)/totals['reference_units']
                base.close(summary['capped_corpus_error_percent'],capped,'Capped target WER differs',atol=1e-12)
    speakers=sorted({r['speaker'] for r in records})
    weights=np.random.default_rng(20261022).multinomial(len(speakers),[1/len(speakers)]*len(speakers),size=10000)
    outcomes=[]
    for defense in conditions:
        values=np.asarray([[sum(r['substitutions']+r['deletions']+r['insertions'] for r in scores
            if r['defense']==defense and r['speaker']==speaker and r['variant']==v) for v in ('candidate','control')]
            +[sum(r['reference_units'] for r in scores if r['defense']==defense and r['speaker']==speaker and r['variant']=='candidate')]
            for speaker in speakers])
        point=100*(values[:,0].sum()-values[:,1].sum())/values[:,2].sum();sample=weights@values
        interval=np.quantile(100*(sample[:,0]-sample[:,1])/sample[:,2],[.025,.975])
        saved=result['comparisons'][defense]
        base.close(saved['family_balanced_delta_wer_points'],float(point),'Transfer macro delta differs',atol=1e-12)
        base.verify(np.allclose(saved['speaker_bootstrap95'],interval,atol=1e-12,rtol=0),'Transfer bootstrap differs')
        passed=bool(point>=0 and interval[0]>=-3)
        base.verify(saved['preservation_screen']==passed,'Transfer preservation flag differs')
        if defense!='waveguard2000':outcomes.append(passed)
    base.verify(result['preservation_status']==('pass' if all(outcomes) else 'not_established'),'Overall preservation verdict differs')
    raw=result['comparisons']['raw']
    base.verify(result['raw_transfer_superiority_signal']==bool(raw['family_balanced_delta_wer_points']>=3 and raw['speaker_bootstrap95'][0]>0),
        'Raw transfer superiority flag differs')
    base.verify(result['candidate_reselected'] is False and result['human_intelligibility_established'] is False
        and result['paper_training_defenses_measured'] is False,'Unsupported scientific scope claim')
    value={'status':'fail' if base.ISSUES else 'pass','count':76,'scores_checked':len(scores),'issues':base.ISSUES,
        'hashes_checked':len(base.HASHES),'scope':'Independent target word recount, post-defense asset/hash/MOS bindings, denominators, bootstrap and scientific verdict; no neural replay'}
    (folder/'audit.json').write_text(__import__('json').dumps(value,indent=2)+'\n')
    print(__import__('json').dumps(value,indent=2));return bool(base.ISSUES)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True)
    phase=p.parse_args().phase
    raise SystemExit(audit_transfer() if phase=='transfer' else audit(phase))
