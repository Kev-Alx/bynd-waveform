"""Controlled full-trajectory loss ablations at the paper's feature budget.

These are diagnostic ablations, not replacement full-objective candidates.
Production sources and all previous research results remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from beyondwaveform.attacks import freeze,project_relative_ball_,relative_radius
from beyondwaveform.config import AttackConfig,SpectralConfig
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,save_audio,seed_everything,write_jsonl
from beyondwaveform.models.whisper import load_whisper
from beyondwaveform.pipeline import config_fingerprint
from parallel_jobs import Supervisor,now
from probe_deterministic_execution import ExplicitReflectionReference,equivalent_surrogate,save_tensor,tensor_tree
from probe_numerical_repeatability import backend_manifest,backend_mode,tensor_value_sha256
import run_deterministic_execution as numerical
import run_quality_research as protocol
import run_quality_refinement as stages
import run_stable_quality_research as stable

DEFAULT=ROOT/'reproduction-runs/paper-quality-stable-ablations'
PARENT=stable.DEFAULT
BASELINE=PARENT/'pilot/strong-sum-lr-0.1'
NAMES=('without-tv','without-hf','without-perceptual')
REVIEW=ROOT/'reproduction-runs/stable-execution-completion-review-20261020'


def cases(base):
    result=[]
    for name,tv,hf in zip(NAMES,(False,True,False),(True,False,False)):
        config=copy.deepcopy(base)
        config['attack'].update(temporal=tv,high_frequency=hf)
        result.append({'name':name,'config':config})
    return result


def active_terms(attack):
    return {'asr':attack.attack,'tv':attack.temporal and attack.lambda_perc>0,
        'hf':attack.high_frequency and attack.lambda_perc>0 and attack.alpha_hf>0}


def ablation_objective(q,delta,text,vocoder,surrogate,attack,reference):
    audio=vocoder(q+delta)
    nll=surrogate.nll(audio,text)
    enabled=active_terms(attack)
    tv,hf=reference.terms(q+delta,audio,temporal=enabled['tv'],high_frequency=enabled['hf'])
    loss=-nll+attack.lambda_perc*(tv+attack.alpha_hf*hf)
    protocol.require(torch.isfinite(loss).item(),'Nonfinite ablation objective')
    return loss,torch.stack([loss.detach(),nll.detach(),tv.detach(),hf.detach()])


def collect(q,x,text,vocoder,surrogate,attack,spectral,directory):
    protocol.require(attack.steps==50 and attack.optimizer=='adam' and attack.lr_schedule=='constant' and
        attack.initialization=='zero' and attack.attack,'Only the fixed fifty-update Adam protocol is supported')
    q,x=q.detach(),x.detach()
    delta=torch.zeros_like(q,requires_grad=True)
    reference=ExplicitReflectionReference(q,x,spectral,attack.eps)
    optimizer=torch.optim.Adam([delta],lr=attack.learning_rate)
    trace=[]
    for step in range(1,51):
        optimizer.zero_grad(set_to_none=True)
        loss,values=ablation_objective(q,delta,text,vocoder,surrogate,attack,reference)
        gradient,=torch.autograd.grad(loss,delta)
        protocol.require(torch.isfinite(gradient).all().item(),'Nonfinite ablation gradient')
        delta.grad=gradient
        optimizer.step()
        project_relative_ball_(delta,q,attack.rho,attack.eps)
        radius=relative_radius(delta,q,attack.eps).item()
        protocol.require(torch.isfinite(delta).all().item() and radius<=attack.rho+1e-6,'Invalid projected iterate')
        trace.append({'step':step,'loss_values':values.tolist(),'loss_value_sha256':tensor_value_sha256(values),
            'gradient_value_sha256':tensor_value_sha256(gradient),'radius':radius,
            'optimizer_value_receipt':tensor_tree(optimizer.state_dict()),
            'delta':save_tensor(delta,directory/f'delta-{step:02d}.pt')})
        if step in (1,10,25,50):
            print(f'Ablation trajectory {directory.name}: update {step}/50, radius={radius:.8f}',flush=True)
    with torch.no_grad():audio=vocoder(q+delta)
    final_audio=save_tensor(audio,directory/'final-audio.pt')
    state_path=directory/'final-optimizer.pt';torch.save(optimizer.state_dict(),state_path)
    protocol.require(not any(p.grad is not None for m in (vocoder,surrogate) for p in m.parameters()),'Accumulated model gradients')
    return {'trace':trace,'actual_optimization_updates':50,'active_terms':active_terms(attack),'final_audio':final_audio,
        'final_optimizer':{'path':str(state_path.resolve()),'file_sha256':digest(state_path),
            'value_receipt':tensor_tree(optimizer.state_dict())}}


def prepare(output):
    protocol.require(not (output/'plan.json').exists(),'Plan already exists')
    parent=protocol.read(PARENT/'plan.json');stable.check(parent)
    protocol.require(protocol.read(PARENT/'job.json')['status']=='complete','Parent job not complete')
    review=protocol.read(REVIEW/'review.json')
    protocol.require(review['status']=='complete' and review['all_checks_passed'] and
        review['plan_sha256']==digest(PARENT/'plan.json') and review['frozen_inputs_verified']==len(parent['frozen_files']) and
        review['projected_iterates_verified']==3600 and review['score_rows_and_WAV_bindings_verified']==216 and
        review['audit_source_sha256']==digest(REVIEW/'audit.py'),'Complete independent saved-output review first')
    base=copy.deepcopy(parent['cases'][0]['config'])
    stable.validate_case({'name':stable.NAMES[0],'config':base})
    entries=cases(base)
    files=[Path(__file__).resolve(),ROOT/'tests/test_stable_ablation_research.py',ROOT/'docs/QUALITY_STABLE_ABLATIONS.md',
        PARENT/'plan.json',PARENT/'job.json',PARENT/'results.json',PARENT/'report.md',
        REVIEW/'review.json',REVIEW/'report.md']
    spectral_review=ROOT/'reproduction-runs/hf-pooling-saved-audio-20261020'
    files.extend(spectral_review/name for name in ('audit.py','plan.json','report.json','report.md'))
    files.extend([ROOT/'beyondwaveform/hf_pooling_diagnostics.py',ROOT/'tests/test_hf_pooling_diagnostics.py'])
    files.extend(p for p in REVIEW.glob('*.py'))
    files.extend(BASELINE/name for name in ('run.json','generated.jsonl','evaluation/summary.json','evaluation/utterances.jsonl','verification.json'))
    records=read_jsonl(BASELINE/'generated.jsonl')
    for index,record in enumerate(records):
        receipt=protocol.read(record['trajectory_receipt']['path'])
        files.extend(Path(asset['path']) for asset in record['audio'].values())
        files.extend([Path(record['delta']['path']),Path(record['trajectory_receipt']['path']),
            Path(receipt['final_audio']['path']),Path(receipt['final_optimizer']['path'])])
        if index<3:files.extend(Path(row['delta']['path']) for row in receipt['trace'])
    for entry in entries:
        path=output/'configs'/(entry['name']+'.json');atomic_json(path,entry['config']);path.chmod(0o444)
        entry['config_path']=str(path);files.append(path)
    frozen=dict(parent['frozen_files']);frozen.update({str(p.resolve()):digest(p) for p in files})
    full=next(r for r in protocol.read(PARENT/'results.json')['results'] if r['name']==stable.NAMES[0])
    plan={'created_at':now(),'frozen_files':frozen,'runtime':parent['runtime'],
        'additional_dependencies':parent['additional_dependencies'],'pilot':parent['pilot'],
        'cache_audit':parent['cache_audit'],'feature_artifacts':parent['feature_artifacts'],'quality':parent['quality'],
        'execution':parent['execution'],'base_config':base,'cases':entries,'baseline':full,'discord_enabled':False,
        'control_preflight_ids':parent['pilot']['ids'][:3],
        'scope':'Full-trajectory diagnostic ablations on the same fixed 24 development clips: remove TV, remove HF, remove both perceptual terms. Fixed rho=.1, lambda=1, alpha=1, Adam LR=.1, fifty updates, native references/decoder/scorers and strict explicit-reflection generation. Cached repeatable full-objective control; no parameter selection, candidate promotion, fresh48 or 999 test. Ablated objectives cannot establish reproduction of the full paper objective.'}
    atomic_json(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    return plan


def check(plan):
    import importlib.metadata
    protocol.require(protocol.runtime()==plan['runtime'] and all(importlib.metadata.version(k)==v
        for k,v in plan['additional_dependencies'].items()),'Changed runtime')
    protocol.require(plan['execution']=={'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True} and
        len(plan['pilot']['ids'])==24 and plan['control_preflight_ids']==plan['pilot']['ids'][:3], 'Changed execution/cohort')
    wanted=cases(plan['base_config'])
    protocol.require([{k:r[k] for k in ('name','config')} for r in plan['cases']]==wanted,'Changed ablation definitions')
    stable.validate_case({'name':stable.NAMES[0],'config':plan['base_config']})
    for path,expected in plan['frozen_files'].items():
        protocol.require(digest(path)==expected,f'Changed frozen ablation input: {path}')
    protocol.require(digest(plan['pilot']['manifest'])==plan['pilot']['sha256'] and
        [r['id'] for r in read_jsonl(plan['pilot']['manifest'])]==plan['pilot']['ids'],'Changed pilot manifest')


def models(plan,config):
    vocoder=stable.load_vocoder(plan,{'config':config})
    native,metadata=load_whisper(config['surrogate'],'cuda');freeze(native)
    protocol.require(metadata==protocol.read(BASELINE/'run.json')['surrogate'],'Changed surrogate artifacts')
    return vocoder,equivalent_surrogate(native),metadata


def control_preflight(output,plan):
    path=output/'control-preflight.json'
    protocol.require(not path.exists() and not (output/'control-trajectories').exists(),'Preserve completed/partial preflight evidence')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Fresh CUDA process needs workspace config')
    base=plan['base_config'];cached={r['id']:r for r in protocol.read(plan['cache_audit'])['items']}
    records={r['id']:r for r in read_jsonl(BASELINE/'generated.jsonl')}
    source=read_jsonl(plan['pilot']['manifest'])[:3];new=[];old=[]
    with backend_mode('strict_math'):
        seed_everything(base['seed']);vocoder,surrogate,_=models(plan,base);flags=backend_manifest()
        for row in source:
            uid=row['id'];record=records[uid];seed_everything(record['seed'])
            q=torch.load(cached[uid]['q_path'],map_location='cuda',weights_only=True)
            x=load_audio(row['audio']).cuda()
            result=collect(q,x,row['text'],vocoder,surrogate,AttackConfig(**base['attack']),SpectralConfig(**base['spectral']),output/'control-trajectories'/uid)
            new.append({'id':uid,'seed':record['seed'],**result})
            old.append(protocol.read(record['trajectory_receipt']['path']))
    comparison=numerical.compare_trajectories({'items':old},{'items':new})
    atomic_json(path,{'status':'complete','backend':flags,'comparison':comparison,'passed':comparison['passed'],
        'scope':'All terms enabled: new flag-aware collector must match all 150 updates and final waveforms/optimizer states of the frozen full-objective control.'})
    protocol.require(comparison['passed'],'New collector does not reproduce the full-objective control')


def generate(output,plan,case):
    folder=output/'pilot'/case['name'];config=case['config']
    protocol.require(not (folder/'run.json').exists(),'Preserve partial/completed generation before retry')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Fresh CUDA process needs workspace config')
    protocol.require(protocol.read(output/'control-preflight.json')['passed'],'Control preflight did not pass')
    original={r['id']:r for r in read_jsonl(BASELINE/'generated.jsonl')}
    cache={r['id']:r for r in protocol.read(plan['cache_audit'])['items']}
    attack,spectral=AttackConfig(**config['attack']),SpectralConfig(**config['spectral'])
    records=[]
    with backend_mode('strict_math'):
        seed_everything(config['seed']);vocoder,surrogate,surrogate_meta=models(plan,config)
        expected=stable.identity(plan,case)
        metadata={**expected,'fingerprint':config_fingerprint(expected),'manifest':plan['pilot']['manifest'],'roles':['pilot'],
            'status':'running','features':plan['feature_artifacts'],'surrogate':surrogate_meta,
            'generation_runtime':protocol.runtime(),'generation_backend':backend_manifest(),'active_terms':active_terms(attack),
            'research_method':'loss ablation, not full-objective candidate','scope':plan['scope']}
        atomic_json(folder/'run.json',metadata)
        for index,row in enumerate(read_jsonl(plan['pilot']['manifest']),1):
            uid=row['id'];parent=original[uid];seed_everything(parent['seed'])
            protocol.require(digest(row['audio'])==row['audio_sha256'] and digest(cache[uid]['q_path'])==cache[uid]['q_sha256'],'Changed q/audio')
            q=torch.load(cache[uid]['q_path'],map_location='cuda',weights_only=True).detach();x=load_audio(row['audio']).cuda()
            value=collect(q,x,row['text'],vocoder,surrogate,attack,spectral,folder/'trajectories'/uid)
            delta=numerical.verify_tensor(value['trace'][-1]['delta']).cuda()
            with torch.no_grad():audio=vocoder(q+delta);reconstruction=vocoder(q)
            protocol.require(tensor_value_sha256(audio)==value['final_audio']['value_sha256'] and
                torch.equal(reconstruction.cpu(),load_audio(parent['audio']['reconstruction']['path'],max_seconds=None)),
                'Resynthesis/clean reconstruction changed')
            assets={k:parent['audio'][k] for k in ('clean','reconstruction')}
            assets['adversarial']=save_audio(folder/'audio/adversarial'/(uid+'.wav'),audio)
            receipt=folder/'trajectories'/uid/'receipt.json'
            atomic_json(receipt,{'id':uid,'seed':parent['seed'],'q_path':cache[uid]['q_path'],'q_sha256':cache[uid]['q_sha256'],
                'q_value_sha256':tensor_value_sha256(q),**value})
            trace=[{'step':r['step'],'loss_total':r['loss_values'][0],'nll':r['loss_values'][1],
                'tv_ratio':r['loss_values'][2],'hf_ratio':r['loss_values'][3],'radius_after_update':r['radius']} for r in value['trace']]
            record={'id':uid,'text':row['text'],'language':row['language'],'role':'pilot','input_audio_sha256':row['audio_sha256'],
                'seed':parent['seed'],'audio':assets,'trace':trace,'delta':value['trace'][-1]['delta'],
                'trajectory_receipt':{'path':str(receipt),'sha256':digest(receipt)},'actual_optimization_updates':50,
                'returned_delta':'projected update 50','active_terms':active_terms(attack)}
            atomic_json(folder/'records'/(uid+'.json'),record);records.append(record)
            print(f'Generated {case["name"]}: {index}/24 [{uid}]',flush=True)
    write_jsonl(folder/'generated.jsonl',records)
    metadata.update(status='complete',count=24,generated_sha256=digest(folder/'generated.jsonl'))
    atomic_json(folder/'run.json',metadata)


def verify_loss_receipt(record,receipt,enabled):
    protocol.require(record['active_terms']==receipt['active_terms']==enabled,'Changed enabled terms')
    protocol.require(len(record['trace'])==len(receipt['trace'])==50 and receipt['actual_optimization_updates']==50,
        'Incomplete loss-removal evidence')
    for step,(row,saved) in enumerate(zip(record['trace'],receipt['trace']),1):
        protocol.require(row['step']==saved['step']==step and
            [row[k] for k in ('loss_total','nll','tv_ratio','hf_ratio')]==saved['loss_values'] and
            row['radius_after_update']==saved['radius'],'Exported trace differs from actual trajectory receipt')
        protocol.require((enabled['tv'] or saved['loss_values'][2]==0) and
            (enabled['hf'] or saved['loss_values'][3]==0),'Removed loss term is still active')


def verify(output,plan,case):
    stable.verify(output,plan,case)
    enabled=active_terms(AttackConfig(**case['config']['attack']))
    folder=output/'pilot'/case['name']
    metadata=protocol.read(folder/'run.json')
    protocol.require(metadata['active_terms']==enabled,'Metadata loss flags changed')
    full=protocol.read(BASELINE/'evaluation/summary.json');measured=protocol.read(folder/'evaluation/summary.json')
    protocol.require(all(measured[k]==full[k] for k in ('target','target_artifacts','metric','normalization',
        'evaluation_scope','evaluation_backend','quality_provenance')),'Evaluation protocol differs from full-objective control')
    for record in read_jsonl(folder/'generated.jsonl'):
        receipt=protocol.read(record['trajectory_receipt']['path'])
        verify_loss_receipt(record,receipt,enabled)
        final_audio=numerical.verify_tensor(receipt['final_audio'])
        protocol.require(torch.equal(final_audio,load_audio(record['audio']['adversarial']['path'],max_seconds=None)),
            'Terminal audio receipt differs from exported WAV')
        state=receipt['final_optimizer']
        protocol.require(digest(state['path'])==state['file_sha256'] and
            tensor_tree(torch.load(state['path'],map_location='cpu',weights_only=True))==
            state['value_receipt']==receipt['trace'][-1]['optimizer_value_receipt'],'Final optimizer receipt changed')
    path=output/'pilot'/case['name']/'verification.json';receipt=protocol.read(path)
    receipt.update(active_terms=enabled,disabled_terms_verified_zero=True);atomic_json(path,receipt)


def report(output,rows,status,plan):
    baseline=plan['baseline']
    atomic_json(output/'results.json',{'status':status,'updated_at':now(),'baseline':baseline,'results':rows,'scope':plan['scope']})
    lines=['# Stable paper-budget loss ablations','',f'Status: **{status}**.','',
        '| Objective | WER (%) | DNSMOS | NISQA | UTMOS | Delta WER (pp) | Delta UTMOS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in [dict(baseline,name='Full objective (frozen control)'),*rows]:
        q=row['quality'];lines.append(f"| {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {row['error_percent']-baseline['error_percent']:+.4f} | {q['utmos']-baseline['quality']['utmos']:+.4f} |")
    lines+=['',plan['scope'],'','These diagnostics test complete trajectories; local moment-matched gradient-removal probes alone did not establish these effects. The paper’s qualitative loss-guidance claim uses WER and DNSMOS. No ablation is promoted as a reproduction of the full objective, regardless of its scores.','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--stage',choices=('control-preflight','generate','evaluate','verify'))
    parser.add_argument('--case',choices=NAMES);args=parser.parse_args()
    os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD']='0';output=args.output.resolve()
    plan=prepare(output) if args.prepare else protocol.read(output/'plan.json');check(plan)
    if args.prepare or args.check_only:
        print(f'Stable ablation preflight passed: {len(plan["frozen_files"])} frozen inputs.',flush=True);return
    if args.stage:
        if args.stage=='control-preflight':control_preflight(output,plan)
        else:
            protocol.require(args.case is not None,'Stage requires a fixed ablation')
            case=next(r for r in plan['cases'] if r['name']==args.case)
            {'generate':generate,'evaluate':stable.evaluate_case,'verify':verify}[args.stage](output,plan,case)
        check(plan);return
    supervisor=Supervisor(output);rows=[]
    try:
        report(output,rows,'running',plan)
        base=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output)]
        stages.phase(supervisor,'full-objective-control-preflight',['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',*base,'--stage','control-preflight'])
        for case in plan['cases']:
            for stage in ('generate','evaluate','verify'):
                command=[*base,'--case',case['name'],'--stage',stage]
                if stage in ('generate','verify'):command=['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',*command]
                stages.phase(supervisor,case['name']+'-'+stage,command)
            folder=output/'pilot'/case['name'];summary=protocol.read(folder/'evaluation/summary.json')
            rows.append({'name':case['name'],'pool':'pilot','run':str(folder),**summary['summary']['adversarial'],
                'summary_sha256':digest(folder/'evaluation/summary.json'),'audit_sha256':digest(folder/'verification.json')})
            report(output,rows,'running',plan)
        check(plan);report(output,rows,'complete',plan);supervisor.finish(scope=plan['scope'])
    except BaseException as error:
        report(output,rows,'failed',plan);supervisor.fail(error);raise


if __name__=='__main__':main()
