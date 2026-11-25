"""Measure three existing attack settings after numerical validation passes.

The execution policy is explicit reflection plus strict math for generation.
Saved-audio decoding and MOS use the unchanged evaluation implementation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from beyondwaveform.attacks import freeze,relative_radius
from beyondwaveform.config import AttackConfig,SpectralConfig
from beyondwaveform.evaluation import aggregate_errors,error_counts
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,save_audio,seed_everything,write_jsonl
from beyondwaveform.models.whisper import load_whisper
from beyondwaveform.pipeline import config_fingerprint,evaluate
from parallel_jobs import Supervisor,now
from probe_deterministic_execution import equivalent_surrogate,trajectory
from probe_feature_gradients import load_frozen_vocoder
from probe_numerical_repeatability import backend_manifest,backend_mode,tensor_value_sha256
import run_deterministic_execution as numerical
import run_quality_research as protocol
import run_quality_refinement as stages

DEFAULT=ROOT/'reproduction-runs/paper-quality-stable-execution'
VALIDATION=numerical.DEFAULT
FINALIZATION=ROOT/'reproduction-runs/paper-quality-finalization'
NAMES=('strong-sum-lr-0.1','quality-sum-lr-0.02','timestamp-sum-lr-0.05')
TARGETS={'wer':75.43,'dnsmos':2.954,'nisqa':3.526,'utmos':3.590}


def cases(previous):
    strong=copy.deepcopy(previous['families'][0]['config'])
    quality=copy.deepcopy(strong);quality['attack']['learning_rate']=.02
    timestamp=copy.deepcopy(previous['families'][1]['config'])
    return [{'name':name,'config':config} for name,config in zip(NAMES,(strong,quality,timestamp))]


def validate_case(case):
    a,s=AttackConfig(**case['config']['attack']),SpectralConfig(**case['config']['spectral'])
    lr,prefix={NAMES[0]:(.1,'no_timestamps'),NAMES[1]:(.02,'no_timestamps'),NAMES[2]:(.05,'initial_timestamp')}[case['name']]
    surrogate=case['config']['surrogate']
    protocol.require(a.steps==50 and a.rho==.1 and a.lambda_perc==1 and a.alpha_hf==1 and a.eps==1e-8 and
        a.optimizer=='adam' and a.learning_rate==lr and a.initialization=='zero' and a.lr_schedule=='constant' and
        a.attack and a.temporal and a.high_frequency and s.sample_rate==16000 and s.cutoff_hz==6000 and
        surrogate['nll_reduction']=='sum' and surrogate.get('nll_timestamp_policy','no_timestamps')==prefix and
        surrogate.get('nll_transcript_policy','original')=='original','Changed predeclared performance anchor')


def prepare(output):
    protocol.require(not (output/'plan.json').exists(),'Plan already exists')
    old=protocol.read(VALIDATION/'plan.json');numerical.check(old)
    protocol.require(protocol.read(VALIDATION/'job.json')['status']=='complete' and
        protocol.read(VALIDATION/'progress.json')['gate_passed'] and
        protocol.read(VALIDATION/'fixed-gradients.json')['passed'] and
        protocol.read(VALIDATION/'trajectory-comparison.json')['passed'],'Complete numerical validation first')
    numerical.verify_fixed(protocol.read(VALIDATION/'fixed-gradients.json'))
    rebuilt=numerical.compare_trajectories(protocol.read(VALIDATION/'trajectory-repeat-0.json'),
        protocol.read(VALIDATION/'trajectory-repeat-1.json'))
    protocol.require(rebuilt==protocol.read(VALIDATION/'trajectory-comparison.json') and rebuilt['passed'],
        'Numerical trajectory evidence no longer matches')
    finalization=protocol.read(FINALIZATION/'plan.json')
    parent=protocol.read(old['parent_plan'])
    entries=cases(finalization)
    files=[Path(__file__).resolve(),ROOT/'tests/test_stable_quality_research.py',
        ROOT/'docs/QUALITY_STABLE_EXECUTION.md',VALIDATION/'plan.json',VALIDATION/'job.json',
        VALIDATION/'progress.json',VALIDATION/'results.json',VALIDATION/'fixed-gradients.json',
        VALIDATION/'trajectory-comparison.json',FINALIZATION/'results.json']
    for name in ('trajectory-repeat-0.json','trajectory-repeat-1.json'):
        path=VALIDATION/name;files.append(path)
        for item in protocol.read(path)['items']:
            files.extend(Path(r['delta']['path']) for r in item['trace'])
            files.extend(Path(item[k]['path']) for k in ('final_audio','final_optimizer'))
    for item in protocol.read(VALIDATION/'fixed-gradients.json')['items']:
        files.extend(Path(r['gradient']['path']) for state in item['states'].values() for r in state['repeats'])
    for entry in entries:
        validate_case(entry)
        path=output/'configs'/(entry['name']+'.json')
        atomic_json(path,entry['config']);path.chmod(0o444);files.append(path)
        entry['config_path']=str(path)
    frozen=dict(old['frozen_files']);frozen.update({str(p.resolve()):digest(p) for p in files})
    plan={'created_at':now(),'frozen_files':frozen,'runtime':old['runtime'],
        'additional_dependencies':old['additional_dependencies'],'parent_plan':old['parent_plan'],
        'pilot':parent['pools']['pilot'],'feature_artifacts':parent['feature_artifacts'],
        'cache_audit':str(Path(old['parent_plan']).parent/'parent-audits/pilot.json'),
        'cases':entries,'quality':finalization['quality'],'targets':TARGETS,'discord_enabled':False,
        'execution':old['execution'],'numerical_validation_plan':str(VALIDATION/'plan.json'),
        'cached_context':[{k:row[k] for k in ('name','run','error_percent','quality','summary_sha256')}
            for row in protocol.read(FINALIZATION/'results.json')['results'] if row['mode']=='terminal' and row['pool']=='pilot'],
        'scope':'Three previously studied settings on the existing fixed 24-clip development pilot. Explicit reflected padding and strict math generation; native saved-audio ASR and fixed MOS scorers. No learning-rate grid, subset changes, automatic promotion, fresh confirmation or 999-clip test. Cached native scores are descriptive context, not paired same-run controls.'}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'plan.json',plan);(output/'plan.json').chmod(0o444)
    return plan


def check(plan):
    import importlib.metadata
    protocol.require(protocol.runtime()==plan['runtime'] and all(importlib.metadata.version(k)==v
        for k,v in plan['additional_dependencies'].items()),'Changed runtime')
    protocol.require([r['name'] for r in plan['cases']]==list(NAMES) and len(plan['pilot']['ids'])==24 and
        plan['targets']==TARGETS and plan['execution']=={'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True},
        'Changed study scope')
    for case in plan['cases']:validate_case(case)
    for path,expected in plan['frozen_files'].items():
        protocol.require(digest(path)==expected,f'Changed frozen stable-quality input: {path}')
    protocol.require(digest(plan['pilot']['manifest'])==plan['pilot']['sha256'] and
        [r['id'] for r in read_jsonl(plan['pilot']['manifest'])]==plan['pilot']['ids'],'Changed pilot cohort')


def identity(plan,case):
    config={'method':'feature_explicit_reflection_execution','parent_config':case['config'],
        'generation_execution':plan['execution'],'returned_delta':'projected update 50','actual_updates':50}
    return {'config':config,'manifest_sha256':plan['pilot']['sha256'],'ids':plan['pilot']['ids']}


def load_vocoder(plan,case):
    return load_frozen_vocoder(case['config']['features'],plan['feature_artifacts'],'cuda')[0]


def generate(output,plan,case):
    folder=output/'pilot'/case['name']
    protocol.require(not (folder/'run.json').exists(),'Do not mix a partial or existing generation; archive before retry')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Fresh process requires cuBLAS workspace')
    config=case['config'];attack,spectral=AttackConfig(**config['attack']),SpectralConfig(**config['spectral'])
    cached={r['id']:r for r in protocol.read(plan['cache_audit'])['items']}
    validated={r['id']:r for r in protocol.read(VALIDATION/'trajectory-repeat-0.json')['items']}
    produced=[]
    with backend_mode('strict_math'):
        seed_everything(config['seed'])
        flags=backend_manifest()
        vocoder=load_vocoder(plan,case)
        native,surrogate_meta=load_whisper(config['surrogate'],'cuda');freeze(native)
        stable=equivalent_surrogate(native)
        expected=identity(plan,case)
        metadata={**expected,'fingerprint':config_fingerprint(expected),'manifest':plan['pilot']['manifest'],
            'roles':['pilot'],'features':plan['feature_artifacts'],'surrogate':surrogate_meta,'status':'running',
            'generation_runtime':protocol.runtime(),'generation_backend':flags,'scope':plan['scope']}
        atomic_json(folder/'run.json',metadata)
        for index,source in enumerate(read_jsonl(plan['pilot']['manifest']),1):
            uid=source['id'];cache=cached[uid]
            protocol.require(digest(cache['q_path'])==cache['q_sha256'] and
                digest(source['audio'])==source['audio_sha256'],'Changed source/q binding')
            seed=(int(config['seed'])+int(hashlib.sha256(uid.encode()).hexdigest()[:8],16))%2**32
            seed_everything(seed)
            q=torch.load(cache['q_path'],map_location='cuda',weights_only=True).detach()
            x=load_audio(source['audio']).cuda().detach()
            values=trajectory(q,x,source['text'],vocoder,stable,attack,spectral,folder/'trajectories'/uid)
            delta_artifact=values['trace'][-1]['delta']
            if case['name']==NAMES[0] and uid in validated:
                protocol.require(delta_artifact['value_sha256']==validated[uid]['trace'][-1]['delta']['value_sha256'] and
                    values['final_audio']['value_sha256']==validated[uid]['final_audio']['value_sha256'],
                    'Strong-anchor generation differs from validated full trajectory')
            delta=numerical.verify_tensor(delta_artifact).cuda()
            with torch.no_grad():
                reconstruction=vocoder(q)
                adversarial=vocoder(q+delta)
            protocol.require(tensor_value_sha256(adversarial)==values['final_audio']['value_sha256'],'Final audio resynthesis mismatch')
            assets={'clean':save_audio(folder/'audio/clean'/(uid+'.wav'),x),
                'reconstruction':save_audio(folder/'audio/reconstruction'/(uid+'.wav'),reconstruction),
                'adversarial':save_audio(folder/'audio/adversarial'/(uid+'.wav'),adversarial)}
            receipt_path=folder/'trajectories'/uid/'receipt.json'
            atomic_json(receipt_path,{'id':uid,'seed':seed,'q_path':cache['q_path'],'q_sha256':cache['q_sha256'],
                'q_value_sha256':tensor_value_sha256(q),**values})
            trace=[{'step':r['step'],'loss_total':r['loss_values'][0],'nll':r['loss_values'][1],
                'tv_ratio':r['loss_values'][2],'hf_ratio':r['loss_values'][3],'radius_after_update':r['radius']} for r in values['trace']]
            record={'id':uid,'text':source['text'],'language':source['language'],'role':'pilot',
                'input_audio_sha256':source['audio_sha256'],'seed':seed,'audio':assets,'trace':trace,
                'delta':delta_artifact,'trajectory_receipt':{'path':str(receipt_path),'sha256':digest(receipt_path)},
                'actual_optimization_updates':50,'returned_delta':'projected update 50'}
            atomic_json(folder/'records'/(uid+'.json'),record);produced.append(record)
            print(f'Generated stable {case["name"]}: {index}/24 {uid}',flush=True)
    write_jsonl(folder/'generated.jsonl',produced)
    metadata.update(status='complete',count=24,generated_sha256=digest(folder/'generated.jsonl'))
    atomic_json(folder/'run.json',metadata)


def evaluate_case(output,plan,case):
    folder=output/'pilot'/case['name']
    protocol.require(not (folder/'evaluation/summary.json').exists(),'Refusing to overwrite evaluation')
    protocol.configure_scorers(plan['quality'])
    flags=backend_manifest()
    result=evaluate(folder,{'kind':'whisper',**case['config']['surrogate']},folder/'evaluation',
        variants=['clean','reconstruction','adversarial'],quality=True,artifact_root=str(ROOT/'artifacts'))
    result.update(research_scorer_manifest=plan['quality'],evaluation_environment=protocol.runtime(),
        evaluation_backend=flags,generation_execution=plan['execution'])
    atomic_json(folder/'evaluation/summary.json',result)


def verify(output,plan,case):
    folder=output/'pilot'/case['name'];metadata=protocol.read(folder/'run.json')
    records=read_jsonl(folder/'generated.jsonl');expected=identity(plan,case)
    protocol.require(metadata['status']=='complete' and metadata['count']==24 and
        metadata['fingerprint']==config_fingerprint(expected) and
        all(metadata[k]==v for k,v in expected.items()) and
        metadata['features']==plan['feature_artifacts'] and metadata['generation_runtime']==plan['runtime'] and
        digest(folder/'generated.jsonl')==metadata['generated_sha256'] and
        [r['id'] for r in records]==plan['pilot']['ids'],'Changed generation identity')
    flags=metadata['generation_backend']
    protocol.require(flags['deterministic_algorithms'] and not flags['deterministic_warn_only'] and
        flags['cublas_workspace_config']==':4096:8' and flags['math_sdp_enabled'] and
        not any(flags[k] for k in ('efficient_sdp_enabled','flash_sdp_enabled','cudnn_sdp_enabled',
            'cuda_matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark')) and flags['cudnn_deterministic'],'Changed generation backend')
    summary=protocol.read(folder/'evaluation/summary.json')
    protocol.require(summary['generation_fingerprint']==metadata['fingerprint'] and
        summary['manifest_sha256']==plan['pilot']['sha256'] and summary['evaluation_ids']==plan['pilot']['ids'] and
        summary['target_artifacts']==metadata['surrogate'] and summary['research_scorer_manifest']==plan['quality'] and
        summary['evaluation_environment']==plan['runtime'] and
        digest(folder/'evaluation/utterances.jsonl')==summary['utterances_sha256'],'Changed evaluation binding')
    scores=[json.loads(line) for line in (folder/'evaluation/utterances.jsonl').read_text().splitlines() if line.strip()]
    protocol.require(len(scores)==72 and {(r['id'],r['variant']) for r in scores}==
        {(uid,v) for uid in plan['pilot']['ids'] for v in ('clean','reconstruction','adversarial')},'Incomplete score coverage')
    by_id={r['id']:r for r in read_jsonl(plan['pilot']['manifest'])}
    all_scores={(r['id'],r['variant']):r for r in scores}
    cache={r['id']:r for r in protocol.read(plan['cache_audit'])['items']}
    resynthesis=[]
    with backend_mode('strict_math'):
        seed_everything(case['config']['seed']);vocoder=load_vocoder(plan,case)
        for record in records:
            uid=record['id'];source=by_id[uid]
            protocol.require(record['text']==source['text'] and record['input_audio_sha256']==source['audio_sha256'] and
                record['actual_optimization_updates']==50 and len(record['trace'])==50 and
                [r['step'] for r in record['trace']]==list(range(1,51)) and
                digest(record['trajectory_receipt']['path'])==record['trajectory_receipt']['sha256'],'Changed trajectory identity')
            q=torch.load(cache[uid]['q_path'],map_location='cuda',weights_only=True)
            delta=numerical.verify_tensor(record['delta']).cuda()
            radius=relative_radius(delta,q,1e-8).item()
            protocol.require(radius<=.100001 and all(r['radius_after_update']<=.100001 for r in record['trace']), 'Exceeded feature ball')
            receipt=protocol.read(record['trajectory_receipt']['path'])
            protocol.require(receipt['trace'][-1]['delta']==record['delta'] and receipt['q_sha256']==cache[uid]['q_sha256'] and
                receipt['actual_optimization_updates']==50,'Changed saved feature/terminal binding')
            if case['name']==NAMES[0]:
                validated={r['id']:r for r in protocol.read(VALIDATION/'trajectory-repeat-0.json')['items']}
                if uid in validated:
                    protocol.require(record['delta']['value_sha256']==validated[uid]['trace'][-1]['delta']['value_sha256'] and
                        receipt['final_audio']['value_sha256']==validated[uid]['final_audio']['value_sha256'],
                        'Strong-anchor integration differs from the validated full trajectory')
            for step in receipt['trace']:
                numerical.verify_tensor(step['delta'])
            with torch.no_grad():wave=vocoder(q+delta)
            saved=load_audio(record['audio']['adversarial']['path'],max_seconds=None).cuda()
            error=(wave-saved).abs().max().item()
            protocol.require(error==0,'Saved adversarial waveform does not match returned delta')
            resynthesis.append({'id':uid,'max_abs_error':error,'radius':radius})
            for variant,asset in record['audio'].items():
                score=all_scores[(uid,variant)]
                protocol.require(digest(asset['path'])==asset['sha256']==score['audio_sha256'] and score['reference']==source['text'],
                    'Changed scored waveform/reference')
                counts=error_counts(source['text'],score['hypothesis'],'en')
                protocol.require(all(score[k]==v for k,v in counts.items()),'Incorrect edit counts')
    for variant in ('clean','reconstruction','adversarial'):
        subset=[r for r in scores if r['variant']==variant]
        rebuilt={'count':24,**aggregate_errors(subset),'quality':{m:sum(r['quality'][m] for r in subset)/24 for m in ('dnsmos','nisqa','utmos')}}
        protocol.require(rebuilt==summary['summary'][variant],'Incorrect score aggregation')
    result={'status':'complete','count':24,'summary_sha256':digest(folder/'evaluation/summary.json'),
        'generated_sha256':digest(folder/'generated.jsonl'),'resynthesis':resynthesis,
        'checked_projected_iterates':1200,'scope':plan['scope']}
    atomic_json(folder/'verification.json',result)


def joint_target(row):
    return row['error_percent']>=TARGETS['wer'] and all(row['quality'][m]>=TARGETS[m] for m in ('dnsmos','nisqa','utmos'))


def report(output,rows,status,plan):
    atomic_json(output/'results.json',{'status':status,'updated_at':now(),'results':rows,'scope':plan['scope']})
    lines=['# WER and quality under validated numerical execution','',f'Status: **{status}**.','',
        '| Setting | WER (%) | DNSMOS | NISQA | UTMOS | Joint descriptive target |',
        '|---|---:|---:|---:|---:|---|','| Paper | 75.4300 | 2.9540 | 3.5260 | 3.5900 | — |']
    for row in rows:
        q=row['quality'];lines.append(f"| {row['name']} | {row['error_percent']:.4f} | {q['dnsmos']:.4f} | {q['nisqa']:.4f} | {q['utmos']:.4f} | {joint_target(row)} |")
    lines+=['',plan['scope'],'','The 24-clip pilot is development data. These scores do not establish paper parity or human sound quality. No fresh confirmation/test run is automatically launched.','',
        'Earlier native execution context (different runs; the earlier optimizer diagnostics found numerical sensitivity):','',
        '| Earlier terminal | WER (%) | UTMOS |','|---|---:|---:|']
    for cached in plan['cached_context']:
        lines.append(f"| {cached['name']} | {cached['error_percent']:.4f} | {cached['quality']['utmos']:.4f} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--stage',choices=('generate','evaluate','verify'))
    parser.add_argument('--case',choices=NAMES)
    args=parser.parse_args();os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD']='0'
    output=args.output.resolve();plan=prepare(output) if args.prepare else protocol.read(output/'plan.json')
    check(plan)
    if args.prepare or args.check_only:
        print(f'Stable quality preflight passed: {len(plan["frozen_files"])} frozen inputs.',flush=True);return
    if args.stage:
        protocol.require(args.case is not None,'Stage needs fixed case')
        case=next(c for c in plan['cases'] if c['name']==args.case)
        {'generate':generate,'evaluate':evaluate_case,'verify':verify}[args.stage](output,plan,case)
        check(plan);return
    supervisor=Supervisor(output);rows=[]
    try:
        report(output,rows,'running',plan)
        for case in plan['cases']:
            for stage in ('generate','evaluate','verify'):
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output),'--case',case['name'],'--stage',stage]
                if stage in ('generate','verify'):command=['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',*command]
                stages.phase(supervisor,case['name']+'-'+stage,command)
            folder=output/'pilot'/case['name'];summary=protocol.read(folder/'evaluation/summary.json')
            adv=summary['summary']['adversarial']
            rows.append({'name':case['name'],'pool':'pilot','run':str(folder),**adv,
                'summary_sha256':digest(folder/'evaluation/summary.json'),'audit_sha256':digest(folder/'verification.json')})
            report(output,rows,'running',plan)
        check(plan);report(output,rows,'complete',plan);supervisor.finish(scope=plan['scope'])
    except BaseException as error:
        report(output,rows,'failed',plan);supervisor.fail(error);raise


if __name__=='__main__':main()
