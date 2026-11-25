"""Revision two: retain the original encoder policy, strict attack execution.

The default production loader and its matching-layer guard are unchanged.
Layers five/seven are experimental inputs to the same frozen layer-6 vocoder;
only a predeclared reconstruction screen permits full-objective attacks.
"""
from __future__ import annotations
import argparse,copy,json,math,os,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from beyondwaveform.config import AttackConfig
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,save_audio,seed_everything,write_jsonl
from beyondwaveform.models.feature import WavLMEncoder,load_feature_models
from beyondwaveform.pipeline import config_fingerprint,evaluate
from parallel_jobs import Supervisor,now
from probe_numerical_repeatability import backend_mode,backend_manifest,tensor_value_sha256
import run_quality_research as protocol
import run_quality_refinement as stages
import run_stable_quality_research as stable
import run_stable_ablation_research as prior
import run_encoder_execution_diagnostics as encoder_diagnostic

DEFAULT=ROOT/'reproduction-runs/paper-quality-feature-layers-v2'
REVIEW=ROOT/'reproduction-runs/stable-ablation-completion-review-20261020'
LAYERS=(5,7)


def layer_contract(base,layer):
    protocol.require(layer in LAYERS,'Only the two predeclared adjacent layers are allowed')
    result=copy.deepcopy(base);result.update(layer=layer,vocoder_training_layer=6,
        status='experimental out-of-training-contract input; matching author/upstream weights not established')
    return result


def reconstruction_gate(measured,baseline):
    rec=measured['reconstruction'];base=baseline['reconstruction']
    return (rec['count']==24 and math.isfinite(rec['error_percent']) and rec['error_percent']<=base['error_percent']+2 and
        all(math.isfinite(rec['quality'][m]) and rec['quality'][m]>=base['quality'][m]-.1 and rec['quality'][m]>2.5
            for m in protocol.METRICS))


def prepare(output):
    protocol.require(not (output/'plan.json').exists(),'Plan already exists')
    old=protocol.read(prior.DEFAULT/'plan.json');prior.check(old)
    diagnostic=encoder_diagnostic.DEFAULT;diagnostic_plan=protocol.read(diagnostic/'plan.json')
    encoder_diagnostic.check(diagnostic_plan)
    measured=protocol.read(diagnostic/'auto.json')
    protocol.require(protocol.read(diagnostic/'job.json')['status']=='complete' and measured['all_24_cache_bitwise_equal'] and
        measured['first_three_repeat_bitwise_equal'] and protocol.read(diagnostic/'completion-review.json')['saved_tensors_verified']==90,
        'Original encoder policy must reproduce frozen features exactly')
    audit=protocol.read(REVIEW/'review.json')
    protocol.require(audit['status']=='complete' and audit['all_checks_passed'] and
        audit['counts']['iterates']==3600 and audit['plan_sha256']==digest(prior.DEFAULT/'plan.json') and
        audit['audit_source_sha256']==digest(REVIEW/'audit.py'),'Review completed ablations first')
    base=copy.deepcopy(old['base_config']);files=[Path(__file__).resolve(),ROOT/'tests/test_feature_layer_research_v2.py',
        ROOT/'tests/test_feature_layer_research.py',ROOT/'docs/QUALITY_FEATURE_LAYERS_V2.md',
        REVIEW/'audit.py',REVIEW/'review.json',REVIEW/'report.md']
    files.extend(diagnostic/name for name in ('plan.json','job.json','results.json','report.md','auto.json','math.json','strict_math.json','completion-review.json'))
    files.extend(encoder_diagnostic.parent.DEFAULT/name for name in ('research.log','report.md','phases/extract-and-reconstruct/stage.log'))
    for mode in encoder_diagnostic.MODES:
        files.extend(Path(a['path']) for item in protocol.read(diagnostic/(mode+'.json'))['items'] for a in item['artifacts'])
    files.extend(prior.DEFAULT/n for n in ('plan.json','results.json','report.md','job.json','control-preflight.json'))
    entries=[]
    for layer in LAYERS:
        config=copy.deepcopy(base);config['features']['layer']=layer
        case={'name':f'layer-{layer}','layer':layer,'config':config}
        path=output/'configs'/(case['name']+'.json');atomic_json(path,config);path.chmod(0o444)
        case['config_path']=str(path);files.append(path);entries.append(case)
    frozen=dict(diagnostic_plan['frozen_files']);frozen.update({str(p.resolve()):digest(p) for p in files})
    plan={'created_at':now(),'frozen_files':frozen,'runtime':old['runtime'],
        'additional_dependencies':old['additional_dependencies'],'base_config':base,'pilot':old['pilot'],
        'baseline_cache_audit':old['cache_audit'],'feature_artifacts':old['feature_artifacts'],
        'quality':old['quality'],'execution':old['execution'],'baseline':old['baseline'],
        'baseline_reconstruction':protocol.read(prior.BASELINE/'evaluation/summary.json')['summary'],
        'cases':entries,'layers':list(LAYERS),'gate':{'WER_loss_pp_max':2.,'MOS_loss_max_each':.1,'min_MOS_exclusive':2.5},
        'layer6_equivalence':{'bitwise_required':True},
        'encoder_execution':{'mode':'auto','CUBLAS_WORKSPACE_CONFIG':None,'strict_algorithms':False},'discord_enabled':False,
        'encoder_backend':measured['backend'],
        'scope':'Two adjacent WavLM output layers (5/7), same frozen layer-6-trained HiFi-GAN. Explicit out-of-training-contract sensitivity probes, not recovered author models. Original auto encoder extraction must reproduce all frozen layer-6 q exactly; separate strict-math attack execution is retained. Fixed 24 development clips; reconstruction gates precede unchanged full-objective fifty-update LR=.1 attacks. Original paper budget/clean denominators/references/ASR/MOS. No layer grid, vocoder retraining, sample filtering, automatic fresh48 or 999 test.'}
    atomic_json(output/'plan.json',plan);(output/'plan.json').chmod(0o444);return plan


def check(plan):
    import importlib.metadata
    protocol.require(protocol.runtime()==plan['runtime'] and all(importlib.metadata.version(k)==v
        for k,v in plan['additional_dependencies'].items()),'Changed runtime')
    protocol.require(plan['layers']==[5,7] and len(plan['pilot']['ids'])==24 and plan['execution']==
        {'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True} and plan['gate']==
        {'WER_loss_pp_max':2.,'MOS_loss_max_each':.1,'min_MOS_exclusive':2.5} and
        plan['layer6_equivalence']=={'bitwise_required':True} and
        plan['encoder_execution']=={'mode':'auto','CUBLAS_WORKSPACE_CONFIG':None,'strict_algorithms':False},'Changed bounded layer protocol')
    for case,layer in zip(plan['cases'],LAYERS):
        expected=copy.deepcopy(plan['base_config']);expected['features']['layer']=layer
        protocol.require(case['config']==expected and case['layer']==layer and case['name']==f'layer-{layer}',
            'Changed feature or attack settings')
    stable.validate_case({'name':stable.NAMES[0],'config':plan['base_config']})
    for path,expected in plan['frozen_files'].items():
        protocol.require(digest(path)==expected,f'Changed frozen layer-study input: {path}')
    protocol.require(digest(plan['pilot']['manifest'])==plan['pilot']['sha256'] and
        [r['id'] for r in read_jsonl(plan['pilot']['manifest'])]==plan['pilot']['ids'],'Changed cohort')


def effective(output,plan,case):
    cache=output/'cache'/case['name']/'audit.json'
    return {**plan,'feature_artifacts':layer_contract(plan['feature_artifacts'],case['layer']),
        'cache_audit':str(cache)}


def extract(output,plan):
    protocol.require(not (output/'extraction.json').exists() and not (output/'cache').exists(),
        'Preserve existing/partial extraction before retry')
    protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None and
        not torch.are_deterministic_algorithms_enabled(),'Extraction must retain the diagnosed original encoder policy')
    original={r['id']:r for r in read_jsonl(prior.BASELINE/'generated.jsonl')}
    cached={r['id']:r for r in protocol.read(plan['baseline_cache_audit'])['items']}
    base=plan['base_config'];items={c['name']:[] for c in plan['cases']};records={c['name']:[] for c in plan['cases']};control=[]
    with backend_mode('auto'):
        seed_everything(base['seed'])
        # Honor the production loader's layer-6 guard. Experimental wrappers
        # are declared separately, with no claim of matching vocoder training.
        encoder,vocoder,metadata=load_feature_models(base['features'],'cuda')
        protocol.require(metadata==plan['feature_artifacts'],'Changed original feature contract')
        encoders={c['layer']:WavLMEncoder(encoder.model,layer=c['layer'],normalize=encoder.normalize).cuda() for c in plan['cases']}
        flags=backend_manifest()
        protocol.require(flags==plan['encoder_backend'],'Encoder execution differs from diagnosed cache-matching policy')
        for index,row in enumerate(read_jsonl(plan['pilot']['manifest']),1):
            uid=row['id'];parent=original[uid];seed_everything(parent['seed'])
            protocol.require(digest(row['audio'])==row['audio_sha256'] and
                digest(cached[uid]['q_path'])==cached[uid]['q_sha256'],'Changed original audio/cache')
            x=load_audio(row['audio']).cuda();q6=torch.load(cached[uid]['q_path'],map_location='cuda',weights_only=True)
            with torch.no_grad():fresh=encoder(x)
            difference=(fresh-q6).double();relative=difference.norm().item()/q6.double().norm().item()
            maximum=difference.abs().max().item()
            protocol.require(fresh.shape==q6.shape and tensor_value_sha256(fresh)==tensor_value_sha256(q6),
                'Fresh layer-6 extraction does not agree with the frozen clean-feature contract')
            control.append({'id':uid,'max_abs':maximum,'relative_l2':relative,'bitwise_equal':torch.equal(fresh,q6)})
            for case in plan['cases']:
                layer,name=case['layer'],case['name']
                with torch.no_grad():q=encoders[layer](x).detach();wave=vocoder(q)
                protocol.require(q.shape==q6.shape and torch.isfinite(q).all() and torch.isfinite(wave).all(),'Invalid layer features/audio')
                if index==1:
                    with torch.no_grad():repeat=encoders[layer](x)
                    protocol.require(torch.equal(q,repeat),'Adjacent-layer fixed-input extraction is not repeatable')
                path=output/'cache'/name/'q'/(uid+'.pt');path.parent.mkdir(parents=True,exist_ok=True);torch.save(q.cpu(),path)
                qc=q.double().flatten();bc=q6.double().flatten()
                item={'id':uid,'input_audio_sha256':row['audio_sha256'],'q_path':str(path),'q_sha256':digest(path),
                    'q_value_sha256':tensor_value_sha256(q),'layer':layer,'relative_l2_to_layer6':(qc-bc).norm().item()/bc.norm().item(),
                    'q_norm_relative_to_layer6':qc.norm().item()/bc.norm().item(),'cosine_to_layer6':qc.dot(bc).item()/(qc.norm().item()*bc.norm().item())}
                items[name].append(item)
                audio={'clean':parent['audio']['clean'],'reconstruction':save_audio(output/'reconstruction'/name/'audio'/(uid+'.wav'),wave)}
                records[name].append({'id':uid,'text':row['text'],'language':row['language'],'role':'pilot',
                    'input_audio_sha256':row['audio_sha256'],'seed':parent['seed'],'audio':audio})
            print(f'Extracted both layer probes: {index}/24 [{uid}]',flush=True)
        protocol.require(not any(p.grad is not None for p in encoder.parameters()),'Feature extraction accumulated gradients')
    for case in plan['cases']:
        name=case['name'];folder=output/'reconstruction'/name;contract=layer_contract(plan['feature_artifacts'],case['layer'])
        atomic_json(output/'cache'/name/'audit.json',{'status':'complete','features':contract,'items':items[name],
            'input_manifest_sha256':plan['pilot']['sha256'],'source_encoder_sha256':metadata['encoder_sha256'],
            'execution':flags,'scope':plan['scope']})
        identity={'config':{'method':'experimental_layer_reconstruction','parent_config':case['config'],
            'vocoder_training_layer':6,'experimental_input_layer':case['layer']},
            'manifest_sha256':plan['pilot']['sha256'],'ids':plan['pilot']['ids']}
        write_jsonl(folder/'generated.jsonl',records[name])
        atomic_json(folder/'run.json',{**identity,'fingerprint':config_fingerprint(identity),'status':'complete','count':24,
            'roles':['pilot'],'features':contract,'surrogate':protocol.read(prior.BASELINE/'run.json')['surrogate'],
            'generated_sha256':digest(folder/'generated.jsonl'),'generation_backend':flags,'scope':plan['scope']})
    atomic_json(output/'extraction.json',{'status':'complete','baseline_layer6_checks':control,'backend':flags,
        'cache_audits':{c['name']:{'path':str(output/'cache'/c['name']/'audit.json'),
            'sha256':digest(output/'cache'/c['name']/'audit.json')} for c in plan['cases']}})


def evaluate_reconstruction(output,plan,case):
    folder=output/'reconstruction'/case['name']
    protocol.require(not (folder/'evaluation/summary.json').exists(),'Preserve existing reconstruction scores')
    protocol.configure_scorers(plan['quality']);backend=backend_manifest()
    result=evaluate(folder,{'kind':'whisper',**case['config']['surrogate']},folder/'evaluation',
        variants=['clean','reconstruction'],quality=True,artifact_root=str(ROOT/'artifacts'))
    result.update(research_scorer_manifest=plan['quality'],evaluation_environment=protocol.runtime(),evaluation_backend=backend)
    atomic_json(folder/'evaluation/summary.json',result)


def verify_cache(output,plan,case,reextract=False):
    entry=effective(output,plan,case);cache_path=Path(entry['cache_audit']);cache=protocol.read(cache_path)
    extraction=protocol.read(output/'extraction.json')
    protocol.require(extraction['cache_audits'][case['name']]['sha256']==digest(cache_path) and
        cache['features']==entry['feature_artifacts'] and [r['id'] for r in cache['items']]==plan['pilot']['ids'],
        'Changed extracted feature cache binding')
    for item in cache['items']:
        protocol.require(digest(item['q_path'])==item['q_sha256'],'Changed cached feature file')
        q=torch.load(item['q_path'],map_location='cpu',weights_only=True)
        protocol.require(tensor_value_sha256(q)==item['q_value_sha256'] and torch.isfinite(q).all(),'Changed cached feature values')
    if reextract:
        protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None and
            not torch.are_deterministic_algorithms_enabled(),'Independent feature extraction needs original encoder policy')
        with backend_mode('auto'):
            seed_everything(plan['base_config']['seed'])
            encoder,vocoder,metadata=load_feature_models(plan['base_config']['features'],'cuda')
            protocol.require(metadata==plan['feature_artifacts'],'Changed encoder/vocoder artifacts')
            model=WavLMEncoder(encoder.model,layer=case['layer'],normalize=encoder.normalize).cuda()
            originals=read_jsonl(plan['pilot']['manifest'])
            seeds={r['id']:r['seed'] for r in read_jsonl(prior.BASELINE/'generated.jsonl')}
            for row,item in zip(originals,cache['items']):
                seed_everything(seeds[row['id']])
                with torch.no_grad():q=model(load_audio(row['audio']).cuda())
                protocol.require(tensor_value_sha256(q)==item['q_value_sha256'],'Independent feature re-extraction differs')
        protocol.require(backend_manifest()==plan['encoder_backend'],'Independent encoder backend changed')
        atomic_json(output/'encoder-verification'/(case['name']+'.json'),{'status':'complete',
            'cache_audit_sha256':digest(cache_path),'input_manifest_sha256':plan['pilot']['sha256'],
            'feature_value_hashes':{r['id']:r['q_value_sha256'] for r in cache['items']},
            'backend':backend_manifest(),'count':24})
    return entry


def verify_encoder_receipt(output,plan,case):
    entry=verify_cache(output,plan,case)
    receipt=protocol.read(output/'encoder-verification'/(case['name']+'.json'))
    cache=protocol.read(entry['cache_audit'])
    protocol.require(receipt['status']=='complete' and receipt['count']==24 and
        receipt['cache_audit_sha256']==digest(entry['cache_audit']) and receipt['input_manifest_sha256']==plan['pilot']['sha256'] and
        receipt['feature_value_hashes']=={r['id']:r['q_value_sha256'] for r in cache['items']} and
        receipt['backend']==plan['encoder_backend'],'Changed independent encoder receipt')
    return entry


def gate(output,plan,case):
    entry=verify_encoder_receipt(output,plan,case)
    folder=output/'reconstruction'/case['name'];result=protocol.read(folder/'evaluation/summary.json')
    protocol.require(result['research_scorer_manifest']==plan['quality'] and result['evaluation_environment']==plan['runtime'] and
        digest(folder/'evaluation/utterances.jsonl')==result['utterances_sha256'],'Changed reconstruction evaluation')
    full=protocol.read(prior.BASELINE/'evaluation/summary.json')
    protocol.require(all(result[k]==full[k] for k in ('target','target_artifacts','metric','normalization','evaluation_scope',
        'evaluation_backend','quality_provenance')) and result['summary']['clean']==full['summary']['clean'],
        'Reconstruction scoring protocol or clean controls changed')
    rows=[json.loads(s) for s in (folder/'evaluation/utterances.jsonl').read_text().splitlines()]
    from beyondwaveform.evaluation import error_counts,aggregate_errors
    records=read_jsonl(folder/'generated.jsonl');pairs={(r['id'],r['variant']):r for r in rows}
    protocol.require(len(rows)==len(pairs)==48 and [r['id'] for r in records]==plan['pilot']['ids'],'Wrong reconstruction coverage')
    for record in records:
        for variant,asset in record['audio'].items():
            score=pairs[(record['id'],variant)]
            protocol.require(digest(asset['path'])==asset['sha256']==score['audio_sha256'] and score['reference']==record['text'] and
                all(score[k]==v for k,v in error_counts(record['text'],score['hypothesis'],'en').items()),'Changed reconstruction wave/counts')
    for variant in ('clean','reconstruction'):
        selected=[r for r in rows if r['variant']==variant]
        rebuilt={'count':24,**aggregate_errors(selected),'quality':{m:sum(r['quality'][m] for r in selected)/24 for m in protocol.METRICS}}
        protocol.require(rebuilt==result['summary'][variant],'Wrong reconstruction aggregation')
    seed_everything(plan['base_config']['seed'])
    with backend_mode('strict_math'):
        vocoder=stable.load_vocoder(entry,case);cache=protocol.read(entry['cache_audit'])
        for record,item in zip(records,cache['items']):
            q=torch.load(item['q_path'],map_location='cuda',weights_only=True)
            with torch.no_grad():wave=vocoder(q)
            protocol.require(torch.equal(wave.cpu(),load_audio(record['audio']['reconstruction']['path'],max_seconds=None)),
                'Reconstruction does not resynthesize from saved features')
    passed=reconstruction_gate(result['summary'],plan['baseline_reconstruction'])
    atomic_json(output/'gates'/(case['name']+'.json'),{'passed':passed,'summary_sha256':digest(folder/'evaluation/summary.json'),
        'cache_audit_sha256':digest(entry['cache_audit']),'summary':result['summary'],'gate':plan['gate'],
        'independent_feature_reextraction':'all 24 bitwise equal','scope':plan['scope']})


def generate_attack(output,plan,case):
    protocol.require(protocol.read(output/'gates'/(case['name']+'.json'))['passed'],'Failed reconstruction screen forbids attack')
    entry=verify_cache(output,plan,case)
    stable.generate(output,entry,case)
    path=output/'pilot'/case['name']/'run.json';metadata=protocol.read(path)
    metadata.update(experimental_vocoder_contract=entry['feature_artifacts'],feature_cache_audit_sha256=digest(entry['cache_audit']))
    atomic_json(path,metadata)


def verify_attack(output,plan,case):
    entry=verify_encoder_receipt(output,plan,case);stable.verify(output,entry,case)
    metadata=protocol.read(output/'pilot'/case['name']/'run.json')
    protocol.require(metadata['feature_cache_audit_sha256']==digest(entry['cache_audit']) and
        metadata['experimental_vocoder_contract']==entry['feature_artifacts'],'Changed experimental contract')


def report(output,plan,rows,status):
    gates={c['name']:protocol.read(output/'gates'/(c['name']+'.json')) for c in plan['cases'] if (output/'gates'/(c['name']+'.json')).exists()}
    atomic_json(output/'results.json',{'status':status,'updated_at':now(),'results':rows,'reconstruction_gates':gates,'scope':plan['scope']})
    lines=['# Adjacent feature-layer compatibility probes','',f'Status: **{status}**.','',
        '| Input layer | Reconstruction WER | Reconstruction UTMOS | Gate | Attack WER | Attack UTMOS |',
        '|---|---:|---:|---|---:|---:|']
    base=plan['baseline_reconstruction']['reconstruction']
    lines.append(f"| 6 (frozen control) | {base['error_percent']:.4f} | {base['quality']['utmos']:.4f} | Matching training layer | {plan['baseline']['error_percent']:.4f} | {plan['baseline']['quality']['utmos']:.4f} |")
    for case in plan['cases']:
        g=gates.get(case['name']);row=next((r for r in rows if r['name']==case['name']),None)
        rec=g['summary']['reconstruction'] if g else None
        lines.append('| '+str(case['layer'])+' | '+('%.4f'%rec['error_percent'] if rec else 'pending')+' | '+
            ('%.4f'%rec['quality']['utmos'] if rec else 'pending')+' | '+(str(g['passed']) if g else 'pending')+' | '+
            ('%.4f'%row['error_percent'] if row else '—')+' | '+('%.4f'%row['quality']['utmos'] if row else '—')+' |')
    lines+=['',plan['scope'],'','The vocoder weights were trained for layer 6. These probes do not establish a matching author contract, even if the empirical screen passes. Production layer validation is unchanged.','']
    (output/'report.md').write_text('\n'.join(lines))


def stage_command(base,stage,case=None):
    command=[*base,'--stage',stage]
    if case is not None:command.extend(['--case',case])
    if stage in ('gate','generate','verify'):return ['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',*command]
    return ['/usr/bin/env','-u','CUBLAS_WORKSPACE_CONFIG',*command]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--stage',choices=('extract','reextract','reconstruction-evaluate','gate','generate','evaluate','verify'))
    parser.add_argument('--case',choices=('layer-5','layer-7'));args=parser.parse_args();os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD']='0'
    output=args.output.resolve();plan=prepare(output) if args.prepare else protocol.read(output/'plan.json');check(plan)
    if args.prepare or args.check_only:print(f'Feature-layer preflight passed: {len(plan["frozen_files"])} frozen inputs.',flush=True);return
    if args.stage:
        if args.stage=='extract':extract(output,plan)
        else:
            protocol.require(args.case is not None,'Stage requires a fixed layer')
            case=next(c for c in plan['cases'] if c['name']==args.case)
            if args.stage=='evaluate':stable.evaluate_case(output,effective(output,plan,case),case)
            elif args.stage=='reextract':verify_cache(output,plan,case,reextract=True)
            else:{'reconstruction-evaluate':evaluate_reconstruction,'gate':gate,'generate':generate_attack,'verify':verify_attack}[args.stage](output,plan,case)
        check(plan);return
    supervisor=Supervisor(output);rows=[]
    try:
        report(output,plan,rows,'running');base=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output)]
        stages.phase(supervisor,'extract-and-reconstruct',stage_command(base,'extract'))
        for case in plan['cases']:
            for stage in ('reconstruction-evaluate','reextract','gate'):
                cmd=stage_command(base,stage,case['name'])
                stages.phase(supervisor,case['name']+'-'+stage,cmd)
            report(output,plan,rows,'running')
        for case in plan['cases']:
            if not protocol.read(output/'gates'/(case['name']+'.json'))['passed']:continue
            for stage in ('generate','evaluate','verify'):
                cmd=stage_command(base,stage,case['name'])
                stages.phase(supervisor,case['name']+'-'+stage,cmd)
            folder=output/'pilot'/case['name'];summary=protocol.read(folder/'evaluation/summary.json')
            rows.append({'name':case['name'],'pool':'pilot','run':str(folder),**summary['summary']['adversarial'],
                'summary_sha256':digest(folder/'evaluation/summary.json'),'audit_sha256':digest(folder/'verification.json')})
            report(output,plan,rows,'running')
        check(plan);report(output,plan,rows,'complete');supervisor.finish(scope=plan['scope'])
    except BaseException as error:report(output,plan,rows,'failed');supervisor.fail(error);raise


if __name__=='__main__':main()
