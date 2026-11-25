"""Immutable expanded research contract, with explicit additional-source access."""
from pathlib import Path
from types import SimpleNamespace
import execution_runtime as rt

OUT=rt.OUT; RUN=OUT/'execution_scaled'
SINGLE=tuple(f'single_{n}' for n in (50,150,200,500,600,1500))
ROBUST=tuple(f'{m}_{n}' for m in ('ensemble_raw','ensemble_eot') for n in (50,200,500))
COHORTS={
 'qualification':{'count':48,'manifest':str(OUT/'execution48/development-48.jsonl'),
    'variants':['clean','fv_reconstruction','fv_quality50','bigvgan_reconstruction']},
 'fvpilot':{'count':48,'manifest':str(OUT/'execution48/development-48.jsonl'),
    'variants':['clean','fv_reconstruction','fv_quality50',*SINGLE,*ROBUST]},
 'melpilot':{'count':48,'manifest':str(OUT/'execution48/development-48.jsonl'),
    'variants':['clean','bigvgan_reconstruction',*(f'mel_eot_{n}' for n in (50,200,500))]},
 'validation':{'count':76,'manifest':str(OUT/'execution_refine/validation-76.jsonl'),
    'variants':['clean','candidate_reconstruction','control_reconstruction','candidate','control']}}


def register():
    old=rt.read(rt.RUN/'plan.json');rt.check(old)
    if rt.read(RUN/'probe-result.json')['status']!='pass': raise ValueError('Real engineering probe has not passed')
    files=[OUT/name for name in ('scaled_components.py','scaled_attack.py','scaled_scope.py','scaled_run.py',
        'scaled_source_evaluate.py','scaled_postprocess.py','scaled_transfer_evaluate.py','supervise_scaled.py',
        'test_scaled_components.py','14_scaled_research_protocol.md','execution_scaled/audit_scaled.py',
        'execution_scaled/component-tests.log','execution_scaled/expanded-tests.log','execution_scaled/assets.json','execution_scaled/download_assets.py',
        'execution_scaled/probe-registration.json','execution_scaled/probe-result.json','scaled_probe.py',
        'execution_scaled/generative-prior-work.md','execution_scaled/transfer-defense-prior-work.md',
        'source_evaluate_refine.py','supervise_pilot48.py','execution_refine/audit_refinement.py',
        'execution48/mechanism-pilot48/generated.jsonl','execution48/mechanism-pilot48/source-summary.json')]
    files += [rt.ROOT/'artifacts/whisper/large-v2.pt',rt.ROOT/'scripts/run_logged.py']
    targets={name:rt.ROOT/f'configs/targets/{name}.yaml' for name in ('whisper_base','whisper_medium','wav2vec2_large','hubert_large')}
    files+=list(targets.values())
    from beyondwaveform.config import load_config
    for target in targets.values():
        config=load_config(target)
        if config['kind']=='whisper':files.append(rt.ROOT/config['checkpoint'])
        else:
            for name,expected in config['artifact_hashes'].items():
                asset=Path(config['model_id'])/name
                if rt.digest(asset)!=expected:raise ValueError('Changed held-out target artifact')
                files.append(asset)
    files += [OUT/'r3_real_runner_v2.py', OUT/'r3_encoder_repair.py',rt.ROOT/'beyondwaveform/defenses.py']
    cohorts={k:dict(v,sha256=rt.digest(v['manifest'])) for k,v in COHORTS.items()}
    for info in cohorts.values():
        r=rt.read_jsonl(info['manifest'])
        if len(r)!=info['count'] or len({x['id'] for x in r})!=info['count']: raise ValueError('Cohort registration changed')
    assets=rt.read(RUN/'assets.json')
    for model in assets['models']:files += [Path(p) for p in model['files']]
    value={'created_at_utc':rt.stamp(),'authorization':'User explicitly authorized larger models, more data and longer runs to exceed75%WER while preserving transfer and defense persistence; old6h limit superseded',
        'maximum_cycle_wall_seconds':172800,'cohorts':cohorts,'source_only':True,'target_models_evaluated':[],
        'generation_sources':['Whisper-small','Whisper-large-v2','facebook/data2vec-audio-base-960h'],
        'source_weights':{'small':.5,'large':.25,'ctc':.25},'checkpoints':[50,200,500],
        'single_source_checkpoints':[50,150,200,500,600,1500],
        'rho':.1,'full_original_tv_hf':True,'fixed_model_gradient_scales':'calibrated once at own clean reconstruction',
        'transforms':['identity','fir6k','resample8k'],'eot':'Seeded uniform draw per update shared across sources',
        'mel_log_box':float(__import__('numpy').log(2)),'optimizer_lrs':{'feature':.1,'mel':.01},
        'target_configs':{k:str(v) for k,v in targets.items()},
        'selection':'Complete48 source-only >75WER and all original automatic quality floors; highestWER, thenUTMOS, thenfewersteps; freeze before76validation. Candidate may be strong known baseline; no automatic novelty claim.',
        'transfer_gate':'After fixed candidate also passes76 source>75 and all quality floors, evaluate4sealed targets and exact named preprocessors on identical candidate/control/clean. Do not use outcomes for reselection.',
        'scientific_limits':'Additional-surrogate/model-pretraining track; known-transform adaptation differs from withheld defense tests; no human study or missing paper defenses claimed.',
        'bindings':{str(p):rt.digest(p) for p in files}}
    path=RUN/'plan.json'
    if path.exists():raise FileExistsError('Preserve expanded registration')
    rt.atomic_json(path,value);path.chmod(0o444);return value


def check():
    p=rt.read(RUN/'plan.json')
    for path,digest in p['bindings'].items():
        if rt.digest(path)!=digest:raise ValueError(f'Changed expanded research binding: {path}')
    if not p['source_only'] or p['target_models_evaluated']:raise ValueError('Generation scope changed')
    return p


def validation_freeze():
    freeze=rt.read(RUN/'validation-freeze.json')
    if freeze['status']!='validate':raise ValueError('No qualifying frozen method')
    for path,digest in freeze['bindings'].items():
        if rt.digest(path)!=digest:raise ValueError('Changed candidate freeze')
    return freeze


def evaluator_scope(name):
    plan=check()
    return SimpleNamespace(OUT=OUT,RUN=RUN,VARIANTS=tuple(plan['cohorts'][name]['variants']),
        check=check,validation_freeze=validation_freeze)
