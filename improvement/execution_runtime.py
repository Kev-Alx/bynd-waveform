"""Isolated execution helpers; only the public Whisper-small surrogate is loaded."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT / 'improvement'
RUN = OUT / 'execution'
sys.path[:0] = [str(OUT), str(ROOT), str(ROOT / 'scripts')]
import torch
from beyondwaveform.attacks import freeze, project_relative_ball_, relative_radius
from beyondwaveform.config import AttackConfig, SpectralConfig
from beyondwaveform.evaluation import QualityScorer, aggregate_errors, error_counts
from beyondwaveform.io import atomic_json, digest, environment, load_audio, read_jsonl, save_audio, seed_everything, write_jsonl
from beyondwaveform.models.whisper import load_whisper
from probe_feature_gradients import load_frozen_vocoder
from probe_deterministic_execution import ExplicitReflectionReference, equivalent_surrogate, tensor_tree
from probe_numerical_repeatability import backend_manifest, backend_mode, tensor_value_sha256

BASE = ROOT / 'reproduction-runs/paper-quality-stable-execution/pilot/strong-sum-lr-0.1'


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'plan.json'
    if path.exists():
        return read(path)
    parent = read(ROOT / 'reproduction-runs/paper-quality-stable-execution/plan.json')
    rows = read_jsonl(parent['pilot']['manifest'])
    ordered = sorted(rows, key=lambda r: (r['duration'], r['id']))
    chosen = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    write_jsonl(RUN / 'development-24.jsonl', rows)
    write_jsonl(RUN / 'calibration-3.jsonl', chosen)
    config = read(BASE / 'run.json')['config']['parent_config']
    final = RUN / 'data/final-reserve-200.jsonl'
    if not final.exists():
        raise ValueError('Freeze final reserve before new model selection')
    inputs = [Path(parent['cache_audit']), BASE / 'run.json', BASE / 'generated.jsonl',
        RUN / 'development-24.jsonl', RUN / 'calibration-3.jsonl', final,
        OUT / '04_experimental_plan.md', OUT / 'execution_runtime.py']
    inputs += list((ROOT / 'beyondwaveform').rglob('*.py'))
    frozen = {str(p): digest(p) for p in inputs}
    # Hash public surrogate/vocoder artifacts before any new optimization.
    frozen[str(ROOT/'artifacts/whisper/small.pt')] = digest(ROOT/'artifacts/whisper/small.pt')
    plan = {'created_at_utc': stamp(), 'status': 'registered_development_phase',
        'authorization': 'User said Continue after steps 1–4; isolated implementation and bounded first-stage experiments.',
        'scope': 'G0/G1 development only. No confirmation48, final200 inference, held-out target models, human recruitment, notifications or external messages.',
        'maximum_initial_gpu_hours': 6, 'r3_initial_training_seconds': 3600,
        'development_manifest': str(RUN/'development-24.jsonl'),
        'calibration_manifest': str(RUN/'calibration-3.jsonl'),
        'calibration_strata': dict(zip([r['id'] for r in chosen], ['short','median','long'])),
        'final_reserve': str(final), 'final_reserve_sha256': digest(final),
        'base_config': config, 'cache_audit': parent['cache_audit'],
        'feature_artifacts': parent['feature_artifacts'], 'quality': parent['quality'],
        'baseline_run': str(BASE), 'execution': parent['execution'],
        'frozen_files': frozen, 'runtime': environment(),
        'r1': {'k':4,'refresh_every':5,'tau':1,'include_eos':True,'learning_rate':0.1,'steps':50,
            'fallback': 'None in canonical calibration; a degenerate result stops this condition.'},
        'r2': {'learning_rate':0.1,'steps':50,'rho':0.1,'cosine_floor':0.95,
            'initialization':'a=0, uniform logits', 'gradient_bootstrap':False},
        'r3': {'seed':11,'linf':0.003,'rms_limit':0.003,'training_hours_data':1,
            'maximum_training_seconds':3600,'maximum_updates':12000,'accumulation':4},
        'no_target_model_selection':True, 'discord_enabled':False}
    atomic_json(path, plan)
    path.chmod(0o444)
    return plan


def check(plan):
    for path, expected in plan['frozen_files'].items():
        if digest(path) != expected:
            raise ValueError(f'Changed registered input: {path}')
    if plan['base_config']['surrogate']['checkpoint'] != 'artifacts/whisper/small.pt':
        raise ValueError('Only registered public source is permitted')
    return True


def models(plan, with_vocoder=True):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Launch a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.set_num_threads(4)
    seed_everything(2026)
    native, metadata = load_whisper(plan['base_config']['surrogate'], 'cuda')
    freeze(native)
    vocoder = None
    if with_vocoder:
        vocoder, _ = load_frozen_vocoder(plan['base_config']['features'], plan['feature_artifacts'], 'cuda')
    return vocoder, equivalent_surrogate(native), native, metadata


def clips(plan, calibration=False):
    rows = read_jsonl(plan['calibration_manifest'] if calibration else plan['development_manifest'])
    cache = {r['id']:r for r in read(plan['cache_audit'])['items']}
    for row in rows:
        cached = cache[row['id']]
        if digest(row['audio']) != row['audio_sha256'] or digest(cached['q_path']) != cached['q_sha256']:
            raise ValueError('Changed input audio/features')
        yield row, torch.load(cached['q_path'], map_location='cuda', weights_only=True).detach(), load_audio(row['audio']).cuda()


def objective(q, x, text, vocoder, surrogate):
    spec = SpectralConfig()
    reference = ExplicitReflectionReference(q, x, spec, 1e-8)
    def apply(z):
        audio = vocoder(z)
        nll = surrogate.nll(audio, text)
        tv, hf = reference.terms(z, audio)
        return {'loss': -nll + (tv+hf), 'nll':nll, 'tv':tv, 'hf':hf}
    return apply


def save_feature_result(name, row, q, x, delta, trace, vocoder, extra=None):
    folder = RUN / name
    with torch.no_grad():
        audio = vocoder(q+delta)
    assets = {'clean':save_audio(folder/'audio/clean'/f'{row["id"]}.wav',x),
        'adversarial':save_audio(folder/'audio/adversarial'/f'{row["id"]}.wav',audio)}
    with torch.no_grad():
        assets['reconstruction'] = save_audio(folder/'audio/reconstruction'/f'{row["id"]}.wav',vocoder(q))
    tensor_path = folder/'deltas'/f'{row["id"]}.pt'
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta.detach().cpu(),tensor_path)
    value = {'id':row['id'],'text':row['text'],'language':'en','role':'pilot',
        'input_audio_sha256':row['audio_sha256'],'audio':assets,'trace':trace,
        'delta':{'path':str(tensor_path),'sha256':digest(tensor_path),'value_sha256':tensor_value_sha256(delta)},
        'actual_relative_radius':relative_radius(delta,q).item(), **(extra or {})}
    atomic_json(folder/'records'/f'{row["id"]}.json',value)
    return value


def baseline_preflight(plan):
    check(plan)
    folder=RUN/'baseline-preflight'
    if (folder/'result.json').exists():
        return read(folder/'result.json')
    old={r['id']:r for r in read_jsonl(BASE/'generated.jsonl')}
    items=[]
    with backend_mode('strict_math'):
        vocoder, surrogate, native, _=models(plan)
        for row,q,x in clips(plan,True):
            uid=row['id']; seed_everything(old[uid]['seed'])
            reference=read(old[uid]['trajectory_receipt']['path'])
            apply=objective(q,x,row['text'],vocoder,surrogate)
            delta=torch.zeros_like(q,requires_grad=True)
            optimizer=torch.optim.Adam([delta],lr=0.1)
            matches=[]; start=time.perf_counter()
            for i in range(50):
                optimizer.zero_grad(set_to_none=True)
                values=apply(q+delta)
                loss=values['loss']
                grad,=torch.autograd.grad(loss,delta)
                delta.grad=grad; optimizer.step(); project_relative_ball_(delta,q,0.1)
                actual={'loss_value_sha256':tensor_value_sha256(torch.stack([values[k].detach() for k in ['loss','nll','tv','hf']])),
                    'gradient_value_sha256':tensor_value_sha256(grad),
                    'delta_value_sha256':tensor_value_sha256(delta),
                    'optimizer_value_receipt':tensor_tree(optimizer.state_dict())}
                expected=reference['trace'][i]
                passed=(actual['loss_value_sha256']==expected['loss_value_sha256'] and
                    actual['gradient_value_sha256']==expected['gradient_value_sha256'] and
                    actual['delta_value_sha256']==expected['delta']['value_sha256'] and
                    actual['optimizer_value_receipt']==expected['optimizer_value_receipt'])
                matches.append({'step':i+1,'passed':passed,**actual})
                if not passed:
                    atomic_json(folder/f'{uid}-mismatch.json',{'step':i+1,'actual':actual,'expected':expected})
                    raise RuntimeError(f'Frozen-control preflight mismatch: {uid} update {i+1}')
            with torch.no_grad():
                audio=vocoder(q+delta)
            audio_match=tensor_value_sha256(audio)==reference['final_audio']['value_sha256']
            if not audio_match:
                raise RuntimeError('Frozen control final audio mismatch')
            item={'id':uid,'steps':matches,'final_audio_matches':audio_match,'seconds':time.perf_counter()-start}
            atomic_json(folder/f'{uid}.json',item); items.append(item)
            print(f'BASELINE EXACT {uid}: 50 updates and final audio match',flush=True)
    result={'status':'passed','items':items,'checked_updates':150,'created_at_utc':stamp()}
    atomic_json(folder/'result.json',result)
    return result


def evaluate_records(plan, name, records):
    """Native saved-waveform source evaluation; never loads an evaluation target."""
    check(plan)
    import run_quality_research as historical
    historical.configure_scorers(plan['quality'])
    _,_,native,metadata=models(plan,with_vocoder=False)
    scorer=QualityScorer(str(ROOT/'artifacts'))
    scores=[]
    for record in records:
        for variant,asset in record['audio'].items():
            if digest(asset['path']) != asset['sha256']:
                raise ValueError('Saved audio changed before evaluation')
            audio=load_audio(asset['path'])
            hypothesis=native.transcribe(audio.cuda())
            scores.append({'id':record['id'],'variant':variant,'reference':record['text'],
                'hypothesis':hypothesis,'audio_sha256':asset['sha256'],
                **error_counts(record['text'],hypothesis),'quality':scorer(audio)})
        print(f'SOURCE SCORED {name}: {record["id"]}',flush=True)
    result={}
    for variant in sorted({r['variant'] for r in scores}):
        pool=[r for r in scores if r['variant']==variant]
        result[variant]={'count':len(pool),**aggregate_errors(pool),
            'quality':{k:sum(r['quality'][k] for r in pool)/len(pool) for k in ['dnsmos','nisqa','utmos']}}
    folder=RUN/name
    write_jsonl(folder/'source-scores.jsonl',scores)
    receipt={'status':'complete','source_only':True,'target_models_evaluated':[],
        'source_metadata':metadata,'scorer_manifest':plan['quality'],'runtime':environment(),
        'evaluation_backend':backend_manifest(),'summary':result,
        'scores_sha256':digest(folder/'source-scores.jsonl')}
    atomic_json(folder/'source-summary.json',receipt)
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','baseline','evaluate'])
    parser.add_argument('--name')
    args=parser.parse_args()
    plan=prepare()
    if args.stage=='baseline': baseline_preflight(plan)
    if args.stage=='evaluate':
        records=read_jsonl(RUN/args.name/'generated.jsonl')
        evaluate_records(plan,args.name,records)
    print(args.stage+' complete',flush=True)
