"""Fresh-process source-only scoring with the historical native evaluation policy.

Launch with CUBLAS_WORKSPACE_CONFIG absent. This module never loads a held-out
recognizer, an attack generator, or a vocoder. All outputs stay in improvement.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT/'improvement'
sys.path[:0] = [str(OUT),str(ROOT),str(ROOT/'scripts')]
import torch
import execution_runtime as rt
from beyondwaveform.evaluation import QualityScorer, aggregate_errors, error_counts
from beyondwaveform.models.whisper import load_whisper
from probe_numerical_repeatability import backend_manifest
import runtime48 as cycle
RUN48 = OUT/'execution48'


def validate_pilot_records(records, allowed):
    if len(records)!=48 or len({r['id'] for r in records})!=48 or set(r['id'] for r in records)!=allowed.keys():
        raise ValueError('Pilot scoring requires all48 registered development IDs exactly once')
    for record in records:
        if record['text'] != allowed[record['id']]['text']:
            raise ValueError('Reference text changed')
        if set(record['audio']) != set(cycle.VARIANTS):
            raise ValueError('Every pilot variant must retain every one of the48 utterances')


def native_policy(expected):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') is not None:
        raise ValueError('Native scoring requires a fresh process with CUBLAS_WORKSPACE_CONFIG unset')
    torch.set_num_threads(expected['cpu_threads'])
    torch.set_num_interop_threads(expected['interop_threads'])
    torch.use_deterministic_algorithms(expected['deterministic_algorithms'],
        warn_only=expected['deterministic_warn_only'])
    torch.backends.cudnn.deterministic=expected['cudnn_deterministic']
    torch.backends.cudnn.benchmark=expected['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=expected['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=expected['cudnn_allow_tf32']
    torch.backends.cuda.enable_flash_sdp(expected['flash_sdp_enabled'])
    torch.backends.cuda.enable_mem_efficient_sdp(expected['efficient_sdp_enabled'])
    torch.backends.cuda.enable_math_sdp(expected['math_sdp_enabled'])
    torch.backends.cuda.enable_cudnn_sdp(expected['cudnn_sdp_enabled'])
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa=expected['whisper_use_sdpa']
    torch.set_float32_matmul_precision(expected['float32_matmul_precision'])
    measured=backend_manifest()
    if measured != expected:
        differences={k:{'expected':expected.get(k),'actual':measured.get(k)}
            for k in expected.keys()|measured.keys() if expected.get(k)!=measured.get(k)}
        raise ValueError(f'Native evaluation policy mismatch: {differences}')
    return measured


def evaluate(name, records_path=None, output_dir=None):
    if not name or Path(name).name!=name:
        raise ValueError('Use one simple registered experiment name')
    plan=rt.read(rt.RUN/'plan.json')
    rt.check(plan)
    registered48=cycle.check()
    folder=(Path(output_dir) if output_dir else RUN48/name).resolve()
    if not folder.is_relative_to(OUT):
        raise ValueError('All score outputs must stay in improvement')
    records_path=Path(records_path) if records_path else RUN48/name/'generated.jsonl'
    if not records_path.resolve().is_relative_to(OUT):
        raise ValueError('New generated manifest must stay in improvement')
    records=rt.read_jsonl(records_path)
    allowed={r['id']:r for r in rt.read_jsonl(registered48['development_manifest'])}
    validate_pilot_records(records, allowed)
    expected=rt.read(rt.BASE/'evaluation/summary.json')['evaluation_backend']
    backend=native_policy(expected)
    # Bind policy, code, inputs, and weights before the first prediction.
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'source-summary.json').exists():
        raise ValueError('Preserve previous scoring; choose a new output directory for a rerun')
    started=time.perf_counter()
    rt.atomic_json(folder/'evaluation-registration.json',{
        'created_at_utc':rt.stamp(),'name':name,'source_only':True,
        'generated_manifest':str(records_path),'generated_manifest_sha256':rt.digest(records_path),
        'evaluator_sha256':rt.digest(Path(__file__)), 'backend':backend,
        'historical_backend':str(rt.BASE/'evaluation/summary.json'),
        'allowed_development_ids':[r['id'] for r in records],
        'target_models_evaluated':[], 'pilot_count':48, 'cycle_plan_sha256':rt.digest(RUN48/'plan.json'), 'scorer_manifest':plan['quality']})
    native,metadata=load_whisper(plan['base_config']['surrogate'],'cuda')
    if metadata!=rt.read(rt.BASE/'run.json')['surrogate']:
        raise ValueError('Source checkpoint/decoder metadata changed')
    torch.cuda.reset_peak_memory_stats()
    import run_quality_research as historical
    historical.configure_scorers(plan['quality'])
    scorer=QualityScorer(str(ROOT/'artifacts'))
    scores=[]
    for index,record in enumerate(records,1):
        for variant,asset in record['audio'].items():
            if rt.digest(asset['path'])!=asset['sha256']:
                raise ValueError('Saved audio hash mismatch')
            audio=rt.load_audio(asset['path'])
            hypothesis=native.transcribe(audio.cuda())
            item={'id':record['id'],'variant':variant,'reference':record['text'],
                'hypothesis':hypothesis,'audio_sha256':asset['sha256'],
                **error_counts(record['text'],hypothesis),'quality':scorer(audio)}
            scores.append(item)
            rt.atomic_json(folder/'score-records'/f'{record["id"]}--{variant}.json',item)
        print(f'Native source scored {name}: {index}/{len(records)} [{record["id"]}]',flush=True)
    summaries={}
    for variant in sorted({r['variant'] for r in scores}):
        pool=[r for r in scores if r['variant']==variant]
        summaries[variant]={'count':len(pool),**aggregate_errors(pool),
            'quality':{k:sum(r['quality'][k] for r in pool)/len(pool)
                for k in ['dnsmos','nisqa','utmos']}}
    rt.write_jsonl(folder/'source-scores.jsonl',scores)
    measured=backend_manifest()
    if measured!=expected:
        raise ValueError('Evaluation policy changed during scoring')
    result={'status':'complete','source_only':True,'target_models_evaluated':[],
        'source_metadata':metadata,'scorer_manifest':plan['quality'],
        'runtime':rt.environment(),'evaluation_backend':measured,'summary':summaries,
        'scores_sha256':rt.digest(folder/'source-scores.jsonl'),
        'resources':{'elapsed_wall_seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()},
        'baseline_policy_exactly_matched':True,'human_intelligibility_established':False}
    rt.atomic_json(folder/'source-summary.json',result)
    print(json.dumps({'status':'complete','summary':summaries},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--records',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    try:
        evaluate(args.name,args.records,args.output_dir)
    except Exception as error:
        folder=(args.output_dir or RUN48/args.name).resolve()
        if folder.is_relative_to(OUT):
            rt.atomic_json(folder/'source-evaluation-failure.json',{
                'status':'failed_infrastructure','error':str(error),
                'error_type':type(error).__name__,'source_only':True,
                'target_models_evaluated':[]})
        raise


if __name__=='__main__':
    main()
