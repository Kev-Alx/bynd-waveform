"""Supervise equivalent spectral padding and full-trajectory CUDA validation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from beyondwaveform.io import atomic_json,digest
from parallel_jobs import Supervisor,now
import run_optimizer_diagnostics as previous
import run_quality_research as protocol
import run_quality_refinement as stages

DEFAULT=ROOT/'reproduction-runs/deterministic-execution-validation'
PREVIOUS=ROOT/'reproduction-runs/optimizer-mechanism-diagnostics-v2'


def prepare(output):
    protocol.require(not (output/'plan.json').exists(),'Plan already exists')
    old=protocol.read(PREVIOUS/'plan.json')
    previous.check(old)
    protocol.require(protocol.read(PREVIOUS/'job.json')['status']=='complete','Previous diagnostics not complete')
    review=ROOT/'reproduction-runs/optimizer-completion-review-20261019/review.json'
    protocol.require(protocol.read(review)['captures_checked']==12,'Complete optimizer evidence review first')
    files=[Path(__file__).resolve(), ROOT/'scripts/probe_deterministic_execution.py',
        ROOT/'beyondwaveform/deterministic_frontend.py',ROOT/'tests/test_deterministic_frontend.py',
        ROOT/'tests/test_deterministic_execution.py',ROOT/'docs/DETERMINISTIC_EXECUTION_VALIDATION.md',review,
        review.with_name('audit.py'),review.with_name('report.md')]
    files.extend(PREVIOUS/name for name in ('plan.json','job.json','results.json','adam-contributions.json',
        'repeatability-auto.json','repeatability-math.json','repeatability-strict_math.json'))
    native=protocol.read(PREVIOUS/'repeatability-math.json')
    files.extend(Path(r['gradient_artifact']['path']) for item in native['items']
        for state in item['states'].values() for r in state['repeats'])
    frozen=dict(old['frozen_files'])
    frozen.update({str(p.resolve()):digest(p) for p in files})
    plan={'created_at':now(),'frozen_files':frozen,'runtime':old['runtime'],
        'additional_dependencies':old.get('additional_dependencies',{}),'parent_plan':old['parent_plan'],
        'parent_plan_sha256':old['parent_plan_sha256'],'ids':old['ids'],
        'native_math_report':str(PREVIOUS/'repeatability-math.json'),
        'fixed_gradient_repeats':3,'trajectory_repeats':2,'updates':50,
        'forward_bitwise_required':True,'gradient_relative_l2_tolerance':1e-4,'gradient_max_abs_tolerance':1e-5,
        'execution':{'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True},
        'discord_enabled':False,
        'scope':'Three fixed existing development clips. Equivalent explicit reflected padding only, strict math CUDA. Native forward equality, repeated fixed gradients, then two fresh-process complete 50-update trajectories. No WER/MOS inference, candidate tuning or new validation/test use.'}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'plan.json',plan)
    (output/'plan.json').chmod(0o444)
    return plan


def check(plan):
    import importlib.metadata
    protocol.require(plan['fixed_gradient_repeats']==3 and plan['trajectory_repeats']==2 and plan['updates']==50 and
        plan['execution']=={'mode':'strict_math','CUBLAS_WORKSPACE_CONFIG':':4096:8','explicit_reflection':True} and
        plan['forward_bitwise_required'] and plan['gradient_relative_l2_tolerance']==1e-4 and
        plan['gradient_max_abs_tolerance']==1e-5,'Changed predeclared validation protocol')
    protocol.require(protocol.runtime()==plan['runtime'] and all(importlib.metadata.version(k)==v
        for k,v in plan['additional_dependencies'].items()),'Changed runtime')
    for path,expected in plan['frozen_files'].items():
        protocol.require(digest(path)==expected,f'Changed frozen execution-validation input: {path}')
    parent=protocol.read(plan['parent_plan'])
    protocol.require(digest(plan['parent_plan'])==plan['parent_plan_sha256'] and
        parent['pools']['pilot']['ids'][:3]==plan['ids'],'Changed input selection')


def verify_common(plan,result,plan_path):
    flags=result['backend']
    protocol.require(result['status']=='complete' and result['plan']==str(plan_path) and result['plan_sha256']==digest(plan_path)
        and result['runtime']==plan['runtime'] and [r['id'] for r in result['items']]==plan['ids'],'Changed result binding')
    protocol.require(flags['deterministic_algorithms'] and not flags['deterministic_warn_only'] and
        flags['cublas_workspace_config']==':4096:8' and flags['math_sdp_enabled'] and
        not any(flags[k] for k in ('flash_sdp_enabled','efficient_sdp_enabled','cudnn_sdp_enabled',
            'cuda_matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark')) and flags['cudnn_deterministic'],
        'Strict execution settings were not retained')
    parent=protocol.read(plan['parent_plan'])
    metadata=protocol.read(Path(parent['parents']['pilot']['run'])/'run.json')
    protocol.require(result['surrogate']==metadata['surrogate'],'Changed surrogate')


def verify_tensor(artifact):
    import torch
    from probe_numerical_repeatability import tensor_value_sha256
    protocol.require(digest(artifact['path'])==artifact['file_sha256'],'Changed saved tensor')
    value=torch.load(artifact['path'],map_location='cpu',weights_only=True)
    protocol.require(torch.isfinite(value).all().item() and tensor_value_sha256(value)==artifact['value_sha256'],
        'Tensor values do not match receipt')
    return value


def verify_fixed(result):
    from probe_numerical_repeatability import compare_gradients
    for item in result['items']:
        protocol.require(set(item['states'])=={'zero_delta','saved_parent_delta'},'Missing fixed state')
        for state in item['states'].values():
            protocol.require(len(state['repeats'])==3 and len(state['pairs'])==3,'Wrong repeat coverage')
            protocol.require({(p['left'],p['right']) for p in state['pairs']}=={(0,1),(0,2),(1,2)},'Wrong repeat pairs')
            values=[verify_tensor(r['gradient']) for r in state['repeats']]
            for pair in state['pairs']:
                rebuilt=compare_gradients(values[pair['left']],values[pair['right']])
                protocol.require(rebuilt==pair['comparison'],'Incorrect repeated-gradient comparison')
                equal=rebuilt['raw_value_sha256_equal'] and state['repeats'][pair['left']]['loss_value_sha256']==state['repeats'][pair['right']]['loss_value_sha256']
                protocol.require(equal==pair['bitwise_equal'],'Incorrect bitwise summary')
            native=verify_tensor({'path':state['native_math_gradient_artifact']['path'],
                'file_sha256':state['native_math_gradient_artifact']['file_sha256'],
                'value_sha256':state['native_math_gradient_artifact']['value_sha256']})
            rebuilt=compare_gradients(native,values[0])
            protocol.require(rebuilt==state['native_math_gradient_comparison'],'Incorrect cross-policy comparison')
            tolerance=rebuilt['relative_l2_difference_to_left']<=1e-4 and rebuilt['max_abs_difference']<=1e-5
            conformance=state['forward_conformance']
            protocol.require(set(conformance['checks'])=={'original_audio_hf','generated_hf','generated_mel','full_objective_scalars'},
                'Incomplete forward conformance')
            protocol.require(all(r['bitwise_equal']==(r['native_value_sha256']==r['explicit_value_sha256']) and
                (not r['bitwise_equal'] or r['max_abs_difference']==0) for r in conformance['checks'].values()),
                'Incorrect forward value comparison')
            protocol.require(conformance['all_bitwise_equal']==all(r['bitwise_equal'] for r in conformance['checks'].values()),
                'Incorrect forward conformance summary')
            passed=conformance['all_bitwise_equal'] and all(p['bitwise_equal'] for p in state['pairs']) and tolerance
            protocol.require(state['passed']==passed and state['native_math_gradient_within_tolerance']==tolerance,'Incorrect fixed-point gate')
        protocol.require(item['passed']==all(s['passed'] for s in item['states'].values()),'Incorrect clip gate')
    protocol.require(result['passed']==all(i['passed'] for i in result['items']),'Incorrect overall gate')


def compare_trajectories(left,right):
    from probe_deterministic_execution import tensor_tree
    import torch
    protocol.require(len(left['items'])==len(right['items'])>0,'Missing trajectory evidence')
    comparisons=[]
    for a,b in zip(left['items'],right['items']):
        protocol.require(a['id']==b['id'] and a['seed']==b['seed'] and a['actual_optimization_updates']==b['actual_optimization_updates']==50
            and len(a['trace'])==len(b['trace'])==50,'Changed trajectory selection or updates')
        rows=[]
        for step,(x,y) in enumerate(zip(a['trace'],b['trace']),1):
            protocol.require(x['step']==y['step']==step and x['radius']<=.100001 and y['radius']<=.100001,'Invalid trajectory trace')
            dx,dy=verify_tensor(x['delta']),verify_tensor(y['delta'])
            scalars=all(x[k]==y[k] for k in ('loss_values','loss_value_sha256','gradient_value_sha256','radius','optimizer_value_receipt'))
            equal=x['delta']['value_sha256']==y['delta']['value_sha256']
            rows.append({'step':step,'delta_bitwise_equal':equal,'trace_and_moments_bitwise_equal':scalars,
                'passed':scalars and equal})
        verify_tensor(a['final_audio']);verify_tensor(b['final_audio'])
        waves=a['final_audio']['value_sha256']==b['final_audio']['value_sha256']
        for item in (a,b):
            state=item['final_optimizer']
            protocol.require(digest(state['path'])==state['file_sha256'],'Changed final optimizer pack')
            rebuilt=tensor_tree(torch.load(state['path'],map_location='cpu',weights_only=True))
            protocol.require(rebuilt==state['value_receipt']==item['trace'][-1]['optimizer_value_receipt'],'Incorrect final moment receipt')
        moments=a['final_optimizer']['value_receipt']==b['final_optimizer']['value_receipt']
        comparisons.append({'id':a['id'],'updates':rows,'final_audio_bitwise_equal':waves,
            'final_optimizer_bitwise_equal':moments,'passed':all(r['passed'] for r in rows) and waves and moments})
    return {'status':'complete','items':comparisons,'passed':all(r['passed'] for r in comparisons)}


def write_report(output,rows,status):
    atomic_json(output/'results.json',{'status':status,'updated_at':now(),'results':rows})
    lines=['# Deterministic execution validation','',f'Status: **{status}**.','',
        '| Stage | Passed | Evidence |','|---|---|---|']
    lines.extend(f"| {r['name']} | {r['passed']} | [JSON]({r['report']}) |" for r in rows)
    lines+=['','This validates numerical execution on three existing development clips. It does not establish WER/quality improvement or author-code parity. No production frontend was changed.','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD']='0'
    output=args.output.resolve();plan_path=output/'plan.json'
    plan=prepare(output) if args.prepare else protocol.read(plan_path)
    check(plan)
    if args.prepare or args.check_only:
        print(f'Execution validation preflight passed: {len(plan["frozen_files"])} frozen inputs.',flush=True)
        return
    supervisor=Supervisor(output);rows=[]
    try:
        write_report(output,rows,'running')
        for name,stage in (('fixed-gradients','fixed'),('trajectory-repeat-0','trajectory'),('trajectory-repeat-1','trajectory')):
            path=output/(name+'.json')
            protocol.require(not path.exists(),'Review existing evidence before rerunning')
            command=['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',sys.executable,'-u',
                str(ROOT/'scripts/probe_deterministic_execution.py'),'--plan',str(plan_path),'--output',str(path),'--stage',stage]
            stages.phase(supervisor,name,command)
            check(plan)
            result=protocol.read(path);verify_common(plan,result,plan_path)
            if stage=='fixed':
                verify_fixed(result)
                passed=result['passed']
            else:
                passed=True  # Stage completeness; exact equality is checked below.
            rows.append({'name':name,'passed':passed,'report':str(path),'sha256':digest(path)})
            write_report(output,rows,'running')
            if not passed:
                write_report(output,rows,'complete: gate failed; trajectories skipped')
                supervisor.finish(gate_passed=False,scope=plan['scope'])
                return
        comparison=compare_trajectories(protocol.read(output/'trajectory-repeat-0.json'),protocol.read(output/'trajectory-repeat-1.json'))
        path=output/'trajectory-comparison.json';atomic_json(path,comparison)
        rows.append({'name':'full-trajectory-bitwise-comparison','passed':comparison['passed'],'report':str(path),'sha256':digest(path)})
        check(plan)
        write_report(output,rows,'complete')
        supervisor.finish(gate_passed=comparison['passed'],scope=plan['scope'])
    except BaseException as error:
        write_report(output,rows,'failed');supervisor.fail(error);raise


if __name__=='__main__':
    main()
