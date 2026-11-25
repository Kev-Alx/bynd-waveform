"""Compare frozen WavLM features with fresh forward execution policies.

Read-only with respect to previous studies. No attack, decoding or MOS.
"""
import argparse,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,seed_everything
from beyondwaveform.models.feature import load_feature_models
from parallel_jobs import Supervisor,now
from probe_numerical_repeatability import backend_mode,backend_manifest,compare_gradients,tensor_value_sha256
from probe_deterministic_execution import save_tensor
import run_feature_layer_research as parent
import run_quality_research as protocol
import run_quality_refinement as stages

DEFAULT=ROOT/'reproduction-runs/encoder-execution-diagnostics'
MODES=('auto','math','strict_math')


def prepare(output):
    protocol.require(not (output/'plan.json').exists(),'Plan already exists')
    previous=protocol.read(parent.DEFAULT/'plan.json');parent.check(previous)
    protocol.require(protocol.read(parent.DEFAULT/'job.json')['status']=='failed','Inspect original failure before this diagnostic')
    frozen=dict(previous['frozen_files'])
    files=[Path(__file__).resolve(),parent.DEFAULT/'plan.json',parent.DEFAULT/'job.json',parent.DEFAULT/'progress.json']
    frozen.update({str(p.resolve()):digest(p) for p in files})
    plan={'created_at':now(),'frozen_files':frozen,'runtime':previous['runtime'],'pilot':previous['pilot'],
        'base_config':previous['base_config'],'feature_artifacts':previous['feature_artifacts'],
        'cache_audit':previous['baseline_cache_audit'],'modes':list(MODES),'repeat_ids':previous['pilot']['ids'][:3],
        'comparison_only':True,'discord_enabled':False,'scope':'Fresh layer-6 forward extraction versus frozen q on all 24 development clips under auto/math/strict-math. Three repeats on first three IDs. No optimizer, layer selection, free decoding or MOS; no changed tolerance or old outputs.'}
    atomic_json(output/'plan.json',plan);(output/'plan.json').chmod(0o444);return plan


def check(plan):
    protocol.require(protocol.runtime()==plan['runtime'] and plan['modes']==list(MODES) and
        plan['repeat_ids']==plan['pilot']['ids'][:3] and plan['comparison_only'],'Changed encoder diagnostic scope/runtime')
    for path,expected in plan['frozen_files'].items():protocol.require(digest(path)==expected,f'Changed frozen input: {path}')


def measure(output,plan,mode):
    path=output/(mode+'.json');directory=output/(mode+'-features')
    protocol.require(not path.exists() and not directory.exists(),'Preserve previous/partial encoder measurements')
    if mode=='strict_math':protocol.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Strict process needs workspace')
    cached={r['id']:r for r in protocol.read(plan['cache_audit'])['items']}
    seeds={r['id']:r['seed'] for r in read_jsonl(parent.prior.BASELINE/'generated.jsonl')}
    rows=[]
    with backend_mode(mode):
        seed_everything(plan['base_config']['seed']);encoder,vocoder,metadata=load_feature_models(plan['base_config']['features'],'cuda')
        protocol.require(metadata==plan['feature_artifacts'],'Changed WavLM/vocoder artifacts');del vocoder
        flags=backend_manifest()
        for source in read_jsonl(plan['pilot']['manifest']):
            uid=source['id'];x=load_audio(source['audio']).cuda()
            protocol.require(digest(source['audio'])==source['audio_sha256'] and
                digest(cached[uid]['q_path'])==cached[uid]['q_sha256'],'Changed frozen waveform/q')
            reference=torch.load(cached[uid]['q_path'],map_location='cpu',weights_only=True)
            repeat_count=3 if uid in plan['repeat_ids'] else 1;values=[];artifacts=[]
            for index in range(repeat_count):
                seed_everything(seeds[uid])
                with torch.no_grad():q=encoder(x).detach().cpu()
                values.append(q);artifacts.append(save_tensor(q,directory/uid/f'repeat-{index}.pt'))
            comparison=compare_gradients(reference,values[0])
            pairs=[compare_gradients(values[0],v) for v in values[1:]]
            rows.append({'id':uid,'reference_value_sha256':tensor_value_sha256(reference),'artifacts':artifacts,
                'frozen_cache_comparison':comparison,'repeat_comparisons':pairs,
                'within_original_tolerance':comparison['max_abs_difference']<=1e-4 and comparison['relative_l2_difference_to_left']<=1e-5})
            print(f'Encoder {mode}: {uid}, maxabs={comparison["max_abs_difference"]:.8g}, relative={comparison["relative_l2_difference_to_left"]:.8g}, exact={comparison["raw_value_sha256_equal"]}',flush=True)
    check(plan)
    atomic_json(path,{'status':'complete','mode':mode,'backend':flags,'items':rows,
        'all_24_cache_bitwise_equal':all(r['frozen_cache_comparison']['raw_value_sha256_equal'] for r in rows),
        'first_three_repeat_bitwise_equal':all(p['raw_value_sha256_equal'] for r in rows for p in r['repeat_comparisons']),
        'all_within_original_tolerance':all(r['within_original_tolerance'] for r in rows),'scope':plan['scope']})


def report(output,plan,rows,status):
    atomic_json(output/'results.json',{'status':status,'results':rows,'scope':plan['scope']})
    lines=['# Encoder execution diagnostics','',f'Status: **{status}**.','',
        '| Policy | 24 cached q exactly equal | First-three repeats exact | Original tolerance |', '|---|---|---|---|']
    lines.extend(f"| {r['name']} | {r['cache_exact']} | {r['repeats_exact']} | {r['within_tolerance']} |" for r in rows)
    lines+=['',plan['scope'],''];(output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--mode',choices=MODES);args=parser.parse_args()
    os.environ['BEYONDWAVEFORM_NOTIFY_DISCORD']='0';output=args.output.resolve()
    plan=prepare(output) if args.prepare else protocol.read(output/'plan.json');check(plan)
    if args.prepare:print(f'Frozen encoder diagnostic: {len(plan["frozen_files"])} inputs.',flush=True);return
    if args.mode:measure(output,plan,args.mode);return
    supervisor=Supervisor(output);rows=[]
    try:
        report(output,plan,rows,'running')
        for mode in MODES:
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output),'--mode',mode]
            if mode=='strict_math':cmd=['/usr/bin/env','CUBLAS_WORKSPACE_CONFIG=:4096:8',*cmd]
            stages.phase(supervisor,mode,cmd);result=protocol.read(output/(mode+'.json'))
            rows.append({'name':mode,'report':str(output/(mode+'.json')),'cache_exact':result['all_24_cache_bitwise_equal'],
                'repeats_exact':result['first_three_repeat_bitwise_equal'],'within_tolerance':result['all_within_original_tolerance']})
            report(output,plan,rows,'running')
        check(plan);report(output,plan,rows,'complete');supervisor.finish(scope=plan['scope'])
    except BaseException as error:report(output,plan,rows,'failed');supervisor.fail(error);raise


if __name__=='__main__':main()
