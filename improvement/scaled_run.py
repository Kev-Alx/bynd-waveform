"""Frozen larger-model generation, decoder qualification and fixed validation."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import torch
import execution_runtime as rt
import scaled_scope as scope
from scaled_components import load_sources, BigVGAN16
from scaled_attack import optimize


def verify_asset(asset):
    if rt.digest(asset['path'])!=asset['sha256']:raise ValueError('Changed saved asset')
    return asset


def validation_cache():
    scope.check();scope.validation_freeze()
    from r3_real_runner_v2 import load_encoder_only
    plan=rt.read(rt.RUN/'plan.json');rt.check(plan)
    if (scope.RUN/'validation-feature-cache.json').exists():raise FileExistsError('Preserve completed feature cache')
    records=[]
    with rt.backend_mode('strict_math'):
        torch.set_num_threads(4);rt.seed_everything(2026)
        native,encoder,metadata=load_encoder_only(plan)
        for row in rt.read_jsonl(scope.COHORTS['validation']['manifest']):
            if rt.digest(row['audio'])!=row['audio_sha256']:raise ValueError('Changed validation input')
            x=rt.load_audio(row['audio']).cuda()
            with torch.no_grad():q=encoder(x).detach()
            p=scope.RUN/'validation-features'/f'{row["id"]}.pt';p.parent.mkdir(parents=True,exist_ok=True)
            torch.save(q.cpu(),p)
            records.append({'id':row['id'],'q':{'path':str(p),'sha256':rt.digest(p)},'audio_sha256':row['audio_sha256']})
            print('Cached validation features',len(records),'/76',flush=True)
    rt.atomic_json(scope.RUN/'validation-feature-cache.json',{'status':'completed','records':records,'encoder_metadata':metadata})


def generate(phase):
    plan=scope.check();info=plan['cohorts'][phase];folder=scope.RUN/phase
    if phase=='validation':scope.validation_freeze()
    if phase=='melpilot' and rt.read(scope.RUN/'decoder-decision.json')['status']!='pass':
        raise ValueError('Decoder did not qualify for attack experiments')
    if (folder/'records').exists():raise FileExistsError('Preserve previous or partial generation')
    folder.mkdir(parents=True,exist_ok=True)
    previous=rt.read(rt.RUN/'plan.json');rt.check(previous)
    history={r['id']:r for r in rt.read_jsonl(rt.OUT/'execution48/mechanism-pilot48/generated.jsonl')}
    reference_metadata=rt.read(rt.OUT/'execution48/mechanism-pilot48/generation-result.json')['source_metadata']
    began=time.perf_counter();result={'status':'running','cohort':phase,'count':info['count'],'actual_count':0,
        'source_only':True,'target_models_evaluated':[],'failures':[],'started_at_utc':rt.stamp(),
        'plan_sha256':rt.digest(scope.RUN/'plan.json'),'source_metadata':reference_metadata}
    rt.atomic_json(folder/'generation-result.json',result)
    def progress(**value):
        state={'updated_at_utc':rt.stamp(),'cohort':phase,'total':info['count'],**value}
        rt.atomic_json(folder/'generation-status.json',state);print(state,flush=True)
    try:
        with rt.backend_mode('strict_math'):
            torch.set_num_threads(4);rt.seed_everything(2026)
            models,source_metadata=({}, {}) if phase=='qualification' else load_sources(previous)
            fv=None;big=None
            if phase in ('fvpilot','validation'):
                fv,_=rt.load_frozen_vocoder(previous['base_config']['features'],previous['feature_artifacts'],'cuda')
            if phase in ('qualification','melpilot') or phase=='validation' and any(r['method']=='mel_eot' for r in scope.validation_freeze()['methods'].values()):
                big=BigVGAN16()
            val_cache={r['id']:r for r in rt.read(scope.RUN/'validation-feature-cache.json')['records']} if phase=='validation' else {}
            if phase=='validation':
                if len(val_cache)!=76:raise ValueError('Incomplete validation feature cache')
                result['validation_feature_cache_sha256']=rt.digest(scope.RUN/'validation-feature-cache.json')
            torch.cuda.reset_peak_memory_stats();records=[]
            for index,row in enumerate(rt.read_jsonl(info['manifest']),1):
                uid=row['id'];start=time.perf_counter()
                if rt.digest(row['audio'])!=row['audio_sha256']:raise ValueError('Changed source audio')
                x=rt.load_audio(row['audio']).cuda();assets={};diagnostics={}
                assets['clean']=rt.save_audio(folder/'audio/clean'/f'{uid}.wav',x)
                if phase!='validation':
                    prior=history[uid]
                    qasset=verify_asset(prior['clean_features'])
                    q=torch.load(qasset['path'],map_location='cuda',weights_only=True).detach()
                    if phase in ('qualification','fvpilot'):
                        assets['fv_reconstruction']=verify_asset(prior['audio']['reconstruction'])
                        assets['fv_quality50']=verify_asset(prior['audio']['fv_quality'])
                    if big is not None:
                        mel=big.encode(x)
                        with torch.no_grad():neutral=big(mel)
                        assets['bigvgan_reconstruction']=rt.save_audio(folder/'audio/bigvgan_reconstruction'/f'{uid}.wav',neutral)
                        diagnostics['bigvgan']={'original_samples':len(x),'model_rate':24000,
                            'native_samples':big.native_samples,'clipped_samples':big.last_clipped_samples}
                else:
                    if val_cache[uid]['audio_sha256']!=row['audio_sha256']:raise ValueError('Validation features reference another input')
                    qasset=verify_asset(val_cache[uid]['q'])
                    q=torch.load(qasset['path'],map_location='cuda',weights_only=True).detach()
                    if big is not None:mel=big.encode(x)
                diagnostics['q']=qasset
                if 'mel' in locals():
                    mp=folder/'mel-features'/f'{uid}.pt';mp.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(mel.detach().cpu(),mp);diagnostics['mel']={'path':str(mp),'sha256':rt.digest(mp)}
                if phase=='qualification':methods=[]
                elif phase=='fvpilot':methods=[('single',1500),('ensemble_raw',500),('ensemble_eot',500)]
                elif phase=='melpilot':methods=[('mel_eot',500)]
                else:methods=[(name,value) for name,value in scope.validation_freeze()['methods'].items()]
                for entry in methods:
                    if phase=='validation':
                        label,configuration=entry;method,endpoint=configuration['method'],configuration['step']
                    else:method,endpoint=entry;label=method
                    renderer,features=(big,mel) if method=='mel_eot' else (fv,q)
                    if phase=='validation':
                        with torch.no_grad():neutral=renderer(features)
                        assets[label+'_reconstruction']=rt.save_audio(folder/'audio'/f'{label}_reconstruction'/f'{uid}.wav',neutral)
                    def on_step(item,label=label):
                        progress(status='running',clip=index,id=uid,variant=label,update=item['step'])
                    def on_checkpoint(step,audio,delta,receipt,label=label,endpoint=endpoint):
                        if phase=='validation' and step!=endpoint:return
                        variant=label if phase=='validation' else f'{label}_{step}'
                        assets[variant]=rt.save_audio(folder/'audio'/variant/f'{uid}.wav',audio)
                        path=folder/'traces'/variant/f'{uid}.json';rt.atomic_json(path,receipt)
                        dp=folder/'deltas'/variant/f'{uid}.pt';dp.parent.mkdir(parents=True,exist_ok=True);torch.save(delta.cpu(),dp)
                        diagnostics[variant]={'receipt':str(path),'sha256':rt.digest(path),
                            'delta':{'path':str(dp),'sha256':rt.digest(dp)},'method':receipt['method'],'step':step}
                    optimize(features,x,row['text'],renderer,models,method=method,uid=uid,steps=endpoint,
                        on_step=on_step,on_checkpoint=on_checkpoint)
                if set(assets)!=set(info['variants']):raise ValueError('Incomplete registered variant coverage')
                record={'id':uid,'text':row['text'],'speaker':row['speaker'],'language':'en','role':phase,'count':info['count'],
                    'source_only':True,'input_audio_sha256':row['audio_sha256'],'audio':assets,'diagnostics':diagnostics,
                    'plan_sha256':result['plan_sha256'],'wall_seconds':time.perf_counter()-start}
                rt.atomic_json(folder/'records'/f'{uid}.json',record);records.append(record)
                result['actual_count']=index;rt.atomic_json(folder/'generation-result.json',result)
                progress(status='running',completed=index,id=uid)
                del x,q,assets,diagnostics
                if 'mel' in locals():del mel
                gc.collect()
            rt.write_jsonl(folder/'generated.jsonl',records)
            result.update(status='completed',wall_seconds=time.perf_counter()-began,generated_sha256=rt.digest(folder/'generated.jsonl'),
                generation_sources_metadata=source_metadata,backend=rt.backend_manifest(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                completed_at_utc=rt.stamp())
            rt.atomic_json(folder/'generation-result.json',result);progress(status='completed',completed=info['count'])
    except Exception as error:
        result.update(status='failed_execution',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            wall_seconds=time.perf_counter()-began)
        rt.atomic_json(folder/'generation-result.json',result);progress(status='failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','generate','cache-validation'])
    p.add_argument('--phase',choices=list(scope.COHORTS),default='qualification');args=p.parse_args()
    if args.stage=='register':print(scope.register())
    elif args.stage=='cache-validation':validation_cache()
    else:generate(args.phase)
