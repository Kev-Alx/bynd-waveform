"""Bounded current-model union adversarial training with matched adapter controls."""
from __future__ import annotations

import gc
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

import torch
from beyondwaveform.evaluation import error_counts,aggregate_errors
from beyondwaveform.io import load_audio,save_audio
from beyondwaveform.models.feature import load_feature_models

from .common import ROOT,WORKSPACE,atomic_json,digest,read_json,read_jsonl,stamp,verify_audio_asset,write_jsonl,initialize_torch
from .whisper_adapter import load_base
from .lora import attach_encoder_lora,adapter_state,load_adapter_state
from .adversary import feature_pgd,waveform_pgd


def feature_models():
    return load_feature_models({'artifact_root':str(WORKSPACE/'artifacts'),'layer':6,'normalize':False,
                                'vocoder_variant':'regular'},'cuda')


def cache_training_features(rows,encoder):
    folder=ROOT/'cache/training_features';folder.mkdir(parents=True,exist_ok=True)
    receipts={}
    for i,row in enumerate(rows):
        path=folder/f'{row["id"]}.pt';receipt=folder/f'{row["id"]}.json'
        if digest(row['audio'])!=row['audio_sha256']:raise ValueError('Changed training waveform')
        if path.exists() and receipt.exists():
            r=read_json(receipt)
            if r['input_audio_sha256']!=row['audio_sha256'] or r['sha256']!=digest(path):
                raise ValueError('Training feature cache mismatch')
        else:
            with torch.no_grad():q=encoder(load_audio(row['audio']).cuda()).detach().cpu()
            torch.save(q,path)
            atomic_json(receipt,{'input_audio_sha256':row['audio_sha256'],'sha256':digest(path),
                                 'shape':list(q.shape),'encoder_contract':'WavLM-Large layer6 normalize=False'})
        receipts[row['id']]=str(path)
        if (i+1)%10==0:print(f'feature-cache {i+1}/{len(rows)}',flush=True)
    return receipts


def load_trained(condition):
    base,meta=load_base()
    if condition=='base':return base,meta
    folder=ROOT/'runs/pilot-v1'/f'train-{condition}'
    result=read_json(folder/'training.json')
    if result['status']!='complete':raise ValueError('Training incomplete')
    if digest(folder/'adapters.pt')!=result['adapter_sha256']:raise ValueError('Changed adapter weights')
    attach_encoder_lora(base.model,rank=4,alpha=8.)
    state=torch.load(folder/'adapters.pt',map_location='cpu',weights_only=True)
    load_adapter_state(base.model,state)
    base.eval()
    return base,{**meta,'adapter':result}


def evaluate_trained(base,folder):
    rows=read_jsonl(ROOT/'data/development48.jsonl')
    results=[]
    for row in rows:
        for variant in ['clean','reconstruction','adversarial']:
            dest=folder/'evaluation/records'/f'{row["id"]}--{variant}.json'
            asset=row['saved_audio'][variant];verify_audio_asset(asset)
            if dest.exists():
                r=read_json(dest)
                if r['audio_sha256']!=asset['sha256']:raise ValueError('Evaluation identity mismatch')
            else:
                x=load_audio(asset['path']).cuda()
                begin=time.time()
                hyp=base.transcribe(x)
                r={'id':row['id'],'speaker':row['speaker'],'variant':variant,'reference':row['text'],
                   'hypothesis':hyp,'audio_sha256':asset['sha256'],**error_counts(row['text'],hyp),
                   'seconds':time.time()-begin}
                atomic_json(dest,r)
            results.append(r)
        print(f'eval {len(results)}/{3*len(rows)} {row["id"]}',flush=True)
    summary={}
    for variant in ['clean','reconstruction','adversarial']:
        rr=[r for r in results if r['variant']==variant]
        summary[variant]={'count':len(rr),**aggregate_errors(rr)}
    write_jsonl(folder/'evaluation/records.jsonl',results)
    atomic_json(folder/'evaluation/summary.json',{'status':'complete','scope':'existing non-adaptive FV, exposed development',
                                               'summary':summary})
    return summary


def train(condition):
    supported={'clean_ft','reconstruction_ft','wave_at','fv_at','union_at'}
    if condition not in supported:raise ValueError(f'Unsupported training condition: {condition}')
    plan=read_json(ROOT/'runs/pilot-v1/plan.json');config=plan['training']
    folder=ROOT/'runs/pilot-v1'/f'train-{condition}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():
        prior=read_json(folder/'training.json')
        if prior['status']=='complete':
            base,_=load_trained(condition)
            if not (folder/'evaluation/summary.json').exists():return evaluate_trained(base,folder)
            return read_json(folder/'evaluation/summary.json')
        raise ValueError('Incomplete training directory; preserve it and choose an explicit recovery path')
    initialize_torch(config['seed'])
    used=[Path(__file__),Path(__file__).with_name('lora.py'),Path(__file__).with_name('adversary.py'),
          Path(__file__).with_name('whisper_adapter.py')]
    receipt={'status':'running','condition':condition,'started_at':stamp(),'config':config,
             'source_hashes':{str(p):digest(p) for p in used},'plan_sha256':digest(ROOT/'runs/pilot-v1/plan.json')}
    atomic_json(folder/'training.json',receipt)
    rows=read_jsonl(ROOT/'data/train-short-1h.jsonl')
    base,base_meta=load_base()
    sample=load_audio(rows[0]['audio']).cuda()
    with torch.no_grad():before=float(base.nll(sample,rows[0]['text']))
    modules=attach_encoder_lora(base.model,config['rank'],config['alpha'])
    base.eval()
    with torch.no_grad():after=float(base.nll(sample,rows[0]['text']))
    if abs(after-before)>1e-6:raise ValueError('Zero-adapter preflight changed base loss')
    parameters=[p for p in base.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=.01)
    initial=adapter_state(base.model)
    encoder,vocoder,feature_meta=feature_models()
    paths=cache_training_features(rows,encoder)
    del encoder;gc.collect();torch.cuda.empty_cache()
    receipt.update({'base':base_meta,'feature_artifacts':feature_meta,'adapter_modules':modules,
                    'trainable_parameters':sum(p.numel() for p in parameters),
                    'total_parameters':sum(p.numel() for p in base.parameters()),
                    'zero_adapter_nll_before':before,'zero_adapter_nll_after':after})
    atomic_json(folder/'training.json',receipt)
    records=[];families=Counter();start=time.time();order=[]
    for update in range(config['updates']):
        if time.time()-start>config['maximum_seconds_per_condition']:break
        if not order:
            order=list(range(len(rows)));random.Random(config['seed']+update).shuffle(order)
        row=rows[order.pop()]
        x=load_audio(row['audio']).cuda()
        q=torch.load(paths[row['id']],map_location='cuda',weights_only=True)
        begin=time.time();inner_calls=0;selected='clean';selection={}
        optimizer.zero_grad(set_to_none=True)
        if condition=='clean_ft':aug=x
        elif condition=='reconstruction_ft':
            with torch.no_grad():aug=vocoder(q).detach()
            selected='reconstruction'
        else:
            attempts={}
            if condition in {'wave_at','union_at'}:
                wave,wt=waveform_pgd(x,lambda u:base.nll(u,row['text']),
                                    steps=config['inner_steps'],epsilon=config['waveform_epsilon'])
                with torch.no_grad():wave_nll=float(base.nll(wave,row['text']))
                attempts['waveform']=(wave,wave_nll,wt)
                inner_calls+=config['inner_steps']
            if condition in {'fv_at','union_at'}:
                feature,delta,ft=feature_pgd(q,x,vocoder,lambda u:base.nll(u,row['text']),
                    steps=config['inner_steps'],rho=config['feature_rho'],learning_rate=config['feature_lr'],
                    lambda_perc=config['feature_lambda_perc'])
                with torch.no_grad():feature_nll=float(base.nll(feature,row['text']))
                attempts['feature']=(feature,feature_nll,ft)
                inner_calls+=config['inner_steps']
            selected=max(attempts,key=lambda k:attempts[k][1]);aug=attempts[selected][0].detach()
            selection={name:{'final_nll':v[1],'trace':v[2]} for name,v in attempts.items()}
        # autograd.grad in inner attacks must never populate outer parameter grads.
        if any(p.grad is not None for p in parameters):raise ValueError('Inner-loop parameter-gradient contamination')
        clean_loss=base.supervised_loss(x,row['text'])
        (.5*clean_loss).backward()
        augmented_loss=base.supervised_loss(aug,row['text'])
        (.5*augmented_loss).backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite adapter training gradient')
        optimizer.step();families[selected]+=1
        record={'update':update+1,'id':row['id'],'selected_family':selected,
                'clean_loss':float(clean_loss.detach()),'augmented_loss':float(augmented_loss.detach()),
                'gradient_norm':float(norm),'inner_backward_calls':inner_calls,'outer_backward_calls':2,
                'selection':selection,'seconds':time.time()-begin,
                'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        records.append(record);atomic_json(folder/'updates'/f'{update+1:04d}.json',record)
        atomic_json(folder/'progress.json',{'status':'running','update':update+1,'maximum_updates':config['updates'],
                    'seconds':time.time()-start,'families':dict(families),'updated_at':stamp()})
        if (update+1)%16==0:print(f'train {condition} {update+1}/{config["updates"]} loss={.5*(record["clean_loss"]+record["augmented_loss"]):.4f} elapsed={time.time()-start:.1f}s',flush=True)
    state=adapter_state(base.model)
    delta_norm=sum((state[k]-initial[k]).square().sum().item() for k in state)**.5
    if not records or delta_norm==0:raise ValueError('No learned adapter update')
    torch.save(state,folder/'adapters.pt')
    torch.save({'optimizer':optimizer.state_dict(),'updates':len(records)},folder/'optimizer.pt')
    receipt.update({'status':'complete','finished_at':stamp(),'updates':len(records),
        'truncated_by_time_cap':len(records)<config['updates'],'seconds':time.time()-start,
        'family_counts':dict(families),'adapter_change_l2':delta_norm,'adapter_sha256':digest(folder/'adapters.pt'),
        'inner_backward_calls':sum(r['inner_backward_calls'] for r in records),
        'outer_backward_calls':sum(r['outer_backward_calls'] for r in records)})
    atomic_json(folder/'training.json',receipt);write_jsonl(folder/'updates.jsonl',records)
    atomic_json(folder/'progress.json',{'status':'complete','updates':len(records),'updated_at':stamp()})
    del optimizer,vocoder;gc.collect();torch.cuda.empty_cache()
    base.requires_grad_(False).eval()
    return evaluate_trained(base,folder)
