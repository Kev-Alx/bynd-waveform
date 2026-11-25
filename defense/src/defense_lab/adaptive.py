"""Direct current-defender FV audits, with retained deltas and hard decoding."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from beyondwaveform.io import load_audio,save_audio
from beyondwaveform.evaluation import error_counts,aggregate_errors
from beyondwaveform.attacks import relative_radius

from .common import ROOT,atomic_json,digest,read_json,read_jsonl,stamp,initialize_torch
from .training import feature_models,load_trained
from .adversary import feature_pgd
from .phone import PhoneRecognizer,PronunciationLexicon,PhoneWordDecoder
from .ctc import ctc_nll,greedy_ctc
from .inference import phone_reconcile
from .whisper_adapter import beam_candidates,candidate_nll


def hard_phone_decode(base,phone,lexicon,word_decoder,audio):
    with torch.no_grad():
        text,candidates,features=beam_candidates(base,audio,8)
        lp=phone.lexical_log_probs(audio)
        for c in candidates:c['origin']='whisper_beam'
        for c in word_decoder.decode(greedy_ctc(lp)):
            if c['text'] not in {r['text'] for r in candidates}:
                candidates.append({**c,'origin':'phone_path_lexicon'})
        for c in candidates:c['asr_nll']=float(candidate_nll(base,features,c['text']))
        return phone_reconcile(text,candidates,lp,lexicon)[0]


def audit():
    plan=read_json(ROOT/'runs/pilot-v1/plan.json')
    output=ROOT/'runs/pilot-v1/adaptive'
    output.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(ROOT/'data/smoke3.jsonl')
    source_paths=[Path(__file__),Path(__file__).with_name('adversary.py'),Path(__file__).with_name('training.py')]
    atomic_json(output/'registration.json',{'created_at':stamp(),'plan_sha256':digest(ROOT/'runs/pilot-v1/plan.json'),
        'conditions':['base','clean_ft','reconstruction_ft','union_at'],
        'source_hashes':{str(p):digest(p) for p in source_paths},
        'steps':[50,200],'restarts':3,'phone_joint_check_steps':50,'phone_joint_restarts':1,
        'scope':'three duration-stratified exposed clips; bounded current-model FV audit, no certificate/content validation',
        'temporal':'Full temporal adaptive promotion requires clean gate; failures are not claimed robust.'})
    records=[];start=time.time()
    for condition in ['base','clean_ft','reconstruction_ft','union_at']:
        base,meta=load_trained(condition)
        base.requires_grad_(False).eval()
        encoder,vocoder,features=feature_models()
        for row in rows:
            x=load_audio(row['audio']).cuda()
            with torch.no_grad():q=encoder(x).detach()
            qpath=output/'features'/f'{row["id"]}.pt'
            qpath.parent.mkdir(parents=True,exist_ok=True)
            if not qpath.exists():torch.save(q.cpu(),qpath)
            qdigest=digest(qpath)
            for restart in range(3):
                dest=output/condition/row['id']/f'restart-{restart}'
                if (dest/'complete.json').exists():
                    for step in [50,200]:records.append(read_json(dest/f'step-{step}.json'))
                    continue
                initialize_torch(20261022+restart)
                def observer(step,audio,delta,trace):
                    asset=save_audio(dest/f'step-{step}.wav',audio)
                    delta_path=dest/f'step-{step}.delta.pt';torch.save(delta.cpu(),delta_path)
                    # Score the exported waveform; avoid assuming export identity.
                    exported=load_audio(asset['path']).cuda()
                    hyp=base.transcribe(exported)
                    record={'id':row['id'],'speaker':row['speaker'],'condition':condition,
                        'attack':'direct adaptive FV, current defended model lexical mean NLL + clean TV/HF',
                        'restart':restart,'step':step,'reference':row['text'],'hypothesis':hyp,
                        **error_counts(row['text'],hyp),'audio':asset,
                        'delta_sha256':digest(delta_path),'delta_path':str(delta_path),
                        'q_path':str(qpath),'q_sha256':qdigest,'q_norm_at_generation':float(q.norm()),
                        'actual_relative_radius':float(relative_radius(delta,q)),
                        'trace':trace,'checkpoint_note':'same uninterrupted Adam trajectory, endpoints 50 and 200',
                        'content_validity':'feature radius verified; human linguistic content not assessed'}
                    atomic_json(dest/f'step-{step}.json',record);records.append(record)
                    print(f'adaptive {condition} {row["id"]} restart={restart} step={step} errors={record["substitutions"]+record["deletions"]+record["insertions"]}/{record["reference_units"]}',flush=True)
                feature_pgd(q,x,vocoder,lambda u:base.nll(u,row['text']),steps=200,rho=.1,
                            learning_rate=.05,lambda_perc=1.,random_start=restart>0,
                            observer=observer,checkpoints=(50,200))
                atomic_json(dest/'complete.json',{'complete':True,'base':meta,'feature_artifacts':features})
        del base,encoder,vocoder;gc.collect();torch.cuda.empty_cache()
    # A smaller joint attack establishes that the phonetic branch is differentiable
    # and is attacked together with the recognizer, not hidden behind hard decoding.
    base,meta=load_trained('base');base.requires_grad_(False)
    encoder,vocoder,_=feature_models();phone=PhoneRecognizer()
    lexicon=PronunciationLexicon(phone.labels)
    word_decoder=PhoneWordDecoder(lexicon,read_json(ROOT/'data/training_word_counts.json'))
    joint=[]
    for row in rows:
        dest=output/'phone-joint'/f'{row["id"]}.json'
        if dest.exists():joint.append(read_json(dest));continue
        labels,unknown=lexicon.encode(row['text'])
        if labels is None:
            atomic_json(dest,{'id':row['id'],'status':'unscorable reference pronunciation','unknown':unknown})
            continue
        x=load_audio(row['audio']).cuda()
        with torch.no_grad():q=encoder(x).detach()
        def joint_loss(audio):
            return base.nll(audio,row['text'])+ctc_nll(phone.lexical_log_probs(audio),labels)
        probe=x.detach().requires_grad_();value=joint_loss(probe)
        grad,=torch.autograd.grad(value,probe)
        if not torch.isfinite(grad).all() or grad.norm()==0:raise ValueError('Broken joint acoustic gradient')
        initialize_torch(20261022)
        audio,delta,trace=feature_pgd(q,x,vocoder,joint_loss,steps=50,rho=.1,learning_rate=.05)
        asset=save_audio(output/'phone-joint'/f'{row["id"]}.wav',audio)
        hypothesis=hard_phone_decode(base,phone,lexicon,word_decoder,load_audio(asset['path']).cuda())
        record={'id':row['id'],'speaker':row['speaker'],'status':'complete','reference':row['text'],
                'hypothesis':hypothesis,**error_counts(row['text'],hypothesis),'audio':asset,
                'relative_radius':float(relative_radius(delta,q)),'gradient_norm':float(grad.norm()),
                'steps':50,'restarts':1,'trace':trace,
                'scope':'joint ASR+phone-CTC proxy; real hard candidate reconciliation evaluated'}
        atomic_json(dest,record);joint.append(record)
        print('phone-joint complete',row['id'],flush=True)
    summary={}
    for condition in ['base','clean_ft','reconstruction_ft','union_at']:
        for step in [50,200]:
            subset=[r for r in records if r['condition']==condition and r['step']<=step]
            envelope=[]
            for row in rows:
                candidates=[r for r in subset if r['id']==row['id']]
                if candidates:envelope.append(max(candidates,key=lambda r:sum(r[k] for k in ['substitutions','deletions','insertions'])))
            summary[f'{condition}/worst_seen_up_to_{step}']={'count':len(envelope),**aggregate_errors(envelope)}
    if any(r.get('status')=='complete' for r in joint):
        complete=[r for r in joint if r.get('status')=='complete']
        summary['phone_joint_50_single_restart']={'count':len(complete),**aggregate_errors(complete)}
    result={'status':'complete','results':summary,'seconds':time.time()-start,
            'limits':['Three exposed clips only','Single learned seed','No human/MOS validity of fresh attacks',
                      'Phone joint audit uses one restart/50 steps; full fusion optimizer not guaranteed optimal',
                      'No held-out generator or final reserve used']}
    atomic_json(output/'summary.json',result)
    return result
