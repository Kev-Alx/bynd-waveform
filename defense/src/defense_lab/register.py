"""Freeze first-stage development scope before any comparison outcomes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.metadata

from .common import ROOT,WORKSPACE,atomic_json,balanced_screen,digest,read_jsonl,write_jsonl,stamp

PLAN=ROOT/'runs/pilot-v1/plan.json'


def register():
    if PLAN.exists():return __import__('json').loads(PLAN.read_text())
    pool=read_jsonl(WORKSPACE/'data/manifests/english-validation.jsonl')
    rows=balanced_screen(pool,48)
    screen_ids={r['id'] for r in rows}
    saved=read_jsonl(WORKSPACE/'reproduction-runs/validation-grid/rho-0.1_lambda-1.0/generated.jsonl')
    by_id={r['id']:r for r in saved}
    train=read_jsonl(WORKSPACE/'improvement/execution/data/train-1h.jsonl')
    # Fixed metadata-only training duration envelope; no outcome filtering.
    train=[r for r in train if r['duration']<=12.]
    if set(r['speaker'] for r in train)&set(r['speaker'] for r in rows):
        raise ValueError('Training/evaluation speaker overlap')
    if set(r['id'] for r in train)&screen_ids:raise ValueError('Training/evaluation ID overlap')
    protected=[WORKSPACE/'improvement/execution/data/final-reserve-200.jsonl',
               WORKSPACE/'reproduction-runs/paper-quality-finalization/data/confirmation/confirmation-48.jsonl']
    for path in protected:
        if screen_ids & {r['id'] for r in read_jsonl(path)}:raise ValueError('Protected reserve consumed')
    for row in rows:
        if row['id'] not in by_id:raise ValueError('Missing saved attack')
        row['saved_audio']=by_id[row['id']]['audio']
        row['role']='defense-development-exposed'
    ordered=sorted(rows,key=lambda r:(r['duration'],r['id']))
    smoke=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    write_jsonl(ROOT/'data/development48.jsonl',rows)
    write_jsonl(ROOT/'data/smoke3.jsonl',smoke)
    write_jsonl(ROOT/'data/train-short-1h.jsonl',train)
    counts=Counter(w.lower() for r in train for w in r['text'].split())
    atomic_json(ROOT/'data/training_word_counts.json',dict(counts))
    files=[WORKSPACE/'beyondwaveform/attacks.py',WORKSPACE/'beyondwaveform/models/feature.py',
           WORKSPACE/'beyondwaveform/models/whisper.py',WORKSPACE/'beyondwaveform/losses.py',
           WORKSPACE/'reproduction-runs/validation-grid/rho-0.1_lambda-1.0/run.json',
           ROOT/'data/development48.jsonl',ROOT/'data/smoke3.jsonl',ROOT/'data/train-short-1h.jsonl',
           ROOT/'weights/timit/download_receipt.json',ROOT/'weights/lexicon/receipt.json']
    plan={'created_at':stamp(),'status':'registered','authorization':'User: Start implementing and testing out these defense designs',
          'scope':'Exposed development and bounded training only; no confirmation/final/human study or robustness certification',
          'development_count':len(rows),'development_speakers':len({r['speaker'] for r in rows}),
          'train_count':len(train),'train_seconds':sum(r['duration'] for r in train),
          'train_speakers':len({r['speaker'] for r in train}),
          'source_train_manifest':str(WORKSPACE/'improvement/execution/data/train-1h.jsonl'),
          'train_filter':'duration <= 12s, before any defense outcomes; smaller than complete 1h',
          'smoke_ids':[r['id'] for r in smoke],
          'saved_attack':{'rho':.1,'lambda_perc':1.,'steps':50,'optimizer':'Adam','learning_rate':.05,
                          'nll_reduction':'mean','kind':'non-adaptive existing public-surrogate attack; logged radii, no retained delta in this historical cache'},
          'decoder':{'language':'en','task':'transcribe','fp16':False,'beam_size':5,'temperature':0.,
                     'without_timestamps':True,'condition_on_previous_text':False},
          'candidate_beam_size':8,'phone_eta':1.,'phone_pronunciation_limit':8,
          'phone_silence':'merge silence with CTC blank','phone_flap':'split dx mass equally to t/d',
          'phone_oov':'fallback to main ASR if its text contains OOV/numeric spelling; do not silently delete such words',
          'phone_expansion':'approximate greedy-phone-to-word edit search; training unigram lexicon; exact CTC rescore; no reference input',
          'temporal':{'width_seconds':8.,'stride_seconds':4.,'fusion':'word-midpoint ownership partition'},
          'training':{'conditions':['clean_ft','reconstruction_ft','union_at'],'rank':4,'alpha':8.,'updates':128,
                      'learning_rate':.0002,'seed':11,'inner_steps':5,'waveform_epsilon':.002,
                      'feature_rho':.1,'feature_lr':.05,'feature_lambda_perc':1.,'clean_outer_weight':.5,
                      'family_selection':'worst current-model lexical NLL among waveform and FV; counts/effects reported',
                      'maximum_seconds_per_condition':2400},
          'adaptive_audit':{'cohort':'smoke3 metadata strata; wider development only if justified',
                            'steps':[50,200],'restarts':3,'waveform_epsilon':.002,'rho':.1,
                            'losses':['main lexical NLL','main plus phone CTC','full plus local-window lexical NLL'],
                            'limits':'bounded development check, not exhaustive worst case or human-content validation'},
          'protected_manifests':{str(p):digest(p) for p in protected},
          'inputs':{str(p):digest(p) for p in files},
          'runtime_packages':{p:importlib.metadata.version(p) for p in ['torch','torchaudio','speechbrain','hyperpyyaml','sentencepiece']}}
    atomic_json(PLAN,plan)
    PLAN.chmod(0o444)
    return plan
