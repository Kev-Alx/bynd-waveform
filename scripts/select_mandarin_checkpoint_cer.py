"""Compare existing Mandarin checkpoints using disjoint validation CER only."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gc
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from beyondwaveform.evaluation import aggregate_errors,error_counts
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,seed_everything,write_jsonl
from beyondwaveform.models.whisper import load_whisper

OUT=ROOT/'reproduction-runs/mandarin-cer-checkpoint-selection'
TAIPEI=timezone(timedelta(hours=8))
SETTINGS={}


def freeze(path,value,manifest=False):
    if path.exists():
        old=read_jsonl(path) if manifest else json.loads(path.read_text())
        assert old==value,f'Frozen input changed: {path}'
    else:
        (write_jsonl if manifest else atomic_json)(path,value)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path(SETTINGS.get('validation_manifest',ROOT/'reproduction-runs/mandarin-independent/manifests/checkpoint-validation.jsonl'))
    rows=read_jsonl(source)
    validation_split=SETTINGS.get('validation_split','dev')
    assert validation_split in {'train','dev'},'Official test data cannot enter checkpoint ranking'
    assert len(rows)==1435 and all(r['role']=='validation' and r['split']==validation_split for r in rows)
    rng=random.Random(2026);pilot=[]
    if SETTINGS.get('pilot_manifest'):
        pilot=read_jsonl(SETTINGS['pilot_manifest'])
        assert {r['id'] for r in pilot}<={r['id'] for r in rows}
    else:
        for speaker in sorted({r['speaker'] for r in rows}):
            ordered=sorted((r for r in rows if r['speaker']==speaker),key=lambda r:(r['duration'],r['id']))
            for i in range(12):
                pilot.append(rng.choice(ordered[i*len(ordered)//12:(i+1)*len(ordered)//12]))
    assert len(pilot)==48 and len({r['id'] for r in pilot})==48
    originals={r['id']:r for r in rows}
    assert all(originals.get(r['id'])==r for r in pilot),'Pilot references must be unchanged validation records'
    train=read_jsonl(SETTINGS.get('training_manifest',ROOT/'reproduction-runs/mandarin-independent/manifests/finetune.jsonl'))
    attack=read_jsonl(SETTINGS.get('attack_manifest',ROOT/'reproduction-runs/mandarin-independent/manifests/attack-reserve.jsonl'))
    assert not {r['id'] for r in rows}&{r['id'] for r in train+attack}
    assert not {r['speaker'] for r in rows}&{r['speaker'] for r in train+attack}
    freeze(OUT/'pilot-48.jsonl',pilot,True)
    freeze(OUT/'validation-1435.jsonl',rows,True)
    targets={}
    checkpoints=SETTINGS.get('checkpoints') or {name:str(ROOT/f'reproduction-runs/mandarin-independent/training/{name}.pt') for name in ['best','last']}
    for name,checkpoint_path in checkpoints.items():
        checkpoint=Path(checkpoint_path)
        target={'checkpoint':str(checkpoint),'language':'zh','nll_reduction':'mean',
                'decoding':{'beam_size':5,'temperature':0.0,'without_timestamps':True}}
        freeze(OUT/f'{name}-target.json',target)
        targets[name]={'target_sha256':digest(OUT/f'{name}-target.json'),'checkpoint_sha256':digest(checkpoint)}
    plan={'status':'frozen','scope':SETTINGS.get('scope','Existing epoch-2 validation-loss winner versus epoch-3 last checkpoint; validation CER selection only'),
          'pilot_count':48,'validation_count':1435,'pilot_sha256':digest(OUT/'pilot-48.jsonl'),
          'validation_sha256':digest(OUT/'validation-1435.jsonl'),'targets':targets,
          'pilot_selection':SETTINGS.get('pilot_selection','Seed 2026; twelve duration-stratified random records per each of four checkpoint-validation speakers'),
          'selection_rule':SETTINGS.get('selection_rule','Lowest corpus CER on all 1,435 held-out checkpoint-validation recordings; tie retains existing best checkpoint. Pilot is a 48-record execution check first. No test CER or attack CER enters checkpoint ranking.')}
    freeze(OUT/'plan.json',plan)
    return plan


def run(stage):
    import torch
    torch.set_num_threads(8);seed_everything(2026)
    plan=prepare();rows=read_jsonl(OUT/('pilot-48.jsonl' if stage=='pilot' else 'validation-1435.jsonl'))
    names=list(plan['targets'])
    if stage=='validation':
        for name in names:
            pilot=json.loads((OUT/'pilot'/name/'results.json').read_text())
            assert pilot['status']=='complete' and pilot['count']==48
    times=[];total=len(names)*len(rows);completed=0
    summaries={}
    for name in names:
        target_cfg=json.loads((OUT/f'{name}-target.json').read_text())
        assert digest(OUT/f'{name}-target.json')==plan['targets'][name]['target_sha256']
        folder=OUT/stage/name;folder.mkdir(parents=True,exist_ok=True)
        target,meta=load_whisper(target_cfg,'cuda')
        assert meta['sha256']==plan['targets'][name]['checkpoint_sha256']
        scored=[]
        for row in rows:
            assert digest(row['audio'])==row['audio_sha256']
            cache=folder/'records'/f"{row['id']}.json"
            if cache.exists():
                saved=json.loads(cache.read_text());assert saved['target']==meta
                item=saved['row'];assert item['reference']==row['text'] and item['audio_sha256']==row['audio_sha256']
                times.append(saved['seconds'])
            else:
                started=time.monotonic();hyp=target.transcribe(load_audio(row['audio']).to('cuda'))
                item={'id':row['id'],'variant':'clean','audio_sha256':row['audio_sha256'],'reference':row['text'],'hypothesis':hyp,
                      **error_counts(row['text'],hyp,'zh')}
                seconds=time.monotonic()-started;times.append(seconds)
                atomic_json(cache,{'target':meta,'row':item,'seconds':seconds})
            scored.append(item);completed+=1
            summary={'count':len(scored),**aggregate_errors(scored)}
            result={'status':'complete' if len(scored)==len(rows) else 'partial','stage':stage,'candidate':name,
                    'metric':'CER','count':len(scored),'target':meta,'summary':{'clean':summary},'plan_sha256':digest(OUT/'plan.json')}
            atomic_json(folder/'results.json',result)
            recent=times[-100:]
            remaining=(total-completed)*sum(recent)/len(recent);now=datetime.now(TAIPEI)
            progress={'status':'running','phase':f'{stage}-{name}','updated_at':now.isoformat(timespec='seconds'),
                      'scope':plan['scope'],'clip_progress':{'completed':completed,'total':total,'percent':round(100*completed/total,1)},
                      'estimated_remaining_seconds':round(remaining),'estimated_remaining_range_seconds':[round(.7*remaining),round(1.8*remaining)],
                      'estimated_completion_taipei':(now+timedelta(seconds=remaining)).isoformat(timespec='seconds'),
                      'eta_basis':'Recent up-to-100 validation ASR observations; excludes future attack work'}
            atomic_json(OUT/'progress.json',progress)
            if completed==1 or completed%20==0:
                sys.path.insert(0,str(ROOT/'scripts'))
                from update_job_status import update
                update(ROOT/'reproduction-runs','paper-quality-coordinate-research')
            (OUT/'JOB_STATUS.md').write_text(f"# Mandarin checkpoint validation CER\n\nUpdated: {progress['updated_at']} (Asia/Taipei).\n\nStage: **{stage}**; checkpoint: **{name}**. Progress: **{completed}/{total}**.\n\nEstimated remaining: **{remaining/60:.1f} minutes**; completion **{progress['estimated_completion_taipei']}**.\n\n{plan['selection_rule']}\n")
        for r in scored:
            assert error_counts(r['reference'],r['hypothesis'],'zh')=={k:r[k] for k in ['substitutions','deletions','insertions','reference_units','reference_normalized','hypothesis_normalized']}
        write_jsonl(folder/'utterances.jsonl',scored)
        atomic_json(folder/'verification.json',{'status':'passed','count':len(scored),'results_sha256':digest(folder/'results.json'),
                                             'utterances_sha256':digest(folder/'utterances.jsonl')})
        summaries[name]=summary
        print(json.dumps({'stage':stage,'candidate':name,'summary':summary},ensure_ascii=False),flush=True)
        del target;gc.collect();torch.cuda.empty_cache()
    winner=min(names,key=lambda n:summaries[n]['error_percent'])
    result={'status':'complete','stage':stage,'metric':'CER','summaries':summaries,
            'selection':winner if stage=='validation' else None,'selection_rule':plan['selection_rule'],'plan_sha256':digest(OUT/'plan.json')}
    atomic_json(OUT/f'{stage}-results.json',result)
    atomic_json(OUT/'results.json',{'status':'complete','metric':'CER','scope':plan['scope'],'summary':summaries})
    progress.update(status='complete',phase=f'{stage}-complete',estimated_remaining_seconds=0,
                    estimated_remaining_range_seconds=[0,0],estimated_completion_taipei=datetime.now(TAIPEI).isoformat(timespec='seconds'))
    atomic_json(OUT/'progress.json',progress)
    scores='\n'.join(f"- {name}: **{summaries[name]['error_percent']:.6f}% CER**" for name in names)
    (OUT/'JOB_STATUS.md').write_text(f"# Mandarin checkpoint validation CER\n\nStage **{stage} complete**; {total}/{total} checkpoint/audio observations.\n\n{scores}\n\n{plan['selection_rule']}\n")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilot','validation'],default='pilot')
    parser.add_argument('--protocol',type=Path,help='Prepared alternate non-test manifests and checkpoint paths')
    args=parser.parse_args();stage=args.stage
    if args.protocol:
        SETTINGS=json.loads(args.protocol.read_text())
        OUT=Path(SETTINGS['output']).resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'evaluation.lock').open('a+') as lock,(ROOT/'reproduction-runs/auxiliary-gpu.lock').open('a+') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(stage)
