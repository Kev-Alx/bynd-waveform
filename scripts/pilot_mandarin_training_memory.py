"""Measure native fp32 training memory on 48 long-token development records.

The temporary model and optimizer are discarded; no weights are saved or used by
the actual trial. Training configuration and checkpoint files remain unchanged.
"""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from beyondwaveform.io import atomic_json,digest,load_audio,read_jsonl,seed_everything
from beyondwaveform.models.whisper import load_whisper

OUT=ROOT/'reproduction-runs/mandarin-cer-full-dev/memory-pilot'


def free_mib():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return int(r.stdout.splitlines()[0].strip())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Inspect the existing pilot receipt before rerunning')
    config=json.loads((OUT.parent/'training-config.json').read_text());plan=json.loads((OUT.parent/'data-plan.json').read_text())
    assert digest(OUT.parent/'training-config.json')==plan['training_config_sha256']
    rows=read_jsonl(OUT.parent/'manifests/finetune.jsonl')
    assert len(rows)==14326 and all(r['split']=='dev' and r['role']=='finetune' for r in rows)
    start_free=free_mib()
    assert start_free>=10*1024,f'Only {start_free} MiB free; pilot needs at least 10 GiB to measure capacity'
    seed_everything(config['seed']);torch.set_num_threads(8)
    model,meta=load_whisper(config['model'],'cuda',trainable=True)
    assert meta['sha256']==plan['raw_whisper_small_sha256']
    ordered=sorted(rows,key=lambda r:(-len(model.tokenizer.encode(r['text'].strip())),r['id']))[:48]
    assert len({r['id'] for r in ordered})==48
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    accumulation=config['gradient_accumulation'];assert accumulation==8
    torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    started=time.monotonic();records=[]
    for i,row in enumerate(ordered):
        assert digest(row['audio'])==row['audio_sha256']
        loss=model.supervised_loss(load_audio(row['audio']).to('cuda'),row['text'])
        assert torch.isfinite(loss)
        (loss/accumulation).backward()
        if (i+1)%accumulation==0:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
            assert torch.isfinite(norm)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        records.append({'id':row['id'],'lexical_tokens':len(model.tokenizer.encode(row['text'].strip())),'loss':loss.item()})
        atomic_json(OUT/'progress.json',{'status':'running','completed':i+1,'total':48,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        print(f"Training-memory pilot {i+1}/48; peak reserved {torch.cuda.max_memory_reserved()/2**20:.0f} MiB",flush=True)
    seconds=time.monotonic()-started
    allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
    result={'status':'passed','count':48,'optimizer_updates':6,'temporary_weights_discarded':True,
            'selection':'48 highest-token-length development training recordings; ID tie-break; no CER-based selection',
            'checkpoint_sha256':meta['sha256'],'training_config_sha256':digest(OUT.parent/'training-config.json'),
            'dtype':'fp32','batch_size':1,'gradient_accumulation':8,'start_free_mib':start_free,
            'peak_allocated_mib':allocated,'peak_reserved_mib':reserved,'elapsed_seconds':seconds,
            'seconds_per_recording':seconds/48,'recommended_minimum_free_mib':math.ceil(reserved+2048),
            'records':records,'finished_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(OUT/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
