"""Prepare a fixed full-development fine-tuning trial; does not train a model."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import soundfile as sf
from beyondwaveform.config import load_config
from beyondwaveform.data import audit_disjoint
from beyondwaveform.io import atomic_json,digest,read_jsonl,write_jsonl

OUT=ROOT/'reproduction-runs/mandarin-cer-full-dev'


def freeze(path,value,manifest=False):
    if path.exists():
        old=read_jsonl(path) if manifest else json.loads(path.read_text())
        assert old==value,f'Prepared protocol differs: {path}'
    else:
        (write_jsonl if manifest else atomic_json)(path,value)


def prepare():
    folder=OUT/'manifests';folder.mkdir(parents=True,exist_ok=True)
    original=ROOT/'reproduction-runs/mandarin-independent'
    source=original/'manifests/dev-all.jsonl'
    receipt=json.loads((original/'data-freeze.json').read_text())
    assert digest(source)==receipt['proofs'][str(source)]
    dev=read_jsonl(source)
    assert len(dev)==14326 and len({r['speaker'] for r in dev})==40
    assert all(r['split']=='dev' and r['role']=='finetune' for r in dev)
    freeze(folder/'finetune.jsonl',dev,True)
    texts=dict(line.split(maxsplit=1) for line in (ROOT/'data/data_aishell/transcript/aishell_transcript_v0.8.txt').read_text().splitlines() if line.strip())
    train_root=ROOT/'data/data_aishell/wav/train'
    speakers=sorted(p.name for p in train_root.iterdir() if p.is_dir())
    assert len(speakers)==340
    rng=random.Random(2026);chosen=rng.sample(speakers,16)
    cv_speakers=sorted(chosen[:8]);attack_speakers=sorted(chosen[8:])

    def eligible(speaker):
        rows=[]
        for audio in sorted((train_root/speaker).glob('*.wav')):
            if audio.stem not in texts:continue
            info=sf.info(audio);duration=info.frames/info.samplerate
            if .045<=duration<=30:
                rows.append({'id':audio.stem,'text':texts[audio.stem].strip(),'audio':str(audio.resolve()),
                             'corpus':'aishell','split':'train','role':'validation','language':'zh','speaker':speaker,
                             'sample_rate':info.samplerate,'samples':info.frames,'duration':duration})
        return rows

    validation=[]
    for i,speaker in enumerate(cv_speakers):
        pool=eligible(speaker);count=180 if i<3 else 179
        assert len(pool)>=count
        validation+=rng.sample(pool,count)
    assert len(validation)==1435
    checkpoint_pilot=[]
    for speaker in cv_speakers:
        ordered=sorted((r for r in validation if r['speaker']==speaker),key=lambda r:(r['duration'],r['id']))
        for i in range(6):
            checkpoint_pilot.append(rng.choice(ordered[i*len(ordered)//6:(i+1)*len(ordered)//6]))
    attack_pilot=[]
    for speaker in attack_speakers:
        ordered=sorted(eligible(speaker),key=lambda r:(r['duration'],r['id']))
        for i in range(6):
            attack_pilot.append(rng.choice(ordered[i*len(ordered)//6:(i+1)*len(ordered)//6]))
    assert len(checkpoint_pilot)==len(attack_pilot)==48
    for rows in [validation,checkpoint_pilot,attack_pilot]:
        for row in rows:row['audio_sha256']=digest(row['audio'])
    freeze(folder/'checkpoint-validation.jsonl',sorted(validation,key=lambda r:r['id']),True)
    freeze(folder/'checkpoint-pilot-48.jsonl',sorted(checkpoint_pilot,key=lambda r:r['id']),True)
    freeze(folder/'attack-pilot-48.jsonl',sorted(attack_pilot,key=lambda r:r['id']),True)
    test=read_jsonl(ROOT/'reproduction-runs/mandarin-cer-corrected/test-5000.jsonl')
    freeze(folder/'test-5000.jsonl',test,True)
    paths=[folder/name for name in ['finetune.jsonl','checkpoint-validation.jsonl','attack-pilot-48.jsonl','test-5000.jsonl']]
    audit=audit_disjoint(paths)
    speaker_sets=[{r['speaker'] for r in read_jsonl(p)} for p in paths]
    assert not any(a&b for i,a in enumerate(speaker_sets) for b in speaker_sets[i+1:])
    config=load_config(ROOT/'configs/mandarin_finetune.yaml')
    config['model']['checkpoint']=str(ROOT/'artifacts/whisper/small.pt')
    config['save_epoch_checkpoints']=True
    freeze(OUT/'training-config.json',config)
    plan={'status':'frozen-data-plan; training not started','scope':'One prespecified full-development-data coverage trial if the existing checkpoint comparison does not resolve the discrepancy',
          'fine_tune_count':14326,'fine_tune_speakers':40,'checkpoint_validation_count':1435,'checkpoint_validation_speakers':cv_speakers,
          'checkpoint_pilot_count':48,'attack_pilot_count':48,'attack_pilot_speakers':attack_speakers,'test_count':5000,
          'manifest_sha256':{str(p):digest(p) for p in paths+[folder/'checkpoint-pilot-48.jsonl']},
          'training_config_sha256':digest(OUT/'training-config.json'),
          'raw_whisper_small_sha256':digest(ROOT/'artifacts/whisper/small.pt'),
          'selection_rule':'Choose the epoch with lowest corpus CER on the fixed 1,435 validation recordings, after a 48-record execution pilot for each checkpoint. Attack settings use only the separate 48-record pilot. Test CER never enters checkpoint or attack-setting ranking.',
          'data_policy':'All 40 official development speakers are used for surrogate fine-tuning. Separate speaker-disjoint official training-split recordings serve exclusively as validation; no official test data enters fine-tuning or ranking. Original transcript strings, CER normalization and the existing 5,000 test IDs remain unchanged.',
          'paper_interpretation':'The paper states AISHELL development-set fine-tuning but does not publish the exact fine-tuning partition or recipe. Using the full development split is an explicit replication interpretation, not an author-recovered recipe.',
          'paper_url':'https://arxiv.org/html/2606.05678v1#S5','split_audit':audit}
    freeze(OUT/'data-plan.json',plan)
    print(json.dumps({k:plan[k] for k in ['status','fine_tune_count','fine_tune_speakers','checkpoint_validation_count','checkpoint_pilot_count','attack_pilot_count','test_count']},indent=2))


if __name__=='__main__':prepare()
