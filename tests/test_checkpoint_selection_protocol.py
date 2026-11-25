import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import select_mandarin_checkpoint_cer as selector
from beyondwaveform.io import write_jsonl


def configure(tmp_path,monkeypatch,split='train'):
    validation=[{'id':f'val-{i}','text':'你好','speaker':f'validation-{i%8}',
                 'role':'validation','split':split,'duration':1+i/10000,
                 'audio':'fixture.wav','audio_sha256':f'hash-{i}'} for i in range(1435)]
    pilot=[r for speaker in sorted({r['speaker'] for r in validation})
           for r in [r for r in validation if r['speaker']==speaker][:6]]
    paths={}
    for key,rows in [('validation',validation),('pilot',pilot),
                     ('training',[{'id':'train','speaker':'training-speaker'}]),
                     ('attack',[{'id':'attack','speaker':'attack-speaker'}])]:
        paths[key]=tmp_path/f'{key}.jsonl';write_jsonl(paths[key],rows)
    checkpoints={}
    for i in range(1,4):
        path=tmp_path/f'epoch-{i:03d}.pt';path.write_bytes(f'checkpoint-{i}'.encode())
        checkpoints[f'epoch-{i:03d}']=str(path)
    output=tmp_path/'selection'
    settings={'validation_manifest':str(paths['validation']),'validation_split':split,
              'pilot_manifest':str(paths['pilot']),'training_manifest':str(paths['training']),
              'attack_manifest':str(paths['attack']),'checkpoints':checkpoints,
              'scope':'Prepared full-development epoch selection','selection_rule':'Lowest validation CER; earliest epoch on a tie'}
    monkeypatch.setattr(selector,'OUT',output);monkeypatch.setattr(selector,'SETTINGS',settings)
    return output,checkpoints


def test_prepared_epoch_protocol_uses_exact_48_pilot_and_all_checkpoints(tmp_path,monkeypatch):
    output,checkpoints=configure(tmp_path,monkeypatch)
    plan=selector.prepare()
    assert list(plan['targets'])==list(checkpoints)
    assert len((output/'pilot-48.jsonl').read_text().splitlines())==48
    assert len((output/'validation-1435.jsonl').read_text().splitlines())==1435
    for name,path in checkpoints.items():
        target=json.loads((output/f'{name}-target.json').read_text())
        assert target['checkpoint']==path
        assert target['decoding']['without_timestamps'] is True


def test_epoch_selection_rejects_test_split_even_when_marked_validation(tmp_path,monkeypatch):
    configure(tmp_path,monkeypatch,split='test')
    with pytest.raises(AssertionError):selector.prepare()
