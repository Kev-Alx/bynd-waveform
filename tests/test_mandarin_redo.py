"""Protect the actual Mandarin inference call against timestamp-prefix drift."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_mandarin_redo import corrected_target, reusable_clean_score
from beyondwaveform.models.whisper import load_whisper
from beyondwaveform.config import load_config


def test_corrected_mandarin_config_reaches_native_transcribe(tmp_path, monkeypatch):
    import beyondwaveform.models.whisper as module

    class Native(torch.nn.Module):
        is_multilingual = True
        num_languages = 99
        dims = SimpleNamespace(n_text_ctx=64)

    checkpoint = tmp_path / 'whisper.pt'
    checkpoint.write_bytes(b'fixture')
    config = {'surrogate': {'checkpoint': str(checkpoint), 'language': 'zh',
                           'decoding': {'beam_size': 5, 'temperature': 0.0}}}
    original = copy.deepcopy(config)
    monkeypatch.setattr(module.whisper, 'load_model', lambda *args, **kwargs: Native())
    calls = []
    monkeypatch.setattr(module.whisper, 'transcribe', lambda model, audio, **kwargs:
                        (calls.append(kwargs) or {'text': '完成'}))
    target, metadata = load_whisper(corrected_target(config), 'cpu')
    assert target.transcribe(torch.zeros(100)) == '完成'
    assert calls[0]['without_timestamps'] is True
    assert calls[0]['beam_size'] == 5 and calls[0]['temperature'] == 0.0
    assert calls[0]['language'] == 'zh' and calls[0]['task'] == 'transcribe'
    assert calls[0]['condition_on_previous_text'] is False
    assert metadata['decoding']['without_timestamps'] is True
    assert config == original


def test_future_mandarin_runs_use_the_supervised_timestamp_prefix():
    model = load_config(Path(__file__).resolve().parents[1]/'configs/mandarin_finetune.yaml')['model']
    assert model['language'] == 'zh'
    assert model['decoding'].get('without_timestamps') is True


def test_clean_reuse_requires_same_reference_audio_and_decoder(tmp_path):
    import json
    cache = tmp_path/'records'
    cache.mkdir()
    metadata = {'sha256':'checkpoint','decoding':{'without_timestamps':True}}
    row = {'id':'one','variant':'clean','reference':'你好','audio_sha256':'audio','hypothesis':'你好'}
    (cache/'one.json').write_text(json.dumps({'target':metadata,'rows':[row]}))
    record = {'id':'one','text':'你好','audio':{'clean':{'sha256':'audio'}}}
    assert reusable_clean_score(tmp_path,record,metadata) == row
    changed = copy.deepcopy(record)
    changed['audio']['clean']['sha256'] = 'other-audio'
    assert reusable_clean_score(tmp_path,changed,metadata) is None
    changed = copy.deepcopy(record)
    changed['text'] = '再见'
    assert reusable_clean_score(tmp_path,changed,metadata) is None
    assert reusable_clean_score(tmp_path,record,{'sha256':'checkpoint','decoding':{'without_timestamps':False}}) is None
