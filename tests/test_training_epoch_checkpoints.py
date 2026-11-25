import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from beyondwaveform.io import digest,save_audio,write_jsonl
from beyondwaveform.training import finetune


@pytest.mark.parametrize('save_epochs',[False,True])
def test_optional_epoch_snapshots_preserve_trained_states_and_hashes(tmp_path,monkeypatch,save_epochs):
    import beyondwaveform.training as module

    class TinySurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.Linear(1,1,bias=False)
            self.model.dims=SimpleNamespace(fixture=1)
            nn.init.constant_(self.model.weight,.8)

        def supervised_loss(self,audio,text):
            return (self.model.weight[0,0]-audio.mean()).square()

    monkeypatch.setattr(module,'load_whisper',lambda *args,**kwargs:(TinySurrogate(),{'sha256':'fixture'}))
    manifests=[]
    for group,values in [('finetune',[.1,.2]),('validation',[.3])]:
        rows=[]
        for i,value in enumerate(values):
            audio=tmp_path/f'{group}-{i}.wav';asset=save_audio(audio,torch.full((1024,),value))
            rows.append({'id':f'{group}-{i}','audio':str(audio),'audio_sha256':asset['sha256'],
                         'text':'你好','corpus':'aishell','split':'dev','role':group,'language':'zh'})
        path=tmp_path/f'{group}.jsonl';write_jsonl(path,rows);manifests.append(path)
    cfg={'model':{'language':'zh'},'epochs':2,'learning_rate':.05,'gradient_accumulation':1,
         'save_epoch_checkpoints':save_epochs}
    output=tmp_path/'training'
    finetune(cfg,*manifests,output,device='cpu')
    metadata=json.loads((output/'training.json').read_text())
    assert metadata['status']=='complete'
    assert (output/'best.pt').exists() and (output/'last.pt').exists()
    if save_epochs:
        snapshots=metadata['epoch_checkpoints']
        assert [r['epoch'] for r in snapshots]==[1,2]
        for item in snapshots:
            assert digest(item['path'])==item['sha256']
        first=torch.load(snapshots[0]['path'],weights_only=True)
        last=torch.load(snapshots[1]['path'],weights_only=True)
        final=torch.load(output/'last.pt',weights_only=True)
        assert not torch.equal(first['model_state_dict']['weight'],last['model_state_dict']['weight'])
        assert torch.equal(last['model_state_dict']['weight'],final['model_state_dict']['weight'])
    else:
        assert not list(output.glob('epoch-*.pt'))
        assert 'epoch_checkpoints' not in metadata
