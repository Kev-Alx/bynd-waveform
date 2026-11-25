"""Released TIMIT filterbank-CRDNN phone recognizer and pinned CMU lexicon."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import torch
from torch import nn

from .common import ROOT,digest,read_json
from .ctc import ctc_nll,greedy_ctc

TIMIT_ROOT=ROOT/'weights/timit/extracted/train/results/augment_noise_CRDNN/2888'
TIMIT_CKPT=TIMIT_ROOT/'save/CKPT+2021-02-06+17-01-30+00'


def read_labels(path):
    labels={}
    for line in Path(path).read_text().splitlines():
        if line.startswith('===='):break
        key,index=line.split(' => ')
        labels[str(ast.literal_eval(key))]=int(index)
    if set(labels.values()) != set(range(40)) or labels.get('sil') != 39 or labels.get('0') != 0:
        raise ValueError('Unexpected checkpoint phone mapping')
    return labels


class PhoneRecognizer(nn.Module):
    """No WavLM/shared SSL extractor; original fixed global statistics retained."""
    def __init__(self,device='cuda'):
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.CRDNN import CRDNN
        from speechbrain.processing.features import InputNormalization
        from speechbrain.nnet.linear import Linear
        receipt=read_json(ROOT/'weights/timit/download_receipt.json')
        used=[TIMIT_CKPT/n for n in ['model.ckpt','output.ckpt','normalizer.ckpt']]
        used += [TIMIT_ROOT/'save/label_encoder.txt',TIMIT_ROOT/'hyperparams.yaml']
        known={r['path']:r['sha256'] for r in receipt['selected_files']}
        for path in used:
            if digest(path) != known[str(path)]:raise ValueError('Changed released phone artifact')
        self.labels=read_labels(TIMIT_ROOT/'save/label_encoder.txt')
        self.id_to_phone={v:k for k,v in self.labels.items()}
        # Exact saved 2021 recipe values, not current mutable recipe defaults.
        self.frontend=Fbank(sample_rate=16000,n_fft=400,n_mels=40)
        self.normalizer=InputNormalization(norm_type='global')
        self.normalizer._load_statistics_dict(torch.load(TIMIT_CKPT/'normalizer.ckpt',map_location='cpu',weights_only=True))
        self.encoder=CRDNN(input_shape=[None,None,40],activation=nn.LeakyReLU,dropout=.15,
                           cnn_blocks=2,cnn_channels=[128,256],cnn_kernelsize=(3,3),
                           time_pooling=True,time_pooling_size=2,rnn_layers=4,
                           rnn_neurons=512,rnn_bidirectional=True,dnn_blocks=2,dnn_neurons=512)
        self.output=Linear(input_size=512,n_neurons=40,bias=True)
        self.encoder.load_state_dict(torch.load(TIMIT_CKPT/'model.ckpt',map_location='cpu',weights_only=True),strict=True)
        self.output.load_state_dict(torch.load(TIMIT_CKPT/'output.ckpt',map_location='cpu',weights_only=True),strict=True)
        self.to(device).eval().requires_grad_(False)
        self.provenance={'architecture':'40-bin Fbank, CRDNN LiGRU, 39 phones + blank',
                         'pretraining':'TIMIT supervised, released 2021 checkpoint',
                         'checkpoint_selection':'lower released validation PER (12.338354886); no local-defense outcomes used',
                         'weight_license':'No separate weight license supplied in archive; source/code Apache-2.0; no redistribution by this task.',
                         'archive_sha256':receipt['sha256'],
                         'files':{str(p):known[str(p)] for p in used},
                         'speechbrain_runtime':'1.0.2; strict state loading required'}

    def forward(self,audio):
        if audio.ndim != 1:raise ValueError('One mono utterance required')
        features=self.frontend(audio[None])
        normalized=self.normalizer(features,torch.ones(1,device=audio.device))
        return self.output(self.encoder(normalized))[0].log_softmax(-1)

    def lexical_log_probs(self,audio):
        lp=self(audio)
        # Marginalize silence as a second blank, preserving all other phone mass.
        silence=self.labels['sil']
        flap=self.labels['dx']
        columns=[torch.logaddexp(lp[:,0],lp[:,silence]) if i==0 else
                 lp[:,i]*0-float('inf') if i in {silence,flap} else
                 torch.logaddexp(lp[:,i],lp[:,flap]-lp.new_tensor(2.).log())
                 if i in {self.labels['t'],self.labels['d']} else lp[:,i] for i in range(lp.shape[1])]
        return torch.stack(columns,-1)

    def phone_sequence(self,audio):
        return [self.id_to_phone[i] for i in greedy_ctc(self(audio)) if i != self.labels['sil']]


class PronunciationLexicon:
    def __init__(self,labels,path=None):
        self.labels=labels
        self.path=Path(path or ROOT/'weights/lexicon/cmudict.dict')
        if path is None:
            receipt=read_json(ROOT/'weights/lexicon/receipt.json')
            expected=next(r['sha256'] for r in receipt['files'] if r['path']==str(self.path))
            if digest(self.path)!=expected:raise ValueError('Changed pinned pronunciation dictionary')
        self.entries={}
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line or line.startswith(';;;'):continue
            fields=line.split('#',1)[0].split()
            word=re.sub(r'\(\d+\)$','',fields[0]).lower()
            phones=[re.sub(r'\d','',p.lower()) for p in fields[1:]]
            phones=[{'ao':'aa'}.get(p,p) for p in phones]
            if phones and all(p in labels and p not in {'sil','0'} for p in phones):
                pronunciation=tuple(labels[p] for p in phones)
                if pronunciation not in self.entries.setdefault(word,[]):self.entries[word].append(pronunciation)

    @staticmethod
    def words(text):
        # Preserve lexical apostrophes; normalize only spelling for dictionary lookup.
        return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())

    def encode(self,text):
        words=self.words(text)
        unknown=[w for w in words if w not in self.entries]
        if unknown or not words:return None,unknown
        return [p for w in words for p in self.entries[w][0]],[]

    def variants(self,text,limit=8):
        if re.search(r'\d',text):return [],['<numeric_spelling>']
        words=self.words(text)
        unknown=[w for w in words if w not in self.entries]
        if unknown or not words:return [],unknown
        paths=[()]
        for word in words:
            paths=[a+b for a in paths for b in self.entries[word]][:limit]
        return [list(p) for p in paths],[]

    def score(self,log_probs,text):
        pronunciations,unknown=self.variants(text)
        if not pronunciations:return None,unknown
        # Equal pronunciation prior; sum alignment probabilities for each alternative.
        values=torch.stack([-ctc_nll(log_probs,p,normalize=False) for p in pronunciations])
        score=torch.logsumexp(values,0)-torch.log(values.new_tensor(len(values)))
        return score/log_probs.shape[0],[]


class PhoneWordDecoder:
    """Approximate lexicon search on a decoded phone path, then exact CTC rescoring.

    Training-text unigram vocabulary and optional inferred ASR candidate words only.
    This is not an exact full posterior lattice decoder; that approximation is logged.
    """
    def __init__(self,lexicon,word_counts,max_edits=1,beam=4):
        self.lexicon=lexicon;self.counts=dict(word_counts);self.max_edits=max_edits;self.beam=beam
        self.trie={};self.total=max(1,sum(self.counts.values()))
        for word in self.counts:
            for pronunciation in lexicon.entries.get(word,[]):
                node=self.trie
                for phone in pronunciation:node=node.setdefault(phone,{})
                node.setdefault('_words',[]).append(word)

    def matches(self,phones,position):
        observed=phones[position:position+20]
        first=list(range(len(observed)+1));found=[]
        def visit(node,previous,depth):
            for phone,child in node.items():
                if phone=='_words':continue
                current=[previous[0]+1]
                for index,value in enumerate(observed,1):
                    current.append(min(current[-1]+1,previous[index]+1,
                                       previous[index-1]+int(phone!=value)))
                if '_words' in child:
                    for consumed,cost in enumerate(current[1:],1):
                        if cost<=self.max_edits:
                            for word in child['_words']:found.append((consumed,word,cost))
                if min(current)<=self.max_edits and depth<19:visit(child,current,depth+1)
        visit(self.trie,first,0)
        return found

    def decode(self,phones):
        paths={0:[(0.,())]}
        for pos in range(len(phones)):
            if pos not in paths:continue
            for consumed,word,edits in self.matches(phones,pos):
                nxt=pos+consumed
                prior=-__import__('math').log((self.counts[word]+1)/(self.total+len(self.counts)))
                for cost,prefix in paths[pos]:
                    candidate=(cost+edits+.05*prior+.05,prefix+(word,))
                    values=paths.setdefault(nxt,[]);values.append(candidate)
                    values.sort(key=lambda item:item[0]);del values[self.beam:]
        return [{'text':' '.join(words),'phone_path_edit_cost':score}
                for score,words in paths.get(len(phones),[]) if words]
