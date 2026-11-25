"""Explicit no-timestamp beam candidates and differentiable candidate scoring."""
from __future__ import annotations

import torch
from whisper.decoding import DecodingOptions,DecodingTask
from beyondwaveform.models.whisper import load_whisper,teacher_forcing_tokens
from beyondwaveform.losses import lexical_nll
from .common import WORKSPACE
from .temporal import windows,stitch_words


def load_base(device='cuda'):
    adapter,provenance=load_whisper({'checkpoint':str(WORKSPACE/'artifacts/whisper/small.pt'),
        'language':'en','nll_reduction':'mean','nll_timestamp_policy':'no_timestamps',
        'decoding':{'beam_size':5,'temperature':0.,'without_timestamps':True}},device)
    adapter.eval()
    return adapter,provenance


class CaptureRanker:
    def __init__(self,original,tokenizer):self.original,self.tokenizer=original,tokenizer;self.candidates=[]
    def rank(self,tokens,logprobs):
        self.candidates=[{'text':self.tokenizer.decode(t.tolist()).strip(),
                          'tokens':t.tolist(),'beam_sum_logprob':float(score)}
                         for t,score in zip(tokens[0],logprobs[0])]
        return self.original.rank(tokens,logprobs)


@torch.no_grad()
def beam_candidates(adapter,audio,beam=8):
    options=DecodingOptions(language='en',task='transcribe',temperature=0.,beam_size=beam,
                            without_timestamps=True,fp16=False)
    task=DecodingTask(adapter.model,options)
    capture=CaptureRanker(task.sequence_ranker,task.tokenizer);task.sequence_ranker=capture
    result=task.run(adapter.mel(audio))[0]
    unique={}
    for item in capture.candidates:
        if item['text']:unique.setdefault(item['text'],item)
    if result.text.strip() and result.text.strip() not in unique:
        unique[result.text.strip()]={'text':result.text.strip(),'tokens':result.tokens,
                                     'beam_sum_logprob':result.avg_logprob*(len(result.tokens)+1)}
    return result.text.strip(),list(unique.values()),result.audio_features[None]


def candidate_nll(adapter,features,text):
    inputs,labels,mask=teacher_forcing_tokens(adapter.tokenizer,text,features.device)
    logits=adapter.model.decoder(inputs,features)
    mask=mask | (labels==adapter.tokenizer.eot)
    return lexical_nll(logits,labels,mask,'mean')


@torch.no_grad()
def local_decode(adapter,audio,width=8.,stride=4.):
    spans=windows(audio.numel()/16000,width,stride)
    parts=[];records=[]
    for span in spans:
        chunk=audio[round(span.start*16000):round(span.end*16000)]
        result=__import__('whisper').transcribe(adapter.model,chunk,language='en',task='transcribe',
            fp16=False,condition_on_previous_text=False,temperature=0.,beam_size=5,
            without_timestamps=True,word_timestamps=True,verbose=None)
        words=[w for segment in result['segments'] for w in segment.get('words',[])]
        parts.append(words)
        records.append({'window':vars(span),'text':result['text'].strip(),'words':words})
    text,selected=stitch_words(parts,spans)
    return text,{'windows':records,'selected_words':selected}
