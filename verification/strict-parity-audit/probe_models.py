import json, gc
from pathlib import Path
import torch
from beyondwaveform.config import load_config, SpectralConfig
from beyondwaveform.io import load_audio, read_jsonl, digest
from beyondwaveform.models.feature import load_feature_models
from beyondwaveform.models.whisper import load_whisper, teacher_forcing_tokens
from beyondwaveform.losses import CleanReference, lexical_nll, high_frequency_fraction

torch.set_num_threads(4)
torch.manual_seed(2026)
config=load_config('configs/english.yaml')
row=read_jsonl('data/manifests/pilot.jsonl')[0]
x=load_audio(row['audio'])
print('Loading actual pretrained feature models on CPU',flush=True)
encoder,vocoder,meta=load_feature_models(config['features'],'cpu')
with torch.no_grad():
 q=encoder(x)
 encoder.normalize=True
 qn=encoder(x)
 encoder.normalize=False
 reconstruction=vocoder(q)
 normreconstruction=vocoder(qn)
result={'pilot_id':row['id'],'pilot_audio_sha256':digest(row['audio']),'input_samples':x.numel(),'features_shape':list(q.shape),'output_samples':reconstruction.numel(),'encoder_checkpoint_normalize':meta['encoder_config']['normalize'],'effective_normalize':encoder.normalize,'normalized_vs_raw_features_relative_l2':((qn-q).norm()/q.norm()).item(),'normalized_vs_raw_reconstruction_relative_l2':((normreconstruction-reconstruction).norm()/reconstruction.norm()).item(),'native_output_rate':meta['vocoder_config']['sampling_rate'],'encoder_sha256':meta['encoder_sha256'],'vocoder_sha256':meta['vocoder_sha256']}
print('Loading actual Whisper-small and probing feature-to-NLL gradient',flush=True)
asr,asrmeta=load_whisper(config['surrogate'],'cpu')
z=q.detach().requires_grad_()
audio=vocoder(z)
nll=asr.nll(audio,row['text'])
ref=CleanReference(q,x,SpectralConfig(),1e-8)
tv,hf=ref.terms(z,audio)
loss=-nll+tv+hf
grad,=torch.autograd.grad(loss,z)
result.update(nll=nll.item(),tv_ratio=tv.item(),hf_ratio=hf.item(),loss=loss.item(),gradient_norm=grad.norm().item(),gradient_all_finite=bool(torch.isfinite(grad).all()),frozen_weights=all(not p.requires_grad and p.grad is None for m in [encoder,vocoder,asr] for p in m.parameters()),surrogate_sha256=asrmeta['sha256'])
with torch.no_grad():
 feats=asr.model.encoder(asr.mel(reconstruction))
 losses={}
 for label,text in [('raw',row['text']),('lowercase',row['text'].lower()),('sentence_case',row['text'].capitalize())]:
  inputs,labels,mask=teacher_forcing_tokens(asr.tokenizer,text,'cpu')
  logits=asr.model.decoder(inputs,feats)
  losses[label]={'text':text,'lexical_tokens':mask.sum().item(),'mean_nll':lexical_nll(logits,labels,mask,'mean').item(),'sum_nll':lexical_nll(logits,labels,mask,'sum').item()}
 result['transcript_case_sensitivity']=losses
# Demonstrate that spectral.sample_rate can silently reinterpret a fixed-16k waveform.
t=torch.arange(16000,dtype=torch.float64)/16000
signal=torch.sin(2*torch.pi*5000*t)
result['sample_rate_guard_probe']={'tone_hz_in_actual_16k_audio':5000,'fraction_with_16k_config':high_frequency_fraction(signal,SpectralConfig()).item(),'fraction_with_22050_config':high_frequency_fraction(signal,SpectralConfig(sample_rate=22050)).item()}
p=Path('verification/strict-parity-audit/model-probes.json');p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
