"""Input-conditioned, single-pass speech-feature policy; no ASR at inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F

RADII=(.025,.05,.1)
class TemporalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.norm=nn.LayerNorm(width);self.expand=nn.Conv1d(width,width*2,1);self.depth=nn.Conv1d(width*2,width*2,5,padding=2*dilation,dilation=dilation,groups=width*2);self.reduce=nn.Conv1d(width,width,1)
    def forward(self,x):
        y=self.norm(x.transpose(1,2)).transpose(1,2);a,b=self.depth(self.expand(y)).chunk(2,dim=1);return x+.1*self.reduce(F.silu(a)*torch.sigmoid(b))
class FeaturePolicy(nn.Module):
    def __init__(self,features=1024,width=384):
        super().__init__();self.features=features;self.width=width;self.norm=nn.LayerNorm(features);self.stem=nn.Conv1d(features,width,1);self.context=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width));self.radius=nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width));self.blocks=nn.ModuleList([TemporalBlock(width,d) for d in (1,2,4,8,16,32,64,1)]);self.head=nn.Conv1d(width,features,1,bias=False);nn.init.normal_(self.head.weight,std=.01)
    def forward(self,q,rho,conditioning=None):
        if q.ndim!=2 or q.shape[1]!=self.features or not 0<float(rho)<=.1:raise ValueError('Invalid feature/radius contract')
        c=q if conditioning is None else conditioning
        if c.shape!=q.shape:raise ValueError('Conditioning must align to original feature shape')
        normalized=self.norm(c);x=self.stem(normalized.T[None]);emb=q.new_tensor([float(rho)/.1,math.log(float(rho)/.1)]);x=x+(self.context(normalized.mean(0))+self.radius(emb))[None,:,None]
        for block in self.blocks:x=block(x)
        raw=self.head(x)[0].T;denom=raw.float().norm().clamp_min(1e-6);delta=raw*(float(rho)*q.detach().float().norm()/denom)
        return delta
    def registration(self):return {'type':'amortized input-conditioned temporal feature policy','features':self.features,'width':self.width,'blocks':8,'dilations':[1,2,4,8,16,32,64,1],'kernel':5,'global_context':True,'bias_free_output':True,'fixed_nonzero_relative_radii':list(RADII),'parameters':sum(p.numel() for p in self.parameters()),'inference':'one frozen WavLM6 encoding, one policy forward, one HiFi-GAN decode; no ASR or per-input optimization'}

def spectral_proxy(audio,reference):
    values=[]
    for size in (256,512,1024):
        def spec(x):
            pad=size//2;padded=torch.cat((x[1:pad+1].flip(0),x,x[-pad-1:-1].flip(0)));return torch.stft(padded,size,hop_length=size//4,window=torch.hann_window(size,device=x.device),center=False,return_complex=True).abs()
        a=spec(audio)
        with torch.no_grad():b=spec(reference)
        values.append(((a+1e-5).log()-(b+1e-5).log()).abs().mean())
    return torch.stack(values).mean()
def envelope_proxy(audio,reference):
    def rms(x):return F.avg_pool1d(x.square()[None,None],400,stride=160).flatten().clamp_min(1e-12).sqrt()
    a=rms(audio)
    with torch.no_grad():b=rms(reference)
    return (a-b).square().mean()/b.square().mean().clamp_min(1e-12)
def quality_gradient_mix(source_gradient,quality_gradient,active):
    """Keep attack descent tangent to an active quality constraint, then restore."""
    if not active:return source_gradient,{'projection_fraction':0.,'quality_correction_relative_norm':0.,'cosine':None}
    sn=source_gradient.double().norm();qn=quality_gradient.double().norm();dot=(source_gradient.double()*quality_gradient.double()).sum();cosine=float(dot/(sn*qn).clamp_min(1e-30))
    if qn<1e-12:return source_gradient,{'projection_fraction':0.,'quality_correction_relative_norm':0.,'cosine':cosine}
    projection=(dot.clamp_max(0)/qn.square()).to(source_gradient.dtype)*quality_gradient
    correction=.25*(sn/qn).to(source_gradient.dtype)*quality_gradient
    combined=source_gradient-projection+correction
    return combined,{'projection_fraction':float(projection.double().norm()/sn.clamp_min(1e-30)),'quality_correction_relative_norm':float(correction.double().norm()/sn.clamp_min(1e-30)),'cosine':cosine}
