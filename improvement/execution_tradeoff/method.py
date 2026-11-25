"""Exact decoder-graph reuse and a fixed one-pass magnitude repair, version 1."""
import time
import numpy as np
import torch
from scipy.signal import stft,istft
import execution_runtime as rt
from scaled_components import ensemble_gradient

def anchored_magnitude(audio,anchor,fraction=.25):
    """Same-vocoder anchor, 512 sample Hann /160 hop, unchanged attacked phase."""
    y=np.asarray(audio,dtype=np.float32);a=np.asarray(anchor,dtype=np.float32);assert y.shape==a.shape
    _,_,z=stft(y,16000,nperseg=512,noverlap=352,boundary='zeros',padded=True)
    _,_,za=stft(a,16000,nperseg=512,noverlap=352,boundary='zeros',padded=True)
    mag=np.exp((1-fraction)*np.log(np.maximum(np.abs(z),1e-7))+fraction*np.log(np.maximum(np.abs(za),1e-7)))
    target=mag*np.exp(1j*np.angle(z));_,restored=istft(target,16000,nperseg=512,noverlap=352,boundary=True)
    out=restored[:len(y)].astype(np.float32);assert len(out)==len(y) and np.isfinite(out).all() and np.max(np.abs(out))<=1
    return out

def optimize(q,x,text,renderer,source,*,once,steps=50,progress=None):
    delta=torch.zeros_like(q,requires_grad=True);opt=torch.optim.Adam([delta],lr=.1)
    ref=rt.ExplicitReflectionReference(q,x,rt.SpectralConfig(),1e-8)
    sums={k:0. for k in ['decoder_no_grad','decoder_graph','source_forward_backward','decoder_vjp_penalties','optimizer_projection','final_render']};gpu=dict(sums)
    def timed(key,fn):
        torch.cuda.synchronize();b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);b.record();t=time.perf_counter();val=fn();e.record();torch.cuda.synchronize();sums[key]+=time.perf_counter()-t;gpu[key]+=b.elapsed_time(e)/1000;return val
    torch.cuda.synchronize();began=time.perf_counter();torch.cuda.reset_peak_memory_stats();trace=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True)
        if once:
            rendered=timed('decoder_graph',lambda:renderer(q+delta));audio=rendered.detach()
        else:
            def render_detached():
                with torch.no_grad():return renderer(q+delta)
            audio=timed('decoder_no_grad',render_detached)
        g,components=timed('source_forward_backward',lambda:ensemble_gradient({'small':source},audio,text,{'small':1.},'identity',single=True))
        if not once:rendered=timed('decoder_graph',lambda:renderer(q+delta))
        def vjp():
            tv,hf=ref.terms(q+delta,rendered);grad,=torch.autograd.grad(-(rendered*g).sum()+tv+hf,delta);return grad,float(tv.detach()),float(hf.detach())
        grad,tv,hf=timed('decoder_vjp_penalties',vjp)
        assert torch.isfinite(grad).all() and grad.double().norm()>0
        def update():
            delta.grad=grad;opt.step()
            with torch.no_grad():rt.project_relative_ball_(delta,q,.1)
        timed('optimizer_projection',update)
        trace.append({'step':step,'nll':components['small']['nll'],'tv':tv,'hf':hf,'gradient_norm':float(grad.double().norm())})
        if progress and (step==1 or step%10==0):progress(step)
        del rendered,audio,g,grad
    def final_render():
        with torch.no_grad():return renderer(q+delta).detach()
    final=timed('final_render',final_render);torch.cuda.synchronize()
    receipt={'once':once,'steps':steps,'wall_seconds':time.perf_counter()-began,'phase_wall_seconds':sums,'phase_gpu_event_seconds':gpu,'gpu_event_seconds':sum(gpu.values()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'source_forward_backward_pairs':steps,'decoder_forwards':steps*(1 if once else 2)+1,'decoder_backwards':steps,'trace':trace}
    return final,delta.detach(),receipt
