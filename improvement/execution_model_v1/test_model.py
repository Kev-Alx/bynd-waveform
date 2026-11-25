"""Critical CPU checks: noncollapse, genuine conditioning, proxy and gradient geometry."""
import torch
from feature_policy import FeaturePolicy,spectral_proxy,envelope_proxy,quality_gradient_mix

def test_noncollapse_conditioning_and_backward():
 torch.manual_seed(11);torch.set_num_threads(2);m=FeaturePolicy(width=32);q=torch.randn(17,1024);other=torch.randn_like(q);delta=m(q,.05);assert torch.isfinite(delta).all();assert abs(float(delta.norm()/q.norm())-.05)<1e-6;assert not torch.allclose(delta,m(q,.05,conditioning=other));(delta*torch.randn_like(delta)).sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert m.head.weight.grad.norm()>0

def test_active_quality_gradient_prevents_first_order_violation():
 source=torch.tensor([1.,2.,-1.]);quality=-source.clone();mixed,diag=quality_gradient_mix(source,quality,True);assert mixed.dot(quality)>0 and diag['projection_fraction']>.999;unchanged,_=quality_gradient_mix(source,quality,False);assert torch.equal(unchanged,source)

def test_quality_is_reference_relative_and_differentiable():
 torch.manual_seed(11);x=torch.randn(4000)*.1;assert float(spectral_proxy(x,x))==0 and float(envelope_proxy(x,x))==0;y=(x+torch.randn_like(x)*.001).requires_grad_();loss=spectral_proxy(y,x)+envelope_proxy(y,x);g,=torch.autograd.grad(loss,y);assert torch.isfinite(g).all() and g.norm()>0

def test_joint_gate_and_efficiency_are_both_required():
 from decide import passes,efficiency
 good={'error_percent':75.01,'quality':{'dnsmos':3.,'nisqa':3.75,'utmos':3.5}}
 assert passes(good);assert not passes({**good,'error_percent':75.})
 assert not passes({**good,'quality':{**good['quality'],'utmos':3.499}})
 rows=[{'id':str(i),'steady_generation_seconds':.1,'peak_allocated_bytes':1024**3,'source_queries_at_inference':0,'per_recording_optimization_updates':0} for i in range(48)]
 assert efficiency(rows)['pass'];rows[0]['source_queries_at_inference']=1;assert not efficiency(rows)['pass'];rows[0]['source_queries_at_inference']=0
 for r in rows:r['steady_generation_seconds']=2.
 assert not efficiency(rows)['pass']
 for r in rows:r['steady_generation_seconds']=.1
 rows[0]['peak_allocated_bytes']=20*1024**3;assert not efficiency(rows)['pass']
