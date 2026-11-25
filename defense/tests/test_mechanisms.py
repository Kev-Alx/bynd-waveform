import itertools
import math
import torch
from torch import nn

from defense_lab.ctc import ctc_nll,greedy_ctc,prefix_beam
from defense_lab.temporal import windows,stitch_words
from defense_lab.lora import LoRALinear
from defense_lab.adversary import waveform_pgd,feature_pgd
from defense_lab.phone import PhoneWordDecoder


def collapse(path):
    return [v for i,v in enumerate(path) if v != 0 and (i == 0 or v != path[i-1])]


def test_ctc_matches_enumerated_alignment_probability_and_gradient():
    logits=torch.tensor([[1.,2.,0.],[2.,1.,0.],[0.,2.,1.]],requires_grad=True)
    lp=logits.log_softmax(-1)
    terms=[torch.stack([lp[t,c] for t,c in enumerate(path)]).sum().exp()
           for path in itertools.product(range(3),repeat=3) if collapse(path)==[1]]
    expected=-torch.stack(terms).sum().log()
    actual=ctc_nll(lp,[1],normalize=False)
    assert torch.allclose(actual,expected,atol=1e-6)
    g1,=torch.autograd.grad(actual,logits,retain_graph=True)
    g2,=torch.autograd.grad(expected,logits)
    assert torch.allclose(g1,g2,atol=1e-6)


def test_prefix_beam_repeated_phone_separated_by_blank():
    lp=torch.tensor([[-10.,0.,-20.],[0.,-10.,-20.],[-10.,0.,-20.]]).log_softmax(-1)
    assert greedy_ctc(lp)==[1,1]
    assert prefix_beam(lp,beam=32,token_limit=3)[0][0]==[1,1]


def test_prefix_beam_matches_exhaustive_path_sums_without_pruning():
    lp=torch.tensor([[.1,.6,.3],[.7,.1,.2],[.2,.5,.3]]).log()
    totals={}
    for path in itertools.product(range(3),repeat=3):
        prefix=tuple(collapse(path));prob=math.exp(sum(float(lp[t,c]) for t,c in enumerate(path)))
        totals[prefix]=totals.get(prefix,0.)+prob
    found=prefix_beam(lp,beam=100,token_limit=3)
    for prefix,score in found:
        assert abs(math.exp(score)-totals.get(tuple(prefix),0.))<1e-6


def test_window_ownership_covers_tail_and_never_duplicates_boundary_word():
    spans=windows(13.,8.,4.)
    assert spans[0].own_start==0 and spans[-1].own_end==13 and spans[-1].end==13
    for a,b in zip(spans,spans[1:]):assert a.own_end==b.own_start
    parts=[]
    for span in spans:
        parts.append([{'word':str(k),'start':k-span.start-.01,'end':k-span.start+.01}
                      for k in range(1,13) if span.start+.01<=k<=span.end-.01])
    text,words=stitch_words(parts,spans)
    assert text.split()==[str(k) for k in range(1,13)]
    assert len(words)==12


def test_zero_lora_preserves_base_then_only_adapter_learns():
    torch.manual_seed(4)
    base=nn.Linear(4,3)
    original={k:v.clone() for k,v in base.state_dict().items()}
    x=torch.randn(5,4);expected=base(x).detach();model=LoRALinear(base,rank=2)
    assert torch.equal(model(x),expected)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    model(x).square().mean().backward();optimizer.step()
    assert not torch.equal(model(x),expected)
    for name,value in base.state_dict().items():assert torch.equal(value,original[name])


def test_inner_attack_preserves_trainable_parameter_state_and_gradients():
    model=nn.Linear(4,1)
    before={k:p.clone() for k,p in model.state_dict().items()}
    x=torch.tensor([.1,-.2,.3,-.4])
    attacked,trace=waveform_pgd(x,lambda a:model(a).sum(),steps=4,epsilon=.02)
    assert float((attacked-x).abs().max())<=.020001
    assert model(attacked)>model(x)
    assert len(trace)==4
    assert all(p.requires_grad and p.grad is None for p in model.parameters())
    for name,value in model.state_dict().items():assert torch.equal(value,before[name])


def test_phone_word_search_generates_text_from_phones_without_reference():
    class Lexicon:
        entries={'cat':[(1,2,3)],'sat':[(4,2,3)],'cap':[(1,2,5)]}
    decoder=PhoneWordDecoder(Lexicon(),{'cat':3,'sat':2,'cap':1},max_edits=0,beam=2)
    assert decoder.decode([1,2,3,4,2,3])[0]['text']=='cat sat'
    assert decoder.decode([8,9])==[]


def test_feature_inner_loop_projects_global_radius_without_freezing_defender():
    torch.manual_seed(5)
    q=torch.randn(8,4)
    x=torch.randn(1024).tanh()
    generator=nn.Sequential(nn.Flatten(0),nn.Linear(32,1024),nn.Tanh()).requires_grad_(False)
    defender=nn.Linear(1024,1)
    before={k:p.clone() for k,p in defender.state_dict().items()}
    audio,delta,trace=feature_pgd(q,x,generator,lambda a:defender(a).sum(),
                                  steps=3,rho=.1,learning_rate=.05)
    assert delta.norm()/q.norm()<=.100001
    assert torch.isfinite(audio).all() and len(trace)==3
    assert all(p.requires_grad and p.grad is None for p in defender.parameters())
    for name,value in defender.state_dict().items():assert torch.equal(value,before[name])


def test_adapter_input_gradient_matches_directional_difference():
    torch.manual_seed(17)
    model=LoRALinear(nn.Linear(4,3),rank=2)
    with torch.no_grad():model.b.normal_(std=.1)
    x=torch.randn(4,requires_grad=True);direction=torch.randn(4)
    loss=model(x).square().sum();grad,=torch.autograd.grad(loss,x)
    eps=.001
    finite=(model(x.detach()+eps*direction).square().sum()-model(x.detach()-eps*direction).square().sum())/(2*eps)
    assert torch.allclose(finite,grad@direction,rtol=1e-3,atol=1e-3)
