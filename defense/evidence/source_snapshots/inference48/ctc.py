"""Exact differentiable CTC transcript likelihood and bounded prefix search."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def ctc_nll(log_probs, labels, blank=0, normalize=True):
    if log_probs.ndim != 2 or not labels:
        raise ValueError('Expected [time,classes] and a nonempty target')
    target = torch.tensor(labels,device=log_probs.device,dtype=torch.long)
    if (target == blank).any():
        raise ValueError('Target must not contain blank')
    repeats = sum(a == b for a,b in zip(labels,labels[1:]))
    if len(labels)+repeats > log_probs.shape[0]:
        return log_probs.new_tensor(float('inf'))
    value = F.ctc_loss(log_probs[:,None],target,
                       torch.tensor([log_probs.shape[0]],device=log_probs.device),
                       torch.tensor([len(labels)],device=log_probs.device),
                       blank=blank,reduction='sum',zero_infinity=False)
    return value/len(labels) if normalize else value


def greedy_ctc(log_probs,blank=0):
    tokens=[];last=None
    for token in log_probs.argmax(-1).tolist():
        if token != last and token != blank:tokens.append(token)
        last=token
    return tokens


def logadd(*values):
    values=[float(v) for v in values if v != -math.inf]
    if not values:return -math.inf
    m=max(values)
    return m+math.log(sum(math.exp(v-m) for v in values))


def prefix_beam(log_probs,blank=0,beam=8,token_limit=12):
    """CTC prefix beam; blank is always included despite token pruning.

    Returns acoustic token candidates; references are never accepted as input.
    token_limit bounds expansion cost and is an explicit search approximation.
    """
    scores=log_probs.detach().cpu()
    prefixes={(): (0.,-math.inf)}
    for frame in scores:
        top=set(frame.topk(min(token_limit,len(frame))).indices.tolist()) | {blank}
        next_scores={}
        def update(prefix,pb=None,pnb=None):
            a,b=next_scores.get(prefix,(-math.inf,-math.inf))
            next_scores[prefix]=(logadd(a,pb) if pb is not None else a,
                                 logadd(b,pnb) if pnb is not None else b)
        for prefix,(pb,pnb) in prefixes.items():
            update(prefix,pb=logadd(pb,pnb)+float(frame[blank]))
            for token in top-{blank}:
                lp=float(frame[token])
                if prefix and token == prefix[-1]:
                    update(prefix,pnb=pnb+lp)
                    update(prefix+(token,),pnb=pb+lp)
                else:
                    update(prefix+(token,),pnb=logadd(pb,pnb)+lp)
        prefixes=dict(sorted(next_scores.items(),key=lambda item:logadd(*item[1]),reverse=True)[:beam])
    return [(list(prefix),logadd(*values)) for prefix,values in sorted(
        prefixes.items(),key=lambda item:logadd(*item[1]),reverse=True)]
