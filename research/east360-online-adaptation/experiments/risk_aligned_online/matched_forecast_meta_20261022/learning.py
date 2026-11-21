"""Source-only teaching through the deployed arrived-signal encoder update.

Forward burn-in replays every preceding within-shot update. Outer derivatives
are truncated to the last two updates; this is TBPTT, not full-shot MAML.
No target outcome or additional risk-correction module is accepted by the
deployable observed_step. The head remains a direct H25 diagnostic predictor.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder
from ..deepcoral_prefix_20261022.learning import forecast_examples

INNER_LR=1e-3


def observed_step(model,signal,count,fast,*,device,create_graph):
    x,m,target,tm,_=forecast_examples(signal,count,model.cfg,device)
    if not tm.any():return fast,model.base.forecast_head.weight.sum()*0.
    forecast=model.predict_fast(x,m,fast)['forecast']
    loss=model.base.forecast_loss(forecast,target,tm)
    gradients=torch.autograd.grad(loss,tuple(fast.values()),create_graph=create_graph)
    norm=(sum(g.square().sum() for g in gradients)+1e-12).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite arrived-signal feedback')
    # The tiny gradient-coordinate epsilon makes double backward finite at
    # zero. It does not set a diagnostic unit scale. Forward agreement with
    # the deployed norm-capped SGD step is checked on CPU and real CUDA data.
    factor=1./norm.clamp_min(1.)
    updated={k:v-INNER_LR*factor*g for (k,v),g in zip(fast.items(),gradients)}
    if not create_graph:updated={k:v.detach().requires_grad_(True) for k,v in updated.items()}
    return updated,loss


def episode(model,signal,query,label,*,device,create_graph):
    if (set(signal)!={'values','available'} or type(query) is not int
            or not 256<=query<len(signal['values']) or label not in (0.,1.)):
        raise ValueError('source-labelled query after first arrived update required')
    counts=list(range(256,query//128*128+1,128))
    tail=counts[-2:];fast=initial_encoder(model);losses=[]
    for count in counts:
        fast,loss=observed_step(model,signal,count,fast,device=device,
                               create_graph=create_graph and count in tail)
        if count in tail:losses.append(loss)
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,query,model.cfg.window)]
    z=model.predict_fast(x,m,fast)['logit']
    y=torch.full_like(z,float(label))
    after=F.binary_cross_entropy_with_logits(z,y)
    # A fresh grad-enabled leaf matches the deployed math-attention dispatch.
    before=F.binary_cross_entropy_with_logits(model.predict_fast(x,m,initial_encoder(model))['logit'],y).detach()
    return dict(risk_before=before,risk_after=after,forecast=torch.stack(losses).mean(),
                fast=fast,all_update_counts=counts,differentiated_counts=tail)


def episode_pools(shots,train_ids):
    if (not shots or len(train_ids)!=len(set(train_ids))
            or any(s['device']!='J-TEXT' or s['shot'] not in set(train_ids) for s in shots)):
        raise ValueError('explicit source TRAIN identities only')
    groups={0:[],1:[]};ledger=[]
    for i,s in enumerate(shots):
        ends=np.arange(256,len(s['values']))
        ends=ends[s['valid'][ends]]
        counts={}
        for label in (0,1):
            take=ends[s['y'][ends]==label];counts[str(label)]=len(take)
            if len(take):groups[label].append((i,take))
        ledger.append(dict(shot=s['shot'],query_counts=counts))
    if any(not groups[k] for k in (0,1)):raise ValueError('both source query labels required')
    return groups,ledger


def draw_episode(shots,pools,rng,label):
    group=pools[label];i,ends=group[int(rng.integers(len(group)))];query=int(rng.choice(ends))
    s=shots[i]
    return s,dict(shot=s['shot'],query=query,label=int(label))
