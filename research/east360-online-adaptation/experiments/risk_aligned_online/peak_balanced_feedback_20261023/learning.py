"""Two controlled changes: safe peak coverage and bounded outer risk gradient.

This is NOT GradNorm reproduction: there is no learned task weight or relative
training-rate target. The per-episode cap bounds Euclidean gradient norms;
Adam preconditioning means it does NOT guarantee monotone forecast loss.
"""
import copy
import numpy as np
import torch


def peak_manifest(manifest,pools,scores):
    byid={r['shot']:r for r in pools}
    safe={k for k,r in byid.items() if not r['is_disrupt']}
    if len(byid)!=len(pools) or set(scores)!=safe:
        raise ValueError('exact distinct historical safe TRAIN scores required')
    peaks={}
    for sid in sorted(safe):
        p=np.asarray(scores[sid])
        blocks=byid[sid]['blocks']
        if p.ndim!=1 or not np.isfinite(p).all() or not blocks or ((p<0)|(p>1)).any():
            raise ValueError('finite probabilities and nonempty eligible TRAIN blocks required')
        if any(not b['ends'] or min(b['ends'])<0 or max(b['ends'])>=len(p) for b in blocks):
            raise ValueError('eligible endpoints outside recorded TRAIN trace')
        block=max(blocks,key=lambda b:(float(p[b['ends']].max()),-b['count']))
        peaks[sid]=dict(shot=sid,**copy.deepcopy(block),peak=float(p[block['ends']].max()))
    revised=copy.deepcopy(manifest);seen={sid:0 for sid in safe}
    for original,row in zip(manifest,revised):
        for e,new in zip(original['episodes'],row['episodes']):
            if (e['shot'] not in byid or e['is_disrupt']!=byid[e['shot']]['is_disrupt']
                    or dict(count=e['count'],ends=e['ends']) not in byid[e['shot']]['blocks']):
                raise ValueError('manifest must contain authorized eligible historical TRAIN episodes')
            if not e['is_disrupt']:
                sid=e['shot'];selected=seen[sid]%2==0;seen[sid]+=1
                if selected:
                    new['count']=peaks[sid]['count'];new['ends']=peaks[sid]['ends'].copy()
                new['selection']='frozen_safe_peak' if selected else 'original_random'
                new['original_count']=e['count']
    if any(n==0 for n in seen.values()):raise ValueError('base manifest did not cover every historical safe shot')
    return revised,list(peaks.values())


def balanced_head_gradient(episode,parameters):
    parameters=tuple(parameters);k=len(episode['differentiated_counts'])
    if not parameters or k not in (1,2):raise ValueError('forecast parameters and one/two differentiated steps required')
    risk=torch.autograd.grad(episode['risk_after'],parameters,retain_graph=True)
    signal=torch.autograd.grad(episode['forecast'],parameters)
    r=tuple(g.detach()/(k*.001) for g in risk);d=tuple(g.detach() for g in signal)
    nr=sum(g.double().square().sum() for g in r).sqrt()
    nd=sum(g.double().square().sum() for g in d).sqrt()
    if not torch.isfinite(nr) or not torch.isfinite(nd):raise FloatingPointError('nonfinite outer gradients')
    # Exact zero cases are mathematical: no risk gradient needs no scaling;
    # zero signal gradient sets the permitted risk-gradient norm to zero.
    scale=1. if float(nr)==0 else min(1.,float(nd/nr))
    total=tuple(a+scale*b for a,b in zip(d,r))
    dot=sum((a.double()*b.double()).sum() for a,b in zip(d,r))
    cosine=float(dot/(nd*nr)) if float(nd*nr)>0 else None
    return total,dict(forecast_norm=float(nd),scaled_risk_norm=float(nr),risk_scale=scale,
                      capped_risk_norm=float(nr)*scale,gradient_cosine=cosine)
