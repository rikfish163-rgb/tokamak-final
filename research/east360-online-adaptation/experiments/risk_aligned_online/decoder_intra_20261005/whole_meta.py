"""Source-only whole-shot risk after causal FFN adaptation.

The outer label may select a source peak, never an EAST inner update. Forward
states replay from index zero; derivatives stop before the final four *mature*
events preceding the selected peak. Early peaks retain their uncontrollable loss.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows
from ..ffn_intra_20261005.stream import predict_shot, signal_step
from .matched_meta import maintenance_loss


def select_peak(shot, scores):
    if shot['device'] != 'J-TEXT':
        raise ValueError('offline source peak selection only')
    scores=np.asarray(scores)
    valid=np.asarray(shot['valid']).copy()
    if scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError('one finite emitted probability per source endpoint required')
    if shot['is_disrupt']:
        valid &= (shot['times']<=shot['down'])
        valid &= np.round((shot['down']-shot['times'])*1000.,9)>30.
    eligible=np.flatnonzero(valid)
    if not len(eligible):
        raise ValueError('source shot has no eligible whole-risk endpoint')
    # Earliest tie on the emitted float32 probabilities; no future state rescoring.
    return int(eligible[np.argmax(scores[eligible])])


def causal_logit(model,shot,query,*,step,radius,create_graph=False):
    if shot['device']!='J-TEXT':
        raise ValueError('offline source replay only')
    if type(query) is not int or not 0<=query<len(shot['times']):
        raise ValueError('query must be an existing source endpoint index')
    if any(not math.isfinite(v) or v<=0 for v in (step,radius)):
        raise ValueError('finite positive FFN budgets required')
    signals={k:shot[k] for k in ('values','available')}
    device=model.center.device;delta=model.initial_delta()
    # origin+25 emits BEFORE its update. First affected risk is origin+26.
    origins=list(range(0,query-25,26))
    burn=max(0,len(origins)-4);missing=executed=differentiated=0
    for j,origin in enumerate(origins):
        x,a=_batch_windows(simple_data,signals,[origin],model,device)
        truth=torch.as_tensor(shot['values'][origin+1:origin+26],
                              device=device,dtype=model.center.dtype)[None]
        observed=torch.as_tensor(shot['available'][origin+1:origin+26],device=device)[None]
        observed=observed & torch.isfinite(truth)
        if observed.any():
            delta,_,_=signal_step(model,delta,x,a,truth,observed,step=step,radius=radius,
                                 create_graph=create_graph and j>=burn)
            executed+=1
            differentiated+=int(create_graph and j>=burn)
        else:
            # Same explicit no-target behavior as the deployed replay: no step.
            missing+=1
        if j<burn or not create_graph:
            delta=delta.detach().requires_grad_()
    x,a=_batch_windows(simple_data,signals,[query],model,device)
    logit=model(x,a,delta)['logit'].reshape(())
    return logit,dict(mature_events=len(origins),burn_in_events=burn,
        unrolled_mature_slots=min(4,len(origins)),missing_target_events=missing,
        executed_updates=executed,differentiated_updates=differentiated,
        delta_norm=float(delta.detach().double().norm()))


def whole_risk(model,shot,*,step,radius,boundary,create_graph=False):
    if not math.isfinite(boundary):
        raise ValueError('finite frozen alarm boundary required')
    # Source OFFLINE outer selection. No target outcomes or full target trace here.
    scores,_=predict_shot(model,shot,intra=True,device=model.center.device,
                         stride=26,step=step,radius=radius,offline_source=True)
    query=select_peak(shot,scores)
    logit,row=causal_logit(model,shot,query,step=step,radius=radius,create_graph=create_graph)
    probability=logit.sigmoid()
    # full_source clips an emitted numpy float32 scalar. Match its EFFECTIVE
    # representable endpoints, especially near 1; exact double 1-1e-7 differs.
    p=probability.double().clamp(float(np.float32(1e-7)),float(np.float32(1-1e-7)))
    margin=torch.logit(p)-boundary
    risk=F.softplus(-margin if shot['is_disrupt'] else margin)
    if not torch.isfinite(risk):
        raise FloatingPointError('nonfinite whole-shot source risk')
    if abs(float(probability.detach())-float(scores[query]))>2e-6:
        raise RuntimeError('causal peak state differs from emitted full replay')
    return risk,dict(**row,shot=shot['shot'],label=int(shot['is_disrupt']),peak_index=query,
        peak_probability=float(probability.detach()),risk_loss=float(risk.detach()),
        nominal_prefix_peak=query<26,before_first_update=row['executed_updates']==0)


def train_pair(model,pairs,optimizer,*,step,radius,boundary):
    allowed={id(p) for p in model.base.forecast_head.parameters()}
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if (set(actual)!=allowed or len(actual)!=len(allowed)
            or {id(p) for p in model.parameters() if p.requires_grad}!=allowed):
        raise ValueError('only forecast_head weight and bias may be optimized')
    if len(pairs)!=2 or sorted(int(s['is_disrupt']) for s,_ in pairs)!=[0,1]:
        raise ValueError('one balanced safe/disruptive source pair required')
    if any(s['device']!='J-TEXT' or e['shot']!=s['shot'] or e['label']!=int(s['is_disrupt'])
           for s,e in pairs):
        raise ValueError('matching source shot/episode identity and outcome required')
    model.zero_grad(set_to_none=True);rows=[]
    for shot,episode in pairs:
        risk,row=whole_risk(model,shot,step=step,radius=radius,boundary=boundary,create_graph=True)
        signal=maintenance_loss(model,shot,episode)
        loss=risk+.20*signal
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite whole-shot head objective')
        (loss/2).backward()
        rows.append(dict(**row,maintenance_loss=float(signal.detach()),objective=float(loss.detach())))
    norm=torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(head_gradient_norm=float(norm),records=rows,
                objective=float(np.mean([r['objective'] for r in rows])))
