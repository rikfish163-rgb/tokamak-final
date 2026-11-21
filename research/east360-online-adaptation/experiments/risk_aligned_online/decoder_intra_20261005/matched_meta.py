"""Source-only head learning through the SAME disposable FFN updates as replay.

No new classifier: only forecast_head changes offline. At EAST replay that head
is fixed; matured observations update the last FFN's disposable delta. Source
labels supervise the outer objective, never the online signal step.
"""
import math

import numpy as np
import torch

from .. import simple_data
from ..redesign_20261004.intra import _batch_windows
from ..ffn_intra_20261005.source import episode_loss
from ..ffn_intra_20261005.loss import signal_components, weighted_signal_loss


def enable_head_only(model):
    model.requires_grad_(False)
    model.base.forecast_head.requires_grad_(True)
    model.eval()


def maintenance_loss(model, shot, episode):
    """Keep the original direct H25 task at zero FFN delta on four fit origins."""
    origins=episode['origins'];device=model.center.device
    x,a=_batch_windows(simple_data,{k:shot[k] for k in ('values','available')},origins,model,device)
    raw=torch.as_tensor(np.stack([shot['values'][o+1:o+26] for o in origins]),
                        dtype=model.center.dtype,device=device)
    observed=torch.as_tensor(np.stack([shot['available'][o+1:o+26] for o in origins]),device=device)
    prediction=model(x,a,model.initial_delta())['forecast']
    return weighted_signal_loss(signal_components(prediction,raw,observed,model.center,model.scale)).mean()


def objective(model,shot,episode,*,mode,step,radius,boundary,create_graph=True):
    if (shot['device']!='J-TEXT' or episode['shot']!=shot['shot']):
        raise ValueError('matching offline source shot identity required')
    if mode not in ('risk_aligned','signal_only'):
        raise ValueError('declared risk_aligned or signal_only objective required')
    if (not all(math.isfinite(v) for v in (step,radius,boundary)) or min(step,radius)<=0):
        raise ValueError('finite positive FFN budget required')
    risk,row=episode_loss(model,shot,episode,step=step,radius=radius,rule=None,
                          boundary=boundary,create_graph=create_graph)
    signal=maintenance_loss(model,shot,episode)
    # .20 is a predeclared engineering tradeoff, NOT a fitted physical constant.
    total=(risk if mode=='risk_aligned' else signal.new_zeros(()))+.20*signal
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite matched-head source objective')
    return total,dict(**row,risk_loss=float(risk.detach()),
                      maintenance_loss=float(signal.detach()),objective=float(total.detach()))


def train_pair(model,pairs,optimizer,*,mode,step,radius,boundary):
    allowed={id(p) for p in model.base.forecast_head.parameters()}
    actual=[id(p) for g in optimizer.param_groups for p in g['params']]
    trainable={id(p) for p in model.parameters() if p.requires_grad}
    if set(actual)!=allowed or len(actual)!=len(allowed) or trainable!=allowed:
        raise ValueError('only forecast_head weight and bias may be optimized')
    if not pairs:raise ValueError('nonempty source pair required')
    model.zero_grad(set_to_none=True)
    rows=[]
    for shot,episode in pairs:
        loss,row=objective(model,shot,episode,mode=mode,step=step,radius=radius,
                           boundary=boundary,create_graph=True)
        (loss/len(pairs)).backward()
        rows.append(row)
    norm=torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(head_gradient_norm=float(norm),records=rows,
                objective=float(np.mean([r['objective'] for r in rows])))
