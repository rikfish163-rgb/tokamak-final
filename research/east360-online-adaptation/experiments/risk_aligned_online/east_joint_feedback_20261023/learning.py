"""EAST preparation can learn both the update recipient and its signal task.

The qualified source archive and fixed risk readout are never modified.
Exact differentiation through the whole observed update prefix is used only
offline; labels are consumed by the outer objective, never observed_step.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..credit_horizon_20261023.learning import observed_step
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..increment_forecast_20261023.learning import feedback_loss
from ..paired_feedback_20261022.model import initial_encoder


def enable(model):
    names=set(model.encoder_names())|{'base.forecast_head.weight','base.forecast_head.bias'}
    for name,p in model.named_parameters():
        p.requires_grad_(name in names);p.grad=None
    model.eval()
    return names


def episode(model,signal,count,ends,disruptive,*,boundary,track,device):
    if (set(signal)!={'values','available'} or type(count) is not int or count<256 or count%128
            or not ends or ends!=sorted(set(ends)) or min(ends)<count
            or max(ends)>=min(count+128,len(signal['values'])) or not math.isfinite(boundary)
            or type(track) is not bool):
        raise ValueError('observed signal-only input and next-block endpoints required')
    prefix={k:v[:count] for k,v in signal.items()}
    x,m,truth,tm,_=forecast_examples(prefix,count,model.cfg,device)
    dtype=model.base.center.dtype;x=x.to(dtype);truth=truth.to(dtype)
    forecast=feedback_loss(model,model(x,m)['forecast'],x,m,truth,tm)
    counts=list(range(256,count+1,128))
    fast=initial_encoder(model,track_initial=track)
    for seen in counts:
        fast,_=observed_step(model,signal,seen,fast,device=device,create_graph=track)
    windows=[simple_data.window(signal,e,128) for e in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)];x=x.to(dtype)
    h=model(x,m,mode='risk_state')['risk_state']
    z=model.base.risk_head(h).flatten().max()-boundary
    before=F.softplus(-z if disruptive else z)
    h=model.predict_fast(x,m,fast,mode='risk_state')['risk_state']
    z=model.base.risk_head(h).flatten().max()-boundary
    after=F.softplus(-z if disruptive else z)
    return dict(before=before,after=after,forecast=forecast,fast=fast,all_counts=counts)


def objective(result,arm):
    if arm not in ('joint','meta'):raise ValueError('joint/meta paired objective required')
    risk=result['before'] if arm=='joint' else .5*(result['before']+result['after'])
    return risk+.2*result['forecast']
