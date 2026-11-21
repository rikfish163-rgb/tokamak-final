"""Predict first, update from arrived window, affect subsequent block only."""
import numpy as np
import torch

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder
from .model import update_observed


def predict_pair(model,shot,*,device,use_event,uniform=False,interval=32,training=False):
    if set(shot)!={'values','available'}:
        raise ValueError('signals only; outcome and metadata fields forbidden')
    if type(interval) is not int or interval<1: raise ValueError('positive interval required')
    n=len(shot['values'])
    if not n or shot['values'].shape!=shot['available'].shape:
        raise ValueError('nonempty matching observations required')
    source=initial_encoder(model); fast=source
    scores={k:[] for k in ('static','intra')}; updates=[]
    for lo in range(0,n,interval):
        hi=min(lo+interval,n)
        windows=[simple_data.window(shot,t,model.cfg.window) for t in range(lo,hi)]
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        for arm,state in (('static',source),('intra',fast)):
            out=model.predict_fast(x,m,state)['logit']
            scores[arm].append(out if training and arm=='intra' else out.detach())
        if hi<model.cfg.window or hi==n: continue
        fast,info=update_observed(model,x[-1:],m[-1:],use_event=use_event,
                                 uniform=uniform,create_graph=training)
        updates.append(dict(last_observation=hi-1,first_affected_prediction=hi,**info))
    result={k:torch.cat(v) for k,v in scores.items()}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite online trace')
    if not training: result={k:v.detach().sigmoid().cpu().numpy() for k,v in result.items()}
    return result,dict(updates=updates,inter_updates=0,reset='fresh source per observation block and shot')
