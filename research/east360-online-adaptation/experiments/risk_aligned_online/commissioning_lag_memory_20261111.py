"""Use a delayed, learned relation prior to encode current observed cross-edges."""

import numpy as np
import torch
from torch import nn

from . import commissioning_lag_data_20261110 as lag
from . import commissioning_lag_source_20261110 as source
from . import io23
from .commissioning_lag_state_20261110 import MODES
from .commissioning_transfer_20261108 import features
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE


class GraphMemory(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.register_buffer('source_mean',torch.from_numpy(mean.copy()))


def train_mean(center,scale,expected):
    rows,ids=source.collect(io23.source_roles()['train'],io23.advanced_times(),center,scale,'mean_prior')
    if ids!=expected['train_shots'] or int(rows['valid'].sum())!=expected['eligible_train_feedback']:
        raise ValueError('source mean must use the same original training episodes')
    return rows['aligned'][rows['valid']].mean(0)


def memory_views(current,q,aligned,shifted,eligible,prior):
    """A squared-error step on the matured relation vector; no new score bias."""
    norm0=np.linalg.norm(prior)
    if not norm0>0.:
        raise ValueError('source relation prior must have a nonzero norm')
    def direction(p):
        return p / np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),norm0)

    scale=np.float32(np.sqrt(len(prior)))
    ref=np.concatenate((current,current*(scale*direction(prior))),axis=1)
    views={k:ref.copy() for k in MODES}
    state=np.stack((prior,prior))
    targets=np.stack((aligned,shifted),axis=1)
    wins=np.zeros(3,np.int64)
    loss=np.zeros(3,np.float64)
    steps=0
    for j,origin in enumerate(q):
        if origin+HORIZON>=len(current):
            break
        if eligible[j]:
            margin=np.concatenate((direction(prior)[None],direction(state)))@aligned[j]
            wins+=margin>0
            loss+=np.logaddexp(0.,-margin)
            state=.85*state+.15*targets[j]
            steps+=1
        first=int(origin+HORIZON+1)
        end=min(first+STRIDE,len(current))
        for k,p in zip(MODES[1:],direction(state),strict=True):
            views[k][first:end,len(prior):]=current[first:end]*(scale*p)
    return views,dict(observed_blocks=steps,updates=dict.fromkeys(MODES[1:],steps),
                      true_order_loss_sum=dict(zip(MODES,loss.tolist(),strict=True)),
                      true_order_wins=dict(zip(MODES,wins.tolist(),strict=True)))


def encode(base,net,signal,center,scale):
    raw,_=features(base,signal,'cuda')
    z,m,*_=source.shot_coordinates(signal,center,scale)
    a,b,valid=lag.feedback(z,m,np.arange(len(z)))
    q=np.arange(FIRST,len(z)-HORIZON,STRIDE)
    ends=q+HORIZON
    views,audit=memory_views(a,q,a[ends],b[ends],valid[ends],net.source_mean.numpy())
    for v in views.values():
        v[:FIRST]=0.
    return raw,views,audit
