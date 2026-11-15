"""Development probe only: an information increment, not an alarm module."""
import numpy as np
import torch
from torch import nn
from ..conditional_forecast_20261023.learning import direct_targets


class EvidenceProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior = nn.Linear(64,1)
        self.update = nn.Sequential(nn.Linear(294,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,h,d,mask):
        if h.ndim!=2 or h.shape[1]!=64 or d.shape!=(len(h),5,23) or mask.shape!=d.shape or mask.dtype!=torch.bool:
            raise ValueError('64-state and five arrived flat23 observations required')
        d=torch.where(mask,d,0.).flatten(1)
        m=mask.to(h.dtype).flatten(1)
        prior=self.prior(h).squeeze(-1)
        increment=(self.update(torch.cat((h,d,m),-1))-
                   self.update(torch.cat((h,torch.zeros_like(d),m),-1))).squeeze(-1)
        return prior,prior+increment


def feedback_arrays(signal,ends,center,scale):
    ends=np.asarray(ends)
    if ends.ndim!=1 or not len(ends) or ends.min()<5:
        raise ValueError('five preceding and arriving points required')
    d,m=direct_targets(signal,ends,center,scale,5)
    old,oldm=direct_targets(signal,ends-5,center,scale,5)
    return d,m,old,oldm


def rotate_within_shot(d,m,ids):
    ids=np.asarray(ids)
    if len(d)!=len(ids) or d.shape!=m.shape:
        raise ValueError('matching observations, masks and shot identities required')
    out=d.copy();mask=m.copy()
    for sid in np.unique(ids):
        use=np.flatnonzero(ids==sid)
        out[use]=np.roll(d[use],len(use)//2,axis=0)
        mask[use]=np.roll(m[use],len(use)//2,axis=0)
    return out,mask
