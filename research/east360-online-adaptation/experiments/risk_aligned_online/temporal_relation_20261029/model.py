"""Paired temporal-relation supervision, separate from the frozen risk head."""
import numpy as np
import torch
from torch import nn

from ..latent_completion_20261028.model import tokens


class OrderJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,h):
        d=h[:,1:16]-h[:,:15]
        return self.net(torch.cat((d[:,:-1],d[:,1:]),-1)).mean((1,2))


def permutation(count,rng,mode='random'):
    if mode not in ('random','interleave'):raise ValueError('declared temporal permutation required')
    rows=[]
    for _ in range(count):
        interior=rng.permutation(np.arange(1,15)) if mode=='random' else np.r_[np.arange(1,15,2),np.arange(2,15,2)]
        if np.array_equal(interior,np.arange(1,15)):interior=np.roll(interior,1)
        rows.append(np.r_[0,interior,15])
    return np.stack(rows)


def paired_views(x,a,p):
    if x.shape!=a.shape or x.shape[1:]!=(128,23) or a.dtype!=torch.bool or not torch.isfinite(x[a]).all():
        raise ValueError('finite observed128x23 windows required')
    p=torch.as_tensor(p,device=x.device,dtype=torch.long)
    if p.shape!=(len(x),16) or not (p[:,0]==0).all() or not (p[:,-1]==15).all() or not torch.equal(p.sort(1).values,torch.arange(16,device=x.device).expand(len(x),-1)):
        raise ValueError('bijective endpoint-preserving patch permutation required')
    b=torch.arange(len(x),device=x.device)[:,None]
    xx=x.reshape(-1,16,8,23)[b,p].reshape_as(x)
    aa=a.reshape(-1,16,8,23)[b,p].reshape_as(a)
    # Constant availability prevents mask-position shortcuts and moving padded
    # startup samples. Uninformative identical pairs never get invented labels.
    eligible=(a==a[:,:1]).all((1,2)) & a.any((1,2)) & ((xx!=x)&a).any((1,2))
    return torch.where(eligible[:,None,None],xx,x),torch.where(eligible[:,None,None],aa,a),eligible


def mean_view(x,a):
    mean=torch.where(a,x,0.).sum(1,keepdim=True)/a.sum(1,keepdim=True).clamp_min(1)
    return torch.where(a,mean.expand_as(x),torch.zeros_like(x))


def roughness(h):
    return -(h[:,1:16]-h[:,:15]).square().mean((1,2))


def margin(base,net,x,a,p,state):
    xx,aa,eligible=paired_views(x,a,p)
    fast=state if state.ndim==1 else torch.cat((state,state))
    h,_=tokens(base,torch.cat((x,xx)),torch.cat((a,aa)),fast)
    real,fake=h.chunk(2,0)
    return net(real)-net(fake),eligible,roughness(real)-roughness(fake)
