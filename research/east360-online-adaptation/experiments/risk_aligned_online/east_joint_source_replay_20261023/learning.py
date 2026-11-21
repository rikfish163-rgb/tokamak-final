"""Reuse only source TRAIN supervision while preparing an EAST model copy."""
import torch
from torch.nn import functional as F


def validate_source(shots,train_ids):
    ids=[s['shot'] for s in shots]
    if (not shots or len(ids)!=len(set(ids)) or len(train_ids)!=len(set(train_ids))
            or ids!=list(train_ids) or any(s['device']!='J-TEXT' for s in shots)):
        raise ValueError('distinct canonical source TRAIN shots in manifest order required')


def replay_loss(model,x,mask,y):
    if y.shape!=(len(x),) or not torch.all((y==0)|(y==1)):
        raise ValueError('one binary source label per training window required')
    h=model(x,mask,mode='risk_state')['risk_state']
    return F.binary_cross_entropy_with_logits(model.base.risk_head(h).flatten(),y)
