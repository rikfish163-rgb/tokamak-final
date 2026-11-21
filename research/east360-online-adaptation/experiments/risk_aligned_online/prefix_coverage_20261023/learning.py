"""Move the safe-shot pre-adaptation risk target, keeping its weight fixed."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..east_joint_feedback_20261023.learning import objective


def prefix_ends(length,size):
    if type(length) is not int or length<256 or type(size) is not int or not 1<=size<=128:
        raise ValueError('at least256 observations and1..128 targets required')
    return np.linspace(0,255,size,dtype=int).tolist()


def prefix_risk(model,signal,size,*,boundary,device):
    if set(signal)!={'values','available'} or not math.isfinite(boundary):
        raise ValueError('signal-only input and finite fixed alarm boundary required')
    ends=prefix_ends(len(signal['values']),size)
    windows=[simple_data.window(signal,e,128) for e in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    h=model(x,m,mode='risk_state')['risk_state']
    margin=model.base.risk_head(h).flatten().max()-boundary
    return F.softplus(margin),ends


def paired_objective(result,early,*,disruptive):
    if disruptive:return objective(result,'meta')
    if early is None or early.ndim!=0:raise ValueError('safe prefix scalar required')
    return .5*(early+result['after'])+.2*result['forecast']
