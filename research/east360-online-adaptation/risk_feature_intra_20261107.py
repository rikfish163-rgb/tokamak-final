"""Matured signal feedback into a temporary source-risk temporal layer."""

from copy import deepcopy

import torch
from torch import nn

from run_joint_trajectory_20261107 import PRIOR_WEIGHT, features


class FeatureIntra(nn.Module):
    """Only the first temporal FFN output matrix is plastic; heads stay fixed."""

    def __init__(self, base, decoder):
        super().__init__()
        self.risk = deepcopy(base).eval().requires_grad_(False)
        self.decoder = deepcopy(decoder).eval().requires_grad_(False)
        self.weight.requires_grad_(True)
        self.register_buffer('initial_weight',self.weight.detach().clone())

    @property
    def weight(self):
        return self.risk.encoder.layers[0].linear2.weight

    def forward(self, x, available):
        tokens,valid,logit = features(self.risk,x,available)
        trajectory = self.decoder(tokens,valid,x[:,-1],available[:,-1])
        return dict(logit=logit,**trajectory)

    def signal_loss(self, e, *, feedback='real'):
        output = self(e['x'],e['a'])
        truth,mask = e['truth'],e['mask']
        if feedback == 'wrong_time':
            truth,mask = truth.flip(1),mask.flip(1)
        elif feedback == 'hold':
            truth = e['x'][:,-1:, :].expand_as(truth)
        elif feedback != 'real':
            raise ValueError('feedback must be real, wrong_time, or hold')
        mask = mask & output['available'] & torch.isfinite(truth)
        center,scale = self.risk.center,self.risk.scale
        target = torch.asinh((torch.where(mask,truth,center)-center)/scale)
        return self.decoder.loss(output['encoded'],output['probabilities'],target,mask).mean()


def update(model, optimizer, e, *, feedback='real'):
    optimizer.zero_grad(set_to_none=True)
    loss = model.signal_loss(e,feedback=feedback)
    prior = (model.weight-model.initial_weight).square().sum()
    objective = loss+PRIOR_WEIGHT*prior
    if not torch.isfinite(objective):
        raise FloatingPointError('nonfinite temporary risk-feature signal objective')
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_([model.weight],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(signal_loss=float(loss.detach()),prior=float(prior.detach()),
                objective=float(objective.detach()),gradient_norm=float(norm))
