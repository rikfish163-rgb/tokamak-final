"""A fixed Armijo diagnostic using only already-matured feedback."""

import math

import torch
from torch.func import functional_call


def scaled_parameters(model, proposal, factor):
    if not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('bounded finite step factor required')
    return {name: (value.detach()+factor*proposal['delta'][name] if factor and name in proposal['delta']
                   else value.detach()) for name,value in model.named_parameters()}


@torch.no_grad()
def select_scale(model, x0, a0, truth, mask, proposal):
    """Pick a descent step on observed feedback, without future queries/labels."""
    before=float(proposal['loss'])
    gradient_norm=float(proposal['gradient_norm'])
    update_norm=float(proposal['update_norm'])
    if not all(math.isfinite(v) for v in (before,gradient_norm,update_norm)):
        raise FloatingPointError('finite local objective and update required')
    if update_norm == 0:
        return dict(factor=0., loss_before=before, loss_after=before, attempts=0, accepted=False)
    for attempt in range(9):
        factor=2.**(-attempt)
        parameters=scaled_parameters(model,proposal,factor)
        prediction=functional_call(model,parameters,(x0,a0))['forecast']
        after=float(model.forecast_loss(prediction,truth,mask))
        if math.isfinite(after) and after <= before-1e-4*factor*gradient_norm*update_norm:
            return dict(factor=factor,loss_before=before,loss_after=after,attempts=attempt+1,accepted=True)
    return dict(factor=0.,loss_before=before,loss_after=before,attempts=9,accepted=False)
