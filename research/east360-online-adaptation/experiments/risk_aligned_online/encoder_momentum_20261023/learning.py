"""Disable only first-moment carryover in the offline encoder optimizer."""
import torch


def optimizer(model):
    return torch.optim.Adam([
        dict(params=[p for k,p in model.named_parameters() if k in model.encoder_names()],
             lr=1e-5,betas=(0.,.999)),
        dict(params=list(model.base.forecast_head.parameters()),lr=1e-4)])
