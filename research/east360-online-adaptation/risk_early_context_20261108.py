"""Causal H25 feature updates beginning with a partially padded history."""

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra, update


def early_curves(base, decoder, signal, *, warm=False):
    """Return observed-window risks with temporary, matured signal feedback."""
    x, a = j.r.latent.make_views(signal, base.center.device)
    with torch.no_grad():
        static = torch.cat([
            base(x[lo:lo+j.BATCH], a[lo:lo+j.BATCH])['logit'].sigmoid()
            for lo in range(0, len(x), j.BATCH)
        ]).cpu().numpy()
    curves = {'static': static, 'real': static.copy()}
    info = dict(releases=[], updates={'real': 0}, weight_norm={'real': 0.},
                gradient_max={'real': 0.})
    if warm or len(x) <= 57:
        return curves, info

    model = FeatureIntra(base, decoder)
    optimizer = torch.optim.AdamW([model.weight], lr=j.INTRA_LR, weight_decay=0.)
    values = torch.as_tensor(signal['values'], device=x.device)
    masks = torch.as_tensor(signal['available'], device=x.device)
    lo = 57
    for end in range(57, len(x), j.STRIDE):
        if lo < end:
            with torch.no_grad():
                curves['real'][lo:end] = model.risk(x[lo:end], a[lo:end])['logit'].sigmoid().cpu().numpy()
        origin = end - j.HORIZON - 1
        episode = dict(x=x[origin:origin+1], a=a[origin:origin+1],
                       truth=values[origin+1:end][None], mask=masks[origin+1:end][None])
        stats = update(model, optimizer, episode, feedback='real')
        info['releases'].append(end)
        info['updates']['real'] += 1
        info['gradient_max']['real'] = max(info['gradient_max']['real'], stats['gradient_norm'])
        lo = end
    with torch.no_grad():
        curves['real'][lo:] = model.risk(x[lo:], a[lo:])['logit'].sigmoid().cpu().numpy()
    info['weight_norm']['real'] = float((model.weight-model.initial_weight).norm().detach())
    if any(not np.isfinite(curve).all() for curve in curves.values()):
        raise FloatingPointError('nonfinite internally adapted current risk')
    return curves, info
