"""Causal partial-H25 signal feedback into a shot-local risk encoder."""

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra, update
from run_risk_feature_intra_20261107 import feature_curves


def partial_curves(base, decoder, signal, *, warm=False):
    """Compare full and two 16-point feedback clocks on one signal-only shot.

    Release indices are the first risk samples affected by an update.  A
    release at the exclusive end of a shot is not executed because it cannot
    affect any risk sample in that shot.
    """
    signal = {'values': signal['values'], 'available': signal['available']}
    reference, reference_info = feature_curves(
        base, decoder, signal, warm=warm, feedbacks=('real',))
    curves = dict(static=reference['static'], full=reference['real'],
                  partial_delayed=reference['static'].copy(),
                  partial_early=reference['static'].copy())
    first_affected = dict(static=None, full=153, partial_delayed=153, partial_early=144)
    info = dict(
        first_affected=first_affected,
        releases=dict(static=[], full=list(reference_info['releases']),
                      partial_delayed=[], partial_early=[]),
        updates=dict(static=0, full=reference_info['updates']['real'],
                     partial_delayed=0, partial_early=0),
        weight_norm=dict(static=0., full=reference_info['weight_norm'].get('real', 0.),
                         partial_delayed=0., partial_early=0.),
        gradient_max=dict(static=0., full=reference_info['gradient_max']['real'],
                          partial_delayed=0., partial_early=0.),
    )
    if warm or len(signal['values']) < j.WINDOW:
        return curves, info

    x, available = j.r.latent.make_views(signal, base.center.device)
    values = torch.as_tensor(signal['values'], device=x.device)
    masks = torch.as_tensor(signal['available'], device=x.device)
    for name, delay in (('partial_delayed', 26), ('partial_early', 17)):
        model = FeatureIntra(base, decoder)
        optimizer = torch.optim.AdamW([model.weight], lr=j.INTRA_LR, weight_decay=0.)
        lo = j.WINDOW - 1
        for end in (*range(first_affected[name], len(x), j.STRIDE), len(x)):
            with torch.no_grad():
                logits = model.risk(x[lo:end], available[lo:end])['logit']
                curves[name][lo:end] = logits.sigmoid().cpu().numpy()
            if end != len(x):
                origin = end - delay
                # Each release precedes the next origin, so the fast weight is
                # unchanged since issuance. As in feature_curves, update
                # recomputes all 25 predictions from that origin's window.
                truth = values.new_empty((1, j.HORIZON, values.shape[1]))
                truth[:] = base.center
                mask = torch.zeros_like(truth, dtype=torch.bool)
                truth[:, :16] = values[origin + 1:origin + 17]
                mask[:, :16] = masks[origin + 1:origin + 17]
                episode = dict(x=x[origin:origin + 1], a=available[origin:origin + 1],
                               truth=truth, mask=mask)
                stats = update(model, optimizer, episode, feedback='real')
                info['releases'][name].append(end)
                info['updates'][name] += 1
                info['gradient_max'][name] = max(info['gradient_max'][name],
                                                 stats['gradient_norm'])
            lo = end
        info['weight_norm'][name] = float((model.weight - model.initial_weight).norm().detach())
    if any(not np.isfinite(values).all() for values in curves.values()):
        raise FloatingPointError('nonfinite internally adapted current risk')
    return curves, info
