"""Shot-local momentum-direction updates with native, accumulated Adam state."""

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra, update
from risk_meta_adam_20261108 import direction_candidate


class DirectionAdamW(torch.optim.AdamW):
    """One internal matrix; exactly the native candidate's displacement length.

    Common update() performs the actual gradient clipping. Adam maintains its
    ordinary constant-size state; no growing gradient-history tape is kept at
    inference. The functional teacher uses the same direction_candidate map.
    """

    def __init__(self, weight, *, lr=1e-4):
        self.matrix = weight
        super().__init__([weight], lr=lr, weight_decay=0., foreach=False)

    @torch.no_grad()
    def step(self):
        if self.matrix.grad is None:
            return super().step()
        before = self.matrix.clone()
        result = super().step()
        candidate = direction_candidate(before, self.state[self.matrix]['exp_avg'], self.matrix)
        self.matrix.copy_(candidate['weight'])
        return result


def direction_curves(base, decoder, signal, *, warm=False):
    """Original 127+32k issuance, complete H25 feedback, first effect at 153.

    The reference clock is deliberately unchanged. Only the optimizer's
    direction differs from feature_curves; all loss code is reused directly.
    """
    x, a = j.r.latent.make_views(signal, base.center.device)
    with torch.no_grad():
        static = torch.cat([base(x[lo:lo+j.BATCH], a[lo:lo+j.BATCH])['logit'].sigmoid()
                            for lo in range(0, len(x), j.BATCH)]).cpu().numpy()
    traces = dict(static=static, real=static.copy())
    info = dict(releases=[], updates=dict(real=0), weight_norm={}, gradient_max=dict(real=0.))
    if warm or len(x) < 128:
        return traces, info
    model = FeatureIntra(base, decoder)
    opt = DirectionAdamW(model.weight, lr=j.INTRA_LR)
    values = torch.as_tensor(signal['values'], device=x.device)
    masks = torch.as_tensor(signal['available'], device=x.device)
    lo = 127
    for end in list(range(153, len(x), 32)) + [len(x)]:
        with torch.no_grad():
            traces['real'][lo:end] = model.risk(x[lo:end], a[lo:end])['logit'].sigmoid().cpu().numpy()
        if end != len(x):
            origin = end-26
            episode = dict(x=x[origin:origin+1], a=a[origin:origin+1],
                           truth=values[origin+1:origin+26][None], mask=masks[origin+1:origin+26][None])
            stats = update(model, opt, episode, feedback='real')
            info['updates']['real'] += 1
            info['gradient_max']['real'] = max(info['gradient_max']['real'], stats['gradient_norm'])
            info['releases'].append(end)
        lo = end
    info['weight_norm']['real'] = float((model.weight-model.initial_weight).norm().detach())
    if any(not np.isfinite(v).all() for v in traces.values()):
        raise FloatingPointError('nonfinite direction-adapted current risk')
    return traces, info
