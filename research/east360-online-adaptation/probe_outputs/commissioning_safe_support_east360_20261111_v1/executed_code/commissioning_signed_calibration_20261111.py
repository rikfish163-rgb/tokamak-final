"""Teach the centered update's direction against matched and alarm-level controls."""

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_split_learning_20261111 as learn
from . import simple_data
from .commissioning_transfer_20261108 import features


def mirror_loss(real, mirror):
    """Reward separation of timely disruption and matched safety, not offset."""
    dr = real[..., 0] - real[..., 1]
    dm = mirror[..., 0] - mirror[..., 1]
    return F.softplus((dm-dr)/.25).mean()


def alarm_loss(logits, lengths, disruptive, boundary):
    peaks = torch.stack([z.max() for z in logits.split(lengths)])
    return .5*(F.softplus(boundary-peaks[disruptive]).mean()
                + F.softplus(peaks[~disruptive]-boundary).mean())


@torch.no_grad()
def cold_alarm_rows(base, net, raw, signal, center, scale, decoder, noise, coef, replay):
    """Sample causal warning windows and safety peaks only from ended shots."""
    row = simple_data.attach_outcome(raw, 'EAST', None)
    ends = np.arange(len(raw['times']))
    tr = learn.track(net, signal, center, scale, ends)
    forecast = learn.base_forecast(tr, decoder)
    states, _ = replay(forecast, learn.normalized_target(tr, noise), tr['observed'], tr['q'], len(ends))
    x, _ = features(base, signal, 'cuda')
    p = torch.from_numpy(tr['pre']).cuda()
    raw_z = torch.from_numpy(x).cuda() @ coef[:-25] + coef[-1]
    if row['is_disrupt']:
        lead = np.round((row['down']-row['times'])*1000., 9)
        q = np.flatnonzero(row['valid'] & (row['y'] > 0.) & (lead > 30.))
    else:
        fixed = raw_z + p.tanh() @ coef[-25:-1]
        peaks = fixed.topk(min(len(ends), 128)).indices.cpu().numpy()
        q = np.union1d(np.arange(0, len(ends), 16), peaks)
        q = np.union1d(q, np.arange(min(160, len(ends))))
    if not len(q):
        raise ValueError(f'no legal cold warning endpoints for {raw["shot"]}')
    return dict(sid=raw['shot'], disruptive=row['is_disrupt'], raw_z=raw_z[q].cpu().numpy(),
                pre=tr['pre'][q], key=tr['key'][q], split=states['split'][q])


def teach(train, cold, alarm, beta, cutoff):
    """One fixed fit with the old recipe plus two predeclared causal constraints."""
    rng = torch.Generator(device='cuda').manual_seed(learn.SEED)
    data = []
    for rows, matched, _ in (train, cold):
        pos, neg = matched[:2]
        _, inv, n = np.unique(rows['shot'][pos], return_inverse=True, return_counts=True)
        prob = torch.from_numpy(1./n[inv]).cuda()
        draws = torch.multinomial(prob, learn.STEPS*learn.BATCH, True, generator=rng).reshape(learn.STEPS, -1)
        table = torch.from_numpy(np.column_stack((pos, neg))).cuda()
        data.append((draws, table, {k: torch.from_numpy(rows[k]).cuda()
                                     for k in ('pre', 'key', 'split', 'delayed', 'reversed')}))
    pre, key, state, offset = (torch.from_numpy(np.concatenate([r[k] for r in alarm])).cuda()
                               for k in ('pre', 'key', 'split', 'raw_z'))
    lengths = [len(r['pre']) for r in alarm]
    positives = torch.tensor([r['disruptive'] for r in alarm], device='cuda', dtype=torch.bool)
    boundary = float(np.log(cutoff/(1.-cutoff)))
    param = torch.nn.Parameter(torch.zeros(4, 20, learn.WIDTH, device='cuda'))
    opt = torch.optim.Adam([param], lr=.012)
    for i in range(learn.STEPS):
        opt.zero_grad(set_to_none=True)
        mapper = learn.bounded_map(param)
        logits = offset + learn.feature_view(pre, key, state, mapper)@beta
        warning = alarm_loss(logits, lengths, positives, boundary)
        loss = .35*warning + .012*mapper.square().sum()
        for wt, (draws, table, r) in zip((.65, .35), data, strict=True):
            j = table[draws[i]]
            p, c = r['pre'][j], r['key'][j]
            real, delay, mirror = (learn.feature_view(p, c, r[k][j], mapper)@beta
                                    for k in ('split', 'delayed', 'reversed'))
            loss = loss + wt*(learn.paired_loss(real, delay, p.tanh()@beta) + .50*mirror_loss(real, mirror))
        loss.backward()
        opt.step()
        if (i+1) % 256 == 0:
            print(dict(stage='signed_fit', steps=i+1, loss=float(loss.detach()),
                       cold_warning=float(warning.detach())), flush=True)
    fitted = learn.bounded_map(param).detach()
    if not torch.isfinite(fitted).all():
        raise FloatingPointError('nonfinite signed calibrated map')
    return fitted.cpu()
