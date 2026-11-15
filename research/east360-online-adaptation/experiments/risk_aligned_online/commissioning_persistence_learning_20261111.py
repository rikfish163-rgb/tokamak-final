"""Fit a small, deadline-aware representation update on ended training shots."""

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_operating_east_20261111 as previous
from . import commissioning_persistence_20261111 as persistent
from . import commissioning_split_learning_20261111 as direct
from . import simple_data
from .commissioning_transfer_20261108 import features


STEPS, BATCH, SEED, RADIUS = 512, 128, 131, 8.
TEACH = ('sustained', 'current_equal_norm', 'spaced', 'opposite')


def observed_track(net, signal, ends, params, mapper):
    center, scale, decoder, noise, context_scale = params
    tr = direct.track(net, signal, center, scale, ends)
    previous_state, _ = previous.estimate(tr, signal, decoder, noise, context_scale)
    b, k = (torch.from_numpy(tr[v]) for v in ('pre', 'key'))
    pre = b + torch.einsum('...k,...f,kfw->...w', k,
                           torch.from_numpy(previous_state['uncalibrated'][ends]), mapper)
    base = direct.base_forecast(tr, decoder)
    levels, observed = persistent.levels(base, direct.normalized_target(tr, noise), tr['observed'])
    q = tr['q']
    mask = signal['available'][:, previous._PHYSICAL]
    count = np.vstack((np.zeros((1, 2), np.int64), np.cumsum(mask, axis=0)))
    end = q+1
    usable = ((count[end]-count[end-16]) >= 8).all(1) & (count[end-16, 0]-count[end-32, 0] >= 8)
    physical = tr['design'][:, direct.WIDTH:direct.WIDTH+3] / context_scale
    ready = levels, observed, q, physical, usable, tr['length']
    return tr, pre, ready


@torch.no_grad()
def labelled(net, raw, signal, advanced, params, mapper, *, base=None, coef=None):
    truth = simple_data.attach_outcome(raw, raw['device'], advanced)
    d, down, t = truth['is_disrupt'], truth['down'], raw['times']
    # This auxiliary pairing task does not replace the source label window.
    ix = direct.pairs.endpoints(t, truth['valid'], d, down, 0., first_effect=persistent.FIRST_EFFECT)
    ends = np.arange(len(t)) if base is not None else np.unique(ix.ravel())
    tr, pre, ready = observed_track(net, signal, ends, params, mapper)
    pos = np.searchsorted(ends, ix)
    keep = tr['matchable'][pos].all(1)
    pos = pos[keep]
    points = ix[keep, 1]
    ctx = tr['context'][pos]
    tail = (not d) & (t[points] >= t[-1] - .125) & (ctx[:, 1, 2] < 0.)
    pairs = dict(pre=pre[pos[:, 1]].numpy(), key=tr['key'][pos[:, 1]], context=ctx,
                 tail=tail, shot=np.full(len(points), raw['shot']),
                 disruptive=np.full(len(points), d),
                 official_positive=truth['y'][points].astype(bool))
    alarm = None
    if base is not None:
        x, _ = features(base, signal, 'cuda')
        offset = torch.from_numpy(x).cuda() @ coef[:-25] + coef[-1]
        if d:
            lead = np.round((down-t)*1000., 9)
            alarms = np.flatnonzero(truth['valid'] & (truth['y'] > 0.) & (lead > 30.))
        else:
            risk = offset + pre.cuda().tanh() @ coef[-25:-1]
            top = risk.topk(min(len(ends), 128)).indices.cpu().numpy()
            alarms = np.union1d(np.arange(0, len(t), 16), top)
            alarms = np.union1d(alarms, np.arange(min(persistent.FIRST_EFFECT, len(t))))
        if not len(alarms):
            raise ValueError('no safe/timely alert support in cold shot')
        alarm = dict(offset=offset[alarms].cpu().numpy(), pre=pre[alarms].numpy(),
                     key=tr['key'][alarms], indices=alarms, disruptive=d, shot=raw['shot'])
    return dict(ready=ready, pairs=pairs, indices=points, alarm=alarm, disruptive=d)


def source(net, ids, advanced, params, mapper, role):
    records = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        if len(raw['times']) > direct.FIRST:
            records.append(labelled(net, raw, raw, advanced, params, mapper))
        if i%450 == 0 or i == len(ids):
            print(dict(stage=role, shots=i, total=len(ids)), flush=True)
    return records


def fit_quiet(records):
    level, mask = [], []
    for r in records:
        if not r['disruptive']:
            values, valid, q, _, _, n = r['ready']
            m = q+persistent.WINDOW+1 < n
            level.append(values[m])
            mask.append(valid[m])
    v, m = np.concatenate(level), np.concatenate(mask)
    return persistent.location_scale(v, m), int(len(v))


def pack(records, quiet, scale):
    rows, alarms = [], []
    for r in records:
        views, _ = persistent.replay(*r['ready'], quiet)
        rows.append(dict(r['pairs'], **{k: v[r['indices']] for k, v in views.items()}))
        if r['alarm'] is not None:
            alarm = r['alarm']
            alarms.append(dict(alarm, state=views['sustained'][alarm['indices']]))
    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    match = direct.pairs.match(rows['context'], rows['disruptive'], rows['tail'], scale)
    return rows, match, alarms


def view(pre, key, state, weights):
    return direct.feature_view(pre, key, state, weights)


def fit(train, cold, beta, cutoff):
    """Protect true alarms and teach helpful directions on near-deadline evidence."""
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    groups = []
    for rows, matched, _ in (train, cold):
        pos, neg = matched[:2]
        _, inv, count = np.unique(rows['shot'][pos], return_inverse=True, return_counts=True)
        a = {k: torch.from_numpy(rows[k]).cuda() for k in ('pre','key', *TEACH)}
        table = torch.from_numpy(np.column_stack((pos, neg))).cuda()
        with torch.no_grad():
            old = a['pre'].tanh() @ beta
            hard = (old[table[:, 0]] <= old[table[:, 1]]).to(torch.float64)
        prob = (1.+hard) / torch.from_numpy(count[inv]).cuda()
        draws = torch.multinomial(prob, STEPS*BATCH, True, generator=rng).reshape(STEPS, BATCH)
        groups.append((a, table[draws]))
    alarm = cold[2]
    offset, p, k, s = (torch.from_numpy(np.concatenate([r[v] for r in alarm])).cuda()
                       for v in ('offset', 'pre', 'key', 'state'))
    lengths = [len(r['indices']) for r in alarm]
    d = torch.tensor([r['disruptive'] for r in alarm], device='cuda', dtype=torch.bool)
    b = float(np.log(cutoff/(1.-cutoff)))
    old_peaks = torch.stack([v.max() for v in (offset+p.tanh() @ beta).split(lengths)])
    priority = 1.+((old_peaks < b) == d).float()
    param = torch.nn.Parameter(torch.zeros(4, 10, direct.WIDTH, device='cuda'))
    opt = torch.optim.Adam([param], lr=.012)
    for j in range(STEPS):
        opt.zero_grad(set_to_none=True)
        weight = param*(RADIUS/param.norm().clamp_min(RADIUS))
        peaks = torch.stack([v.max() for v in (offset+view(p, k, s, weight) @ beta).split(lengths)])
        deadline = .5*((F.softplus((b-peaks[d])/.35)*priority[d]).mean()
                       + (F.softplus((peaks[~d]-b)/.35)*priority[~d]).mean())
        loss = .35*deadline + .012*weight.square().sum()
        for mult, (r, draws) in zip((.65, .35), groups, strict=True):
            z = draws[j]
            pre, key = r['pre'][z], r['key'][z]
            base = pre.tanh() @ beta
            scores = [view(pre, key, r[m][z], weight) @ beta for m in TEACH]
            delta = scores[0]-base
            sep = delta[:, 0]-delta[:, 1]
            same = (F.relu(-delta[:, 0]).square()+F.relu(delta[:, 1]).square()).mean()
            true_rank = scores[0][:, 0]-scores[0][:, 1]
            controls = torch.stack([z[:, 0]-z[:, 1] for z in scores[1:]])
            selectivity = F.softplus((controls-true_rank)/.25).mean()
            loss = loss + mult*(F.softplus(-sep/.35).mean()+.20*same+.65*selectivity)
        loss.backward()
        opt.step()
        if (j+1)%256 == 0:
            print(dict(stage='persistence_teach', steps=j+1, loss=float(loss.detach())), flush=True)
    w = param.detach()*(RADIUS/param.detach().norm().clamp_min(RADIUS))
    if not torch.isfinite(w).all():
        raise FloatingPointError('nonfinite persistence mapper')
    return w.cpu()


@torch.no_grad()
def concordance(pack, beta, mapper):
    rows, match, _ = pack
    a, b = match[:2]
    pre, key = (torch.from_numpy(rows[k]) for k in ('pre', 'key'))
    base = pre.tanh()@beta
    scores = dict(inherited=base, **{
        k: view(pre, key, torch.from_numpy(rows[k]), mapper)@beta for k in persistent.VIEWS})
    result = {}
    for k, z in scores.items():
        gains = z-base
        result[k] = dict(wins=int((z[a]>z[b]).sum()),
                         mean_rank_gain=float((gains[a]-gains[b]).mean()))
    return dict(pairs=len(a), disruptive_shots=len(np.unique(rows['shot'][a])),
                safe_donors=len(np.unique(rows['shot'][b])),
                official_positive_pairs=int(rows['official_positive'][a].sum()), result=result)
