"""Teach the bounded transition learner with J-TEXT TRAIN, audit on VAL only."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .commissioning_event_pairs_20261110 import match
from .commissioning_event_roughness_20261110 import COLUMNS, CONTEXT
from .commissioning_roughness_20261110 import coordinates
from .commissioning_transfer_20261108 import shot_weights
from .commissioning_transition_net_20261110 import FIRST, MODES, STRIDE, TransitionNet
from .history_gradient_metric_20261102.run import sha, write
from .mask_repair_20261104.east_evaluation import configure
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


SEED, STEPS, BATCH = 131, 960, 256


def decision_points(y, valid):
    first = FIRST + 2*STRIDE
    legal = valid & (np.arange(len(y)) >= first)
    positive = np.flatnonzero(y & legal)
    sampled = np.arange(first, len(y), 48)
    return np.unique(np.r_[sampled[valid[sampled]], positive[::3], positive[-1:]])


def shot_coordinates(signal, center, scale):
    cols = COLUMNS + CONTEXT
    z, m = coordinates(signal['values'][:, cols], signal['available'][:, cols], center[cols], scale[cols])
    return z[:, :-2], m[:, :-2], z[:, -2:], m[:, -2:]


def collect(ids, advanced, center, scale, phase):
    from .commissioning_transition_data_20261110 import inputs, targets

    columns = []
    used = []
    for j, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        shot = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        q = decision_points(shot['y'].astype(bool), shot['valid'])
        if len(q):
            z, m, ctx, available = shot_coordinates(raw, center, scale)
            before = q[:, None] + np.array([-2*STRIDE, -STRIDE, 0])
            x = inputs(z, m, ctx, available, before.ravel()).reshape(len(q), 3, -1)
            y, obs = targets(z, m, before[:, :2].ravel())
            y, obs = (a.reshape((len(q), 2, *a.shape[1:])) for a in (y, obs))
            lead = (np.round((shot['down']-shot['times'][q])*1000., 9) if shot['is_disrupt']
                    else np.zeros(len(q), float))
            columns.append((x, y, obs, shot['y'][q], np.full(len(q), sid), q,
                            np.full(len(q), shot['is_disrupt']), lead))
            used.append(sid)
        if j % 300 == 0 or j == len(ids):
            print(dict(stage=f'collect_{phase}', shots=j, total=len(ids)), flush=True)
    if not columns:
        raise ValueError(f'no source episodes in {phase}')
    batch = dict(zip(('x', 'transition', 'observed', 'risk', 'shot', 'endpoint', 'disruptive', 'lead'),
                     (np.concatenate(v) for v in zip(*columns, strict=True)), strict=True))
    return batch, used


def episode(net, x, y, m):
    pre, key = net.design(x)
    s = net.initial(len(x))
    fixed = net.risk(pre[:, 2].tanh()).squeeze(-1)
    for i in range(2):
        s, loss = net.update(pre[:, i], key[:, i], s, y[:, i], m[:, i])
        if i == 0:
            first = loss
    adapted = net.risk(net.project(pre[:, 2], key[:, 2], s)).squeeze(-1)
    return fixed, adapted, first, loss


def train(batch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        net = TransitionNet(len(COLUMNS)).cuda()
    x, y, m, target = (torch.from_numpy(batch[k]).cuda() for k in ('x', 'transition', 'observed', 'risk'))
    w = torch.from_numpy(shot_weights(batch['shot'], batch['risk'].astype(bool))).cuda()
    generator = torch.Generator(device='cuda').manual_seed(SEED)
    draws = torch.multinomial(w, STEPS*BATCH, replacement=True, generator=generator).reshape(STEPS, BATCH)
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    for i, index in enumerate(draws, 1):
        opt.zero_grad(set_to_none=True)
        fixed, adapted, c0, c1 = episode(net, x[index], y[index], m[index])
        bf, ba = (F.binary_cross_entropy_with_logits(z, target[index]) for z in (fixed, adapted))
        loss = .5*(bf+ba) + .35*(c0+c1).mean()/2.
        loss.backward()
        opt.step()
        if i % 240 == 0:
            print(dict(stage='source_fit', step=i, fixed_bce=float(bf.detach()), fast_bce=float(ba.detach()),
                       auxiliary_ce=float(.5*(c0+c1).mean().detach())), flush=True)
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise FloatingPointError('nonfinite source network')
    return net.eval().requires_grad_(False)


@torch.no_grad()
def held(net, batch):
    from .commissioning_transition_data_20261110 import reverse_valid

    ys = batch['transition']
    ms = batch['observed']
    null = reverse_valid(ys.reshape((-1, *ys.shape[-2:])), ms.reshape((-1, *ms.shape[-2:]))).reshape(ys.shape)
    xt, yt, mt, nt = (torch.from_numpy(a).cuda() for a in (batch['x'], ys, ms, null))
    score = []
    for q in range(0, len(xt), 2048):
        x, y, m, rev = (a[q:q+2048] for a in (xt, yt, mt, nt))
        f, r, _, _ = episode(net, x, y, m)
        _, c, _, _ = episode(net, x, rev, m)
        score.append(torch.stack((f, r, c), -1).cpu().numpy())
    score = np.concatenate(score)
    w = shot_weights(batch['shot'], batch['risk'].astype(bool))
    bce = (np.logaddexp(0., score)-batch['risk'][:, None]*score).T @ w
    pos = []
    safe = []

    def row(i):
        ctx = np.r_[batch['x'][i, -1, :2], 0.]
        return dict(shot=int(batch['shot'][i]), endpoint=int(batch['endpoint'][i]),
                    family=0, context=ctx, feature=score[i])

    for s in np.unique(batch['shot']):
        which = batch['shot'] == s
        if batch['disruptive'][which].any():
            i = np.flatnonzero(which & (batch['risk'] > 0) & (batch['lead'] > 10.))
            if len(i):
                lead = batch['lead'][i]
                pos.append(row(i[np.abs(lead-np.median(lead)).argmin()]))
        else:
            safe.extend(row(i) for i in np.flatnonzero(which))
    pairs = match(pos, safe)
    if not pairs:
        raise ValueError('no held operating-context matched pairs')
    margin = np.stack([d['feature']-s['feature'] for d, s in pairs])
    return dict(scope='JTEXT_VAL160_SAMPLED_TWO_STEP_DIAGNOSTIC_NOT_TEST220', rows=len(score),
                eligible_positive_shots=len(pos), matched=len(pairs),
                unique_matched_safe=len({s['shot'] for _, s in pairs}),
                weighted_bce=dict(zip(MODES, bce.tolist(), strict=True)),
                matched_wins=dict(zip(MODES, (margin>0).sum(0).tolist(), strict=True)),
                matched_ties=dict(zip(MODES, (margin==0).sum(0).tolist(), strict=True)),
                same_pairs=[[r['shot'], r['endpoint'], s['shot'], s['endpoint']] for r, s in pairs])


def run(out):
    from . import commissioning_transition_data_20261110 as data
    from . import commissioning_transition_net_20261110 as model

    started = time.monotonic()
    configure()
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy() for v in (base.center, base.scale))
    roles = io23.source_roles()
    advanced = io23.advanced_times()
    out.mkdir(parents=True)
    recipe = dict(scope='SOURCE_TRAIN1353_AUXILIARY_ONLY_NO_TEST220',
                  parent_sha256=PARENT_SHA256, advanced_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
                  steps=STEPS, batch=BATCH, seed=SEED,
                  model_sha256=sha(Path(model.__file__)), data_sha256=sha(Path(data.__file__)),
                  producer_sha256=sha(Path(__file__)))
    write(out / 'protocol.json', recipe)
    batch, used = collect(roles['train'], advanced, center, scale, 'train')
    assert set(used).issubset(roles['train'])
    net = train(batch)
    receipt = dict(source_train_ids=used, train_rows=len(batch['x']), risk_positives=int(batch['risk'].sum()),
                   transition_counts=np.bincount(batch['transition'][batch['observed']], minlength=3).tolist())
    del batch
    hold, val_used = collect(roles['val'], advanced, center, scale, 'val')
    evidence = held(net, hold)
    if set(val_used) & set(used):
        raise ValueError('source validation entered fitting')
    with (out / 'model.npz').open('xb') as f:
        np.savez(f, **{k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    write(out / 'result.json', dict(**receipt, val_ids=val_used, held=evidence,
                                   model_sha256=sha(out / 'model.npz'), seconds=time.monotonic()-started))
    print({k:v for k,v in evidence.items() if k!='same_pairs'}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True, type=Path)
    run(cli.parse_args().output.resolve())
