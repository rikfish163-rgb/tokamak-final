"""Direct waveform forecasting and deadline-aligned learning of a temporal adapter."""

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_ordinal_pairs_20261111 as pairs
from . import commissioning_weak_shape_data_20261111 as shape
from . import commissioning_weak_shape_source_20261111 as weak
from . import simple_data
from .commissioning_innovation_east_20261111 import fit_decoder
from .commissioning_transition_data_20261110 import _window_mean
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE, WIDTH


SHORT = 8
EARLY_EFFECT = FIRST + SHORT + 1
LATE_EFFECT = FIRST + HORIZON + 1
STEPS, SEED, BATCH = 512, 131, 128
MAP_RADIUS = 8.


def direct_targets(z, mask, q):
    """Predict all 25 departures from the previous 16-point observed mean."""
    z, mask = shape._signal(z, mask)
    recent, count = _window_mean(z, mask, q, -15, 16)
    future = q[:, None] + 1 + np.arange(HORIZON)
    indices = np.minimum(future, len(z) - 1)
    available = mask[indices] & (future < len(z))[..., None] & (count[:, None, :] >= 8)
    target = np.where(available, z[indices] - recent[:, None, :], 0.)
    return target.astype(np.float32), available


def early_phase(ends):
    phase = (ends - FIRST) % STRIDE
    return (ends >= EARLY_EFFECT) & (phase > SHORT) & (phase <= HORIZON)


def feature_view(pre, key, state, mapper):
    """Move the 24-dimensional temporal representation before its nonlinearity."""
    return (pre + torch.einsum('...k,...f,kfw->...w', key, state, mapper)).tanh()


def bounded_map(w):
    return w * (MAP_RADIUS / w.norm().clamp_min(MAP_RADIUS))


def paired_loss(split, delayed, fixed):
    gain = split - fixed
    advance = split - delayed
    separation = F.softplus(-(gain[..., 0] - gain[..., 1]) / .40).mean()
    timing = F.softplus(-(advance[..., 0] - advance[..., 1]) / .25).mean()
    protection = (F.relu(-gain[..., 0]).square() + F.relu(gain[..., 1]).square()).mean()
    return separation + .75 * timing + .20 * protection


@torch.no_grad()
def track(net, signal, center, scale, ends):
    z, m, context, cm = weak.shot_coordinates(signal, center, scale)
    q = np.arange(FIRST, len(z), STRIDE)
    indices = np.union1d(q, ends)
    x, coords = shape.inputs(z, m, context, cm, indices)
    all_pre = net.stem(torch.from_numpy(x))
    q_rows, end_rows = (np.searchsorted(indices, a) for a in (q, ends))
    pre = all_pre[end_rows].clone()
    pre[ends < FIRST] = 0.
    recent, nr = _window_mean(z, m, q, -15, 16)
    prior, np_ = _window_mean(z, m, q, -31, 16)
    velocity = np.where((nr >= 8) & (np_ >= 8), recent - prior, 0.)
    design = np.column_stack((all_pre[q_rows].tanh().numpy(), np.tanh(coords[q_rows]),
                              np.tanh(velocity))).astype(np.float32)
    cond = coords[end_rows]
    key = .5 * np.column_stack((np.ones(len(ends)), np.tanh(cond[:, 0]), np.tanh(cond[:, 1]),
                                np.tanh(4 * cond[:, 2]))).astype(np.float32)
    y, obs = direct_targets(z, m, q)
    return dict(q=q, design=design, target=y, observed=obs, length=len(z), ends=ends,
                pre=pre.numpy(), context=cond, matchable=x[end_rows, -3:].astype(bool).all(1), key=key)


def episode(net, raw, device, advanced, center, scale, *, signal=None):
    actual = simple_data.attach_outcome(raw, device, advanced)
    d, down, t = actual['is_disrupt'], actual['down'], raw['times']
    valid = actual['valid'] & early_phase(np.arange(len(t)))
    # Source anchors are before their official positive window; EAST cold
    # anchors are inside (30,125] ms, as auxiliary decision-time teaching.
    exclusion = actual['horizon'] if device == 'J-TEXT' else 0.
    ix = pairs.endpoints(t, valid, d, down, exclusion, first_effect=EARLY_EFFECT)
    tr = track(net, raw if signal is None else signal, center, scale, ix.ravel())
    if d:
        future = tr['q'][:, None] + 1 + np.arange(HORIZON)
        ix_valid = np.minimum(future, len(t) - 1)
        tr['observed'] &= actual['valid'][ix_valid, None]
    tail = (not d) & (t[ix[:, 1]] >= t[-1] - weak.TAIL_S) & (tr['context'][1::2, 2] < 0)
    return dict(tr, indices=ix, sid=raw['shot'], disruptive=d, falling_tail=tail)


def collect_source(net, ids, advanced, center, scale, role):
    tracks = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        if len(raw['times']) > FIRST:
            tracks.append(episode(net, raw, 'J-TEXT', advanced, center, scale))
        if i % 450 == 0 or i == len(ids):
            print(dict(stage=f'collect_{role}', shots=i, total=len(ids)), flush=True)
    return tracks


def normalized_target(track, scale):
    return np.clip(track['target'] / scale, -8., 8.).astype(np.float32)


def fit_forecast(train):
    safe = [r for r in train if not r['disruptive']]
    target = np.concatenate([r['target'] for r in safe]).reshape(-1, 10)
    observed = np.concatenate([r['observed'] for r in safe]).reshape(-1, 10)
    scale = np.array([max(.025, np.quantile(np.abs(v[m]), .75))
                      for v, m in zip(target.T, observed.T, strict=True)], np.float32)
    x, y, m = (np.concatenate(xs) for xs in zip(*(
        (r['design'], normalized_target(r, scale).reshape(-1, HORIZON * 10),
         r['observed'].reshape(-1, HORIZON * 10)) for r in train), strict=True))
    enough = m.any(1)
    decoder = fit_decoder(x[enough], y[enough], m[enough], ridge=.03)
    return decoder, scale, dict(training_origins=int(enough.sum()), training_shots=len(train),
                                training_observed_points=int(m.sum()))


def base_forecast(tr, decoder):
    """Single forward evaluation, all 25 points from a causal origin."""
    w, b = decoder
    return (tr['design'] @ w.T + b).reshape(-1, HORIZON, 10)


def pack(tracks, decoder, scale, replay, match_scale):
    rows = []
    squared = np.zeros((2, 3), np.float64)
    for r in tracks:
        pred = base_forecast(r, decoder)
        y, mask = normalized_target(r, scale), r['observed']
        for i, sl in enumerate((slice(0, SHORT), slice(SHORT, HORIZON))):
            ref = y[:, sl][mask[:, sl]].astype(np.float64)
            err = pred[:, sl][mask[:, sl]].astype(np.float64) - ref
            squared[i] += np.array([len(ref), np.square(err).sum(), np.square(ref).sum()])
        keep = r['matchable'].reshape(-1, 2).all(1)
        if not keep.any():
            continue
        state, _ = replay(pred, y, mask, r['q'], r['length'])
        ix = r['indices'][keep]
        near = ix[:, 1]
        rows.append(dict(pre=r['pre'][1::2][keep], key=r['key'][1::2][keep],
                         context=r['context'].reshape(-1, 2, 7)[keep],
                         tail=r['falling_tail'][keep], disruptive=np.full(len(ix), r['disruptive']),
                         shot=np.full(len(ix), r['sid']), near=near,
                         **{k: v[near] for k, v in state.items()}))
    if not rows:
        raise ValueError('no matchable early-phase episodes')
    rows = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    m = pairs.match(rows['context'], rows['disruptive'], rows['tail'], match_scale)
    if not len(m[0]):
        raise ValueError('no timely disruption-safe matched pairs')
    return rows, m, squared


def pair_receipt(rows, matched):
    pos, neg, _, distance = matched
    return dict(pairs=len(pos), disruptive_shots=np.unique(rows['shot'][pos]).tolist(),
                safe_donors=np.unique(rows['shot'][neg]).tolist(), median_distance=float(np.median(distance)),
                nonidentical_latest_vs_delayed=int(np.any(rows['split'][pos] != rows['delayed'][pos], axis=1).sum()))


def teach(train, cold, beta):
    """Joint source and ended-cold teaching; all evaluation views share this map."""
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    data = []
    for rows, matched, _ in (train, cold):
        pos, neg = matched[:2]
        _, inv, n = np.unique(rows['shot'][pos], return_inverse=True, return_counts=True)
        sample = torch.from_numpy(1. / n[inv]).cuda()
        draws = torch.multinomial(sample, STEPS * BATCH, True, generator=rng).reshape(STEPS, BATCH)
        table = torch.from_numpy(np.column_stack((pos, neg))).cuda()
        data.append((draws, table, {k: torch.from_numpy(rows[k]).cuda()
                                     for k in ('pre', 'key', 'split', 'delayed')}))
    w = torch.nn.Parameter(torch.zeros(4, 20, WIDTH, device='cuda'))
    opt = torch.optim.Adam([w], lr=.012)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        mapper = bounded_map(w)
        loss = .012 * mapper.square().sum()
        for weight, (draws, table, r) in zip((.65, .35), data, strict=True):
            ids = table[draws[step]]
            pre, key, real, late = (r[k][ids] for k in ('pre', 'key', 'split', 'delayed'))
            y, delayed = (feature_view(pre, key, s, mapper) @ beta for s in (real, late))
            loss = loss + weight * paired_loss(y, delayed, pre.tanh() @ beta)
        loss.backward()
        opt.step()
        if (step + 1) % 256 == 0:
            print(dict(stage='deadline_fit', steps=step+1, loss=float(loss.detach())), flush=True)
    value = bounded_map(w).detach()
    if not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite latent adapter')
    return value.cpu()


@torch.no_grad()
def check_pairs(rows, matched, mapper, beta):
    p, n = matched[:2]
    pre, key = (torch.from_numpy(rows[k]) for k in ('pre', 'key'))
    h = pre.tanh()
    fixed = (h @ beta).numpy()
    out = dict(fixed=int((fixed[p] > fixed[n]).sum()))
    for mode in ('split', 'delayed', 'reversed'):
        effect = (feature_view(pre, key, torch.from_numpy(rows[mode]), mapper) - h) @ beta
        after = fixed + effect.numpy()
        out[mode] = dict(pairs_ordered=int((after[p] > after[n]).sum()),
                         mean_gain=float((effect[p] - effect[n]).mean()))
    return out
