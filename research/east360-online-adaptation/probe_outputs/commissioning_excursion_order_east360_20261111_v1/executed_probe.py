"""Does observed excursion order add information beyond its endpoints?"""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree
from scipy.special import expit, logit
import torch

from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .commissioning_transition_source_20261110 import shot_coordinates
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


GROUPS = np.asarray(lag.GROUP)
N_GROUPS = 4
FLOOR, PENALTY = .05, .10
CALIPERS = np.array([.50, .35, .35, .50, .20, .20])
FAMILIES = ('magnetic', 'position', 'radiation', 'loop_voltage')
EPSILON = 1e-6


def describe_excursion(reference, initial, follow):
    """Separate order-odd terms from an exactly invariant endpoint/motion view."""
    symmetric, odd = [], []
    before = initial - reference
    after = follow - reference[:, None]
    for group in range(N_GROUPS):
        use = GROUPS == group
        old, trajectory = before[:, use], after[:, :, use]
        amp = np.square(old).mean(1)
        # A small excitation has small influence; never divide by raw displacement alone.
        projection = np.tanh(((trajectory - old[:, None])*old[:, None]).mean(2)
                             / (amp[:, None]+FLOOR**2))
        radius = np.tanh(np.sqrt(np.square(trajectory).mean(2)))
        symmetric.append(np.column_stack((np.tanh(np.sqrt(amp)), radius[:, 2], projection[:, 2],
            .5*(radius[:, 0]+radius[:, 1]), np.abs(radius[:, 1]-radius[:, 0]),
            .5*(projection[:, 0]+projection[:, 1]), np.abs(projection[:, 1]-projection[:, 0]))))
        odd.append(np.column_stack((projection[:, 1]-projection[:, 0], radius[:, 1]-radius[:, 0])))
    amplitude = np.tanh(np.sqrt(np.column_stack((np.square(before).mean(1),
                                                 np.square(after[:, 2]).mean(1)))))
    return np.concatenate(symmetric, 1), np.concatenate(odd, 1), amplitude


def mature_views(x, observed):
    """Every row is timestamped at q+26 after seeing the whole 25-point block."""
    q = np.arange(lag.FIRST, len(x)-lag.H-1, lag.STRIDE, dtype=np.int64)
    observed = observed & np.isfinite(x)
    slices = ((-127, 64), (-7, 8), (1, 8), (9, 8), (17, 9))
    means, good = [], np.ones(len(q), bool)
    for offset, n in slices:
        indices = q[:, None] + offset + np.arange(n)
        good &= observed[indices].all(axis=(1, 2))
        means.append(np.where(observed[indices], x[indices], 0.).mean(axis=1, dtype=np.float64))
    ref, initial, *y = (v[good] for v in means)
    return q[good]+lag.H+1, *describe_excursion(ref, initial, np.stack(y, axis=1))


def disjoint_pairs(positive, safe, positive_shots, safe_shots, calipers):
    """Match the greatest number of disjoint shots, then minimize distance."""
    if not len(positive) or not len(safe):
        return np.empty((0, 2), dtype=np.int64)
    x, y = positive/calipers, safe/calipers
    _, a = np.unique(positive_shots, return_inverse=True)
    _, b = np.unique(safe_shots, return_inverse=True)
    n, m = a.max()+1, b.max()+1
    dummy = (len(calipers)+1.)*(n+1)
    cost = np.full((n, m+n), 2.*dummy)
    cost[:, m:] = dummy
    pair = np.full((n, m, 2), -1, dtype=np.int64)
    neighbors = cKDTree(y).query_ball_point(x, r=1.+1e-12, p=np.inf)
    for i, js in enumerate(neighbors):
        if not len(js):
            continue
        js = np.asarray(js)
        dx = np.abs(positive[i]-safe[js])
        ok = (dx <= calipers).all(1)
        js, dist = js[ok], np.square(dx[ok]/calipers).sum(1)
        order = np.lexsort((js, dist, b[js]))
        pick = order[np.unique(b[js[order]], return_index=True)[1]]
        take = pick[dist[pick] < cost[a[i], b[js[pick]]]]
        j = js[take]
        cost[a[i], b[j]] = dist[take]
        pair[a[i], b[j], 0] = i
        pair[a[i], b[j], 1] = j
    i, j = linear_sum_assignment(cost)
    keep = j < m
    out = pair[i[keep], j[keep]]
    if (out < 0).any():
        raise AssertionError('infeasible pair selected')
    return out


def pair_fit(positive, safe):
    """Shot-balanced pair logistic fit with a fixed ridge and no intercept."""
    scale = np.maximum(np.concatenate((positive, safe)).std(axis=0), .05)
    d = (positive-safe)/scale

    def objective(w):
        z = d@w
        return (np.logaddexp(0., -z).mean()+.5*PENALTY*np.square(w).sum(),
                -(d.T @ expit(-z))/len(d)+PENALTY*w)

    fit = minimize(objective, np.zeros(d.shape[1]), jac=True, method='L-BFGS-B',
                   options={'gtol': 1e-9, 'ftol': 1e-13})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'pair fit did not converge: {fit.message}')
    return fit.x/scale


def _view(signal, center, scale):
    x, available, phys, mask = shot_coordinates(signal, center, scale)
    t, sym, ordered, amp = mature_views(x, available)
    recent, n = _window_mean(phys, mask, t, -15, 16)
    before, m = _window_mean(phys, mask, t, -31, 16)
    ctx = np.tanh(np.column_stack((recent, 4.*(recent[:, 0]-before[:, 0]))))
    valid = (n >= 8).all(1) & (m[:, 0] >= 8)
    return t[valid], sym[valid], ordered[valid], amp[valid], ctx[valid]


def _table(raw, points, z):
    end, sym, ordered, amp, ctx = points
    return dict(shot=np.full(len(end), raw['shot']), endpoint=end, time=raw['times'][end],
                match=np.c_[z, ctx, amp], unordered=np.c_[z, ctx, sym], ordered=ordered)


def _stack(parts):
    return {k: np.concatenate([r[k] for r in parts], axis=0) for k in parts[0]}


@torch.no_grad()
def _source(ids, base, center, scale, role):
    rows = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        points = _view(raw, center, scale)
        if len(points[0]):
            x, m = lag._windows(base, raw, points[0])
            z = base(x, m)['logit'].cpu().numpy()
            rows.append(_table(raw, points, np.clip(z, logit(EPSILON), logit(1.-EPSILON))))
        if i == len(ids) or i % 450 == 0:
            print(dict(stage='collect_'+role, shots=i, usable=len(rows)), flush=True)
    return _stack(rows)


def _outcomes(rows, ids, device):
    advanced = io23.advanced_times() if device == 'J-TEXT' else {}
    disruptive, lead, horizon = np.zeros(len(rows['shot']), bool), np.full(len(rows['shot']), np.nan), {}
    truth = {}
    for sid in ids:
        truth[sid] = io23.read_outcome(device, sid)
        if truth[sid][0]:
            take = rows['shot'] == sid
            disruptive[take] = True
            lead[take] = np.round((truth[sid][1]-rows['time'][take])*1000, 9)
            horizon[sid] = io23.effective_horizon_ms(sid, advanced)[0] if device == 'J-TEXT' else 125.
    horizon_vector = np.asarray([horizon.get(i, 0.) for i in rows['shot']])
    return disruptive, lead, horizon_vector, truth


def _pairs(rows, outcome, deadline, *, selected=None):
    d, lead, horizon, _ = outcome
    keep = d & (lead > deadline) & (lead <= horizon+1e-6)
    if selected is not None:
        keep &= np.isin(rows['shot'], selected)
    a, b = np.flatnonzero(keep), np.flatnonzero(~d)
    p = disjoint_pairs(rows['match'][a], rows['match'][b], rows['shot'][a], rows['shot'][b], CALIPERS)
    pairs = np.column_stack((a[p[:, 0]], b[p[:, 1]]))
    return pairs, dict(positive_windows=len(a), safe_windows=len(b),
                       positive_shots=len(np.unique(rows['shot'][a])), safe_shots=len(np.unique(rows['shot'][b])))


def _score(rows, w0, w1):
    unordered, ordered = rows['unordered'], rows['ordered']
    n = unordered.shape[1]
    common, extra = unordered@w1[:n], ordered@w1[n:]
    return dict(parent=unordered[:, 0], unordered_fit=unordered@w0, ordered_true=common+extra,
                same_model_no_order=common, swap_first_two=common-extra)


def _evaluate(rows, pairs, coverage, scores):
    pairs = np.asarray(pairs)
    if not len(pairs):
        return dict(**coverage, pairs=0)
    positive, safe = pairs.T
    receipt = {name: np.asarray(values)[positive]-np.asarray(values)[safe] for name, values in scores.items()}
    metrics = {k: dict(wins=int((v > 0).sum()), ties=int((v == 0).sum()),
                      agreement=float(((v>0)+.5*(v==0)).mean()),
                      mean_pair_log_loss=float(np.logaddexp(0.,-v).mean())) for k,v in receipt.items()}
    # Descriptive paired shot bootstrap, not an independent deployment interval.
    x = (receipt['ordered_true'] > 0).astype(float)+.5*(receipt['ordered_true']==0)
    delta = {k: x-((receipt[k] > 0).astype(float)+.5*(receipt[k]==0))
             for k in ('unordered_fit', 'swap_first_two')}
    resample = np.random.default_rng(131).integers(len(pairs), size=(2000, len(pairs)))
    return dict(**coverage, pairs=len(pairs), scores=metrics,
                paired_agreement_delta={k: dict(point=float(v.mean()), descriptive_95=np.quantile(
                    v[resample].mean(axis=1), [.025,.975]).tolist()) for k,v in delta.items()},
                median_matching_gap=np.median(np.abs(rows['match'][positive]-rows['match'][safe]), 0).tolist(),
                matched=np.stack((rows['shot'][pairs],rows['endpoint'][pairs]),axis=2).tolist())


def _east(lock, ref, protocol, seal, previous, old_seal, center, scale):
    rows, prefix = [], {}
    for rank, sid in enumerate(lock['ids'], 1):
        name = f'{rank:04d}_{sid}.npz'
        path, old = ref/'shots'/name, previous/'shots'/name
        if (sha(path) != seal['curve_sha256'][name] or sha(old) != old_seal['curve_sha256'][name]
                or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]):
            raise ValueError('sealed clock, baseline or signal changed')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        signal = lag.prepare_actual(raw, rank, lock['mapping'])
        points = _view(signal, center, scale)
        with np.load(path, allow_pickle=False) as g, np.load(old, allow_pickle=False) as r:
            np.testing.assert_array_equal(raw['times'], g['times'])
            np.testing.assert_array_equal(raw['times'], r['times'])
            key = 'fixed' if rank <= protocol['startup'] else lag.BASES[-1]
            if rank > protocol['startup'] and not np.isin(points[0], g['release']).all():
                raise ValueError('new feature outside the agreed feedback clock')
            y = logit(np.clip(r[key][points[0]].astype(np.float64), EPSILON, 1.-EPSILON))
        if rank in (21,120,360):
            end = min(766, len(raw['times']))
            sub = {k: v[:end] for k,v in signal.items() if k in ('values', 'available')}
            take = points[0] < end
            for a,b in zip(_view(sub, center, scale), points, strict=True):
                np.testing.assert_array_equal(a, b[take])
            prefix[str(sid)] = end
        rows.append(_table(raw, points, y))
        if rank%120 == 0:
            print(dict(stage='east_features', shots=rank), flush=True)
    return _stack(rows), prefix


def _subset(table, mask):
    return {k: v[mask] for k,v in table.items()}


def _save(output, filename, table, **extra):
    with (output/filename).open('xb') as f:
        np.savez_compressed(f, **table, **extra)


def run(ref, output):
    lag.configure()
    started = time.monotonic()
    lock, roles = lag.load_inputs(), io23.source_roles()
    p, seal = (read(ref/k) for k in ('protocol.json', 'seal.json'))
    ids = lock['ids']
    previous = Path(p['previous_teacher'])
    if (p['scope'] != 'EAST360_SIGNED_POINT_VS_PATH_GUARD_DEVELOPMENT' or p['order'] != ids
            or p['source_sha256'] != PARENT_SHA256 or sha(previous/'seal.json') != p['previous_seal']
            or p['held_source'] != roles['val'] or len(seal['curve_sha256']) != 360
            or p['fit_roles'] != {'J-TEXT/train': roles['train'], 'EAST/cold': ids[:lag.STARTUP]}):
        raise ValueError('exact sealed EAST360 development stream required')
    old = read(previous/'seal.json')
    base, checkpoint = load_parent(DEFAULT_PARENT)
    frozen = state_digest(base)
    if (frozen != lock['baseline_protocol']['source_state_sha256']
            or checkpoint['threshold'] != p['original']):
        raise ValueError('frozen source changed')
    center, scale = (v.numpy().copy() for v in (base.center, base.scale))
    base.cuda()
    output.mkdir(parents=True)
    copyfile(Path(__file__), output/'executed_probe.py')
    write(output/'protocol.json', dict(scope='EAST360_SHADOW_EXCURSION_ORDER_NOT_ALARMS',
        reference=str(ref), reference_seal=sha(ref/'seal.json'), source_sha256=PARENT_SHA256,
        producer_sha256=sha(Path(__file__)), fit=roles['train'], held_source=roles['val'],
        east_order=ids, startup=lag.STARTUP, first_effect=lag.FIRST_EFFECT, stride=lag.STRIDE,
        first_two_followup_lengths=[8,8], unchanged_tail_length=9, calipers=CALIPERS.tolist(),
        caliper_keys=['frozen_logit','Ip','Bt','short_Ip_change','start_radius','terminal_radius'],
        l2_penalty=PENALTY, excursion_floor=FLOOR, families=FAMILIES,
        match='maximum disjoint shot pairs; minimum standardized distance within calipers',
        fit_scope='J-TEXT TRAIN only; source label windows unchanged; strict >10 ms',
        held_scope='J-TEXT VAL at >10 and >30 ms; EAST (30,125] ms; no target fit',
        scoring='held endpoint rank comparison; no live alarm, no risk head change, no gradient update'))
    train = _source(roles['train'], base, center, scale, 'source_train')
    train_pairs, cov = _pairs(train, _outcomes(train, roles['train'], 'J-TEXT'), 10.)
    a,b = train_pairs.T
    u = train['unordered']
    x = np.c_[u,train['ordered']]
    w0,w1 = pair_fit(u[a],u[b]), pair_fit(x[a],x[b])
    _save(output, 'source_train.npz', train, pairs=train_pairs, unordered_weight=w0, ordered_weight=w1)
    table = dict(train_source_10=_evaluate(train, train_pairs, cov, _score(train,w0,w1)))
    print(dict(stage='fit_locked', pairs=len(train_pairs), seconds=round(time.monotonic()-started,2)), flush=True)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('held HDF5 outcomes are unsealed')):
        val = _source(roles['val'], base, center, scale, 'source_val')
        east, witnesses = _east(lock, ref, p, seal, previous, old, center, scale)
        for name, data in (('source_val.npz',val),('east_windows.npz',east)):
            _save(output, name, data, **{f'score_{k}':v for k,v in _score(data,w0,w1).items()})
    if state_digest(base) != frozen:
        raise ValueError('source model changed')
    write(output/'seal.json', dict(prefix_witnesses=witnesses, assets={k: sha(output/k) for k in
           ('protocol.json','source_train.npz','source_val.npz','east_windows.npz','executed_probe.py')}))
    val_truth = _outcomes(val, roles['val'], 'J-TEXT')
    for horizon in (10, 30):
        pairs,cov = _pairs(val,val_truth,horizon)
        table[f'held_source_{horizon}'] = _evaluate(val,pairs,cov,_score(val,w0,w1))
    old_shots = read(ref/'per_shot.json')['0.2'][lag.BASES[-1]][lag.STARTUP:]
    missed = [r['shot'] for r in old_shots if r['disruptive'] and not r['timely']]
    cold = np.isin(east['shot'], ids[:lag.STARTUP])
    for name, keep, cohort in (('east_cold',cold,ids[:lag.STARTUP]),('east_suffix',~cold,ids[lag.STARTUP:])):
        data = _subset(east,keep)
        truth = _outcomes(data,cohort,'EAST')
        score = _score(data,w0,w1)
        pairs,cov = _pairs(data,truth,30.)
        table[name] = dict(**_evaluate(data,pairs,cov,score), cohort=len(cohort),
                           n_disruptive=sum(v[0] for v in truth[3].values()))
        if name == 'east_suffix':
            pairs,cov = _pairs(data,truth,30.,selected=missed)
            table['east_legacy_misses'] = dict(**_evaluate(data,pairs,cov,score), cohort=len(missed))
    write(output/'result.json', dict(status='COMPLETED_EXCURSION_ORDER_SHADOW',
          held_suffix_trained=False, seconds=time.monotonic()-started, comparisons=table))
    print({k:{n:v[n] for n in ('pairs','scores','median_matching_gap') if n in v}
           for k,v in table.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    run(cli.reference.resolve(),cli.output.resolve())
