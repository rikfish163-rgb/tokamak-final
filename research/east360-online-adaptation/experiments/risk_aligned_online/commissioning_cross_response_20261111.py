"""Shot-disjoint conditional cross-diagnostic forecast probe, not an alarm model."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import io23
from .commissioning_east_observability_20261111 import (
    FAMILIES, FIRST, STRIDE, WINDOW, write,
)


CHANNELS = ('ip', 'Bt', *FAMILIES['motion'], *FAMILIES['magnetic'])
MOTION, MAGNETIC = (2, 3), (4, 5)
DIRECTIONS = ('magnetic_to_motion', 'motion_to_magnetic')
ARMS = ('persistence', 'own', 'real', 'lagged')
LAGS = (32, 64, 96)
MAX_CONTEXT_DISTANCE = 1.
RIDGE = .02


def _history(x, m, q):
    j = q[..., None] + np.arange(-31, 1)
    h = x[j].reshape(*q.shape, 4, 8, x.shape[1]).mean(-2)
    c = np.stack((h[..., -1, 0], h[..., -1, 1],
                  h[..., -1, 0]-h[..., 0, 0]), axis=-1)
    return h, c, m[j].all(axis=(-2, -1))


def predictors(x, mask, q, target):
    """Issue features using only indices at or before each forecast origin."""
    source = MAGNETIC if target == MOTION else MOTION
    h, c, present = _history(x, mask, q)
    candidates = q[:, None] - np.asarray(LAGS)
    prior, c_prior, prior_present = _history(x, mask, candidates)
    d = np.linalg.norm(c_prior-c[:, None], axis=-1)
    d = np.where(prior_present, d, np.inf)
    chosen = d.argmin(1)
    row = np.arange(len(q))
    distance = d[row, chosen]
    return dict(q=q, own=np.column_stack((c, h[..., target].reshape(len(q), 8))),
                real=h[..., source].reshape(len(q), 8),
                lagged=prior[row, chosen][..., source].reshape(len(q), 8),
                anchor=h[:, -1, target], context=c, lag=np.asarray(LAGS)[chosen],
                distance=distance, prefix_present=present,
                support=present & (distance <= MAX_CONTEXT_DISTANCE))


def observations(x, mask, issued, target):
    """Score only complete observed targets, with the same mask in all arms."""
    q = issued['q']
    mature = q+WINDOW+1 < len(x)
    use = np.flatnonzero(mature & issued['support'])
    j = q[use, None] + np.arange(1, WINDOW+1)
    valid = mask[j][..., target].all(axis=(-2, -1))
    use = use[valid]
    y = x[j[valid]][..., target]-issued['anchor'][use, None]
    return dict(use=use, target=y.reshape(len(use), 2*WINDOW),
                mature=int(mature.sum()), supported=int((mature & issued['support']).sum()))


def _location_scale(x, w):
    center = w @ x
    return center, np.maximum(1.e-3, np.sqrt(w @ np.square(x-center)))


def _linear(x, y, w, regularization=RIDGE):
    b = w @ y
    a = x.T * w
    coef = np.linalg.solve(a @ x + regularization*np.eye(x.shape[1]), a @ (y-b))
    return b, coef


def fit(own, real, lagged, y, shots):
    _, ix, counts = np.unique(shots, return_inverse=True, return_counts=True)
    w = 1./(len(counts)*counts[ix])
    loc, scale = _location_scale(own, w)
    x = (own-loc)/scale
    b, B = _linear(x, y, w)
    err = y-(b+x@B)
    results = dict(loc=loc, scale=scale, b=b, B=B)
    for name, raw in (('real', real), ('lagged', lagged)):
        d0, D = _linear(x, raw, w)
        residual = raw-(d0+x@D)
        r0, rs = _location_scale(residual, w)
        b0, C = _linear((residual-r0)/rs, err, w)
        results.update({name+'_'+k: v for k, v in
                        dict(d0=d0, D=D, r0=r0, rs=rs, b0=b0, C=C).items()})
    return results


def forecasts(fit_values, own, real, lagged):
    f = fit_values
    x = (own-f['loc'])/f['scale']
    y = f['b']+x@f['B']
    joint = []
    for name, raw in (('real', real), ('lagged', lagged)):
        d0, D, r0, rs, b0, C = (f[name+'_'+k] for k in ('d0', 'D', 'r0', 'rs', 'b0', 'C'))
        residual = (raw-(d0+x@D)-r0)/rs
        joint.append(y+b0+residual@C)
    return np.stack((np.zeros_like(y), y, *joint), axis=1)


def load_signal(sid):
    s = io23.read_signals('EAST', sid, rate_contract='nominal')
    ix = [io23.RAW23.index(k) for k in CHANNELS]
    x = s['v'][:, ix].astype(np.float64)
    x[:, :2] = abs(x[:, :2])
    return x, s['a'][:, ix], s['times']


def coordinates(shots):
    samples = np.concatenate([np.where(m[::32], x[::32], np.nan) for x, m, _ in shots])
    p, center, q = np.nanquantile(samples, [.25, .5, .75], axis=0)
    if not np.isfinite(center).all():
        raise ValueError('no normal reference measurements for one or more selected channels')
    return center, np.maximum(q-p, 1.e-6*np.maximum(abs(center), 1.))


def _rows(signal, loc, scale, group):
    x, mask, t = signal
    z = np.where(mask, np.arcsinh((x-loc)/scale), 0.)
    issued = predictors(z, mask, np.arange(FIRST, len(z), STRIDE), group)
    obs = observations(z, mask, issued, group)
    ix = obs['use']
    q = issued['q'][ix]
    later = np.median(x[q[:, None]+np.arange(-7, 1), 0], axis=1)
    earlier = np.median(x[q[:, None]+np.arange(-31, -23), 0], axis=1)
    return dict(own=issued['own'][ix], real=issued['real'][ix], lagged=issued['lagged'][ix],
                release=t[q+WINDOW+1], q=q, context=issued['context'][ix],
                lag=issued['lag'][ix], distance=issued['distance'][ix],
                decreasing=later < .99*earlier, y=obs['target']), obs['mature'], obs['supported']


def fit_normal(signals):
    loc, scale = coordinates(signals.values())
    fits = {}
    for direction, group in zip(DIRECTIONS, (MOTION, MAGNETIC), strict=True):
        rows = {s: _rows(v, loc, scale, group)[0] for s, v in signals.items()}
        shot = np.concatenate([np.full(len(r['q']), s) for s, r in rows.items()])
        fits[direction] = fit(*[np.concatenate([r[k] for r in rows.values()])
                                for k in ('own', 'real', 'lagged', 'y')], shot)
    return loc, scale, fits


def score_shot(normal, signal, sid):
    loc, scale, fits = normal
    parts = {}
    for direction, group in zip(DIRECTIONS, (MOTION, MAGNETIC), strict=True):
        r, mature, supported = _rows(signal, loc, scale, group)
        yhat = forecasts(fits[direction], r['own'], r['real'], r['lagged'])
        e = abs(r['y'][:, None]-yhat)
        h = np.where(e <= 1., .5*e**2, e-.5)
        parts[direction] = dict(q=r['q'], release=r['release'], lag=r['lag'],
                                distance=r['distance'], context=r['context'], decreasing=r['decreasing'],
                                shot=np.full(len(r['q']), sid), huber=h.mean(-1), mae=e.mean(-1))
    return parts, dict(shot=sid, mature=mature, paired_origin=supported, scored=len(r['q']))


def _summary(rows, take):
    ix = np.flatnonzero(take)
    shots = np.unique(rows['shot'][ix])
    if not len(shots):
        return dict(shots=0, windows=0)
    summary = dict(shots=len(shots), windows=len(ix))
    for metric in ('huber', 'mae'):
        v = np.stack([rows[metric][ix[rows['shot'][ix] == sid]].mean(0) for sid in shots])
        avg = v.mean(0)
        out = dict(mean=dict(zip(ARMS, avg.tolist(), strict=True)))
        for k in ('own', 'lagged', 'persistence'):
            diff = v[:, ARMS.index(k)]-v[:, ARMS.index('real')]
            b = np.random.default_rng(131).choice(diff, size=(4000, len(diff))).mean(1)
            out['real_vs_'+k] = dict(
                mean_reduction=float(diff.mean()), median_reduction=float(np.median(diff)),
                improved_shots=int((diff > 0.).sum()),
                descriptive_shot_bootstrap95=np.quantile(b, [.025, .975]).tolist())
        summary[metric] = out
    return summary


def metrics(sealed, cold_safe, outcomes):
    by_direction = {}
    for direction, rows in sealed.items():
        is_cold = np.isin(rows['shot'], list(cold_safe))
        dis = np.array([outcomes[int(s)][0] if int(s) in outcomes else False for s in rows['shot']])
        down = np.array([outcomes[int(s)][1] if d else np.nan
                         for s, d in zip(rows['shot'], dis, strict=True)])
        lead = (down-rows['release'])*1000.
        safe = ~is_cold & ~dis
        cohorts = dict(cold_safe_leave_one_out=is_cold, suffix_safe=safe,
                       suffix_safe_decreasing=safe & rows['decreasing'],
                       suffix_safe_not_decreasing=safe & ~rows['decreasing'],
                       suffix_disruptive_early=dis & (lead > 125.+1.e-6),
                       suffix_disruptive_30_125=dis & (lead > 30.+1.e-6) & (lead <= 125.+1.e-6),
                       suffix_disruptive_0_30=dis & (lead > 0.) & (lead <= 30.+1.e-6))
        by_direction[direction] = {k: _summary(rows, v) for k, v in cohorts.items()}
    return by_direction


def run(reference, output):
    start = time.monotonic()
    blob = reference.read_bytes()
    p = json.loads(blob)
    order = p['order']
    if (p['scope'] != 'EAST360_MEASUREMENT_COVERAGE_NOT_A_PREDICTION_EXPERIMENT'
            or len(set(order)) != len(order) or len(order) != 360 or p['startup'] != 20):
        raise ValueError('this probe is bound to the fixed EAST360 measurement audit')
    cold = {s: io23.read_outcome('EAST', s) for s in order[:20]}
    normals = {s: load_signal(s) for s in cold if not cold[s][0]}
    if len(normals) != 12:
        raise ValueError('the ended preparation prefix must have 12 non-disruptive shots')
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(
        status='EAST360_CROSS_DIAGNOSTIC_SIGNAL_DEVELOPMENT_NO_ALARMS',
        reference=str(reference), reference_sha256=hashlib.sha256(blob).hexdigest(),
        code_sha256={Path(k).name: hashlib.sha256(Path(k).read_bytes()).hexdigest()
                     for k in (__file__, io23.__file__)},
        order=order, startup=20, normal_fit=list(normals), lags=LAGS, context_max=MAX_CONTEXT_DISTANCE,
        directions=DIRECTIONS, primary_direction=DIRECTIONS[0], arms=ARMS, channels=CHANNELS,
        first=FIRST, horizon=WINDOW, stride=STRIDE, ridge=RIDGE,
        scaling='asinh of cold-normal per-channel median/IQR; within model weighted feature standardization',
        controls='matched earlier history chosen from three within-shot lag windows, identical fit capacity',
        evaluation='paired complete targets; each shot equal; cold safe leave-one-shot-out',
        event_labels_read=False, suffix_fit=False, risk_prediction=False, cross_shot_updates=0))
    full = fit_normal(normals)
    loc, scale, fits = full
    np.savez_compressed(output/'normal_fit.npz', loc=loc, scale=scale,
                        **{f'{d}_{k}': v for d, f in fits.items() for k, v in f.items()})
    all_parts = {d: [] for d in DIRECTIONS}
    coverage, witnesses = [], {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('forecast records not sealed')):
        for i, sid in enumerate(order):
            if i < 20 and sid not in normals:
                # The 8 disrupted cold shots are not a normal reference or a test cohort here.
                continue
            if sid in normals:
                normal = fit_normal({s: v for s, v in normals.items() if s != sid})
                signal = normals[sid]
            else:
                normal, signal = full, load_signal(sid)
            parts, counted = score_shot(normal, signal, sid)
            coverage.append(counted)
            for d, block in parts.items():
                all_parts[d].append(block)
            if i in (20, 119, 359):
                short = tuple(v[:min(380, len(signal[0]))] for v in signal)
                prefix, _ = score_shot(normal, short, sid)
                for d, part in prefix.items():
                    n = len(part['q'])
                    if not n:
                        raise ValueError('empty real prefix witness')
                    for k, x in part.items():
                        np.testing.assert_allclose(x, parts[d][k][:n], atol=1.e-12, rtol=0.)
                    witnesses[f'{sid}_{d}'] = n
            if (i+1) % 60 == 0:
                print(dict(stage='measurements', through=i+1, elapsed=time.monotonic()-start), flush=True)
    sealed = {d: {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
              for d, pieces in all_parts.items()}
    np.savez_compressed(output/'issued_error_seal.npz',
                        **{f'{d}_{k}': v for d, r in sealed.items() for k, v in r.items()})
    seal_digest = hashlib.sha256((output/'issued_error_seal.npz').read_bytes()).hexdigest()
    outcomes = {s: io23.read_outcome('EAST', s) for s in order[20:]}
    if sum(d for d, _ in outcomes.values()) != 172:
        raise ValueError('EAST suffix must have the fixed 172 disruptive outcomes')
    write(output/'result.json', dict(
        status='COMPLETE_SIGNAL_PROBE_NOT_ALARM_EVIDENCE', score_seal_sha256=seal_digest,
        scores=metrics(sealed, normals, outcomes), coverage=coverage,
        prefix_replay=witnesses, seconds=time.monotonic()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_protocol.resolve(), args.output.resolve())
