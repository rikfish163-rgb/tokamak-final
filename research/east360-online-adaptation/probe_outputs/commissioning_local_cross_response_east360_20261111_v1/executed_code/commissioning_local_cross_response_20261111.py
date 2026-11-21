"""Shot-reset identification of conditional short-horizon diagnostic responses."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import commissioning_cross_response_20261111 as fixed, io23


ARMS = ('persistence', 'own', 'real', 'lagged_source', 'late_feedback')
RETENTION, RIDGE, ENERGY_FLOOR = .875, .25, 1.e-4


def _changes(block):
    x = block.reshape(len(block), 4, 2)
    return np.column_stack((x[:, 3]-x[:, 2], x[:, 3]-x[:, 0]))


def replay(own, real, lagged, y, observed):
    """Predict from prior completed blocks; never learn from the issued block.

    On the fixed grid the previous target ends 7 points before the current
    origin. The deliberate late-feedback arm uses the second previous block.
    All four regressors begin with the same zero-prediction prior.
    """
    aligned = np.column_stack((own, real))
    keys = (own, aligned, np.column_stack((own, lagged)), aligned)
    cov = [np.zeros((x.shape[1], x.shape[1])) for x in keys]
    moment = [np.zeros((x.shape[1], y.shape[1])) for x in keys]
    prediction = np.zeros((len(y), len(ARMS), y.shape[1]))
    leverage = np.zeros((len(y), len(keys)))
    for i in range(len(y)):
        for a, (x, G, F) in enumerate(zip(keys, cov, moment, strict=True)):
            G *= RETENTION
            F *= RETENTION
            j = i-(2 if a == 3 else 1)
            if j >= 0 and observed[j]:
                G += np.outer(x[j], x[j])
                F += np.outer(x[j], y[j])
            precision = G + RIDGE * np.diag(np.maximum(np.diag(G), ENERGY_FLOOR))
            scaled_key = np.linalg.solve(precision, x[i])
            confidence = float(x[i]@scaled_key)
            leverage[i, a] = confidence
            prediction[i, a+1] = (scaled_key@F)/np.sqrt(max(1., confidence))
    return prediction, leverage


def score_shot(signal, coords, sid):
    x, mask, t = signal
    center, scale = coords
    z = np.where(mask, np.arcsinh((x-center)/scale), 0.)
    q = np.arange(fixed.FIRST, len(z), fixed.STRIDE)
    result = {}
    for direction, group in zip(fixed.DIRECTIONS, (fixed.MOTION, fixed.MAGNETIC), strict=True):
        issued = fixed.predictors(z, mask, q, group)
        obs = fixed.observations(z, mask, issued, group)
        ix = obs['use']
        target = np.zeros((len(q), 2*fixed.WINDOW))
        present = np.zeros(len(q), dtype=bool)
        present[ix], target[ix] = True, obs['target']
        own = np.column_stack((issued['context'][:, 2], _changes(issued['own'][:, 3:])))
        hat, leverage = replay(own, _changes(issued['real']), _changes(issued['lagged']), target, present)
        delta = abs(hat[ix]-target[ix, None])
        past_ip, last_ip = (np.median(x[q[ix, None]+np.arange(a, b), 0], axis=1)
                            for a, b in ((-31,-23), (-7,1)))
        result[direction] = dict(
            q=q[ix], release=t[q[ix]+fixed.WINDOW+1], shot=np.full(len(ix), sid),
            huber=np.where(delta <= 1., .5*delta**2, delta-.5).mean(-1),
            mse=np.square(delta).mean(-1), decreasing=last_ip < .99*past_ip,
            leverage=leverage[ix], update_before_issue=np.any(hat[ix,1:] != 0., axis=-1))
    return result


def describe(r, mask, archived):
    shots = np.unique(r['shot'][mask])
    if not len(shots):
        return dict(shots=0, windows=0)
    choices = [np.flatnonzero(mask & (r['shot'] == s)) for s in shots]
    result = dict(shots=len(shots), windows=int(mask.sum()),
                  real_nonzero_windows=int(r['update_before_issue'][mask,1].sum()),
                  median_real_leverage=float(np.median(r['leverage'][mask,1])))
    for k in ('huber', 'mse'):
        per = np.stack([r[k][ix].mean(0) for ix in choices])
        by = dict(mean=dict(zip(ARMS, per.mean(0).tolist(), strict=True)))
        comparators = {arm: per[:,i] for i,arm in enumerate(ARMS) if arm != 'real'}
        if k == 'huber':
            comparators['cold_real'] = np.array([archived[ix, fixed.ARMS.index('real')].mean()
                                                  for ix in choices])
            by['mean']['cold_real'] = float(comparators['cold_real'].mean())
        for arm, v in comparators.items():
            delta = v-per[:,ARMS.index('real')]
            rng = np.random.default_rng(131)
            boot = rng.choice(delta,size=(4000,len(delta))).mean(1)
            by['real_vs_'+arm] = dict(
                reduction=float(delta.mean()), better_shots=int((delta>0.).sum()),
                descriptive_shot_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        result[k] = by
    return result


def summarize(sealed, previous, cold_safe, outcomes):
    result = {}
    for d, r in sealed.items():
        cold = np.isin(r['shot'], list(cold_safe))
        dis = np.array([outcomes[int(s)][0] if int(s) in outcomes else False for s in r['shot']])
        down = np.array([outcomes[int(s)][1] if y else np.nan for s,y in zip(r['shot'],dis,strict=True)])
        lead = (down-r['release'])*1000.
        safe = ~cold & ~dis
        scopes = dict(cold_safe_leave_one_out=cold, suffix_safe=safe,
                      suffix_safe_decreasing=safe & r['decreasing'],
                      suffix_disruptive_early=dis & (lead>125.+1.e-6),
                      suffix_disruptive_30_125=dis & (lead>30.+1.e-6) & (lead<=125.+1.e-6),
                      suffix_disruptive_0_30=dis & (lead>0.) & (lead<=30.+1.e-6))
        result[d] = {k: describe(r, sel, previous[d+'_huber']) for k, sel in scopes.items()}
    return result


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(reference, output):
    started = time.monotonic()
    p, r = (json.loads((reference/k).read_bytes()) for k in ('protocol.json','result.json'))
    if (p['status'] != 'EAST360_CROSS_DIAGNOSTIC_SIGNAL_DEVELOPMENT_NO_ALARMS'
            or r['score_seal_sha256'] != _sha(reference/'issued_error_seal.npz')
            or p['code_sha256'][Path(fixed.__file__).name] != _sha(Path(fixed.__file__))):
        raise ValueError('fixed cold comparison must match the sealed signal probe')
    order, startup = p['order'], p['startup']
    with np.load(reference/'issued_error_seal.npz', allow_pickle=False) as f:
        previous = {k:f[k] for k in f.files}
    with np.load(reference/'normal_fit.npz', allow_pickle=False) as f:
        coords = f['loc'], f['scale']
    cold = {s: io23.read_outcome('EAST',s) for s in order[:startup]}
    cold_safe = [s for s,(d,_) in cold.items() if not d]
    if cold_safe != p['normal_fit']:
        raise ValueError('cold safe outcomes changed')
    normals = {s: fixed.load_signal(s) for s in cold_safe}
    output.mkdir(parents=True)
    fixed.write(output/'protocol.json',dict(
        scope='SHOT_RESET_LOCAL_CROSS_RESPONSE_SIGNAL_TEST_NO_RISK',
        reference=str(reference), assets={k:_sha(reference/k) for k in
                                          ('protocol.json','issued_error_seal.npz','normal_fit.npz')},
        code_sha256={f.name:_sha(f) for f in map(Path,(__file__,fixed.__file__,io23.__file__))},
        arms=ARMS, order=order, startup=startup, scale_fit=cold_safe,
        first=fixed.FIRST, horizon=fixed.WINDOW, stride=fixed.STRIDE,
        keys='asinh last-minus-previous and last-minus-first 8-point means; own current slope',
        local_reference='past-block exponentially weighted ridge, no cold response weights',
        ridge=RIDGE, energy_floor=ENERGY_FLOOR, retention=RETENTION, leverage_cap=1.,
        same_shot_only=True, event_labels_read=False, risk_predictions_computed=False,
        matched_suffix=340))
    by_direction = {d: [] for d in fixed.DIRECTIONS}
    witnesses = {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for i, sid in enumerate(order):
            if i < startup and sid not in normals:
                continue
            if i < startup:
                scale = fixed.coordinates(v for s,v in normals.items() if s != sid)
                signal = normals[sid]
            else:
                scale, signal = coords, fixed.load_signal(sid)
            parts = score_shot(signal,scale,sid)
            for d, part in parts.items():
                sel = previous[d+'_shot']==sid
                for k in ('q','release','decreasing'):
                    np.testing.assert_array_equal(part[k],previous[d+'_'+k][sel])
                by_direction[d].append(part)
            if i in (20,119,359):
                short = score_shot(tuple(x[:380] for x in signal),scale,sid)
                for d, part in short.items():
                    n = len(part['q'])
                    if not n:
                        raise ValueError('empty local prefix witness')
                    for k,x in part.items():
                        np.testing.assert_array_equal(x, parts[d][k][:n])
                    witnesses[f'{sid}_{d}'] = n
            if (i+1)%60 == 0:
                print(dict(stage='local_response',through=i+1,seconds=time.monotonic()-started),flush=True)
    sealed = {d:{k:np.concatenate([p[k] for p in v]) for k in v[0]} for d,v in by_direction.items()}
    np.savez_compressed(output/'local_error_seal.npz',
                        **{f'{d}_{k}':v for d,r in sealed.items() for k,v in r.items()})
    digest = _sha(output/'local_error_seal.npz')
    outcomes = {s:io23.read_outcome('EAST',s) for s in order[startup:]}
    if sum(d for d,_ in outcomes.values()) != 172:
        raise ValueError('fixed EAST360 outcome contract changed')
    fixed.write(output/'result.json',dict(
        status='COMPLETE_LOCAL_SIGNAL_PROBE_NOT_ALARM_EVIDENCE', score_seal_sha256=digest,
        scores=summarize(sealed,previous,cold_safe,outcomes), prefix_replay=witnesses,
        seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference.resolve(),args.output.resolve())
