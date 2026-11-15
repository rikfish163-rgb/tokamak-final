"""Predict the first jointly positive spectral record after a gap, without changing alarms."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from . import io23, simple_data
from .commissioning_coincident_support_audit_20261111 import _raw_truth
from .commissioning_event_roughness_20261110 import COLUMNS
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write


REFERENCE = io23.ROOT/'probe_outputs/commissioning_bias_dynamics_east360_20261112_v1'
PARENT = 'bias_only'
SPECTRUM = tuple(io23.RAW23.index(k) for k in ('Mir_avg_amp', 'Mir_avg_fre'))
OTHER = tuple(c for c in COLUMNS if c not in SPECTRUM)
AGES, H, STALE, MINIMUM = (0, 8), 25, 32, 64
ARMS = ('hold', 'history', 'stale', 'recent', 'recent_local')


def gap_landmarks(zero, positive, age, minimum=MINIMUM):
    """Index a gap only after a positive anchor and age+1 consecutive joint zeros."""
    start = np.flatnonzero(zero & np.r_[False, positive[:-1]])
    q = start + age
    admissible = (start >= minimum) & (q < len(zero))
    start, q = start[admissible], q[admissible]
    counts = np.r_[0, np.cumsum(zero)]
    observed = counts[q+1] - counts[start]
    usable = observed == age+1
    return start[usable], q[usable]


def _exits(zero, issue):
    points = np.flatnonzero(np.r_[~zero, True])
    return points[np.searchsorted(points, issue, side='right')]


def return_targets(zero, positive, issue, horizon=H):
    """Unknown exit and incomplete follow-up are not a known zero event."""
    issue = np.asarray(issue)
    end = _exits(zero, issue)
    deadline = issue + horizon
    event = np.where((end <= deadline) & np.r_[positive, False][end], end, -1)
    known = (event >= 0) | ((deadline < len(zero)) & (end > deadline))
    return event, known


def lag_matched_deltas(positive, log_spec, start, event):
    """Same lag from a previously observed all-positive span, for descriptive comparison."""
    edge = np.diff(np.r_[False, positive, False].astype(np.int8))
    lo, hi = np.flatnonzero(edge > 0), np.flatnonzero(edge < 0)-1
    out = np.full((len(start), 2), np.nan)
    for k in np.flatnonzero(event >= 0):
        lag = event[k] - start[k] + 1
        past = np.flatnonzero((hi < start[k]) & (hi-lo >= lag))
        if len(past):
            b = hi[past[-1]]
            out[k] = log_spec[b] - log_spec[b-lag]
    return out


def _diagnostics(x, seen, q):
    changes = []
    for width in (4, 16):
        pos = q[:, None] + np.arange(1-2*width, 1)
        mask = seen[pos].reshape(len(q), 2, width, len(OTHER))
        val = x[pos].reshape(mask.shape).astype(np.float64)
        n = mask.sum(2)
        mean = np.where(mask, val, 0.).sum(2)/n.clip(1)
        diff = mean[:, 1] - mean[:, 0]
        scale = np.abs(mean).sum(1)
        change = np.divide(diff, scale, out=np.zeros_like(diff), where=(scale>0.) & (n>0).all(1))
        changes.extend((change, n[:, 0]/width, n[:, 1]/width))
    return np.column_stack(changes)


def examples(values, available, age):
    spec = values[:, SPECTRUM]
    known = (available[:, SPECTRUM] & np.isfinite(spec)).all(1)
    zero, positive = known & (spec == 0.).all(1), known & (spec > 0.).all(1)
    start, q = gap_landmarks(zero, positive, age)
    log_spec = np.zeros(spec.shape, dtype=np.float64)
    np.log(spec.astype(np.float64), out=log_spec, where=positive[:, None])
    anchor = log_spec[start-1]
    history = [np.log1p(q)]
    for width in (4, 32):
        i = start[:, None] + np.arange(-width, 0)
        mask = positive[i]
        n = mask.sum(1)
        mu = (log_spec[i]*mask[..., None]).sum(1)/n[:, None].clip(1)
        history.extend((anchor-mu, n/width))
    x, seen = values[:, OTHER], available[:, OTHER]
    recent, stale = _diagnostics(x, seen, q), _diagnostics(x, seen, q-STALE)
    event, target_known = return_targets(zero, positive, q)
    delta = np.full((len(q), 2), np.nan)
    valid = event >= 0
    delta[valid] = log_spec[event[valid]] - anchor[valid]
    exit_point = _exits(zero, q)
    exit_kind = np.where(exit_point == len(zero), 0, np.where(np.r_[positive, False][exit_point], 1, 2))
    return dict(start=start, issue=q, event=event, known=target_known, exit_kind=exit_kind,
                history=np.column_stack(history), recent=recent, stale=stale, delta=delta,
                matched=lag_matched_deltas(positive, log_spec, start, event))


def matured_offsets(issue, event, delta, prediction):
    """Add a shrunk correction from past errors on up to eight observed returns."""
    indices = np.flatnonzero(event >= 0)
    indices = indices[np.argsort(event[indices], kind='stable')]
    n = np.searchsorted(event[indices], issue, side='left')
    prev = np.maximum(n-8, 0)
    residual = np.clip(delta[indices]-prediction[indices], -2., 2.)
    cumulative = np.concatenate((np.zeros((1, 2)), np.cumsum(residual, axis=0)))
    return (cumulative[n]-cumulative[prev])/(4+n-prev)[:, None]


def _design(rows, key):
    return rows['history'] if key == 'history' else np.column_stack((rows['history'], rows[key]))


def _weights(shot):
    _, groups, sizes = np.unique(shot, return_inverse=True, return_counts=True)
    return (len(shot)/len(sizes))/sizes[groups]


def fit(rows):
    known, returned = rows['known'], rows['event'] >= 0
    if not returned.any() or not np.any(known & ~returned):
        raise ValueError('training requires observed returns and complete windows with no return')
    w, v = (_weights(rows['shot'][m]) for m in (known, returned))
    prevalence = float(np.average(returned[known], weights=w))
    fitted = {}
    for key in ('history', 'stale', 'recent'):
        x = _design(rows, key)
        mu = np.average(x[known], axis=0, weights=w)
        sd = np.sqrt(np.average((x[known]-mu)**2, axis=0, weights=w)).clip(.05)
        z = (x-mu)/sd
        probability = LogisticRegression(C=1., max_iter=300).fit(z[known], returned[known], sample_weight=w)
        mark = Ridge(alpha=50.).fit(z[returned], rows['delta'][returned], sample_weight=v)
        fitted[key] = (mu, sd, probability, mark)
    return prevalence, fitted


def forecast(rows, fitted):
    count = len(rows['issue'])
    prior, heads = fitted
    if not count:
        return dict(prob=np.empty((0, len(ARMS))), marks=np.empty((0, len(ARMS), 2)))
    p = [np.full(count, prior)]
    d = [np.zeros((count, 2))]
    for key in ARMS[1:-1]:
        mu, sd, probability, mark = heads[key]
        z = (_design(rows, key)-mu)/sd
        p.append(probability.predict_proba(z)[:, 1])
        d.append(mark.predict(z))
    d.append(d[-1]+matured_offsets(rows['issue'], rows['event'], rows['delta'], d[-1]))
    return dict(prob=np.column_stack([*p, p[-1]]), marks=np.stack(d, axis=1))


def stack(rows):
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def _shot(raw, age):
    rows = examples(raw['values'], raw['available'], age)
    return dict(rows, shot=np.full(len(rows['issue']), raw['shot'], dtype=np.int64))


def _training(device, ids):
    grouped = {age: [] for age in AGES}
    for j, sid in enumerate(ids, 1):
        raw = simple_data.read_shot(device, sid)
        for age in AGES:
            grouped[age].append(_shot(raw, age))
        if j % 450 == 0:
            print(dict(stage='source_signals', processed=j), flush=True)
    return {a: fit(stack(rows)) for a, rows in grouped.items()}


def seal(out):
    start = time.monotonic()
    p = read(REFERENCE/'protocol.json')
    ids, cold_count = p['order'], p['startup']
    if len(ids) != 360 or len(set(ids)) != len(ids) or cold_count != 20:
        raise ValueError('reference does not match the locked cold-20/360 stream')
    roles = io23.source_roles()
    out.mkdir()
    copyfile(Path(__file__), out/'executed_probe.py')
    write(out/'protocol.json', dict(order=ids, cold_count=cold_count, source_train=roles['train'],
        source_validation=roles['val'], ages=AGES, horizon=H, minimum_start=MINIMUM, stale_lag=STALE,
        channels=[io23.RAW23[c] for c in OTHER], arms=ARMS, input='direct raw values, no interpolation',
        reference=str(REFERENCE), reference_seal=sha(REFERENCE/'seal.json'),
        source='fit only J-TEXT TRAIN signals; VAL excluded from fitting',
        east='issue source-only for cold20; separate later readout fits completed cold20 signals, no East outcomes',
        adaptation='same-shot past residuals from strictly earlier completed returns, local mark only',
        cutoff='known continuous zero to first jointly positive; unknown exit is censored',
        scope='observation probe; never changes alarms or a deployed weight'))
    saved, cold = ({age: [] for age in AGES} for _ in range(2))
    verified = []
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before seal')):
        source = _training('J-TEXT', roles['train'])
        for rank, sid in enumerate(roles['val'], 1):
            raw = simple_data.read_shot('J-TEXT', sid)
            for age in AGES:
                data = _shot(raw, age)
                pred = forecast(data, source[age])
                saved[age].append(_row(data, pred, pred, rank, device=0))
        for rank, sid in enumerate(ids, 1):
            if rank == cold_count+1:
                commissioned = {a: fit(stack(rows)) for a, rows in cold.items()}
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            for age in AGES:
                rows = _shot(raw, age)
                model = source[age] if rank <= cold_count else commissioned[age]
                src, adapted = forecast(rows, source[age]), forecast(rows, model)
                if rank <= cold_count:
                    cold[age].append(rows)
                if rank in (21, 120, 360) and len(rows['issue']):
                    end = min(len(raw['values']), int(rows['issue'][len(rows['issue'])//2])+1)
                    take = rows['issue'] < end
                    short = examples(raw['values'][:end], raw['available'][:end], age)
                    np.testing.assert_array_equal(short['issue'], rows['issue'][take])
                    for key, val in forecast(short, model).items():
                        np.testing.assert_allclose(val, adapted[key][take], rtol=0., atol=1e-11)
                    verified.append(dict(shot=sid, age=age, prefix_points=end, issue_count=int(take.sum())))
                saved[age].append(_row(rows, src, adapted, rank, device=1))
            if rank % 120 == 0:
                print(dict(stage='east_issued', rank=rank, seconds=round(time.monotonic()-start, 2)), flush=True)
    evidence = {f'{age}_{key}': val for age, rows in saved.items() for key, val in stack(rows).items()}
    with (out/'forecasts.npz').open('xb') as f:
        np.savez_compressed(f, **evidence)
    write(out/'seal.json', dict(forecasts=sha(out/'forecasts.npz'), protocol=sha(out/'protocol.json'),
        source_snapshot=sha(out/'executed_probe.py'), prefix_checks=verified, seconds=time.monotonic()-start))


def _row(data, source, available, rank, device):
    keep = ('shot', 'start', 'issue', 'event', 'known', 'exit_kind', 'delta', 'matched')
    n = len(data['issue'])
    return dict({k: data[k] for k in keep}, rank=np.full(n, rank), device=np.full(n, device),
                **{f'{f}_{k}': v for f, m in (('source', source), ('commissioned', available)) for k, v in m.items()})


def _shot_average(shot, values):
    keys, groups, counts = np.unique(shot, return_inverse=True, return_counts=True)
    result = np.zeros((len(keys), *values.shape[1:]))
    np.add.at(result, groups, values)
    return result / counts.reshape((-1,)+(1,)*(values.ndim-1))


def summarize(data, take, method):
    scored = take & data['known']
    returned = take & (data['event'] >= 0)
    positive = data['event'][scored] >= 0
    p = data[f'{method}_prob'][scored]
    score = _shot_average(data['shot'][scored], (p-positive[:, None])**2)
    errors = np.abs(data[f'{method}_marks'][returned]-data['delta'][returned, None])
    e = _shot_average(data['shot'][returned], errors)
    matched = returned & np.isfinite(data['matched']).all(1)
    pairs = np.stack((np.abs(data['delta'][matched]), np.abs(data['matched'][matched])), 1)
    continuity = _shot_average(data['shot'][matched], pairs)
    return dict(issued=int(take.sum()), observed=int(scored.sum()), returned=int(returned.sum()),
        shots=int(len(e)), never_seen_again=int((take & (data['exit_kind'] == 0)).sum()),
        ambiguous_exit=int((take & (data['exit_kind'] == 2)).sum()),
        brier={k: float(score[:, i].mean()) if len(score) else None for i, k in enumerate(ARMS)},
        log_mae={k: e[:, i].mean(0).tolist() if len(e) else None for i, k in enumerate(ARMS)},
        relative_error={k: _shot_average(data['shot'][returned], np.tanh(errors[:, i]/2)).mean(0).tolist()
                        if len(e) else None for i, k in enumerate(ARMS)},
        shots_beating_hold={k: (e[:, i]<e[:, 0]).sum(0).tolist() for i, k in enumerate(ARMS)},
        matched_rows=int(matched.sum()), matched_log_mae=continuity.mean(0).tolist() if len(continuity) else None)


def score(out):
    p, s = read(out/'protocol.json'), read(out/'seal.json')
    if (sha(out/'forecasts.npz') != s['forecasts'] or sha(out/'protocol.json') != s['protocol']
            or sha(REFERENCE/'seal.json') != p['reference_seal']):
        raise ValueError('forecast or warning evidence changed after seal')
    with np.load(out/'forecasts.npz', allow_pickle=False) as f:
        rows = {a: {k.split('_', 1)[1]: f[k] for k in f.files if k.startswith(f'{a}_')} for a in AGES}
    old = read(REFERENCE/'per_shot.json')['0.2'][PARENT]
    oldp, olds = read(REFERENCE/'protocol.json'), read(REFERENCE/'seal.json')
    late, false = {a: np.zeros(len(rows[a]['shot']), bool) for a in AGES}, {
        a: np.zeros(len(rows[a]['shot']), bool) for a in AGES}
    late_n, false_n = 0, 0
    for rank, (sid, archived) in enumerate(zip(p['order'], old, strict=True), 1):
        d, down, t = _raw_truth(sid)
        file = REFERENCE/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(file) != olds['curve_sha256'][file.name]:
            raise ValueError('changed parent curve')
        with np.load(file, allow_pickle=False) as f:
            np.testing.assert_array_equal(t, f['times'])
            v = f[PARENT]
        tau = oldp['original'] if rank <= p['cold_count'] else oldp['calibration']['0.2']
        found = dict(shot=sid, disruptive=d, down_s=down, **score_curve(t, v, d, down, tau))
        if found != archived:
            raise ValueError('warning archive no longer matches HDF5')
        if rank <= p['cold_count']:
            continue
        missed, fp = d and not found['timely'], not d and found['alarm']
        late_n += int(missed)
        false_n += int(fp)
        for a, data in rows.items():
            i = np.flatnonzero((data['device'] == 1) & (data['rank'] == rank) & (data['event'] >= 0))
            use = data['event'][i]+1
            keep = use < len(t)
            i, use = i[keep], use[keep]
            if missed:
                lead = np.round((down-t[use])*1000., 9)
                late[a][i] = (lead > 30.) & (lead <= 125.)
            if fp:
                false[a][i] = use <= np.flatnonzero(v >= tau)[0]
    result = {}
    for a, data in rows.items():
        src = data['device'] == 0
        suf = ~src & (data['rank'] > p['cold_count'])
        splits = dict(source_val=src, east_cold=~src & ~suf, east_suffix=suf,
                      late_timely_window=late[a], before_false_alarm=false[a])
        result[str(a)] = {name: {fit_role: summarize(data, m, fit_role) for fit_role in (
            ('source', 'commissioned') if name in ('east_suffix', 'late_timely_window', 'before_false_alarm') else ('source',))
            } for name, m in splits.items()}
        result[str(a)]['opportunity_ids'] = {name: np.unique(data['shot'][m]).tolist()
            for name, m in (('late_timely_window', late[a]), ('before_false_alarm', false[a]))}
    write(out/'result.json', dict(status='OBSERVATION_ONLY', verified_hdf5_shots=len(old),
        after_cold=dict(parent_late_or_missed=late_n, parent_false=false_n), landmarks=result,
        note='Positive-mark metrics condition on a jointly positive record in the next H samples; '
             'opportunity is not a rescue or an alarm rerun. A constant zero has relative error 1 on these marks.'))
    print(dict(stage='sealed_and_scored', verified=len(old), late_or_missed=late_n, false_alarms=false_n), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    target = parser.parse_args().out.resolve()
    seal(target)
    score(target)
