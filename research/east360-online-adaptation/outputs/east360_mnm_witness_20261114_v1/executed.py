"""Test archived MNM as an auxiliary observation without exposing it to risk."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from time import monotonic

import numpy as np

from . import io23
from .history_gradient_metric_20261102.run import read, write


ROOT = io23.ROOT
REFERENCE = ROOT/'outputs/east360_opposed_feedback_20261114_v2'
COLUMNS = ('MNM', 'Mir_avg_amp', 'Mir_avg_fre', 'n=1 amplitude',
           'sxr_var', 'xuv_var', 'v_loop', 'ip', 'Bt')
RAW = [io23.RAW23.index(name) for name in COLUMNS]
MIR = (0, 1, 2)
OUT = (4, 5, 6)
FIRST, STRIDE, RELEASE, H = 127, 32, 39, 25
ARMS = ('hold', 'without_mnm', 'current', 'level_only', 'reversed', 'earlier')


def issued(x, valid, risk):
    """Complete MNM packets at the old handoff; no sample after r is used."""
    q = np.arange(FIRST, len(x)-RELEASE, STRIDE, dtype=np.int64)
    r = q+RELEASE
    f = q[:, None]+np.arange(1, 1+H)
    triple = valid[:, MIR]
    detected = triple.all(1) & (x[:, MIR]!=0.).all(1)
    active = detected[f]
    complete = active.all(1)
    # Candidate is an archived number, not a discrete mode class. Preserve
    # the common detection mask in every arm, and match on it before testing
    # whether the nonzero values themselves offer additional information.
    z = np.arcsinh(x.astype(np.float64))
    m = z[f, 0].reshape(-1, 5, 5).mean(2)
    m[~complete] = 0.
    previous = np.maximum.accumulate(np.r_[-1, np.where(complete, np.arange(len(q)), -1)])
    old = previous[np.searchsorted(q+H, q, side='right')]
    old_idx = np.maximum(old, 0)
    previous_at, previous_values = q[old_idx]+H, m[old_idx].copy()
    previous_values[old<0] = 0.
    previous_at[old<0] = -1
    indices = r[:, None]+np.arange(-47, 1)
    history = z[indices, 1:].reshape(-1, 3, 16, len(COLUMNS)-1).mean(2)
    supported = valid[indices, 1:].all((1, 2))
    gate = detected[indices].reshape(-1, 3, 16).mean(2)
    # Same frozen parent, elapsed record time, and recently observed
    # non-MNM diagnostics across matched comparisons.
    common = np.c_[history[:, -1], np.diff(history, axis=1).reshape(len(q), 2*(len(COLUMNS)-1)),
                   gate, risk[r], np.log1p(r)]
    return dict(q=q, r=r, features=common, available=complete & supported,
                records=complete, any_live=active.any(1), true=m, reversed=m[:, ::-1],
                level=np.repeat(m.mean(1, keepdims=True), 5, axis=1),
                earlier=previous_values, earlier_at=previous_at)


def mature(x, valid, r):
    """A directly forecast 25-point response can be scored only at r+26."""
    observed = valid[:, OUT]
    usable = np.flatnonzero(r+H+1<len(x))
    future = r[usable, None]+np.arange(1, H+1)
    recent = r[usable, None]+np.arange(-15, 1)
    good = observed[future].all((1, 2)) & observed[recent].all((1, 2))
    value = np.arcsinh(x[:, OUT].astype(np.float64))
    target = np.zeros((len(r), 5*len(OUT)))
    out_valid = np.zeros(len(r), bool)
    out_valid[usable[good]] = True
    future = value[future[good]].reshape(-1, 5, 5, len(OUT)).mean(2)
    center = value[recent[good]].mean(1)
    target[out_valid] = (future-center[:, None]).reshape(-1, 5*len(OUT))
    return target, out_valid


def model(x, y, shot):
    _, j, number = np.unique(shot, return_inverse=True, return_counts=True)
    weight = 1./(len(number)*number[j])
    center = weight@x
    scale = np.sqrt(np.maximum(weight@np.square(x-center), 1e-10))
    z = np.clip((x-center)/scale, -5., 5.)
    z = np.c_[z, np.ones(len(z))]
    a = z.T@(weight[:, None]*z)+.1*np.diag(np.r_[np.ones(x.shape[1]), 0.])
    return center, scale, np.linalg.solve(a, z.T@(weight[:, None]*y))


def predict(x, fit):
    center, scale, weight = fit
    z = np.clip((x-center)/scale, -5., 5.)
    return np.c_[z, np.ones(len(z))]@weight


def observed(sid, archive):
    z = io23.read_signals('EAST', sid, rate_contract='nominal')
    # io23 returns finite-cleaned values together with the availability mask.
    x, m = z['v'][:, RAW], z['a'][:, RAW]
    with np.load(archive, allow_pickle=False) as f:
        risk = f['bias_only']
    if len(x)!=len(risk):
        raise ValueError(f'EAST signal and archived parent clocks differ: {sid}')
    return x, m, risk, z['times']


def design(issued, real='true'):
    return np.c_[issued['features'], issued[real]]


def paired(rows):
    if not rows:
        return dict(shots=0, windows=0, mae=None)
    losses = np.stack([a.mean(0) for a in rows])
    return dict(shots=len(rows), windows=sum(len(a) for a in rows),
                mae=dict(zip(ARMS, losses.mean(0).tolist(), strict=True)),
                current_wins={b:int((losses[:, 2]<losses[:, i]).sum())
                              for i, b in enumerate(ARMS) if b!='current'})


def full_forecast(r, pair, reduced, extended):
    ac = pair['available'] & (pair['earlier_at']>=0)
    return np.column_stack((np.zeros((len(r), 5*len(OUT))),
        predict(pair['features'], reduced), *(predict(design(pair, k), extended)
           for k in ('true', 'level', 'reversed', 'earlier')))).reshape(len(r), len(ARMS), 5*len(OUT)), ac


def run(out):
    start = monotonic()
    p, s = (read(REFERENCE/k) for k in ('protocol.json', 'seal.json'))
    if len(p['order'])!=360 or p['startup']!=20 or s['order']!=p['order'] or p['hand_off']!=8:
        raise ValueError('same sealed EAST360 and handoff required')
    order = p['order']
    filenames = [f'{rank:04d}_{sid}.npz' for rank, sid in enumerate(order, 1)]
    if filenames!=s['curves']:
        raise ValueError('archived predictions do not match this order')
    X, Y, S = [], [], []
    for sid, name in zip(order[:20], filenames[:20], strict=True):
        x, m, risk, _ = observed(sid, REFERENCE/'shots'/name)
        a = issued(x, m, risk)
        y, good = mature(x, m, a['r'])
        known = good & a['available'] & (a['earlier_at']>=0)
        X.append(design(a)[known])
        Y.append(y[known])
        S.append(np.repeat(sid, known.sum()))
    X, Y, S = map(np.concatenate, (X, Y, S))
    if len(np.unique(S))<2:
        raise ValueError('requires completed unlabeled packets from more than one preparation shot')
    reduced = model(X[:, :-5], Y, S)
    extended = model(X, Y, S)
    # Channel scales are based on the label-free preparation segment only.
    scale = np.sqrt(np.maximum(np.mean(Y.reshape(-1, 5, len(OUT))**2, axis=(0, 1)), 1e-10))
    out.mkdir()
    copyfile(__file__, out/'executed.py')
    write(out/'protocol.json', dict(reference=str(REFERENCE), order=order, source=p['source_sha256'],
        fit_shots=int(len(np.unique(S))), fit_windows=len(S), fit_ids=np.unique(S).tolist(),
        release=RELEASE, target_available=H+1, output=[COLUMNS[i] for i in OUT],
        models={k:[v.tolist() for v in f] for k, f in (('reduced', reduced), ('full', extended))},
        scale=scale.tolist(), arms=ARMS, changed_parent_alarms=False))
    evidence = {}
    gate_stats = np.zeros(4, np.int64)
    witnesses = {}
    for rank, (sid, name) in enumerate(zip(order, filenames, strict=True), 1):
        x, m, risk, t = observed(sid, REFERENCE/'shots'/name)
        a = issued(x, m, risk)
        r = a['r']
        py, co = full_forecast(r, a, reduced, extended)
        actual, mature_mask = mature(x, m, r)
        error = np.abs(py-actual[:, None]).reshape(len(r), len(ARMS), 5, len(OUT))/scale
        known = co & mature_mask
        first = np.flatnonzero(risk>=(p['original'] if rank<=20 else p['calibration']['0.2']))
        gate_stats += (len(x), int(((x[:, 0]==0.)==(x[:, 1]==0.)).sum()),
                       int(((x[:, 0]==0.)==(x[:, 2]==0.)).sum()),
                       int((x[:, 0]!=0.).sum()))
        if sid in (90874, 74970, 85422):
            n = min(600, len(risk))
            short = issued(x[:n], m[:n], risk[:n])
            short_pred, short_support = full_forecast(short['r'], short, reduced, extended)
            k = len(short['r'])
            np.testing.assert_array_equal(short_support, co[:k])
            np.testing.assert_allclose(short_pred, py[:k], rtol=0., atol=1e-12)
            ys, loaded = mature(x[:n], m[:n], short['r'])
            np.testing.assert_array_equal(ys[loaded], actual[:k][loaded])
            witnesses[str(sid)] = dict(samples=n, issues=k, matured=int(loaded.sum()))
        obs = t[np.minimum(r+H+1, len(t)-1)]
        raise_ca = (obs<=t[r]) & mature_mask
        if raise_ca.any():
            raise ValueError('scored a target before its issue')
        evidence[str(sid)] = dict(r=t[r].tolist(), target_at=[
            float(v) if ok else None for v, ok in zip(obs, mature_mask, strict=True)],
            any_live=a['any_live'].tolist(), complete=a['records'].tolist(),
            ready=co.tolist(), known=known.tolist(), errors=error[known].mean((2, 3)).tolist(),
            first=float(t[first[0]]) if len(first) else None)
    write(out/'forecast_sealed.json', dict(shots=evidence, witnesses=witnesses, gate=gate_stats.tolist()))
    # Reference outcomes are opened only after complete forecast and support
    # evidence is on disk. They are never used to fit or issue any forecast.
    evaluate(out, read(out/'forecast_sealed.json'), p)
    print(dict(stage='mnm_witness_complete', shots=len(evidence), seconds=monotonic()-start), flush=True)


def evaluate(out, saved, p):
    labels = read(REFERENCE/'per_shot.json')['0.2']
    rows, coverage = defaultdict(list), defaultdict(list)
    for rank, (sid, base, old) in enumerate(zip(p['order'], labels['bias_only'], labels['current_5'],
                                                strict=True), 1):
        y, down = io23.read_outcome('EAST', sid)
        if (sid, y, down)!=(base['shot'], base['disruptive'], base['down_s']) or sid!=old['shot']:
            raise ValueError('invalid frozen outcome record')
        obs = saved['shots'][str(sid)]
        release = np.asarray(obs['r'])
        arrived = np.asarray([v if v is not None else np.inf for v in obs['target_at']])
        support = np.asarray(obs['known'], bool)
        if rank<=20:
            continue
        kind = ('timely' if old['timely'] else 'late' if old['alarm'] else 'never') if y else (
            'false' if base['alarm'] else 'safe')
        error = np.asarray(obs['errors']).reshape(-1, len(ARMS))
        pool = np.ones(len(release), bool)
        until = np.inf
        if y:
            ms = np.round((down-release)*1000., 9)
            pool = (ms>30.) & (ms<=125.)
            until = down-.030
        elif kind=='false':
            since = np.round((obs['first']-release)*1000., 9)
            pool = (since>=0.) & (since<=95.)
            until = obs['first']
        target_in_time = (np.round((until-arrived)*1000., 9)>0.) if np.isfinite(until) else np.isfinite(arrived)
        covered = {k:int((pool & np.asarray(obs[v], bool)).sum())
                   for k, v in (('starts', 'any_live'), ('complete', 'complete'), ('common', 'ready'))}
        coverage[kind].append(dict(shot=sid, epochs=int(pool.sum()), **covered,
                                   timely_audit=int((pool & support & target_in_time).sum())))
        masks = ((kind, pool), (f'{kind}_before_cutoff', pool & target_in_time), ('suffix', np.ones_like(pool)))
        for tag, select in masks:
            indices = select[support]
            if indices.any():
                rows[tag].append(error[indices])
    write(out/'result.json', dict(groups={k:paired(rows[k]) for k in ('suffix', *coverage,
          *(f'{k}_before_cutoff' for k in coverage))}, coverage=coverage,
          horizon_note='HDF5 sample grid only; no deployment latency claim',
          no_new_warning=True))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    run(cli.parse_args().output.resolve())
