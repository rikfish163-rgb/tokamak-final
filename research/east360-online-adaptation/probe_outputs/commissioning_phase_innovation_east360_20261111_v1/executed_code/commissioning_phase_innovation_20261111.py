"""Compare current-point and path-matched forecast innovations in EAST360."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


IP = io23.RAW23.index('ip')
HISTORY, NEIGHBORS = 12, 2
PHASE_SCALE, STRENGTH, HUBER_CLIP = .10, .50, .20
CURRENT, PHASE, SHIFT, CONTEXT = NEW = (
    'current_innovation_true', 'path_innovation_true', 'path_innovation_shifted',
    'path_prefix_no_gradient',
)
COMPARISONS = ((lag.BASES[-1], PHASE), (guard.PATH, PHASE), (CURRENT, PHASE),
               (SHIFT, PHASE), (CONTEXT, PHASE))


def phase_path(ip, observed, q):
    """Three earlier changes and drawdown from the running 32-sample peak."""
    observed = observed & np.isfinite(ip)
    t = np.arange(1, len(ip)+1)
    left = np.maximum(0, t-32)
    c = np.r_[0, np.cumsum(observed)]
    v = np.r_[0., np.cumsum(np.where(observed, np.abs(ip), 0.), dtype=np.float64)]
    count = c[t]-c[left]
    mean = np.divide(v[t]-v[left], count, out=np.zeros(len(t)), where=count >= 16)
    mean[:31] = 0.
    peak = np.maximum.accumulate(mean)[q]
    ends = np.maximum(q[:, None]-np.array([96, 64, 32, 0]), 0)
    track = mean[ends]
    valid = (q >= 127) & (count[ends] >= 16).all(1) & (peak > 0.)
    delta = np.column_stack((np.diff(track, axis=1), track[:, -1]-peak))
    result = np.divide(delta, peak[:, None], out=np.zeros_like(delta), where=valid[:, None])
    return result.astype(np.float32), valid


def past_bias(q, current, phase, usable, error):
    """At each issue time match two earlier, already matured windows per arm."""
    supported = np.flatnonzero(usable)
    stop = np.searchsorted(q[supported]+lag.H+1, q, side='right')
    clipped = np.clip(error, -HUBER_CLIP, HUBER_CLIP)
    bias = np.zeros((2, *error.shape), dtype=error.dtype)
    donors = np.full((len(q), 2, NEIGHBORS), -1, dtype=np.int64)
    for j in np.flatnonzero(usable & (stop >= NEIGHBORS)):
        pool = supported[max(0, stop[j]-HISTORY):stop[j]][::-1]
        d = np.square((current[pool]-current[j])/teacher.CALIPER[1:]).mean(1)
        path = np.square((phase[pool]-phase[j])/PHASE_SCALE).mean(1)
        for a, distance in enumerate((d, d+path)):
            k = pool[np.argsort(distance, kind='stable')[:NEIGHBORS]]
            donors[j, a] = k
            bias[a, j] = STRENGTH*clipped[k].mean(0)
    return bias, donors


def innovation_losses(pred, target, shifted, bias):
    """Use the same phase prior in both fresh-feedback and wrong-time arms."""
    return (teacher.grouped_error(pred+bias[0], target),
            teacher.grouped_error(pred+bias[1], target),
            teacher.grouped_error(pred+bias[1], shifted))


def _release(raw, signal, args, decoder, *, source=False):
    tr, h, z, beta = guard._track(signal, args, source=source)
    q, key, prior, target = teacher._mature(tr, args[2][2], args[2][3])
    phase, valid = phase_path(raw['values'][:, IP], raw['available'][:, IP], q)
    def tensor(a):
        return torch.as_tensor(a, device=beta.device)
    with torch.no_grad():
        pred = lag.ordered_signature(lag.motion(lag._predict(tensor(h[q]), tensor(prior), decoder)))
        y = tensor(target)
        real, wrong = (lag.ordered_signature(lag.motion(y, shift=s)) for s in (False, True))
    err = (real-pred).cpu().numpy()
    bias, donors = past_bias(q, 2.*key[:, 1:], phase, valid & tr['matchable'][q], err)
    mse = np.stack([np.square(e, dtype=np.float64).mean(1)
                    for e in (err, err-bias[0], err-bias[1])], axis=1)
    return dict(q=q, h=h, z=z, beta=beta, key=tensor(key), prior=tensor(prior), target=real,
                wrong=wrong, bias=tensor(bias), prefix=tensor(np.c_[key, .5*np.tanh(5.*phase)]),
                supported=donors[:, 0, 0] >= 0,
                changed=(np.sort(donors[:, 0], axis=1) != np.sort(donors[:, 1], axis=1)).any(1),
                mse=mse)


def _states(base, signal, released, decoder, fitted, rows):
    """Different residual targets backpropagate to the same shot-local matrix."""
    n = len(lag.CHANNELS)
    out = []
    for i in range(0, len(rows), 96):
        row = rows[i:i+96]
        x, mask = lag._windows(base, signal, released['q'][row])
        theta = x.new_zeros((len(row), n, n), requires_grad=True)
        h = lag._features(base, x, mask, theta)
        p = lag.ordered_signature(lag.motion(lag._predict(h, released['prior'][row], decoder)))
        e = torch.cat(innovation_losses(p, released['target'][row], released['wrong'][row],
                                        released['bias'][:, row]), dim=1)
        jac = torch.stack([torch.autograd.grad(e[:, j].sum(), theta, retain_graph=j+1<e.shape[1])[0]
                           for j in range(e.shape[1])], dim=1)
        d = torch.stack([guard.signed_direction(*guard.basis(g), released['key'][row], fitted)
                         for g in jac.split(teacher.GROUPS, dim=1)])
        common = d.ne(0.).any(dim=(2, 3)).all(0)
        out.append(-lag.RADIUS*d*common[None, :, None, None])
    return torch.cat(out, dim=1) if out else base.center.new_empty((3, 0, n, n))


def _prediction(base, signal, r, matrices):
    q, h, z, beta = (r[k] for k in ('q', 'h', 'z', 'beta'))
    slots, _ = lag.delayed_slots(q, np.arange(len(z)))
    z = torch.as_tensor(z, device=beta.device)
    before = z.sigmoid().cpu().numpy()
    out = {}
    for name, m in zip(NEW, matrices, strict=True):
        n = len(lag.CHANNELS)
        effective = np.concatenate((np.zeros((1, n, n), np.float32), m.cpu().numpy()))[slots]
        delta = lag._embed_active(base, signal, h, effective)-h
        after = (z+(torch.as_tensor(delta, device=beta.device)*beta).sum(-1)).sigmoid().cpu().numpy()
        after[~effective.any((1, 2))] = before[~effective.any((1, 2))]
        out[name] = after
    return before, out


def _replay(raw, signal, args, decoder, fitted, context):
    r = _release(raw, signal, args, decoder)
    d = _states(args[0], signal, r, decoder, fitted, np.arange(len(r['q'])))
    n = len(lag.CHANNELS)
    prefix = -lag.RADIUS*guard._unit((r['prefix'] @ context).reshape(-1, n, n))
    state = torch.cat((d, prefix[None]))
    before, out = _prediction(args[0], signal, r, state)
    return before, out, dict(release=r['q']+lag.H+1, supported=r['supported'],
                             changed=r['changed'], forecast_mse=r['mse'].astype(np.float32),
                             radius=state.norm(dim=(-2, -1)).T.cpu().numpy())


def _historical_cases(pairs, role, allowed, lock, args, decoder, fitted, cutoff, context=None):
    """Use the same disjoint, historically sealed hard pairs as the prior teacher."""
    source = role != 'cold'
    device = 'J-TEXT' if source else 'EAST'
    advanced = io23.advanced_times() if source else {}
    features, directions, records = [], [], []
    used = set()
    for pair in pairs:
        for positive, entry in zip((True, False), pair, strict=True):
            sid, q, t = (entry[k] for k in ('shot', 'origin', 'action'))
            if sid not in allowed or sid in used or entry['role'] != role:
                raise ValueError('phase reference outside its sealed historical role')
            used.add(sid)
            if not source and sha(io23.path_for(device, sid)) != lock['hashes'][str(sid)]:
                raise ValueError('historical EAST data changed')
            raw = simple_data.read_shot(device, sid, rate_contract='strict' if source else 'nominal')
            truth = simple_data.attach_outcome(raw, device, advanced)
            if truth['is_disrupt'] != positive or not truth['valid'][t]:
                raise ValueError('archived reference contradicts the historical HDF5 outcome')
            if positive:
                lead = np.round((truth['down']-raw['times'][t])*1000., 9)
                if lead <= (10. if source else 30.) or not truth['y'][t]:
                    raise ValueError('historical positive outside the timely supervision window')
            signal = raw if source else lag.prepare_actual(raw, lag.STARTUP+1, lock['mapping'])
            r = _release(raw, signal, args, decoder, source=source)
            j = int(np.searchsorted(r['q']+lag.H+1, t, side='right'))-1
            if j < 0 or r['q'][j] != q:
                raise ValueError('historical reference did not match the same released block')
            margin = r['z'][t]-np.log(cutoff/(1.-cutoff))
            np.testing.assert_allclose(margin, entry['match'][0], atol=2e-6, rtol=0.)
            mat = _states(args[0], signal, r, decoder, fitted, np.array([j]))[:, 0]
            k = r['prefix'][j]
            features.append(k)
            directions.append(mat[1].flatten())
            if context is not None:
                no_grad = -lag.RADIUS*guard._unit((k @ context).reshape(1, *mat.shape[1:]))
                mat = torch.cat((mat, no_grad))
            with torch.no_grad():
                x, m = lag._windows(args[0], signal, np.full(len(mat), t))
                after = lag._features(args[0], x, m, mat)
                gain = ((after-torch.as_tensor(r['h'][t], device=x.device))*r['beta']).sum(1)
                gain[~mat.ne(0.).any(dim=(1, 2))] = 0.
            records.append(dict(shot=sid, role=role, disruptive=positive, origin=q, action=t,
                                margin=float(margin), gain=gain.cpu().tolist(),
                                support=bool(r['supported'][j]), changed=bool(r['changed'][j]),
                                forecast_mse=r['mse'][j].tolist()))
    return torch.stack(features), torch.stack(directions), records


def _historical_summary(records, arms):
    d = np.asarray([r['gain'] for r in records]).reshape(-1, 2, len(arms))
    m = np.asarray([r['margin'] for r in records]).reshape(-1, 2)
    supported = np.asarray([r['support'] for r in records])
    pred = np.asarray([r['forecast_mse'] for r in records])
    return dict(pairs=len(d), supported=int(supported.sum()), changed=sum(r['changed'] for r in records),
                positive_below=int((m[:, 0] < 0).sum()),
                forecast_mse_supported=pred[supported].mean(0).tolist() if supported.any() else None,
                direction={name: dict(mean_positive=float(v[:, 0].mean()),
                                      mean_safe=float(v[:, 1].mean()),
                                      both_right=int(((v[:, 0] > 0)&(v[:, 1] < 0)).sum()),
                                      positive_cross=int(((m[:, 0] < 0)&(m[:, 0]+v[:, 0] >= 0)).sum()),
                                      safe_new_cross=int(((m[:, 1] < 0)&(m[:, 1]+v[:, 1] >= 0)).sum()))
                           for name, v in zip(arms, d.transpose(2, 0, 1), strict=True)})


def run(previous, output):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    ids, roles = lock['ids'], io23.source_roles()
    p0, seal, fit = (read(previous/k) for k in ('protocol.json', 'seal.json', 'teacher.json'))
    p_teacher = Path(p0['previous_teacher'])
    earlier = read(p_teacher/'protocol.json')
    lp = read(Path(earlier['previous_lag'])/'protocol.json')
    coords = next(Path(a).parent for a in p0['assets']
                  if Path(a).parent.name.startswith('commissioning_local_cross_response_'))
    p, _, args, assets = lag.parent._frozen(Path(lp['reference']), coords, lock)
    if (p0['scope'] != 'EAST360_SIGNED_POINT_VS_PATH_GUARD_DEVELOPMENT'
            or tuple(p0['arms']) != guard.ARMS or p0['order'] != ids or p0['assets'] != assets
            or p0['original'] != p['original'] or p0['calibration'] != p['calibration']
            or p0['fit_roles'] != {'J-TEXT/train': roles['train'], 'EAST/cold': ids[:lag.STARTUP]}
            or p0['held_source'] != roles['val'] or len(seal['curve_sha256']) != len(ids)
            or sha(previous/'teacher.json') != p0['teacher_sha256']
            or sha(p_teacher/'teacher.json') != p0['previous_pairs_sha256']
            or sha(p_teacher/'seal.json') != p0['previous_seal']):
        raise ValueError('exact sealed phase-control reference required')
    decoder = lag._bind_forecast(Path(lp['source_decoder']).parent, p)
    archived = read(p_teacher/'teacher.json')['pairs']
    for rank, sid in enumerate(ids[:lag.STARTUP], 1):
        name = f'{rank:04d}_{sid}.npz'
        if sha(previous/'shots'/name) != seal['curve_sha256'][name]:
            raise ValueError('cold startup must seal before historical fitting')
    base = args[0]
    frozen = tuple(state_digest(m) for m in args[:2])
    teacher_path = torch.as_tensor(fit['path'], device=base.center.device, dtype=base.center.dtype)
    original, thresholds = p['original'], p['calibration']
    producer = Path(__file__)
    lag.parent._require_hashes(producer.parent, p0['code_sha256'])
    (output/'shots').mkdir(parents=True)
    (output/'executed_code').mkdir()
    copyfile(producer, output/'executed_code'/producer.name)
    code = dict(p0['code_sha256'], **{producer.name: sha(producer)})
    train, cold = (_historical_cases(archived[k], k, set(ids[:lag.STARTUP] if k == 'cold' else roles[k]),
                                    lock, args, decoder, teacher_path,
                                    thresholds['0.2'] if k == 'cold' else original)
                   for k in ('train', 'cold'))
    key, d = (torch.cat((train[i], cold[i])) for i in (0, 1))
    # Prefix-only control fits to the historically observed update directions.
    # At replay it never reads the latest diagnostic residual or runs backward.
    context = torch.linalg.solve(key.T @ key + torch.eye(key.shape[1], device=key.device),
                                 -(key.T @ d))
    _, _, val = _historical_cases(archived['val'], 'val', set(roles['val']), lock, args,
                                  decoder, teacher_path, original, context)
    summary = dict(train=_historical_summary(train[2], NEW[:3]),
                   cold=_historical_summary(cold[2], NEW[:3]), val=_historical_summary(val, NEW))
    write(output/'historical.json', dict(summary=summary, context=context.cpu().tolist(),
                                        train=train[2], cold=cold[2], val=val))
    write(output/'protocol.json', dict(scope='EAST360_PHASE_CONDITIONED_OBSERVED_INNOVATION_DEVELOPMENT',
        order=ids, startup=lag.STARTUP, original=original, calibration=thresholds,
        source_sha256=p['source_sha256'], assets=assets, code_sha256=code,
        previous_guard=str(previous), previous_seal=sha(previous/'seal.json'),
        historical_sha256=sha(output/'historical.json'), fit_roles=p0['fit_roles'],
        held_source=roles['val'], new=NEW, arms=(*guard.ARMS, *NEW), comparisons=COMPARISONS,
        history=HISTORY, neighbors=NEIGHBORS, strength=STRENGTH, clip=HUBER_CLIP,
        phase_scale=PHASE_SCALE, horizon=lag.H, first_effect=lag.FIRST_EFFECT, stride=lag.STRIDE,
        radius=lag.RADIUS, input_rms=lag.INPUT_RMS, persistent_updates=0,
        phase='four successive 32-sample absolute Ip means; three differences and drawdown from causal running peak',
        current='present scaled Ip, Bt and short Ip change, same candidate windows as phase',
        context='past path and current state regression on ended shots; matched release clock; no backpropagation',
        feedback='estimate signature bias at origin from two earlier mature same-shot blocks; differentiate after H25',
        outcomes='no EAST suffix HDF5 outcome until all 360 predictions seal'))
    print(dict(stage='historical_locked', summary=summary, seconds=round(time.monotonic()-start, 2)), flush=True)
    hashes, witnesses = {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('unsealed EAST outcome')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            f = previous/'shots'/name
            if sha(f) != seal['curve_sha256'][name]:
                raise ValueError('sealed reference changed')
            with np.load(f, allow_pickle=False) as saved:
                times = saved['times']
                if rank <= lag.STARTUP:
                    curves, trace = dict.fromkeys(NEW, saved[guard.PATH]), {}
                else:
                    if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                        raise ValueError('locked EAST signal changed')
                    raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                    np.testing.assert_array_equal(raw['times'], times)
                    signal = lag.prepare_actual(raw, rank, lock['mapping'])
                    before, curves, trace = _replay(raw, signal, args, decoder, teacher_path, context)
                    np.testing.assert_array_equal(before, saved[lag.BASES[-1]])
                    release = trace['release']
                    np.testing.assert_array_equal(release, saved['release'])
                    stop = int(release[0]) if len(release) else len(times)
                    for v in curves.values():
                        np.testing.assert_array_equal(v[:stop], before[:stop])
                    for other in (1, 2):
                        np.testing.assert_allclose(trace['radius'][:, 0], trace['radius'][:, other],
                                                   atol=3e-7, rtol=1e-6)
                    if rank in (21, 120, 360):
                        end = min(len(times), lag.FIRST_EFFECT+19*lag.STRIDE+5)
                        r0, s0 = ({k: v[k][:end] for k in ('values', 'available')}
                                  for v in (raw, signal))
                        _, short, seen = _replay(r0, s0, args, decoder, teacher_path, context)
                        valid = release < end
                        for k in ('release', 'supported', 'changed'):
                            np.testing.assert_array_equal(seen[k], trace[k][valid])
                        for k in NEW:
                            np.testing.assert_allclose(short[k], curves[k][:end], atol=3e-6, rtol=0.)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(short[k] >= tau, curves[k][:end] >= tau)
                        witnesses[str(sid)] = end
            if not all(np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite phase-innovation prediction')
            target = output/'shots'/name
            with target.open('xb') as fd:
                np.savez_compressed(fd, times=times, **curves, **trace)
            hashes[name] = sha(target)
            if rank == 21 or rank%40 == 0:
                print(dict(stage='sealed', rank=rank, seconds=round(time.monotonic()-start, 2)), flush=True)
    if tuple(state_digest(m) for m in args[:2]) != frozen:
        raise ValueError('frozen model changed')
    write(output/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = read(previous/'per_shot.json')
    for table in per.values():
        table.update({k: [] for k in NEW})
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        f = output/'shots'/name
        if sha(f) != hashes[name]:
            raise ValueError('phase-innovation curve changed after sealing')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(f, allow_pickle=False) as z:
            for band, table in per.items():
                tau = original if rank <= lag.STARTUP else thresholds[band]
                for k in NEW:
                    table[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                         **score_curve(z['times'], z[k], d, down, tau)))
    write(output/'per_shot.json', per)
    metrics = {band: {k: timing_decomposition(v) for k, v in table.items()} for band, table in per.items()}
    paired = {band: {f'{a}__to__{b}': flips(table[a], table[b]) for a, b in COMPARISONS}
              for band, table in per.items()}
    write(output/'lead_sensitivity.json', report([output/'per_shot.json'], (*guard.ARMS, *NEW)))
    write(output/'result.json', dict(status='COMPLETED_PHASE_INNOVATION_DEVELOPMENT',
        metrics=metrics, paired=paired, seconds=time.monotonic()-start))
    print({band: {k: (m['tp'], m['fp']) for k, m in table.items()}
           for band, table in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    run(cli.previous.resolve(), cli.output.resolve())
