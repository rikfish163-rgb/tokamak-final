"""Teach an observed-feedback update on matched, already ended discharges."""

import argparse
from functools import partial
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_lag_mixer_20261111 as lag
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


FAMILY_PAIRS = tuple((i, j) for i in range(4) for j in range(i+1, 4))
MEMBERS = torch.stack([
    (torch.minimum(lag.GROUP[lag.ORDERED[0]], lag.GROUP[lag.ORDERED[1]]) == i) &
    (torch.maximum(lag.GROUP[lag.ORDERED[0]], lag.GROUP[lag.ORDERED[1]]) == j)
    for i, j in FAMILY_PAIRS], dim=1)
MEMBERS = torch.block_diag(MEMBERS, MEMBERS).float()
GROUPS = MEMBERS.shape[1]
CALIPER = np.array([.50, .35, .35, .50])
STEPS, SEED = 192, 131
TRUE, SHIFT, HOLD = 'taught_true', 'taught_phase_shifted', 'taught_hold_previous'
ARMS = (*lag.ARMS, TRUE, SHIFT, HOLD)
COMPARISONS = ((lag.BASES[-1], TRUE), (lag.TRUE, TRUE), (SHIFT, TRUE), (HOLD, TRUE))


def grouped_error(p, y):
    error = F.smooth_l1_loss(p, y, beta=.2, reduction='none')
    return error @ MEMBERS.to(device=p.device, dtype=p.dtype) / p.shape[1]


def weights(params, context):
    w = torch.exp(.85 * (context @ params.T).tanh())
    return w / w.mean(1, keepdim=True)


def allocate(group_gradient, w):
    g = torch.einsum('bg,bgij->bij', w, group_gradient)
    n = g.norm(dim=(1, 2), keepdim=True)
    return -lag.RADIUS * torch.where(n > 1e-8, g/n.clamp_min(1e-8), 0.)


def direction_loss(delta, margin):
    # Two classes contribute equally even when their parent scores are already correct.
    up, down = delta.T
    positive = F.softplus(-up/.10)
    negative = F.softplus(down/.10)
    protect = F.relu(down - (-margin[:, 1]).clamp_min(0.))/.10
    separation = F.softplus((down-up)/.20)
    return (.5*(positive+negative) + .50*protect + .25*separation).mean()


def match_shots(positive, safe):
    """Maximum disjoint shot-pair count, then lowest risk/operating distance."""
    if not positive or not safe:
        return np.empty((0, 2), dtype=np.int64)
    p, s = (np.asarray([r['match'] for r in rows], dtype=float) for rows in (positive, safe))
    p_shots, ip = np.unique([r['shot'] for r in positive], return_inverse=True)
    s_shots, js = np.unique([r['shot'] for r in safe], return_inverse=True)
    n, m = len(p_shots), len(s_shots)
    # All admissible costs are at most four. Dummy cost gives cardinality priority.
    dummy = 5.*(n+1)
    cost = np.full((n, m+n), 2.*dummy)
    cost[:, m:] = dummy
    for i, z in enumerate(p):
        gap = (s-z)/CALIPER
        ok = (abs(gap) <= 1.).all(1)
        np.minimum.at(cost[ip[i], :m], js[ok], np.square(gap[ok]).sum(1))
    rows, cols = linear_sum_assignment(cost)
    result = []
    for a, b in zip(rows, cols, strict=True):
        if b >= m:
            continue
        i, j = np.flatnonzero(ip==a), np.flatnonzero(js==b)
        gap = (p[i, None]-s[j])/CALIPER
        err = np.where((abs(gap) <= 1.).all(2), np.square(gap).sum(2), np.inf)
        x, y = np.unravel_index(err.argmin(), err.shape)
        if not np.isfinite(err[x, y]):
            raise ValueError('assignment has no in-caliper endpoint')
        result.append((i[x], j[y]))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def _mature(tr, prior, noise):
    q = tr['q']
    ok = tr['observed'].all((1, 2)) & (q+lag.H+1 < tr['length'])
    key = tr['key'][np.searchsorted(tr['ends'], q[ok])]
    return q[ok], key, lag.direct.base_forecast(tr, prior)[ok], lag.direct.normalized_target(tr, noise)[ok]


def _signature_problem(base, x, m, prior, decoder, y):
    theta = torch.zeros((len(x), len(lag.CHANNELS), len(lag.CHANNELS)), device=x.device,
                        dtype=x.dtype, requires_grad=True)
    h = lag._features(base, x, m, theta)
    p = lag.ordered_signature(lag.motion(lag._predict(h, prior, decoder)))
    target, shifted = (lag.ordered_signature(lag.motion(y, shift=b)) for b in (False, True))
    return theta, grouped_error(p, target), grouped_error(p, shifted)


def online_feedback(base, signal, tr, forecast, noise, decoder, *, teacher):
    """Same complete-window clock and matrix budget as the uniform lag mixer."""
    q, keys, pred, target = _mature(tr, forecast, noise)
    device = base.center.device
    c = len(lag.CHANNELS)
    states = np.zeros((2, len(q)+1, c, c), np.float32)
    trace = np.zeros((len(q), 3), np.float32)
    for start in range(0, len(q), 128):
        end = start+128
        x, m = lag._windows(base, signal, q[start:end])
        z, y, k = (torch.from_numpy(a[start:end]).to(device) for a in (pred, target, keys))
        theta, e, wrong = _signature_problem(base, x, m, z, decoder, y)
        w = weights(teacher, k)
        g, s = (torch.autograd.grad((err*w).sum(), theta, retain_graph=i==0)[0]
                for i, err in enumerate((e, wrong)))
        a, b = lag.matched_steps(g, s)
        states[:, start+1:end+1] = torch.stack((a, b)).cpu().numpy()
        trace[start:end] = torch.stack((g.norm(dim=(1, 2)), s.norm(dim=(1, 2)),
                                      e.detach().sum(1)), dim=1).cpu().numpy()
    if not (np.isfinite(states).all() and np.isfinite(trace).all()):
        raise FloatingPointError('nonfinite task-taught lag update')
    return q, states, trace, pred, target


def _risk_context(raw, signal, args, is_source):
    base, net, params, old, quiet, mapper, coef, _ = args
    ends = np.arange(len(raw['times']))
    if is_source:
        tr = lag.direct.track(net, signal, params[0], params[1], ends)
        _, z = extract_shot(base, signal, 'cuda')
        beta = base.risk_head.weight.detach().flatten().cpu().numpy()
    else:
        tr, inherited, ready = lag.history.observed_track(net, signal, ends, params, old)
        state, _ = lag.persistent.replay(*ready, quiet)
        latest = inherited + torch.einsum('...k,...f,kfw->...w', torch.from_numpy(tr['key']),
                    4.*torch.from_numpy(state['current_equal_norm']), mapper)
        x, _ = lag.features(base, signal, 'cuda')
        v = np.concatenate((x, latest.tanh().numpy()), axis=1)
        z = (torch.from_numpy(v).cuda()@coef[:-1]+coef[-1]).cpu().numpy()
        beta = coef[:base.cfg.dim].cpu().numpy()
    return tr, z, beta


def _collect(role, ids, args, lock, original, cutoff):
    result = ([], [])
    is_source = role != 'cold'
    device = 'J-TEXT' if is_source else 'EAST'
    advanced = io23.advanced_times() if is_source else {}
    for i, sid in enumerate(ids, 1):
        if not is_source and sha(io23.path_for(device, sid)) != lock['hashes'][str(sid)]:
            raise ValueError('cold-start HDF5 changed')
        raw = simple_data.read_shot(device, sid, rate_contract='strict' if is_source else 'nominal')
        truth = simple_data.attach_outcome(raw, device, advanced)
        d, down = truth['is_disrupt'], truth['down']
        signal = raw if is_source else lag.prepare_actual(raw, lag.STARTUP+1, lock['mapping'])
        tr, z, beta = _risk_context(raw, signal, args, is_source)
        q, k, prior, target = _mature(tr, args[2][2], args[2][3])
        ends = q[:, None] + lag.H+1 + np.arange(0, lag.STRIDE, 8)
        legal = truth['valid'].copy() & tr['matchable']
        lead = np.round((down - raw['times'])*1000., 9) if d else None
        if d:
            legal &= truth['y'].astype(bool) & (lead > (10. if is_source else 30.))
        margin = z - logit(original if is_source else cutoff)
        candidates = []
        for j, row in enumerate(ends):
            ix = row[row < len(z)]
            ix = ix[legal[ix]]
            if len(ix):
                t = int(ix[np.argmin(abs(margin[ix]))])
                candidates.append((abs(margin[t]), j, t))
        for _, j, t in sorted(candidates)[:6]:
            at_origin = simple_data.window(signal, int(q[j]), lag.W)
            at_action = simple_data.window(signal, t, lag.W)
            context = tr['context'][t, :3].astype(float)
            context[2] *= 4.
            result[0 if d else 1].append(dict(shot=int(sid), role=role,
                origin=int(q[j]), action=t, lead_ms=float(lead[t]) if d else None,
                match=np.r_[float(margin[t]), np.tanh(context)].tolist(),
                key=k[j], prior=prior[j], target=target[j], beta=beta,
                xq=at_origin[0], mq=at_origin[1], xt=at_action[0], mt=at_action[1]))
        if i % 350 == 0 or i == len(ids):
            print(dict(stage='candidate_' + role, scanned=i, total=len(ids)), flush=True)
    pos, safe = result
    pairs = [(pos[i], safe[j]) for i, j in match_shots(pos, safe)]
    coverage = dict(positive_windows=len(pos), safe_windows=len(safe),
        positive_shots=len({r['shot'] for r in pos}), safe_shots=len({r['shot'] for r in safe}),
        matched_shots_per_class=len(pairs))
    return pairs, coverage


def _packed(pairs, base, decoder):
    records = [r for pair in pairs for r in pair]
    device = base.center.device
    data = {k: torch.as_tensor(np.stack([r[k] for r in records]), device=device)
            for k in ('xq', 'mq', 'xt', 'mt', 'key', 'prior', 'target', 'beta')}
    g, s = [], []
    for a in range(0, len(records), 64):
        sl = slice(a, a+64)
        theta, real, sham = _signature_problem(base, data['xq'][sl], data['mq'][sl],
                                               data['prior'][sl], decoder, data['target'][sl])
        both = torch.cat((real, sham), dim=1)
        jac = [torch.autograd.grad(both[:, i].sum(), theta, retain_graph=i+1<both.shape[1])[0]
               for i in range(both.shape[1])]
        real_g, sham_g = torch.stack(jac, dim=1).split(GROUPS, dim=1)
        g.append(real_g)
        s.append(sham_g)
    data['g'], data['s'] = torch.cat(g), torch.cat(s)
    data['margin'] = torch.as_tensor([r['match'][0] for r in records],
                                    device=device, dtype=base.center.dtype)
    with torch.no_grad():
        zero = data['g'].new_zeros((len(records), len(lag.CHANNELS), len(lag.CHANNELS)))
        data['h'] = torch.cat([lag._features(base, x, m, w)
            for x,m,w in zip(data['xt'].split(128), data['mt'].split(128), zero.split(128), strict=True)])
    return data


def _gain(base, data, params, rows, kind='g'):
    w = weights(params, data['key'][rows])
    theta = allocate(data[kind][rows], w)
    h = lag._features(base, data['xt'][rows], data['mt'][rows], theta)
    return ((h-data['h'][rows])*data['beta'][rows]).sum(-1).reshape(-1, 2)


def _teach(base, data):
    params = torch.nn.Parameter(data['key'].new_zeros((GROUPS, data['key'].shape[1])))
    rng = np.random.default_rng(SEED)
    opt = torch.optim.Adam([params], lr=.025)
    n = len(data['key'])//2
    last = None
    for _ in range(STEPS):
        pair = rng.choice(n, min(n, 24), replace=False)
        ids = torch.as_tensor((2*pair[:, None]+np.arange(2)).ravel(), device=params.device)
        opt.zero_grad(set_to_none=True)
        loss = direction_loss(_gain(base, data, params, ids), data['margin'][ids].reshape(-1,2))
        loss = loss + .025*params.square().mean()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    if not torch.isfinite(params).all():
        raise FloatingPointError('nonfinite offline teacher')
    return params.detach(), last


@torch.no_grad()
def _describe(base, data, fitted):
    out = {}
    for key, params, kind in (('uniform', 0.*fitted, 'g'),
                              (TRUE, fitted, 'g'), (SHIFT, fitted, 's')):
        z = torch.cat([_gain(base, data, params, j, kind)
                       for j in torch.arange(len(data['key']), device=fitted.device).split(128)])
        values = z.cpu().numpy()
        m = data['margin'].cpu().numpy().reshape(-1, 2)
        out[key] = dict(pairs=len(values), mean_delta=values.mean(0).tolist(),
            median_delta=np.median(values, axis=0).tolist(),
            both_right=int(((values[:, 0]>0.) & (values[:, 1]<0.)).sum()),
            safe_new_cross=int(((m[:, 1]<0.) & (m[:, 1]+values[:, 1]>=0.)).sum()),
            timely_reaches=int(((m[:, 0]<0.) & (m[:, 0]+values[:, 0]>=0.)).sum()),
            median_positive_minus_safe=float(np.median(values[:, 0]-values[:, 1])))
    return out


def run(reference, coordinates, forecast, lag_result, output):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ids = lock['ids']
    p, _, args, assets = lag.parent._frozen(reference, coordinates, lock)
    decoder = lag._bind_forecast(forecast, p)
    pp, seal = (read(lag_result/k) for k in ('protocol.json', 'seal.json'))
    if (pp['order'] != ids or tuple(pp['arms']) != lag.ARMS or pp['assets'] != assets
            or pp['calibration'] != p['calibration'] or pp['original'] != p['original']
            or pp['decoder_sha256'] != sha(forecast/'forecast.npz')
            or pp['code_sha256'][Path(lag.__file__).name] != sha(Path(lag.__file__))):
        raise ValueError('exact preceding lag-mixer reference required')
    for i, sid in enumerate(ids[:lag.STARTUP], 1):
        f = lag_result/'shots'/f'{i:04d}_{sid}.npz'
        if sha(f) != seal['curve_sha256'][f.name]:
            raise ValueError('cannot use unsealed cold-start outcomes')
    producer = Path(__file__)
    code = dict(pp['code_sha256'], **{producer.name: sha(producer)})
    (output/'shots').mkdir(parents=True)
    (output/'executed_code').mkdir()
    for name, value in code.items():
        target = output/'executed_code'/name
        copyfile(producer.parent/name, target)
        if sha(target) != value:
            raise ValueError('archived producer changed: '+name)
    roles = io23.source_roles()
    frozen = tuple(state_digest(m) for m in args[:2])
    original, thresholds = p['original'], p['calibration']
    bundles = {}
    for name, selection in (('train', roles['train']), ('cold', ids[:lag.STARTUP])):
        bundles[name] = _collect(name, selection, args, lock, original, thresholds['0.2'])
    fit_pairs = bundles['train'][0] + bundles['cold'][0]
    if not fit_pairs:
        raise ValueError('no labeled hard cases in common support')
    base = args[0]
    fit_data = _packed(fit_pairs, base, decoder)
    teacher, loss = _teach(base, fit_data)
    fit_score = _describe(base, fit_data, teacher)
    del fit_data
    bundles['val'] = _collect('val', roles['val'], args, lock, original, thresholds['0.2'])
    val = _describe(base, _packed(bundles['val'][0], base, decoder), teacher) if bundles['val'][0] else None
    write(output/'teacher.json', dict(weights=teacher.cpu().tolist(), loss=loss, seed=SEED, steps=STEPS,
        fitted=fit_score, held_source=val, coverage={k:v[1] for k,v in bundles.items()},
        pairs={role: [[{k:r[k] for k in ('shot', 'role', 'origin', 'action', 'lead_ms', 'match')}
                       for r in pair] for pair in rows] for role, (rows, _) in bundles.items()}))
    print(dict(stage='teacher_sealed', counts={k:v[1] for k,v in bundles.items()},
               held=val, seconds=round(time.monotonic()-started, 2)), flush=True)
    write(output/'protocol.json', dict(scope='EAST360_MATCHED_OFFLINE_DIRECTION_TEACHING_DEVELOPMENT',
        order=ids, source_sha256=p['source_sha256'], assets=assets,
        startup=lag.STARTUP, original=original, calibration=thresholds, arms=ARMS,
        previous_lag=str(lag_result), previous_seal=sha(lag_result/'seal.json'),
        teacher_sha256=sha(output/'teacher.json'), code_sha256=code,
        fit_roles={'J-TEXT/train': roles['train'], 'EAST/cold': ids[:lag.STARTUP]},
        held_source=roles['val'], caliper=CALIPER.tolist(), comparison=COMPARISONS,
        radius=lag.RADIUS, input_rms=lag.INPUT_RMS, first_effect=lag.FIRST_EFFECT, stride=lag.STRIDE,
        phase_control='same fitted weights with groupwise cyclic phase shift; equal matrix norm',
        hold_control='one most recent complete update skipped',
        outcomes='no suffix HDF5 outcome until all predictions seal', persistent_updates=0))
    hashes, witnesses = {}, {}
    rule = partial(online_feedback, teacher=teacher)
    with patch.object(lag, 'feedback', rule), patch.object(io23, 'read_outcome',
            side_effect=RuntimeError('unsealed EAST outcome')):
        for rank, sid in enumerate(ids, 1):
            name = f'{rank:04d}_{sid}.npz'
            prev = lag_result/'shots'/name
            if sha(prev) != seal['curve_sha256'][name]:
                raise ValueError('old lag curves changed')
            with np.load(prev, allow_pickle=False) as f:
                times = f['times']
                curves = {k:f[k] for k in lag.ARMS}
                old_trace = {k:f[k] for k in ('release','radius')} if rank>lag.STARTUP else {}
            trace = {}
            if rank <= lag.STARTUP:
                curves.update(dict.fromkeys((TRUE, SHIFT, HOLD), curves['fixed']))
            else:
                if sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('locked signal changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'], times)
                sig = lag.prepare_actual(raw, rank, lock['mapping'])
                scored, trace = lag.encode(sig, args, decoder)
                for k in lag.BASES:
                    np.testing.assert_array_equal(scored[k], curves[k])
                np.testing.assert_array_equal(trace['release'], old_trace['release'])
                np.testing.assert_allclose(trace['radius'], old_trace['radius'], atol=2e-7, rtol=1e-6)
                if rank in (21, 120, 360):
                    end = lag.FIRST_EFFECT+3*lag.STRIDE+5
                    short, audit = lag.encode({k:sig[k][:end] for k in ('values','available')}, args, decoder)
                    for k in lag.ARMS:
                        np.testing.assert_allclose(short[k], scored[k][:end], atol=3e-6, rtol=0.)
                        for tau in thresholds.values():
                            np.testing.assert_array_equal(short[k]>=tau, scored[k][:end]>=tau)
                    np.testing.assert_array_equal(audit['release'], trace['release'][trace['release']<end])
                    witnesses[str(sid)] = end
                curves.update({k:scored[j] for k,j in zip((TRUE,SHIFT,HOLD), lag.ARMS[3:], strict=True)})
            if any(not np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite score')
            target = output/'shots'/name
            with target.open('xb') as f:
                np.savez_compressed(f, times=times, **curves, **trace)
            hashes[name] = sha(target)
            if rank%40==0 or rank==lag.STARTUP+1:
                print(dict(stage='sealed', rank=rank, seconds=round(time.monotonic()-started, 2)), flush=True)
    if tuple(state_digest(m) for m in args[:2]) != frozen:
        raise ValueError('frozen model changed')
    write(output/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses))
    per = {c: {k: [] for k in ARMS} for c in thresholds}
    updates, phase = 0, np.zeros(4)
    for rank, sid in enumerate(ids, 1):
        name = f'{rank:04d}_{sid}.npz'
        f = output/'shots'/name
        if sha(f) != hashes[name]:
            raise ValueError('curve not sealed')
        d, down = io23.read_outcome('EAST', sid)
        with np.load(f, allow_pickle=False) as z:
            if rank>lag.STARTUP:
                updates += int((z['radius'][:,0]>0.).sum())
                phase += z['next_phase_error'].sum(0)
            for c, group in per.items():
                tau = original if rank <= lag.STARTUP else thresholds[c]
                for k in ARMS:
                    group[k].append(dict(shot=sid, disruptive=d, down_s=down,
                                         **score_curve(z['times'], z[k], d, down, tau)))
    old = read(lag_result/'per_shot.json')
    if any(per[c][k] != old[c][k] for c in thresholds for k in lag.ARMS):
        raise ValueError('frozen prior alarm scoring changed')
    write(output/'per_shot.json', per)
    metrics = {c: {k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    paired = {c: {f'{a}__to__{b}':flips(g[a],g[b]) for a,b in COMPARISONS} for c,g in per.items()}
    write(output/'lead_sensitivity.json', report([output/'per_shot.json'], ARMS))
    write(output/'result.json', dict(status='COMPLETED_MATCHED_LAG_TEACHING_DEVELOPMENT',
        metrics=metrics, paired=paired, updates=updates, phase_error_sum=phase.tolist(),
        seconds=time.monotonic()-started))
    print({c:{k:(m['tp'],m['fp']) for k,m in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('reference', 'coordinates', 'forecast', 'lag-result', 'output')
    for name in names:
        parser.add_argument('--'+name, type=Path, required=True)
    cli = parser.parse_args()
    run(*(getattr(cli, name.replace('-', '_')).resolve() for name in names))
