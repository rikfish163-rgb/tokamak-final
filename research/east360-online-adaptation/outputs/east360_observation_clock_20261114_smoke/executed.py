"""Separate late writes, feedback releases and radiation availability on EAST360."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_history_span_20261114 as archive
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_group_carry_20261112 import HISTORY
from .commissioning_transfer_20261108 import contrast
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = archive.group, archive.lag, archive.layer
WAIT, DELAY = archive.WAIT, 8
RADIATION = tuple(i for i, k in enumerate(io23.RAW23) if k.startswith(('sxr_', 'xuv_')))
FEEDBACK = tuple(i for i, c in enumerate(lag.CHANNELS) if c in RADIATION)
CONTEXT = tuple(io23.RAW23.index(k) for k in ('ip', 'Bt'))
SCENARIOS = ('nominal', 'write', 'feedback', 'observation')
ARMS = (*archive.OLD, 'write_current', 'write_earlier',
        'feedback_parent', 'feedback_current', 'feedback_earlier',
        'observation_parent', 'observation_current', 'observation_earlier')
CASES = (70320, 73405, 76956, 79649, 71076)


def available_windows(values, mask, ends, width, delay, columns):
    """Insert arrived values at their physical times; leave pending slots missing."""
    slot = ends[:, None]-np.arange(width-1, -1, -1)
    row = slot.clip(min=0)
    visible = mask[row] & (slot >= 0)[..., None]
    if delay:
        visible[:, -delay:, columns] = False
    return np.where(visible, values[row], 0.), visible


def windows(signal, ends, width, delay):
    return available_windows(signal['values'], signal['available'], ends, width, delay, RADIATION)


def input_design(z, m, ctx, cm, ends, delay):
    if not delay:
        return lag.direct.shape.inputs(z, m, ctx, cm, ends)
    # Each packed row is one independently observed snapshot. The original
    # 128-point design sees exactly that row, no observation from another row.
    channels = z.shape[1]
    out, context = [], []
    for i in range(0, len(ends), 256):
        use = ends[i:i+256]
        x, mask = available_windows(z, m, use, lag.W, delay, FEEDBACK)
        c, avail = available_windows(ctx, cm, use, lag.W, 0, ())
        endpoint = np.arange(1, len(use)+1)*lag.W-1
        a, b = lag.direct.shape.inputs(x.reshape(-1, channels), mask.reshape(-1, channels),
            c.reshape(-1, 2), avail.reshape(-1, 2), endpoint)
        out.append(a)
        context.append(b)
    return np.concatenate(out), np.concatenate(context)


def motion(signal, ends, delay):
    x, m = signal['values'], signal['available']
    result = contrast(x, m)[ends]
    if delay:
        # The 32-point comparison still ends at the decision time. Pending
        # pairs are absent; replacing them by older pairs changes the task.
        v, have = x[:, RADIATION].astype(np.float64), m[:, RADIATION]
        paired = have[32:] & have[:-32]
        d = np.where(paired, v[32:]-v[:-32], 0.)
        a, b = (ends-63).clip(min=0), (ends-31-delay).clip(min=0)
        n, net, energy = (
            (s := np.pad(np.cumsum(w, axis=0), ((1, 0), (0, 0))))[b]-s[a]
            for w in (paired, d, d*d))
        moved = np.divide(net, np.sqrt(np.maximum(n*energy, 0.)),
                          out=np.zeros_like(net), where=(n>=16) & (energy>0.))
        result[:, RADIATION] = np.clip(moved, -1., 1.)
    return result


def innovation(q, release, residual, eligible):
    """Freeze the historical baseline at issue, not when the target arrives."""
    lo = np.searchsorted(q, q-group.bias.HISTORY*lag.STRIDE)
    hi = np.searchsorted(release, q, side='right')
    n = np.r_[0, np.cumsum(eligible)]
    count = n[hi]-n[lo]
    used = residual.masked_fill(~torch.as_tensor(eligible, device=residual.device)[:, None, None], 0.)
    c = torch.cat((used.new_zeros((1, *used.shape[1:]), dtype=torch.float64),
                   used.cumsum(0, dtype=torch.float64)))
    expected = ((c[hi]-c[lo]) /
                torch.as_tensor(count+1, device=residual.device)[:, None, None]).to(residual.dtype)
    current = residual-expected
    indices = np.where(eligible & (count>=2), np.arange(len(q)), -1)
    prev = np.maximum.accumulate(np.r_[-1, indices])[hi]
    older = torch.cat((residual.new_zeros((1, *residual.shape[1:])), current))[prev+1]
    return expected, current, older, eligible & (count>=2) & (prev>=lo)


def shift_state(x, delay):
    return torch.nn.functional.pad(x, (0, 0, delay, 0))[:len(x)]


def _batch_windows(base, signal, end, age):
    if not age:
        return lag._windows(base, signal, end)
    return tuple(torch.as_tensor(a, device=base.center.device)
                 for a in windows(signal, end, base.cfg.window, age))


@torch.no_grad()
def _observed_track(signal, net, params, age):
    cen, scale = params[:2]
    ends = np.arange(len(signal['values']))
    if not age:
        return lag.direct.track(net, signal, cen, scale, ends)
    z, m, ctx, cm = lag.direct.weak.shot_coordinates(signal, cen, scale)
    q = np.arange(lag.FIRST, len(z), lag.STRIDE)
    x, coords = input_design(z, m, ctx, cm, ends, age)
    pre = net.stem(torch.from_numpy(x)).numpy()
    pre[:lag.FIRST] = 0.
    recent, nr = _window_mean(z, m, q, -15, 16)
    prior, np_ = _window_mean(z, m, q, -31, 16)
    # DELAY=8 affects just the newest half of a 16-point statistic at issue.
    now, n = _window_mean(z[:, FEEDBACK], m[:, FEEDBACK], q, -15, 16-age)
    recent[:, FEEDBACK], nr[:, FEEDBACK] = now, n
    velocity = np.where((nr>=8)&(np_>=8), recent-prior, 0.)
    design = np.column_stack((np.tanh(pre[q]), np.tanh(coords[q]), np.tanh(velocity))).astype(np.float32)
    physical_key = .5*np.column_stack((np.ones(len(z)), np.tanh(coords[:, 0]),
                                      np.tanh(coords[:, 1]), np.tanh(4*coords[:, 2]))).astype(np.float32)
    future = q[:, None]+np.arange(1, lag.H+1)
    index = future.clip(max=len(z)-1)
    observed = m[index] & (future<len(z))[..., None] & (nr[:, None, :]>=8)
    y = np.where(observed, z[index]-recent[:, None, :], 0.).astype(np.float32)
    return dict(q=q, design=design, pre=pre, key=physical_key, target=y, observed=observed, length=len(z))


@torch.no_grad()
def _radiation_view(base, signal, age):
    if not age:
        return lag.features(base, signal, 'cuda')[0]
    hidden = []
    handle = base.risk_head.register_forward_pre_hook(lambda _m, a: hidden.append(a[0].cpu().numpy()))
    try:
        ends = np.arange(len(signal['values']))
        for j in range(0, len(ends), 256):
            base(*_batch_windows(base, signal, ends[j:j+256], age))
    finally:
        handle.remove()
    return np.column_stack((np.concatenate(hidden), motion(signal, ends, age)))


@torch.no_grad()
def parent_features(signal, args, age, delay):
    base, net, params, mapper, quiet, persist_map, coef, _ = args
    tr = _observed_track(signal, net, params, age)
    op = lag.history.previous
    state, _ = op.estimate(tr, signal, *params[2:])
    use, key = torch.from_numpy(tr['pre']), torch.from_numpy(tr['key'])
    old = shift_state(torch.from_numpy(state['uncalibrated']), delay)
    use = use + torch.einsum('...k,...f,kfw->...w', key, old, mapper)
    p, q = lag.persistent, tr['q']
    forecast = lag.direct.base_forecast(tr, params[2])
    target = lag.direct.normalized_target(tr, params[3])
    values, mask = p.levels(forecast, target, tr['observed'])
    cm = signal['available'][:, CONTEXT]
    count = np.vstack((np.zeros((1, 2), np.int64), np.cumsum(cm, axis=0)))
    end = q+1
    usable = ((count[end]-count[end-16])>=8).all(1) & ((count[end-16, 0]-count[end-32, 0])>=8)
    context = tr['design'][:, lag.direct.WIDTH:lag.direct.WIDTH+3]/params[4]
    state, _ = p.replay(values, mask, q, context, usable, tr['length'], quiet)
    fresh = shift_state(torch.from_numpy(state['current_equal_norm']), delay)
    use += torch.einsum('...k,...f,kfw->...w', key, 4.*fresh, persist_map)
    features = _radiation_view(base, signal, age)
    scores = torch.from_numpy(np.column_stack((features, use.tanh().numpy()))).to(coef.device)
    return tr, scores[:, :base.cfg.dim], scores @ coef[:-1]+coef[-1], coef[:base.cfg.dim]


def bias_estimate(q, release, mean_error):
    lo = np.searchsorted(q, q-group.bias.HISTORY*lag.STRIDE)
    hi = np.searchsorted(release, q, side='right')
    x = np.clip(mean_error, -.5, .5)
    s = np.concatenate((np.zeros((1, x.shape[1]), dtype=x.dtype), np.cumsum(x, axis=0)))
    n = hi-lo
    return np.where(n[:, None]>=2, (s[hi]-s[lo])/(n+1)[:, None], 0.).astype(x.dtype)


def _packet_fields(signal, params, q, age, device):
    eligible, fields, waveform = group.feedback_waveform(signal, q, params, device)
    if not age:
        return eligible, fields, waveform
    z, m, *_ = lag.direct.weak.shot_coordinates(signal, *params[:2])
    reduced, count = _window_mean(z[:, FEEDBACK], m[:, FEEDBACK], q, -15, 16-age)
    recent, center, spread, noise = fields
    recent[:, FEEDBACK] = torch.as_tensor(reduced, device=device, dtype=recent.dtype)
    # The disjoint reference used by feedback_waveform ends q-10. All of it
    # has arrived by q in this eight-sample stress scenario.
    return eligible & (count>=8).all(1), (recent, center, spread, noise), waveform


def packets(signal, args, decoder, path, weights, age, delay):
    """Build the same parent and transport each issued residual at its legal release."""
    base, _, params = args[:3]
    if age:
        assert age<=group.recovery.REFERENCE_LAG
    if delay:
        tr, h, z, beta = parent_features(signal, args, age, delay)
        q, key, prior, target = group.teacher._mature(tr, params[2], params[3])
        keep = q+lag.H+1+delay < len(h)
        q, key, prior, target = (t[keep] for t in (q, key, prior, target))
    else:
        q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    rel = q+lag.H+1+delay
    device = z.device
    if delay:
        prior, target = (torch.as_tensor(t, device=device) for t in (prior, target))
        real = []
        for j in range(0, len(q), group.BATCH):
            w = slice(j, j+group.BATCH)
            x,m = _batch_windows(base, signal, q[w], age)
            theta, loss, _ = group.teacher._signature_problem(base, x, m, prior[w], decoder, target[w])
            jac = torch.stack([torch.autograd.grad(loss[:, i].sum(), theta, retain_graph=i+1<loss.shape[1])[0]
                               for i in range(loss.shape[1])], 1)
            real.append(group.guard.basis(jac))
        phase = z.new_zeros((len(q)+1, len(lag.CHANNELS), len(lag.CHANNELS)))
        if real:
            pairs = (torch.cat(t) for t in zip(*real, strict=True))
            phase[1:] = -lag.RADIUS*group.guard.signed_direction(*pairs, torch.as_tensor(key, device=device), path)
        error = (target-lag._predict(h[q], prior, decoder)).mean(1).cpu().numpy()
        b = bias_estimate(q, rel, error)
        correction = torch.as_tensor(np.pad(b, ((1, 0), (0, 0))), device=device)*torch.as_tensor(
            params[3], device=device)
    eligible, fields, waveform = _packet_fields(signal, params, q, age, device)
    force = z.new_zeros((len(q), lag.H//group.BIN, len(group.USE)))
    e, k = base.cfg.window//base.cfg.patch-1, len(group.TRUSTED)
    states = z.new_zeros((2, len(q)+1, e, k))
    _, _, _, common = innovation(q, rel, force, eligible)
    recent, cen, spread, noise = fields
    slots = np.searchsorted(rel, q, side='right')
    indices = np.flatnonzero(eligible)
    for j in range(0, len(indices), group.BATCH):
        i = indices[j:j+group.BATCH]
        x,m = _batch_windows(base, signal, q[i], age)
        ss = slots[i]
        theta = z.new_zeros((len(i),e,k), requires_grad=True)
        issued = group.forward_hidden(base, x, m, layer.patcher(phase[ss], correction[ss]), state=theta)
        waveform_pred = lag._predict(issued, prior[i], decoder)*noise+recent[i, None]
        pred = group.signed_levels(waveform_pred, cen[i], spread[i])
        target = group.signed_levels(waveform[i], cen[i], spread[i])
        force[i] = ((target-pred.detach())/group.BETA).clamp(-1., 1.)
        _, now, old, _ = innovation(q, rel, force, eligible)
        x,m = _batch_windows(base, signal, rel[i], age)
        ds = torch.zeros_like(theta, requires_grad=True)
        a = group.forward_hidden(base, x, m, layer.patcher(phase[i+1], correction[i+1]), state=ds)
        gate = torch.as_tensor(common[i], device=device)[:, None, None]
        specs = tuple((pred, group.shape_force(f[i], weights)*gate) for f in (now,old))
        states[:, i+1] = group.signed_directions(issued,a,ds,specs)
    return h, z, beta, q, rel, phase, correction, common, states


@torch.no_grad()
def render(base, signal, track, age, extra_write=0):
    h,z,beta,q,rel,phase,correction,common,states = track
    clock = rel+WAIT+extra_write
    n = len(z)
    parent = z.sigmoid().cpu().numpy()
    out = np.broadcast_to(parent, (2,n)).copy()
    ends = np.arange(n)
    pos = np.searchsorted(rel, ends, side='right')
    slot = np.searchsorted(clock, ends, side='right')
    active = np.flatnonzero(pos>0)
    dose = np.zeros(n,np.float32)
    error = 0.
    for j in range(0,len(active),group.BATCH):
        t = active[j:j+group.BATCH]
        p,s = pos[t],slot[t]
        old = (s-1).clip(min=0)
        # The frozen parent can change as soon as feedback is released. Only
        # the additional grouped update observes its extra write wait.
        on = torch.as_tensor(np.r_[False, common][s],device=z.device)
        frac = group.handoff_fraction(np.r_[-1, clock][s],t,base.cfg.patch)
        x,m = _batch_windows(base, signal, t, age)
        v,d,e = group.forward_hidden(base,x,m,layer.patcher(phase[p],correction[p]),writes=states[:,s],
            handoff=(states[:,old]*on[None,:,None,None],torch.as_tensor(frac,device=z.device,dtype=z.dtype)))
        y = (z[t]+((v-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        parent[t] = y[0]
        dose[t] = d.cpu().numpy()
        out[:,t] = np.where(dose[t]>0.,y[1:],y[0])
        error = max(error,float(e))
    if error>1e-5 or not np.isfinite(out).all():
        raise ValueError('nonfinite prediction or unequal first-block token dose')
    return np.vstack((parent,out)),dose,error


def evaluate(signal, args, decoder, path, weights):
    base = args[0]
    nominal = packets(signal, args, decoder, path, weights, 0, 0)
    a, da, ea = render(base, signal, nominal, 0)
    b, db, eb = render(base, signal, nominal, 0, extra_write=DELAY)
    np.testing.assert_array_equal(a[0], b[0])
    scenarios = ((0, DELAY), (DELAY, DELAY))
    rendered, dose, errors = [a, b], [da, db], [ea, eb]
    release, eligible = [nominal[4], nominal[4]], [nominal[-2], nominal[-2]]
    for age, wait in scenarios:
        last = packets(signal, args, decoder, path, weights, age, wait)
        c, d, e = render(base, signal, last, age)
        rendered.append(c)
        dose.append(d)
        errors.append(e)
        release.append(last[4])
        eligible.append(last[-2])
    keys = ARMS[1:]
    curves = np.concatenate((rendered[0], rendered[1][1:], *rendered[2:]))
    return dict(zip(keys, curves, strict=True)), dict(dose=np.stack(dose),
        **{f'{k}_{n}': v for k, a in (('release',release), ('eligible',eligible))
           for n,v in zip(SCENARIOS,a,strict=True)}), max(errors)


def empty_trace(n):
    return dict(dose=np.zeros((len(SCENARIOS),n),np.float32),
                **{f'release_{s}': np.empty(0,np.int64) for s in SCENARIOS},
                **{f'eligible_{s}': np.empty(0,bool) for s in SCENARIOS})


def seal(out, rank_only=None):
    start = time.monotonic()
    lock, protocol, previous, args, decoder, path = archive.inputs()
    asset = HISTORY/'teaching.json'
    if sha(asset)!=read(HISTORY/'protocol.json')['teaching_sha256']:
        raise ValueError('original frozen diagnostic teacher changed')
    weights = torch.as_tensor(read(asset)['weights'],device=args[0].center.device,dtype=torch.float32)
    thresholds = tuple(protocol['calibration'].values())
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(scope='EAST360_RADIATION_AVAILABILITY_DEVELOPMENT',
        **{k:protocol[k] for k in archive.LOCK}, arms=ARMS, rank_only=rank_only,
        reference=str(archive.ARCHIVE), reference_seal=sha(archive.ARCHIVE/'seal.json'),
        radiation=[io23.RAW23[c] for c in RADIATION],
        affected_feedback=[io23.RAW23[lag.CHANNELS[c]] for c in FEEDBACK],
        delays_samples=dict(nominal=0, late_write=DELAY, late_feedback=DELAY, late_observation=DELAY),
        timing=dict(issuance_first=int(lag.FIRST), stride=lag.STRIDE, horizon=lag.H,
                    normal_release=lag.H+1, handoff=args[0].cfg.patch, original_wait=WAIT),
        meaning={
            'write':'normal observations and all forecast matching; delay only last-layer-input write',
            'feedback':'normal prediction observations; delayed complete diagnostic packet to every adaptation path',
            'observation':'same delayed packets; radiation samples enter all inputs only after eight ticks, '
                          'at their original physical indices, no imputation'},
        history='use only completed packets at issue; shift inherited uncalibrated and persistent release '
                'states, not the current diagnostic features or the time axis',
        scoring='fixed original clock, cold20, HDF5 >30ms after sealing; zero new persistent updates',
        dose='equal across real and older residual within each scenario; the available basis, '
             'handoff and lag can change dose across scenarios'))
    files, witnesses = {}, {}
    replay_err = write_err = 0.
    frozen = [state_digest(m) for m in args[:2]]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome')):
        for rank, sid in enumerate(protocol['order'],1):
            if rank_only is not None and rank!=rank_only:
                continue
            source = archive.archived_shot(rank,sid,previous)
            if rank<=protocol['startup']:
                trace = empty_trace(len(source['fixed']))
                curves = dict.fromkeys(ARMS, source['fixed'])
            else:
                raw_file = io23.path_for('EAST',sid)
                if sha(raw_file)!=lock['hashes'][str(sid)]:
                    raise ValueError('EAST signal changed relative to fixed benchmark')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                curves, trace, err = evaluate(sig,args,decoder,path,weights)
                curves['fixed'] = source['fixed']
                np.testing.assert_array_equal(trace['release_nominal'], source['release'])
                np.testing.assert_array_equal(trace['eligible_nominal'], source['eligible'])
                np.testing.assert_allclose(trace['dose'][0],source['dose'],atol=3e-6,rtol=0.)
                for k in archive.OLD[1:]:
                    np.testing.assert_allclose(curves[k],source[k],atol=3e-6,rtol=0.)
                    for th in thresholds:
                        np.testing.assert_array_equal(curves[k]>=th,source[k]>=th)
                    replay_err = max(replay_err,float(abs(curves[k]-source[k]).max()))
                write_err = max(write_err,err)
                if rank in (21,120,360) or sid in CASES or rank_only==rank:
                    n = min(620,len(sig['values']))
                    short, st, _ = evaluate({k:v[:n] for k,v in sig.items()},args,decoder,path,weights)
                    pe = 0.
                    for k in short:
                        np.testing.assert_allclose(short[k],curves[k][:n],atol=3e-6,rtol=0.)
                        for th in thresholds:
                            np.testing.assert_array_equal(short[k]>=th,curves[k][:n]>=th)
                        pe = max(pe,float(abs(short[k]-curves[k][:n]).max()))
                    for scenario in SCENARIOS:
                        rel = trace[f'release_{scenario}']
                        for k in ('release','eligible'):
                            np.testing.assert_array_equal(st[f'{k}_{scenario}'],
                                                          trace[f'{k}_{scenario}'][rel<n])
                    np.testing.assert_allclose(st['dose'],trace['dose'][:,:n],atol=3e-6,rtol=0.)
                    witnesses[str(sid)] = dict(points=n,max_score_error=pe)
            name = f'{rank:04d}_{sid}.npz'
            f = out/'shots'/name
            with f.open('xb') as dest:
                np.savez_compressed(dest,**curves,**trace)
            files[name] = sha(f)
            if rank in (21,60,120,180,240,300,360) or rank_only:
                print(dict(stage='sealed',rank=rank,elapsed_s=round(time.monotonic()-start,2)),flush=True)
    if [state_digest(m) for m in args[:2]]!=frozen:
        raise ValueError('frozen parent mutated')
    write(out/'seal.json',dict(curves=files,prefix=witnesses,
          max_frozen_replay_error=replay_err,max_token_dose_error=write_err,elapsed_s=time.monotonic()-start))


def score(out):
    p, manifest = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if (p['rank_only'] is not None or len(manifest['curves'])!=360
            or p['reference_seal']!=sha(archive.ARCHIVE/'seal.json')):
        raise ValueError('complete frozen prediction stream required before HDF5 scoring')
    previous = read(archive.ARCHIVE/'per_shot.json')
    per = {band:{a:[] for a in ARMS} for band in p['calibration']}
    cases = []
    paths = (
        ('bias_only','current_5','earlier_5'),
        ('bias_only','write_current','write_earlier'),
        ('feedback_parent','feedback_current','feedback_earlier'),
        ('observation_parent','observation_current','observation_earlier'))
    coverage = {s:dict(endpoints=0, shots=0, common_prior_pulses=0) for s in SCENARIOS}
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        f = out/'shots'/name
        if sha(f)!=manifest['curves'][name]:
            raise ValueError('sealed curve changed')
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(f,allow_pickle=False) as q:
            nominal_late = rank>p['startup'] and y and not previous['0.2']['current_5'][rank-1]['timely']
            near = (np.round((down-t)*1000.,9)>30.) & (np.round((down-t)*1000.,9)<=125.) if y else None
            if nominal_late:
                m = dict(shot=sid)
            for s,(parent,real,old),dose in zip(SCENARIOS,paths,q['dose'],strict=True):
                rel, eligible = q[f'release_{s}'],q[f'eligible_{s}']
                wait = WAIT+(DELAY if s=='write' else 0)
                slots = np.searchsorted(rel+wait,np.flatnonzero(dose>0.),side='right')
                if not np.r_[False,eligible][slots].all():
                    raise ValueError('state write without a completed observation packet')
                for k in (real,old):
                    np.testing.assert_array_equal(q[k][dose==0.],q[parent][dose==0.])
                v = coverage[s]
                v['endpoints'] += int(np.count_nonzero(dose))
                v['shots'] += bool(dose.any())
                v['common_prior_pulses'] += int(eligible.sum())
                if nominal_late:
                    apply = near & (dose>0.)
                    m[s] = dict(active=int(apply.sum()),parent_gap=None,max_upward=None)
                    if apply.any():
                        up = logit(np.clip(q[real][apply].astype(float),1e-7,1.-1e-7))
                        base = logit(np.clip(q[parent][apply].astype(float),1e-7,1.-1e-7))
                        m[s].update(parent_gap=float(logit(p['calibration']['0.2'])-base.max()),
                                    max_upward=float((up-base).max()))
            if nominal_late:
                cases.append(m)
            for b, tau in p['calibration'].items():
                threshold = p['original'] if rank<=p['startup'] else tau
                for a in ARMS:
                    row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,q[a],y,down,threshold))
                    if a in archive.OLD and row!=previous[b][a][rank-1]:
                        raise ValueError('nominal first alarm differs from frozen control')
                    per[b][a].append(row)
    write(out/'per_shot.json',per)
    comparisons = (('current_5','write_current'), ('write_current','feedback_current'),
                   ('feedback_parent','observation_parent'), ('feedback_current','observation_current'),
                   ('bias_only','feedback_parent'),
                   *((b,a) for b,a,c in paths), *((c,a) for b,a,c in paths))
    summaries = {b:{a:lead.summarize(rows) for a,rows in group_.items()} for b,group_ in per.items()}
    pair = {b:{f'{x}__to__{z}':dict(flips=flips(g[x],g[z]),leads=lead.paired(g[x],g[z]))
               for x,z in comparisons} for b,g in per.items()}
    write(out/'result.json',dict(status='COMPLETED_EAST360_SYNTHETIC_ARRIVAL_STRESS',
          raw_scored=360*len(ARMS)*len(per), metrics=summaries, paired=pair,
          coverage=coverage, late_or_missed_nominal=cases))
    print({b:{a:(v['by_deadline']['30']['tp'],v['fp']) for a,v in r.items()}
           for b,r in summaries.items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    ap.add_argument('--rank',type=int,choices=range(1,361),metavar='1..360')
    cli = ap.parse_args()
    seal(cli.out.resolve(),cli.rank)
    if cli.rank is None:
        score(cli.out.resolve())
