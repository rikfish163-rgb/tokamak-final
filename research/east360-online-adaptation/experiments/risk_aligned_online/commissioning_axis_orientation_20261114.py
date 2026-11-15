"""Separate AXUV-skew motion sensitivity from same-shot directional adaptation."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_axis_forecast_20261114 as axis
from . import commissioning_history_span_20261114 as history
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = history.group, history.lag, history.layer
PATCH, WAIT, OLD = history.PATCH, history.WAIT, history.OLD
BLOCKS = 4
STATIC = ('flatten_motion', 'double_motion', 'reflect_motion', 'triple_motion')
STATIC_CHANGE = (-1., 1., -2., 2.)
LIVE = ('axis_current', 'axis_wrong_order', 'axis_previous', 'same_dose_reduce', 'same_dose_enhance')
ARMS = (*OLD, *STATIC, 'legacy_matched', *LIVE)
PREDICTORS = ('cross_diagnostic', 'wrong_order', 'no_change')
CASES = (70320, 73405, 79649, 71076)


def lane_basis(patches, embedding):
    """Project observed, slot-centered AXUV motion to time tokens before attention."""
    x = patches.unflatten(-1, (PATCH, 46))
    c = axis.TARGET_INDEX
    valid = x[..., 23+c].bool()
    z = x[..., c].masked_fill(~valid, 0.)
    center = z.sum(1, keepdim=True)/valid.sum(1, keepdim=True).clamp_min(1)
    motion = (z-center).masked_fill(~valid, 0.)
    return motion @ embedding.unflatten(-1, (PATCH, 46))[..., c].T


def rolling_alignment(q, prediction, target, valid):
    """Direction from the last four matched blocks; hold previous only for comparison."""
    n = len(q)
    out = np.zeros((3, n), dtype=np.float64)
    common = np.zeros(n, dtype=bool)
    if n < BLOCKS+1:
        return out, common
    i = np.arange(BLOCKS-1, n)
    window = i[:, None]-np.arange(BLOCKS-1, -1, -1)
    p, y = np.nan_to_num(prediction[window]), np.nan_to_num(target[window])
    scale = np.sqrt(np.square(p).sum((1, 2))*np.square(y).sum((1, 2)))
    eligible = valid[window].all(1) & (scale > 1e-12)
    score = np.zeros((2, n), dtype=np.float64)
    for a, yy in enumerate((y, np.roll(y, 2, axis=2))):
        look = (p*yy).sum((1, 2))
        score[a, i] = np.divide(look, scale, out=np.zeros_like(look), where=eligible).clip(-1., 1.)
    ok = np.r_[np.zeros(BLOCKS-1, dtype=bool), eligible]
    common[1:] = ok[1:] & ok[:-1]
    out[:, 1:] = np.vstack((score[:, 1:], score[0, :-1]))
    return out, common


def handoff(rho, qualified, q, endpoints):
    release = q+axis.RELEASE_OFFSET+WAIT
    slot = np.searchsorted(release, endpoints, side='right')
    full = np.c_[np.zeros(3), rho] * np.r_[False, qualified]
    if not len(q):
        return full[:, slot]
    frac = group.handoff_fraction(np.r_[0, release][slot], endpoints, PATCH)
    old = full[:, np.maximum(slot-1, 0)]
    # No action in an unsupported slot, even if the preceding one was valid.
    return (old+(full[:, slot]-old)*frac)*np.r_[False, qualified][slot]


def aligned_write(basis, rho, cap):
    """Match the realized current-motion dose across both old clocks and fixed signs."""
    norm = basis.flatten(1).norm(dim=1)
    # The true state requests g = 1 + 2*rho. Only its direction, not dose,
    # can differ across the older, phase-displaced and fixed-sign controls.
    supported = (rho.abs()>group.EPS).all(0) & (norm>group.EPS)
    dose = torch.minimum(2.*rho[0].abs()*norm, cap).masked_fill(~supported, 0.)
    size = dose/norm.clamp_min(group.EPS)
    sign = torch.cat((rho.sign(), -torch.ones_like(rho[:1]), torch.ones_like(rho[:1])))
    return sign[..., None, None]*size[:, None, None]*basis, dose, 1.+sign[0]*size


def issued_blocks(signal):
    q = np.arange(lag.FIRST, len(signal['values']), lag.STRIDE)
    return (q, *axis.make_samples({k: signal[k] for k in ('values', 'available')}, q))


def forecast_errors(prediction, target):
    p = np.stack((prediction, np.roll(prediction, -2, axis=1), np.zeros_like(prediction)), axis=1)
    e = np.abs(p-target[:, None])
    return np.where(e<.2, e*e/.4, e-.1).mean(axis=2).astype(np.float32)


def fit_source(out):
    roles = simple_data.source_roles()
    fit_support, val = {}, []

    def samples(role):
        for sid in roles[role]:
            signal = simple_data.read_shot('J-TEXT', sid)
            _, x, issued, y, mature = issued_blocks(signal)
            both = issued & mature
            yield sid, x[both], y[both]

    def packets():
        for sid, x, y in samples('train'):
            fit_support[sid] = len(x)
            yield x, y

    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('forecast uses signals only')):
        coefficient = axis.fit_weighted(packets())
        for sid, x, y in samples('val'):
            pred = axis.predict(x, coefficient)
            e = forecast_errors(pred, y)
            active = (abs(y[:, -1]-y[:, 0])>1e-8) & (abs(pred[:, -1]-pred[:, 0])>1e-8)
            correct = (y[:, -1]-y[:, 0])*(pred[:, -1]-pred[:, 0])>0.
            val.append(dict(shot=sid, blocks=len(x), compared_signs=int(active.sum()),
                            matching_signs=int((active & correct).sum()),
                            error=e.mean(0).tolist() if len(e) else None))
    used = np.asarray([r['error'] for r in val if r['error'] is not None])
    metrics = dict(usable_train_shots=sum(k>0 for k in fit_support.values()),
                   train_blocks=sum(fit_support.values()), usable_val_shots=len(used),
                   val_blocks=sum(r['blocks'] for r in val),
                   val_signs=[sum(r[k] for r in val) for k in ('matching_signs', 'compared_signs')],
                   val_shot_average=dict(zip(PREDICTORS, used.mean(0).tolist(), strict=True)),
                   val_wins_against_zero=int((used[:, 0]<used[:, 2]).sum()))
    out.mkdir(parents=True)
    write(out/'source_forecast.json', dict(
        fit_ids=roles['train'], val_ids=roles['val'], features=axis.FEATURE_NAMES,
        ridge=axis.RIDGE, coefficient=coefficient.tolist(), train_support=fit_support, val=val, metrics=metrics))
    copyfile(axis.__file__, out/'executed_forecast.py')
    copyfile(__file__, out/'executed.py')
    print(dict(stage='source_forecast', **metrics), flush=True)


@torch.no_grad()
def encoder(base, signal, ends, q, phase, correction, *, rho=None, cap=None):
    x, mask = lag._windows(base, signal, ends)
    slot = lag.delayed_slots(q, ends)[0]
    transform = layer.patcher(phase[slot], correction[slot])
    batch, count = len(x), 1+len(STATIC if rho is None else LIVE)
    ret, trace = [], {}

    def prepare(_m, args):
        adjusted = transform(args[0][:batch])
        basis = lane_basis(adjusted, base.embedding.weight)
        if rho is None:
            wanted = x.new_tensor(STATIC_CHANGE)[:, None, None, None]*basis
            trace['dose'] = torch.empty(0, device=x.device)
            trace['gain'] = torch.empty(0, device=x.device)
        else:
            inherited = F.linear(adjusted-args[0][:batch],base.embedding.weight).flatten(1).norm(dim=1)
            wanted, trace['dose'], trace['gain'] = aligned_write(
                basis, torch.as_tensor(rho, device=x.device, dtype=x.dtype),
                torch.minimum(torch.as_tensor(cap, device=x.device), inherited))
        trace['delta'] = wanted
        return (adjusted.repeat(count, 1, 1),)

    def inject(_m, args):
        v = args[0]
        before = v[batch:, :-1].unflatten(0, (count-1, batch))
        change = trace['delta']
        actual = ((before+change)-before).flatten(2).norm(dim=2)
        if rho is None:
            err = actual-change.flatten(2).norm(dim=2)
        else:
            err = actual-trace['dose'][None]
        trace['error'] = float(err.abs().max())
        joined = torch.cat((torch.zeros_like(change[:1]), change)).flatten(0, 1)
        return (layer.add_to_patches(v, joined), *args[1:])

    hooks = (base.embedding.register_forward_pre_hook(prepare),
             base.encoder.layers[0].register_forward_pre_hook(inject),
             base.risk_head.register_forward_pre_hook(lambda _m, a: ret.append(a[0])))
    try:
        base(x.repeat(count, 1, 1), mask.repeat(count, 1, 1))
    finally:
        for handle in hooks:
            handle.remove()
    return (ret[0].unflatten(0, (count, batch)), trace['dose'].cpu().numpy(),
            trace['gain'].cpu().numpy(), trace['error'])


def track_shot(signal, args, decoder, path, archive, coefficient):
    base = args[0]
    q, h, z, beta, _, phase, correction = group.parent_track(signal, args, decoder, path)
    np.testing.assert_array_equal(q+lag.H+1, archive['release'])
    issued, design, available, measured, mature = issued_blocks(signal)
    prediction = axis.predict(design, coefficient)
    valid = available & mature
    rho, qualified = rolling_alignment(issued, prediction, measured, valid)
    scored = np.full((len(issued), len(PREDICTORS)), np.nan, dtype=np.float32)
    scored[valid] = forecast_errors(prediction[valid], measured[valid])
    parent = archive['bias_only']
    curves = np.broadcast_to(parent, (len(STATIC)+len(LIVE), len(parent))).copy()
    legacy = parent.copy()
    dose, gain = np.zeros_like(parent), np.ones_like(parent)
    maximum = np.zeros(2)

    def evaluate(pos, *, dynamic):
        keys = slice(len(STATIC), None) if dynamic else slice(0, len(STATIC))
        rm = handoff(rho, qualified, issued, pos) if dynamic else None
        for o in range(0, len(pos), group.BATCH):
            j = slice(o, o+group.BATCH)
            i = pos[j]
            view, used, ge, error = encoder(base, signal, i, q, phase, correction,
                rho=rm[:, j] if dynamic else None, cap=archive['dose'][i] if dynamic else None)
            prob = (z[i]+((view-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
            chosen = used>0. if dynamic else np.ones(len(i), dtype=bool)
            where = i[chosen]
            curves[keys, where] = prob[1:, chosen]
            if dynamic:
                dose[i], gain[i] = used, ge
                legacy[where] = archive['current_5'][where]
            maximum[:] = np.maximum(maximum, (error, abs(prob[0]-parent[i]).max()))

    evaluate(np.arange(lag.FIRST, len(parent)), dynamic=False)
    points = np.flatnonzero(archive['dose']>0.)
    active_slot = np.searchsorted(issued+axis.RELEASE_OFFSET+WAIT, points, side='right')
    evaluate(points[np.r_[False, qualified][active_slot]], dynamic=True)
    # 2x reflection/amplification may use a larger norm than the permitted
    # adaptive action; the finite probes are not an online calibration attempt.
    if maximum[0]>5e-5 or maximum[1]>3e-6 or not np.isfinite(curves).all():
        raise ValueError(f'changed frozen parent or finite token action: {maximum}')
    return (dict(legacy_matched=legacy, **dict(zip((*STATIC, *LIVE), curves, strict=True))),
        dict(q=issued, valid=valid, qualified=qualified, errors=scored, rho=rho,
             dose=dose, gain=gain), maximum)


def replay(out, *, smoke_rank=None):
    start = time.monotonic()
    lock, p, s, args, decoder, path = history.inputs()
    asset = read(out/'source_forecast.json')
    roles = simple_data.source_roles()
    if (asset['fit_ids']!=roles['train'] or asset['val_ids']!=roles['val']
            or asset['features']!=list(axis.FEATURE_NAMES) or axis.HORIZON!=lag.H
            or io23.RAW23[axis.TARGET_INDEX]!='xuv_skew'):
        raise ValueError('fixed source-only AXUV forecast required')
    coefficient = np.asarray(asset['coefficient'])
    (out/'shots').mkdir()
    write(out/'protocol.json', dict(scope='EAST360_CAUSAL_AXIS_MOTION_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK}, arms=ARMS, smoke_rank=smoke_rank,
        reference=str(history.ARCHIVE), reference_seal=sha(history.ARCHIVE/'seal.json'),
        source_forecast='source_forecast.json', channel=io23.RAW23[axis.TARGET_INDEX],
        fixed_gains=dict(zip(STATIC, (0, 2, -1, 3), strict=True)),
        clock=dict(first=lag.FIRST, stride=lag.STRIDE, horizon=axis.HORIZON,
                   release=axis.RELEASE_OFFSET, wait=WAIT, blocks=BLOCKS, handoff=PATCH),
        update='g = 1 + 2 * cosine(predicted, observed); limit embedded dose to archived old '
               'token dose; controls use same realized dose',
        reset='shot-local adaptation; unchanged parent, cold20, thresholds, and no inter-shot updates'))
    frozen = [state_digest(m) for m in args[:2]]
    files, witnesses, maxima = {}, {}, np.zeros(2)
    for rank, sid in enumerate(p['order'],1):
        if smoke_rank is not None and rank!=smoke_rank:
            continue
        arc = history.archived_shot(rank, sid, s)
        parent = arc['bias_only']
        if rank<=p['startup']:
            curves = dict.fromkeys(ARMS[len(OLD):], parent)
            detail = dict(q=np.empty(0, np.int64), valid=np.empty(0, bool),
                          qualified=np.empty(0, bool), errors=np.empty((0, len(PREDICTORS))),
                          rho=np.empty((3, 0)), dose=np.zeros_like(parent), gain=np.ones_like(parent))
        else:
            with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before seal')):
                if sha(io23.path_for('EAST', sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST record changed')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                curves, detail, error = track_shot(signal, args, decoder, path, arc, coefficient)
                maxima = np.maximum(maxima, error)
                if rank in (21,120,360) or sid in CASES or smoke_rank==rank:
                    n = min(len(parent), 620)
                    short = {k:signal[k][:n] for k in ('values','available')}
                    sub = {k:arc[k][:n] for k in (*OLD,'dose')}
                    m = arc['release']<n
                    sub.update(release=arc['release'][m], eligible=arc['eligible'][m])
                    trunc, d, _ = track_shot(short, args, decoder, path, sub, coefficient)
                    for k, v in trunc.items():
                        np.testing.assert_allclose(v, curves[k][:n], atol=3e-6, rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(v>=tau, curves[k][:n]>=tau)
                    np.testing.assert_allclose(d['gain'], detail['gain'][:n], atol=3e-6, rtol=1e-6)
                    np.testing.assert_array_equal(d['dose']>0., detail['dose'][:n]>0.)
                    matured = d['q']+axis.RELEASE_OFFSET<n
                    for name in ('q', 'valid', 'qualified', 'errors'):
                        np.testing.assert_array_equal(d[name][matured], detail[name][:len(matured)][matured])
                    np.testing.assert_array_equal(d['rho'][:, matured], detail['rho'][:, :len(matured)][:, matured])
                    witnesses[str(sid)] = dict(points=n, active=int((d['dose']>0.).sum()),
                        max_error=max(float(np.max(abs(v-curves[k][:n]))) for k,v in trunc.items()))
        name = f'{rank:04d}_{sid}.npz'
        with (out/'shots'/name).open('xb') as f:
            np.savez_compressed(f, **{k:arc[k] for k in OLD}, **curves, **detail)
        files[name] = sha(out/'shots'/name)
        if rank in (21,60,120,180,240,300,360) or smoke_rank==rank:
            print(dict(stage='axis_seal', rank=rank, active=int((detail['dose']>0.).sum()),
                       seconds=round(time.monotonic()-start,2)), flush=True)
    if frozen != [state_digest(m) for m in args[:2]]:
        raise ValueError('frozen source or inherited network changed')
    write(out/'seal.json',dict(curves=files, witnesses=witnesses,
                              maximum_error=maxima.tolist(), elapsed_s=time.monotonic()-start))


def score(out):
    p, s = (read(out/f'{name}.json') for name in ('protocol','seal'))
    if p['smoke_rank'] is not None or len(s['curves'])!=360 or p['reference_seal']!=sha(history.ARCHIVE/'seal.json'):
        raise ValueError('complete sealed stream on frozen references required')
    archived = read(history.ARCHIVE/'per_shot.json')
    per = {b:{k:[] for k in ARMS} for b in p['calibration']}
    forecast, coverage, missed = defaultdict(list), [], []
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        file = out/'shots'/name
        if sha(file)!=s['curves'][name]:
            raise ValueError('sealed predictions changed')
        y, down, times = group.hdf._raw_truth(sid)
        with np.load(file,allow_pickle=False) as f:
            active = f['dose']>0.
            for a in ('legacy_matched', *LIVE):
                np.testing.assert_array_equal(f[a][~active], f['bias_only'][~active])
            for a in STATIC:
                n = len(times) if rank<=p['startup'] else lag.FIRST
                np.testing.assert_array_equal(f[a][:n], f['bias_only'][:n])
            for b,tau in p['calibration'].items():
                for a in ARMS:
                    row = dict(shot=sid, disruptive=y, down_s=down, **score_curve(
                        times, f[a], y, down, p['original'] if rank<=p['startup'] else tau))
                    if a in OLD and archived[b][a][rank-1]!=row:
                        raise ValueError('archived first alarm changed')
                    per[b][a].append(row)
            q, valid, qual, rho = f['q'], f['valid'], f['qualified'], f['rho']
            coverage.append(dict(shot=sid, forecast_blocks=int(valid.sum()), learned=int(qual.sum()),
                                 active=int(active.sum()), gain_negative=int((f['gain']<0.).sum()),
                                 reduce=int((active & (f['gain']<1.)).sum()),
                                 true_direction_differs_wrong=int((rho[0,qual]*rho[1,qual]<0.).sum()),
                                 true_direction_differs_previous=int((rho[0,qual]*rho[2,qual]<0.).sum())))
            e = f['errors']
            if valid.any():
                forecast['all_EAST'].append(e[valid])
            base = per['0.2']['current_5'][-1]
            tau = p['calibration']['0.2']
            if y and rank>p['startup'] and not base['timely']:
                dt = np.round((down-times)*1000., 9)
                legal = valid & (q+axis.RELEASE_OFFSET+WAIT+PATCH<len(times))
                reach = q[legal]+axis.RELEASE_OFFSET+WAIT+PATCH
                near = (dt[reach]>30.) & (dt[reach]<=125.)
                if near.any():
                    forecast['old_miss_mature_in_30_to_125ms'].append(e[legal][near])
                ok = (dt>30.) & (dt<=125.)
                use = ok & active
                if use.any():
                    fixed = logit(f['current_5'][use].clip(1e-6,1-1e-6))
                    missed.append(dict(shot=sid, active=int(use.sum()),
                        old_gap=float(logit(tau)-fixed.max()),
                        finite_action_gain={a:float((logit(f[a][use].clip(1e-6,1-1e-6))-fixed).max())
                                            for a in LIVE},
                        crossings={a:bool((f[a][ok]>=tau).any()) for a in (*STATIC, *LIVE)}))
            elif not y and rank>p['startup'] and base['alarm']:
                alarm = np.flatnonzero(f['current_5']>=tau)[0]
                before = valid & (q+axis.RELEASE_OFFSET+WAIT+PATCH<=alarm)
                if before.any():
                    forecast['old_safe_false_alarm_before_alarm'].append(e[before])

    reports = {}
    for k, errors in forecast.items():
        means = np.stack([e.mean(0, dtype=np.float64) for e in errors])
        reports[k] = dict(shots=len(means), blocks=sum(map(len,errors)),
            huber=dict(zip(PREDICTORS, means.mean(0).tolist(), strict=True)),
            wins_vs_zero=int((means[:,0]<means[:,2]).sum()))
    metrics = {b:{a:lead.summarize(v) for a,v in rows.items()} for b,rows in per.items()}
    pairs = (('current_5', 'axis_current'), ('bias_only', 'axis_current'),
             ('legacy_matched', 'axis_current'),
             *((k, 'axis_current') for k in LIVE[1:]),
             ('flatten_motion','double_motion'), ('reflect_motion','triple_motion'),
             *(('bias_only', k) for k in STATIC))
    paired = {b:{f'{a}__to__{c}':dict(flips=flips(rows[a],rows[c]), lead=lead.paired(rows[a],rows[c]))
                 for a,c in pairs} for b,rows in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json', dict(metrics=metrics, forecast=reports, coverage=coverage,
                                 missed=missed, paired=paired, raw_verified=360*len(ARMS)*len(per)))
    print({b:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in rows.items()}
           for b,rows in metrics.items()},flush=True)
    print(dict(forecast=reports, active=sum(r['active'] for r in coverage),
               reflected=sum(r['gain_negative'] for r in coverage)), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('source', 'replay', 'score'), required=True)
    parser.add_argument('--smoke-rank', type=int)
    a = parser.parse_args()
    out = a.output.resolve()
    if a.stage=='source':
        fit_source(out)
    elif a.stage=='replay':
        replay(out, smoke_rank=a.smoke_rank)
    else:
        score(out)
