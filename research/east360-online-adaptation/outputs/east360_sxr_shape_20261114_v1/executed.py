"""Test gain-invariant SXR shape and direct H25 feedback before EAST first alarms."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit

from . import commissioning_excursion_order_20261111 as matching
from . import commissioning_history_span_20261114 as history
from . import commissioning_sxr_moments_20261114 as moments
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, write


group, lag, layer = history.group, history.lag, history.layer
WAIT, PATCH, OLD = history.WAIT, history.PATCH, history.OLD
RIDGE = .1
NON_SXR = tuple(io23.RAW23.index(k) for k in ('ip', 'Bt', 'v_loop', 'n=1 amplitude'))
NON_SXR_COORDS = np.array([1e5, 1., 1., 1.])
REGRESSION = dict(sxr_history=tuple(range(8)), all_observed=tuple(range(16)))
METHODS = ('hold', 'linear_trend', *REGRESSION)
FEATURES = ('gain_activity', 'late_shape_distance', 'mean_shape_distance',
            'shape_retained_minus_early', 'late_shape_wrong_order', 'wrong_retained_minus_early')
CASES = (70320, 73405, 79649, 71076)


def make_packet(signal, q):
    values, mask = signal['values'], signal['available']
    c, y, issued, mature = moments.packet(values, mask, q)
    z = np.arcsinh(values[:, NON_SXR].astype(np.float64)/NON_SXR_COORDS)
    m = mask[:, NON_SXR]
    new, n = _window_mean(z, m, q, -15, 16)
    old, o = _window_mean(z, m, q, -31, 16)
    ok = (n>=8).all(1) & (o>=8).all(1)
    x = np.c_[c, new, new-old]
    return x, y, issued & ok, mature & ok


def fit_decoders(packets):
    n = len(packets)
    xmean = np.mean([x.mean(0) for x,y in packets], axis=0)
    xscale = np.sqrt(np.mean([np.square(x-xmean).mean(0) for x,y in packets], axis=0)).clip(1e-6)
    result = {}
    for name, keep in REGRESSION.items():
        gram, cross = np.zeros((len(keep)+1, len(keep)+1)), np.zeros((len(keep)+1, 20))
        for x, y in packets:
            h = np.c_[((x-xmean)/xscale)[:, keep], np.ones(len(x))]
            gram += (h.T @ h)/len(x)
            cross += (h.T @ y.reshape(-1,20))/len(x)
        reg = np.diag([RIDGE]*len(keep)+[0.])
        weight = np.linalg.solve(gram/n+reg, cross/n)
        result[name] = (xmean[list(keep)], xscale[list(keep)], weight)
    return result


def predict(design, model, name):
    mu, sd, w = model[name]
    x = design[:, REGRESSION[name]]
    return (np.c_[(x-mu)/sd, np.ones(len(x))] @ w).reshape(-1, 5, 4)


def shape_loss(prediction, target, scale):
    # Each component is scaled by its TRAIN-only RMS delta before Huber loss.
    e = np.abs((prediction-target)/scale)
    return np.where(e<1., .5*e*e, e-.5)


def assess(x, target, model, scale):
    x0 = np.zeros_like(target)
    # Two trailing means have centers 16 points apart. The first forecast
    # five-point mean lies 10.5 points beyond the recent-mean center.
    linear = x[:, None, 4:8]*(10.5+5*np.arange(5))[None, :, None]/16.
    outputs = (x0, linear, *(predict(x, model, k) for k in REGRESSION))
    return np.stack([shape_loss(z,target,scale).mean(1) for z in outputs], axis=1)


def feedback_views(target, scale):
    """Independent of any outcome: common gain, shape, and order-null shape views."""
    z = target/scale
    gain = np.arcsinh(np.sqrt(np.square(z[...,0]).mean(1)))[:, None]
    norm = np.sqrt(np.square(z[...,1:]).mean(2))
    # The shape contrasts share the exact common gain, all cell magnitudes
    # and the temporal order of the gain. Only SXR shape's phase is shuffled.
    def describe(radius):
        end = radius[:,-2:].mean(1)
        return np.arcsinh(np.c_[end, radius.mean(1), end-radius[:,:2].mean(1)])
    return gain, describe(norm), describe(np.roll(norm, 2, axis=1))


def previous_rows(q, valid):
    """Return the preceding 32-point slot if it actually had complete feedback."""
    if not len(q):
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(q, q-lag.STRIDE)
    return np.where((q[pos]==q-lag.STRIDE) & valid[pos], pos, -1)


def clock(releases, qualified, endpoints):
    """Both controls use the same action clock, including eight-point handoff."""
    slot = np.searchsorted(releases, endpoints, side='right')
    frac = group.handoff_fraction(np.r_[-1,releases][slot], endpoints, PATCH)
    known = np.r_[False, qualified]
    use = known[slot] & ((frac>0.) | known[np.maximum(slot-1,0)])
    return use, slot, frac


def fit_source(out):
    roles = simple_data.source_roles()
    samples, counts = {}, {}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('source signal-only fit')):
        for role in ('train','val'):
            data, rows = [], []
            for sid in roles[role]:
                sig = simple_data.read_shot('J-TEXT',sid)
                q = np.arange(lag.FIRST,len(sig['values']),lag.STRIDE)
                x,y,issued,valid = make_packet(sig,q)
                rows.append(dict(shot=sid, issues=len(q), issued=int(issued.sum()), mature=int(valid.sum())))
                if valid.any():
                    data.append((x[valid],y[valid]))
            samples[role], counts[role] = data, rows
    tr, va = samples['train'], samples['val']
    scale = np.sqrt(np.mean([np.square(y).mean((0,1)) for x,y in tr], axis=0)).clip(1e-6)
    model = fit_decoders(tr)
    err = np.stack([assess(x,y,model,scale).mean(0) for x,y in va])
    out.mkdir(parents=True)
    copyfile(__file__, out/'executed.py')
    copyfile(moments.__file__,out/'executed_moments.py')
    stats = summarize_forecasts(err, sum(len(x) for x,y in va))
    stats['usable_train_shots'] = len(tr)
    stats['train_windows'] = sum(len(x) for x,y in tr)
    write(out/'source_forecast.json',dict(counts=counts,scale=scale.tolist(),ridge=RIDGE,
        model={m:[k.tolist() for k in v] for m,v in model.items()}, val=stats))
    print(dict(stage='source',**stats),flush=True)


def summarize_forecasts(e, n):
    return dict(shots=len(e), windows=n,
        errors=dict(zip(METHODS,e.mean(0).tolist(),strict=True)),
        by_method_shape=dict(zip(METHODS,e[...,1:].mean((0,2)).tolist(),strict=True)),
        shape_better_than_hold=dict(zip(METHODS,(e[...,1:].mean(2)<e[:,:1,1:].mean(2)).sum(0).tolist(),
                                        strict=True)))


def trace_observations(signal, parent, tau, fit, scale):
    """Seal the first method-independent descriptors at the fully usable handoff."""
    q = np.arange(lag.FIRST,len(parent),lag.STRIDE)
    x,y,issued,valid = make_packet(signal,q)
    previous = previous_rows(q,valid)
    usable = valid & (previous>=0) & (q+lag.H+1+WAIT+PATCH < len(parent))
    matched = np.full((len(q), 10), np.nan)
    valid_idx = np.flatnonzero(valid)
    err = np.full((len(q),len(METHODS),4), np.nan,dtype=np.float32)
    err[valid_idx] = assess(x[valid_idx],y[valid_idx],fit,scale)
    m = np.flatnonzero(usable)
    a = q[m]+lag.H+1+WAIT+PATCH
    v, mask = signal['values'], signal['available']
    context, n = _window_mean(v[:,NON_SXR[:2]], mask[:,NON_SXR[:2]], a, -15, 16)
    before, o = _window_mean(v[:,NON_SXR[:1]], mask[:,NON_SXR[:1]], a, -31, 16)
    ok = (n>=8).all(1) & (o>=8).all(1)
    adjustment = np.c_[logit(parent[a].clip(1e-6,1.-1e-6))-logit(tau),
                           np.arcsinh(context[:,0]/1e5),np.arcsinh(context[:,1]),
                           np.arcsinh((context[:,0]-before[:,0])/1e4)]
    g, real, wrong = feedback_views(y[m],scale)
    matched[m[ok]] = np.c_[adjustment, g, real, wrong[:,(0,2)]][ok]
    return q,x,y,issued,valid,usable,previous,err,matched


def row_selection(protocol, seal, folder, fit, scale, lock):
    """Feature collection reads signals and archived predictions, never outcomes."""
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome before SXR feature seal')):
        for rank,sid in enumerate(protocol['order'], 1):
            # The preparation segment's historical predictions remain uncopied.
            if rank<=protocol['startup']:
                continue
            arc = history.archived_shot(rank,sid,seal)
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            sig = lag.prepare_actual(raw,rank,lock['mapping'])
            q,_,_,issued,valid,ok,_,errors,context=trace_observations(
                sig,arc['bias_only'],protocol['calibration']['0.2'],fit,scale)
            name=f'{rank:04d}_{sid}.npz'
            with (folder/name).open('xb') as h:
                np.savez_compressed(h,q=q,issued=issued,valid=valid,usable=ok,errors=errors,context=context)
            if rank==21 or sid in CASES or rank in (120,360):
                # Real truncation: scores at already fully handed-off landmarks
                # cannot change if later recorded signals are removed.
                n = min(len(raw['times']), 620)
                trim = {k:sig[k][:n] for k in ('values','available')}
                p = trace_observations(trim,arc['bias_only'][:n],protocol['calibration']['0.2'],fit,scale)
                fully_released = p[0]+lag.H+1+WAIT+PATCH < n
                for a,b in zip((p[0],p[7],p[8]),(q,errors,context),strict=True):
                    np.testing.assert_array_equal(a[fully_released],b[:len(a)][fully_released])
            yield name


def seal_observation(out, fit, scale):
    start = time.monotonic()
    lock = lag.load_inputs()
    p,s = (read(history.ARCHIVE/f'{name}.json') for name in ('protocol','seal'))
    if (p['order']!=lock['ids'] or p['startup']!=20 or len(s['curve_sha256'])!=360
            or p['source_sha256']!=lock['baseline_protocol']['source_sha256']):
        raise ValueError('original frozen stream and source needed')
    folder = out/'observed'
    folder.mkdir()
    names = list(row_selection(p,s,folder,fit,scale,lock))
    write(out/'observed_seal.json',dict(order=p['order'], names=names, startup=p['startup'],
        source_sha256=p['source_sha256'], features=('margin','ip','Bt','delta_ip',*FEATURES),
        history_window=32, forecast=25, feedback_delay=26, wait=WAIT, handoff=PATCH,
        elapsed_s=time.monotonic()-start))
    return p,s


def diagnose(out, p, s):
    """Postseal diagnostics: are large changes or persistent shape more informative?"""
    seal = read(out/'observed_seal.json')
    files = out/'observed'
    ids = seal['order']
    if p['order']!=ids or len(seal['names'])!=340:
        raise ValueError('all signal-only rows must be sealed before scoring')
    per = read(history.ARCHIVE/'per_shot.json')['0.2']['current_5']
    taus = p['calibration']
    delay = lag.H+1+WAIT+PATCH
    category, sample = defaultdict(list), defaultdict(list)
    match = defaultdict(list)
    for rank,sid in enumerate(ids,1):
        y, down, times = group.hdf._raw_truth(sid)
        archived = per[rank-1]
        if (archived['shot'],archived['disruptive'],archived['down_s'])!=(sid,y,down):
            raise ValueError('fixed evaluations must use original HDF5 outcomes')
        if rank<=p['startup']:
            continue
        with np.load(files/f'{rank:04d}_{sid}.npz',allow_pickle=False) as d:
            q,e,valid,c=(d[k] for k in ('q','errors','valid','context'))
        if valid.any():
            sample['all_EAST'].append(e[valid])
        supported = np.flatnonzero(np.isfinite(c).all(1))
        t = q[supported]+delay
        if y and not archived['timely']:
            dt = np.round((down-times[t])*1000.,9)
            ok = (dt>30.) & (dt<=125.)
            bucket = 'miss'
        elif not y and archived['alarm']:
            # Compare demand and false alarm at the same unchanged warning rule.
            arc = history.archived_shot(rank,sid,s)
            alarm = int(np.flatnonzero(arc['current_5']>=np.float32(taus['0.2']))[0])
            ok = t<=alarm
            bucket = 'false'
        else:
            continue
        if y:
            arc = history.archived_shot(rank,sid,s)
        assert score_curve(times,arc['current_5'],y,down,taus['0.2'])=={
            k:archived[k] for k in ('alarm','lead_ms','timely')}
        j = supported[ok]
        if len(j):
            sample[bucket].append(e[j])
            category[bucket].append(dict(shot=sid, n=len(j)))
            match[bucket].extend((sid,int(q[i]+delay),c[i]) for i in j)

    def pairs_for(keys, tolerances):
        a,b = (match[k] for k in ('miss','false'))
        if not a or not b:
            return dict(pairs=0)
        idx = matching.disjoint_pairs(*(np.stack([v[2][keys] for v in r]) for r in (a,b)),
            *(np.array([v[0] for v in r]) for r in (a,b)),np.asarray(tolerances))
        pairs = [(a[i],b[j]) for i,j in idx]
        deltas = np.stack([x[2][4:]-z[2][4:] for x,z in pairs]) if len(pairs) else np.empty((0,len(FEATURES)))
        return dict(pairs=len(pairs), ids=[[i,t,j,u] for (i,t,x),(j,u,y) in pairs],
            columns=FEATURES, wins=dict(zip(FEATURES,((deltas>0).sum(0)+.5*(deltas==0).sum(0)).tolist(),strict=True)),
            deltas=deltas.tolist())

    four = [.50,.35,.35,.50]
    out_dict = dict(group_shots=dict(category),
        forecast={a:summarize_forecasts(np.stack([m.mean(0,dtype=np.float64) for m in v]),
                                            sum(map(len,v))) for a,v in sample.items()},
        pairs=dict(context=pairs_for(np.arange(4),four),
                   matched_gain=pairs_for(np.arange(5),[*four,.25])))
    write(out/'observability.json',out_dict)
    print({k:{a:v for a,v in d.items() if a not in ('ids','deltas')} for k,d in out_dict['pairs'].items()},flush=True)
    print(out_dict['forecast'],flush=True)


def load_forecast(out):
    asset = read(out/'source_forecast.json')
    roles = simple_data.source_roles()
    if (asset['ridge']!=RIDGE or any([v['shot'] for v in asset['counts'][k]]!=roles[k] for k in ('train','val'))):
        raise ValueError('forecast must be fitted on the fixed TRAIN only, with held VAL')
    return {a:tuple(map(np.asarray,v)) for a,v in asset['model'].items()},np.asarray(asset['scale'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--stage',choices=('source','observe'),required=True)
    a=cli.parse_args()
    out=a.output.resolve()
    if a.stage=='source':
        fit_source(out)
    else:
        fit,scale=load_forecast(out)
        diagnose(out,*seal_observation(out,fit,scale))
