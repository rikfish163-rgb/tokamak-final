"""Is a local diagnostic reference useful before the EAST warning deadline?"""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, write
from .operating_point_audit import best_at_count
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent


PREVIOUS = io23.ROOT/'outputs/east360_weak_signal_probe_20261114_v2'
PAIRED = io23.ROOT/'outputs/east360_two_prefix_coverage_20261114/result.json'
CHANNELS = tuple(probe.COLUMNS)
WINDOW, BIN, HISTORY = 64, 8, 64
ARMS = ('absolute','recent_shift','recent_rank','opening_rank')


def rank_position(query, reference):
    """An empirical CDF centered at zero, with half mass for ties."""
    d = reference[:,None]-query[:,:,None]
    return 2.*((d<0).mean(2)+.5*(d==0).mean(2))-1.


def describe(signal,end,center,spread):
    """The current 64 points and either disjoint 64-point reference."""
    cols = list(CHANNELS)
    x,m = (signal[k][:,cols] for k in ('values','available'))
    i = end[:,None]+np.arange(1-WINDOW,1)
    past = i-HISTORY
    ok = m[i].all((1,2)) & m[past].all((1,2)) & m[:HISTORY].all()
    # Apply the exact same convention for all ordered current blocks.
    current = x[i].reshape(len(i),WINDOW//BIN,BIN,len(cols)).mean(2,dtype=np.float64)
    prev = x[past]
    early = np.broadcast_to(x[:HISTORY],prev.shape)
    center,spread = center[cols],spread[cols]
    absolute = np.arcsinh((current-center)/spread)
    residual = np.arcsinh((current-np.median(prev,axis=1)[:,None])/spread)
    v = (absolute, residual,rank_position(current,prev),rank_position(current,early))
    return {k:x.reshape(len(i),len(cols)*(WINDOW//BIN)) for k,x in zip(ARMS,v,strict=True)},ok


def score_blocks(signal,t,center,spread,fits=None):
    x,okay = describe(signal,t,center,spread)
    ctx,operating = probe.operating(signal,t)
    good = okay & operating
    absolute = x['absolute'][good]
    # For the recency and centering comparison, fit the same form of readout
    # with all absolute coordinates retained in each arm.
    features = [absolute if k=='absolute' else np.c_[absolute,x[k][good]] for k in ARMS]
    if fits is None:
        return good,ctx[good],features
    values = np.full((len(t),len(ARMS)),np.nan,np.float32)
    values[good] = np.column_stack([probe.apply_probe(ctx[good],v,fits[k])
                                    for k,v in zip(ARMS,features,strict=True)])
    return values,good


def source_pools(center,spread,role):
    aux = simple_data.advanced_times()
    pools = []
    for sid in role:
        r,t,y = probe.source_training_points(sid,aux)
        if not len(t):
            continue
        ok,ctx,features = score_blocks(r,t,center,spread)
        if ok.any():
            pools.append((np.full(len(ctx),sid),np.full(len(ctx),y,bool),ctx,*features))
    sid,y,ctx,*features = [np.concatenate(x) for x in zip(*pools,strict=True)]
    return sid,y,ctx,features


def source_fit(center,spread):
    roles = simple_data.source_roles()
    sid,y,ctx,features = source_pools(center,spread,roles['train'])
    counts = {'train':probe.train_counts(sid,y)}
    fit = {k:probe.fit_probe(ctx,x,sid,y) for k,x in zip(ARMS,features,strict=True)}
    sid,y,ctx,features = source_pools(center,spread,roles['val'])
    counts['val'] = probe.train_counts(sid,y)
    v = np.column_stack([probe.apply_probe(ctx,x,fit[k]) for k,x in zip(ARMS,features,strict=True)])
    ids = np.unique(sid)
    p = np.stack([v[sid==k].max(0) for k in ids])
    y = np.asarray([y[sid==k][0] for k in ids])
    a,b = y.sum(),(~y).sum()
    auc = dict(zip(ARMS,((rankdata(p,axis=0)[y].sum(0)-a*(a+1)/2)/(a*b)).tolist(),strict=True))
    return dict(fit=fit,counts=counts,held_positive_window_and_safe_peak_auc=auc)


def observe(out):
    base,_ = load_parent(DEFAULT_PARENT)
    center,spread = base.center.numpy(),base.scale.numpy()
    protocol = read(PREVIOUS/'protocol.json')
    lock = probe.lag.load_inputs()
    if protocol['order']!=lock['ids'] or len(protocol['order'])!=360:
        raise ValueError('same fixed360 required')
    fit = source_fit(center,spread)
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'source.json',fit)
    done,prefixes = [],{}
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('EAST outcome not observable')),
          patch.object(probe.group.hdf,'_raw_truth',side_effect=RuntimeError('EAST outcome not observable'))):
        for rank,sid in enumerate(protocol['order'],1):
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            q = probe.end_grid(len(raw['times']))
            x = probe.lag.prepare_actual(raw,rank,lock['mapping'])
            scores,eligible = score_blocks(x,q,center,spread,fit['fit'])
            f = f'{rank:04d}_{sid}.npz'
            if rank in (21,120,360) and len(q)>9:
                t = int(q[9])+1
                truncated = {k:v[:t] for k,v in x.items() if k in ('values','available')}
                a,b = score_blocks(truncated,q[:10],center,spread,fit['fit'])
                np.testing.assert_allclose(a,scores[:10],atol=1e-6,rtol=0.,equal_nan=True)
                np.testing.assert_array_equal(b,eligible[:10])
                prefixes[str(sid)] = t
            with (out/'shots'/f).open('xb') as file:
                np.savez_compressed(file,ends=q,scored=eligible,score=scores)
            done.append(f)
            if rank in (20,120,240,360):
                print('sealed',rank,flush=True)
    write(out/'seal.json',dict(order=protocol['order'],cold=protocol['startup'],completed=done,views=ARMS,
        channels=[io23.RAW23[k] for k in CHANNELS],history=HISTORY,window=WINDOW,
        bin=BIN,prefixes=prefixes,outcome='EAST not loaded until after the measurements are sealed'))


def compare(pairs, rows):
    at = {k:[] for k in ('no_first_alarm','late_first_alarm')}
    missing = dict.fromkeys(at,0)
    for p in pairs:
        values = []
        group = p['positive_group']
        for time in ('earlier_32','current'):
            anchor = p['anchors'][time]
            a,b = (rows[s] for s in (p['positive_shot'],p['safe_shot']))
            i,j = (int(np.searchsorted(v['ends'],anchor[name])) for v,name in (
                (a,'positive_index'),(b,'safe_index')))
            if (a['ends'][i],b['ends'][j])!=(anchor['positive_index'],anchor['safe_index']):
                raise ValueError('matched comparison not on the same sealed sample clock')
            values.append(a['score'][i]-b['score'][j])
        values = np.stack(values)
        if np.isfinite(values).all():
            at[group].append(dict(positive=p['positive_shot'],safe=p['safe_shot'],margins=values.tolist()))
        else:
            missing[group] += 1
    report = {}
    for group,data in at.items():
        p = np.asarray([r['margins'] for r in data]).reshape(-1,2,len(ARMS))>0.
        report[group] = dict(n=len(data),without_signal=missing[group],
            earlier=p[:,0].sum(0).tolist(),later=p[:,1].sum(0).tolist(),both=p.all(1).sum(0).tolist(),
            pairs=data)
    return report


def diagnose(out):
    p = read(out/'seal.json')
    old = read(probe.history.ARCHIVE/'per_shot.json')['0.2']['current_5']
    cases = read(PAIRED)
    expected = [f'{i:04d}_{sid}.npz' for i,sid in enumerate(p['order'],1)]
    if p['completed']!=expected or [r['shot'] for r in old]!=p['order'] or len(expected)!=360:
        raise ValueError('must seal the complete common stream before HDF5 scoring')
    groups = ('timely_all_history','timely_in_125ms')
    peaks = {g:{k:[] for k in ARMS} for g in groups}
    rows = {}
    support = []
    for rank,(sid,file) in enumerate(zip(p['order'],expected,strict=True),1):
        d,down,t = probe.group.hdf._raw_truth(sid)
        if (d,down)!=(old[rank-1]['disruptive'],old[rank-1]['down_s']):
            raise ValueError('HDF5 does not match the archived truth')
        with np.load(out/'shots'/file,allow_pickle=False) as r:
            v = rows[sid] = dict(r)
        e,seen = v['ends'],v['scored']
        np.testing.assert_array_equal(e,probe.end_grid(len(t)))
        if rank<=p['cold']:
            continue
        reach = np.round((down-t[e])*1e3,9) if d else np.full(len(e),np.inf)
        allowed = (seen & (reach>30.), seen & ((reach<=125.) | (not d)) & (reach>30.))
        kind = ('timely' if old[rank-1]['timely'] else 'late' if old[rank-1]['alarm'] else 'never') if d else (
            'false' if old[rank-1]['alarm'] else 'safe')
        support.append(dict(shot=sid,kind=kind,all=int(allowed[0].sum()),close=int(allowed[1].sum())))
        for group,common in zip(groups,allowed,strict=True):
            if not d and not common.any():
                raise ValueError('safe shot has no scored prefix for these comparisons')
            for k,dense in zip(ARMS,v['score'].T,strict=True):
                peak = float(dense[common].max()) if common.any() else None
                # Rank scores are not probabilities. A monotone map makes them
                # compatible with the reusable tied-peak operating-point audit.
                prob = float(1./(1.+np.exp(-peak))) if peak is not None else None
                peaks[group][k].append(dict(shot=sid,disruptive=d,peak=prob))
    # The cutoff is retrospective and used only to compare ranking, not deployed.
    equal = {g:{str(n):{k:{m:v[m] for m in ('tp','fp')} for k,r in a.items()
                        for v in [best_at_count(r,n)]} for n in (12,21,30)} for g,a in peaks.items()}
    pairs = {k:compare(v['pairs'],rows) for k,v in cases['comparisons'].items()}
    write(out/'result.json',dict(scope='held diagnostic readout; no alarms generated',
        view_order=ARMS,paired=pairs,comparable_windows=support,
        downstream_only_equal_false_peaks=equal,held_source=read(out/'source.json')[
            'held_positive_window_and_safe_peak_auc']))
    print(equal,flush=True)
    for k,v in pairs.items():
        print(k,{q:{a:b for a,b in w.items() if a!='pairs'} for q,w in v.items()},flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    observe(a.output.resolve())
    diagnose(a.output.resolve())
