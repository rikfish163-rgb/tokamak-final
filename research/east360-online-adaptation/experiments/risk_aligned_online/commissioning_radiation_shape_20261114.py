"""Do recorded high-order AXUV statistics add information beyond lower moments?"""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, write


WEAK = io23.ROOT/'outputs/east360_weak_signal_probe_20261114_v2'
PAIRED = io23.ROOT/'outputs/east360_two_prefix_coverage_20261114/result.json'
REF = io23.ROOT/'outputs/east360_peak_feedback_20261114_v1'
COLS = tuple(io23.RAW23.index(name) for name in ('xuv_skew','xuv_var','xuv_kurt'))
WIDTH = 32
VIEWS = ('lower_moments','with_high_order')


def measure(x,mask,end):
    """No imputation. The signed moment gap is a recorded feature, not a validity rule."""
    frames = end[:,None]+np.arange(1-WIDTH,1)
    values = x[frames][...,COLS].astype(np.float64)
    ok = mask[frames][...,COLS].all((1,2)) & np.isfinite(values).all((1,2))
    s,v,k = values.transpose(2,0,1)
    d = k+2.-np.square(s)
    raw = np.median(np.stack((s, np.arcsinh(v), d),axis=-1),axis=1)
    invalid_fraction = (d < -1e-5).mean(1)
    result = np.c_[raw,invalid_fraction]
    result[~ok] = np.nan
    return result,ok


def source():
    advanced = io23.advanced_times()
    role = simple_data.source_roles()
    rows = {}
    for label in ('train','val'):
        samples = []
        for sid in role[label]:
            r,t,disrupted = probe.source_training_points(sid,advanced)
            if not len(t):
                continue
            f,ok = measure(r['values'],r['available'],t)
            work,valid = probe.operating(r,t)
            i = ok & valid
            if i.any():
                samples.append((np.full(i.sum(),sid),np.full(i.sum(),disrupted),work[i],f[i,:3]))
        rows[label] = [np.concatenate(a) for a in zip(*samples,strict=True)]
    sid,y,x,data = rows['train']
    w = probe.training_weights(sid,y)
    scale = np.sqrt(np.maximum(w@np.square(data[:,:2]-w@data[:,:2]),.05**2))
    a = {name:probe.fit_probe(x,data[:,:2+i],sid,y) for i,name in enumerate(VIEWS)}
    sid,y,x,data = rows['val']
    vals = apply(a,x,data)
    ids = np.unique(sid)
    peak = np.stack([vals[sid==k].max(0) for k in ids])
    flag = np.array([y[sid==k][0] for k in ids],bool)
    n = int(flag.sum())
    rank = rankdata(peak,axis=0)
    auc = (rank[flag].sum(0)-n*(n+1)/2)/(n*(len(flag)-n))
    return dict(models=a,scale=scale.tolist(),lower_moment_caliper_sd=.5, roles={k:
        probe.train_counts(v[0],v[1]) for k,v in rows.items()},
        source_val_peak_auc=dict(zip(VIEWS,auc.tolist(),strict=True)))


def apply(models,work,f):
    return np.column_stack([probe.apply_probe(work,f[:,:2+i],models[k])
                            for i,k in enumerate(VIEWS)])


def audit(out):
    scheme = read(WEAK/'protocol.json')
    old = read(REF/'protocol.json')
    if any(scheme[k]!=old[k] for k in probe.history.LOCK):
        raise ValueError('paired observation must use the same fixed EAST stream')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    fit = source()
    write(out/'source.json',fit)
    seen,prefix = [],{}
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('no outcome while measuring shapes')),
          patch.object(probe.group.hdf,'_raw_truth',side_effect=RuntimeError('no outcome while measuring shapes'))):
        for rank,sid in enumerate(scheme['order'],1):
            k = f'{rank:04d}_{sid}.npz'
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            t = probe.end_grid(len(raw['times']))
            f,good = measure(raw['values'],raw['available'],t)
            scores = np.empty((len(t),2),np.float32)
            scores[:] = np.nan
            with np.load(WEAK/'shots'/k,allow_pickle=False) as r:
                if rank>scheme['startup']:
                    np.testing.assert_array_equal(r['ends'],t)
                    usable = good & r['valid']
                    scores[usable] = apply(fit['models'],r['context'][usable,1:4],f[usable])
            if rank in (21,120,360) and len(t)>3:
                stop = t[3]+1
                a,b = measure(raw['values'][:stop],raw['available'][:stop],t[:4])
                np.testing.assert_array_equal(a,f[:4])
                np.testing.assert_array_equal(b,good[:4])
                prefix[str(sid)] = int(stop)
            with (out/'shots'/k).open('xb') as fp:
                np.savez_compressed(fp,end=t,valid=good,value=f,scores=scores)
            seen.append(k)
    write(out/'seal.json',dict(order=scheme['order'],completed=seen,views=VIEWS,source_role='J-TEXT TRAIN',
        feature='32-point trailing median. Retain negative gap and report count; no outcome used for EAST.',
        prefix_witness=prefix))


def assess(out):
    sealed = read(out/'seal.json')
    fit = read(out/'source.json')
    names = [f'{rank:04d}_{sid}.npz' for rank,sid in enumerate(sealed['order'],1)]
    if len(names)!=360 or sealed['completed']!=names:
        raise ValueError('complete locked stream measurement required')
    ids = dict(zip(sealed['order'],names,strict=True))
    threshold = np.asarray(fit['scale'])*fit['lower_moment_caliper_sd']
    reference = read(REF/'per_shot.json')['0.2']['current_5']
    paired = read(PAIRED)['comparisons']
    caps,raw = {},{}
    for sid, file in ids.items():
        with np.load(out/'shots'/file,allow_pickle=False) as r:
            raw[sid] = dict(r)
        d,down,t = probe.group.hdf._raw_truth(sid)
        old = reference[len(caps)]
        if (sid,d,down)!=(old['shot'],old['disruptive'],old['down_s']):
            raise ValueError('outcomes differ from locked HDF5 reference')
        np.testing.assert_array_equal(raw[sid]['end'],probe.end_grid(len(t)))
        caps[sid] = dict(npz=file, slots=int(raw[sid]['valid'].sum()))
    comparisons = {}
    for class_id,v in paired.items():
        rows = []
        for z in v['pairs']:
            d,s = z['positive_shot'],z['safe_shot']
            anchors = {}
            for tag,x in z['anchors'].items():
                records = []
                for sid,j in ((d,x['positive_index']),(s,x['safe_index'])):
                    r = raw[sid]
                    i = int(np.searchsorted(r['end'],j))
                    if r['end'][i]!=j:
                        raise ValueError('paired anchor outside sealed clock')
                    records.append((r['value'][i],r['scores'][i]))
                a,b = records
                support = bool(np.isfinite(np.r_[a[0],a[1],b[0],b[1]]).all())
                row = dict(supported=support)
                if support:
                    delta = np.abs(a[0][:2]-b[0][:2])
                    row |= dict(compatible_low=bool((delta<=threshold).all()), lower_difference=delta.tolist(),
                                values={'disruptive':a[0].tolist(),'non_disruptive':b[0].tolist()},
                                score_margin=dict(zip(VIEWS,(a[1]-b[1]).tolist(),strict=True)))
                anchors[tag] = row
            rows.append(dict(positive=d,safe=s,kind=z['positive_group'],anchors=anchors))
        stats = {}
        for g,desc in v['strata'].items():
            source = [r for r in rows if r['kind']==g]
            ready = [r for r in source if all(x['supported'] for x in r['anchors'].values())]
            eligible = [r for r in ready if all(x['compatible_low'] for x in r['anchors'].values())]
            stats[g] = dict(eligible_abs_difference_both=len(eligible), original_pairs=len(source),
                            no_high_order_observation=len(source)-len(ready),
                            base_disruptive_denominator=desc['denominator_shots'])
            for label,subset in (('all_exact',ready),('low_matched',eligible)):
                stats[g][label] = {
                    k:{tag:sum(r['anchors'][tag]['score_margin'][k]>0 for r in subset)
                       for tag in ('earlier_32','current')} |
                    {'both':sum(all(a['score_margin'][k]>0 for a in r['anchors'].values()) for r in subset)}
                    for k in VIEWS}
        comparisons[class_id] = dict(stats=stats,rows=rows)
    write(out/'result.json',dict(total_shots=len(ids),lower_moment_threshold=threshold.tolist(),
        comparisons=comparisons, measured_shots=caps, source=fit['source_val_peak_auc']))
    for c,v in comparisons.items():
        print(c,v['stats'],flush=True)


def rematch(positive,safe,limits):
    """Maximum shot-disjoint support, then smallest context distance."""
    if not positive or not safe:
        return []
    pa,sa = (np.stack([r['context'] for r in rows])/limits for rows in (positive,safe))
    p,s = (sorted({r['shot'] for r in rows}) for rows in (positive,safe))
    pi,si = ({k:i for i,k in enumerate(a)} for a in (p,s))
    w = (len(limits)+1)*(len(p)+1)
    cost = np.full((len(p),len(s)+len(p)),2.*w)
    cost[:,len(s):] = w
    edge = {}
    for j,neighbors in enumerate(cKDTree(sa).query_ball_point(pa,1.+1e-12,p=np.inf)):
        for k in neighbors:
            diff = abs(pa[j]-sa[k])
            if (diff>1.+1e-12).any():
                continue
            a,b = positive[j],safe[k]
            cell = pi[a['shot']],si[b['shot']]
            d = float(diff@diff)
            K = (d,a['end'],b['end'])
            if cell not in edge or K<edge[cell][0]:
                edge[cell] = K,(j,k)
                cost[cell] = d
    i,j = linear_sum_assignment(cost)
    return [edge[a,b][1] for a,b in zip(i,j,strict=True) if (a,b) in edge]


def balanced(out):
    """Re-match all available handoffs, allowing lower-moment partners to change."""
    complete = read(out/'seal.json')
    source = read(out/'source.json')
    bind = read(probe.history.ARCHIVE/'seal.json')
    old = read(REF/'per_shot.json')['0.2']['current_5']
    w = read(WEAK/'protocol.json')
    peak = {'never':[], 'late':[], 'false':[], 'unwarned_safe':[]}
    if len(complete['completed'])!=360 or complete['order']!=w['order']:
        raise ValueError('full shape seal required')
    cut = w['calibration']['0.2']
    lim = np.r_[probe.CALIPERS,np.asarray(source['scale'])*source['lower_moment_caliper_sd']]
    # Name of the strict in-time edge, identical for both safe groups.
    for rank,r in enumerate(old,1):
        if rank<=w['startup'] or (r['disruptive'] and r['timely']):
            continue
        sid,d = r['shot'],r['disruptive']
        with (np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as a,
              np.load(WEAK/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as b):
            frame,obs,data,estimate = (a[k] for k in ('end','valid','value','scores'))
            state,ctx,valid = (b[k] for k in ('score','context','valid'))
        arc = probe.history.archived_shot(rank,sid,bind)
        f,down,t = probe.group.hdf._raw_truth(sid)
        if (f,down)!=(d,r['down_s']):
            raise ValueError('original HDF5 outcome changed')
        # At or before each full handoff. The nearest observation on the
        # existing probe grid is ≤3 samples earlier, preserving both times.
        handoff = arc['release']+probe.history.WAIT+probe.history.PATCH
        handoff = handoff[handoff<len(t)]
        handoff = handoff[arc['dose'][handoff]>0]
        if d:
            ahead = np.round((down-t[handoff])*1e3,9)
            handoff = handoff[(ahead>30.)&(ahead<=125.)]
            g = 'late' if r['alarm'] else 'never'
        elif r['alarm']:
            u = np.flatnonzero(arc['current_5']>=cut)[0]
            handoff = handoff[handoff<=u]
            g = 'false'
        else:
            g = 'unwarned_safe'
        keep = obs & valid
        for h in handoff:
            b = int(np.searchsorted(frame,h,'right')-1)
            a = b-8
            if a<0 or not keep[[a,b]].all() or arc['dose'][frame[[a,b]]].min()<=0.:
                continue
            low = np.c_[ctx[[a,b]],data[[a,b],:2]]
            score = np.c_[estimate[[a,b]],state[[a,b],:4]]
            peak[g].append(dict(shot=sid, end=int(h), anchors=frame[[a,b]].tolist(),
                                context=low.reshape(-1),score=score))
    groups = {'never':12,'late':21,'false':28,'unwarned_safe':140}
    for c,a in peak.items():
        print(c,len(a),len({v['shot'] for v in a}),flush=True)
    compare = {}
    for g in ('false','unwarned_safe'):
        a,b = peak['never']+peak['late'],peak[g]
        rows = []
        for j,k in rematch(a,b,np.tile(lim,2)):
            p,s = a[j],b[k]
            rows.append(dict(positive=p['shot'],safe=s['shot'],positive_frames=p['anchors'],safe_frames=s['anchors'],
                positive_kind='never' if j<len(peak['never']) else 'late',
                margins=(p['score']-s['score']).tolist(),
                match_difference=abs(p['context']-s['context']).reshape(2,-1).tolist()))
        new = {}
        for k in ('never','late'):
            m = np.array([v['margins'] for v in rows if v['positive_kind']==k]).reshape(-1,2,6)>0
            new[k] = dict(n=len(m),earlier=m[:,0].sum(0).tolist(),now=m[:,1].sum(0).tolist(),
                          both=m.all(1).sum(0).tolist())
        compare[g] = dict(rows=rows,scores=new)
    write(out/'rematched.json',dict(safe_comparators=('false','unwarned_safe'),full_group_sizes=groups,
        eligible={k:dict(shots=len({r['shot'] for r in a}),windows=len(a)) for k,a in peak.items()},
        columns=(*VIEWS,*probe.NAMES),two_prefix_caliper=lim.tolist(),comparisons=compare))
    print({k:v['scores'] for k,v in compare.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--rematch-only',action='store_true')
    a = parser.parse_args()
    target = a.output.resolve()
    if a.rematch_only:
        balanced(target)
    else:
        audit(target)
        assess(target)
