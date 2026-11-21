"""Held-shot phase affinity. This is not a calibrated disruption probability."""
import numpy as np


def distances(a,b):
    a=np.asarray(a,dtype='float64');b=np.asarray(b,dtype='float64')
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0.)


def phase_score(query,early,late,shots):
    if (query.ndim!=2 or early.ndim!=2 or early.shape!=late.shape
            or query.shape[1]!=early.shape[1] or shots.shape!=(len(early),)
            or not all(np.isfinite(v).all() for v in (query,early,late))):
        raise ValueError('finite paired feature arrays and shot identities required')
    identifiers=np.unique(shots)
    if len(identifiers)<3:
        raise ValueError('at least three historical phase trajectories required')
    separation=((early-late)**2).sum(1)
    positive=separation[separation>1e-12]
    if not len(positive):
        raise ValueError('phase difference not identifiable from historical coordinates')
    temperature=float(np.median(positive))
    de=distances(query,early);dl=distances(query,late)
    chosen=np.stack([np.flatnonzero(shots==sid)[np.minimum(de[:,shots==sid],dl[:,shots==sid]).argmin(1)] for sid in identifiers],1)
    er=np.take_along_axis(de,chosen,1);lr=np.take_along_axis(dl,chosen,1)
    phase=np.exp(-np.logaddexp(0.,-(er-lr)/temperature))
    logw=-np.minimum(er,lr)/temperature
    weights=np.exp(logw-logw.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
    return (phase*weights).sum(1),dict(selected=chosen,weights=weights,phase=phase,temperature=temperature)


def make_fold(shots,labels,early_ends,disruptive,safe):
    if (shots.shape!=labels.shape or shots.shape!=early_ends.shape or not np.isin(labels,[0,1]).all()
            or not np.all(labels[shots==disruptive]==1) or not np.all(labels[shots==safe]==0)):
        raise ValueError('whole-shot disruptive/safe roles required')
    late=np.flatnonzero(shots==disruptive);safe_rows=np.flatnonzero(shots==safe)
    if not len(late) or not len(safe_rows):
        raise ValueError('both held-out identities must exist')
    _,first,inverse=np.unique(early_ends[late],return_index=True,return_inverse=True)
    return dict(history=np.flatnonzero(~np.isin(shots,[disruptive,safe])),late=late,
                safe=safe_rows,early_unique=late[first],pair_to_unique=inverse)


def auc(positive,negative):
    diff=np.asarray(positive)[:,None]-np.asarray(negative)[None,:]
    if not diff.size or not np.isfinite(diff).all():
        raise ValueError('nonempty finite classes required')
    return float(np.mean((diff>0)+.5*(diff==0)))


def summarize(rows):
    per={}
    for sid in sorted({r['held_shots'][0] for r in rows}):
        group=[r for r in rows if r['held_shots'][0]==sid]
        per[str(sid)]={key:float(np.mean([r[key] for r in group])) for key in ('phase_auc','late_safe_auc','paired_order')}
    result={key:float(np.mean([r[key] for r in per.values()])) for key in ('phase_auc','late_safe_auc','paired_order')}
    return dict(result,per_disruptive=per,folds=len(rows))


def eligible(results):
    p=results['risk']['phase'];old=results['risk']['late_only']
    return bool(len(p['per_disruptive'])==4
        and all(r['phase_auc']>.5 and r['paired_order']>.5 for r in p['per_disruptive'].values())
        and p['phase_auc']>old['phase_auc'] and p['paired_order']>old['paired_order']
        and p['late_safe_auc']>=old['late_safe_auc']
        and all(p['phase_auc']>results[a]['phase']['phase_auc'] for a in ('mean','reverse','raw_mean'))
        and all(results[f'risk_{seed}']['phase']['phase_auc']>.5 for seed in (131,132,133)))
