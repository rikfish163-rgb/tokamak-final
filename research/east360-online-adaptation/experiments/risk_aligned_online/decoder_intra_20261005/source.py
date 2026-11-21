"""OFFLINE source-head fitting only. Future-dependent strata are never online gates."""
import numpy as np
import torch
from torch.nn import functional as F


def source_origins(shot):
    if shot['device']!='J-TEXT':
        raise ValueError('source TRAIN only; no EAST outcome access')
    origins=np.arange(0,max(0,len(shot['times'])-26),26,dtype=int)
    keep=shot['valid'][origins+26].copy()
    if shot['is_disrupt']:
        keep &= np.round((shot['down']-shot['times'][origins+26])*1000.,9)>30.
    return origins[keep]


def weighted_quantile(values,weights,q):
    values,weights=np.asarray(values,float),np.asarray(weights,float)
    if (values.shape!=weights.shape or not len(values) or not 0<=q<=1
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights<0).any() or weights.sum()<=0):
        raise ValueError('finite nonempty values and positive total weight required')
    order=np.argsort(values,kind='stable')
    idx=np.searchsorted(np.cumsum(weights[order])/weights.sum(),q,side='left')
    return float(values[order[min(idx,len(values)-1)]])


def activity_thresholds(rows):
    thresholds={}
    for label in (0,1):
        group=[r for r in rows if r['label']==label]
        counts={sid:sum(r['shot']==sid for r in group) for sid in {r['shot'] for r in group}}
        thresholds[label]=weighted_quantile([r['activity'] for r in group],
                                            [1/counts[r['shot']] for r in group],.75)
    return thresholds


def sampling_pools(rows,thresholds):
    pools={(c,h):{} for c in (0,1) for h in (0,1)}
    for i,row in enumerate(rows):
        key=(row['label'],int(row['activity']>thresholds[row['label']]))
        pools[key].setdefault(row['shot'],[]).append(i)
    if any(not p for p in pools.values()):
        raise ValueError('empty source class/activity stratum; candidate not estimable')
    return pools


def sample_indices(pools,rng,count):
    indices=[]
    for _ in range(count):
        group=pools[(int(rng.integers(2)),int(rng.integers(2)))]
        sid=rng.choice(list(group))
        indices.append(int(rng.choice(group[sid])))
    return indices


def normalized_level_loss(prediction,target,mask):
    """One value/event: mean of per-channel H25 masked SmoothL1 means.

    Inputs already use the immutable source coordinates. Empty channels do not
    contribute; missing targets are sanitized before arithmetic. No label input.
    """
    observed=mask & torch.isfinite(target)
    if (observed & ~torch.isfinite(prediction)).any():
        raise FloatingPointError('observed nonfinite forecast')
    p=torch.where(observed,prediction,0.)
    t=torch.where(observed,target.detach(),0.)
    loss=F.smooth_l1_loss(p,t,reduction='none')
    count=observed.sum(1)
    channel=loss.sum(1)/count.clamp_min(1)
    return channel.sum(-1)/(count>0).sum(-1).clamp_min(1)
