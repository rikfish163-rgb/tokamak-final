"""Historical signal-only budget selection, not an online danger detector."""
import numpy as np


def describe(signal, center, scale):
    if set(signal)!={'values','available'}:
        raise ValueError('signal-only input required')
    x=np.asarray(signal['values']);m=np.asarray(signal['available'])
    if x.ndim!=2 or x.shape[1]!=23 or x.shape!=m.shape or m.dtype!=bool or not len(x):
        raise ValueError('nonempty flat23 signal and boolean mask required')
    center=np.asarray(center);scale=np.asarray(scale)
    if center.shape!=(23,) or scale.shape!=(23,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('finite source center and positive scales required')
    m=m&np.isfinite(x);z=np.arcsinh((np.where(m,x,center)-center)/scale)
    result=np.zeros((5,23));result[4]=m.mean(0)
    for c in range(23):
        if m[:,c].any():result[:3,c]=np.quantile(z[m[:,c],c],[.1,.5,.9])
        paired=m[:-1,c]&m[1:,c]
        if paired.any():result[3,c]=np.median(np.abs(np.diff(z[:,c])[paired]))
    return result.reshape(-1)


def rank_coordinates(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite candidate descriptors required')
    out=np.full_like(x,.5)
    if len(x)==1:return out
    for c in range(x.shape[1]):
        _,inv,count=np.unique(x[:,c],return_inverse=True,return_counts=True)
        lower=np.r_[0,np.cumsum(count)[:-1]]
        out[:,c]=(lower[inv]+(count[inv]-1)/2)/(len(x)-1)
    return out


def select(ids, descriptors, budget, method, seed):
    ids=np.asarray(ids);x=np.asarray(descriptors)
    if (ids.ndim!=1 or len(set(ids))!=len(ids) or x.ndim!=2 or len(x)!=len(ids)
            or type(budget)!=int or not 1<=budget<=len(ids) or method not in ('uniform','diverse')):
        raise ValueError('distinct candidates, matched descriptors, valid budget and method required')
    order=np.argsort(ids);ids=ids[order];x=rank_coordinates(x[order])
    if method=='uniform':return sorted(int(v) for v in np.random.default_rng(seed).choice(ids,budget,replace=False))
    distance=((x[:,None]-x[None])**2).sum(-1)
    average=distance.mean(1)
    first=int(np.flatnonzero(np.isclose(average,average.min(),rtol=0,atol=1e-12))[0])
    chosen=[first];nearest=distance[:,first].copy()
    while len(chosen)<budget:
        remaining=np.ones(len(ids),bool);remaining[chosen]=False
        maximum=nearest[remaining].max()
        index=int(np.flatnonzero(remaining&np.isclose(nearest,maximum,rtol=0,atol=1e-12))[0])
        chosen.append(index);nearest=np.minimum(nearest,distance[:,index])
    return [int(ids[i]) for i in chosen]


def identity_folds(ids):
    ids=sorted(ids)
    if len(ids)!=85 or len(set(ids))!=85:raise ValueError('fixed distinct historical85 required')
    return [dict(fold=f,held=ids[f::4],candidate=[s for s in ids if s not in set(ids[f::4])]) for f in range(4)]
