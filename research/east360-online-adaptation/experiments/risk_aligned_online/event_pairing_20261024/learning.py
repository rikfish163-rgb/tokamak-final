"""Event-conditioned risk similarity, not a risk-score correction module."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def features(x, mask, center, scale):
    if (np.shape(x)!=(128,23) or np.shape(mask)!=(128,23) or mask.dtype!=bool
            or np.shape(center)!=(23,) or np.shape(scale)!=(23,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale<=0).any()):
        raise ValueError('past128 flat23 window and positive source scales required')
    valid=mask.copy() & np.isfinite(x)
    # EAST MNM is unavailable under the current data contract. Apply the
    # identical omission to this auxiliary SOURCE representation only.
    valid[:,2]=False
    z=np.arcsinh((np.where(valid,x,center).astype('float64')-center)/scale)
    count=valid.sum(0);mean=z.sum(0)/np.maximum(count,1)
    bins=valid.reshape(8,16,23).sum(1)
    path=z.reshape(8,16,23).sum(1)/np.maximum(bins,1)-mean
    return np.r_[mean,np.where(bins>0,path,0.).reshape(-1),(bins/16).reshape(-1)].astype('float32')


def reverse_features(x):
    if x.shape[-1]!=391:raise ValueError('391-dimensional descriptor required')
    out=x.copy();shape=x.shape[:-1]+(8,23)
    out[...,23:207]=x[...,23:207].reshape(shape)[...,::-1,:].reshape(x.shape[:-1]+(184,))
    out[...,207:]=x[...,207:].reshape(shape)[...,::-1,:].reshape(x.shape[:-1]+(184,))
    return out


def mean_features(x):
    if x.shape[-1]!=391:raise ValueError('391-dimensional descriptor required')
    out=x.copy();out[...,23:207]=0.
    m=x[...,207:].reshape(x.shape[:-1]+(8,23)).mean(-2,keepdims=True)
    out[...,207:]=np.broadcast_to(m,x.shape[:-1]+(8,23)).reshape(x.shape[:-1]+(184,))
    return out


def pair_candidates(means,shots,y,family,*,mode):
    n=len(y)
    if (means.ndim!=2 or len(means)!=n or shots.shape!=(n,) or family.shape!=(n,)
            or mode not in ('risk','family') or not np.isfinite(means).all()
            or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2):
        raise ValueError('finite matching TRAIN inputs, both hazard states and declared pairing required')
    positive=np.full((n,4),-1,int);negative=positive.copy();pn=np.zeros(n,int);nnn=pn.copy()
    weight=np.empty(n,float)
    for f in np.unique(family):
        for label in (0,1):
            group=np.flatnonzero((family==f)&(y==label))
            if not len(group):raise ValueError('each supplied event family needs both risk states')
            ids,counts=np.unique(shots[group],return_counts=True)
            for sid,count in zip(ids,counts):weight[group[shots[group]==sid]]=1./(len(ids)*count)
    norm=(means*means).sum(1)
    for lo in range(0,n,128):
        dist=np.maximum(norm[lo:lo+128,None]+norm[None,:]-2*means[lo:lo+128]@means.T,0.)
        for row,i in enumerate(range(lo,min(lo+128,n))):
            different=shots!=shots[i]
            pos=different & (family==family[i]) & (y==y[i])
            neg=different & ((family==family[i]) & (y!=y[i]) if mode=='risk'
                             else (family!=family[i]) & (y==y[i]))
            for allow,output,count in ((pos,positive,pn),(neg,negative,nnn)):
                candidates=np.flatnonzero(allow)
                if not len(candidates):raise ValueError(f'no distinct-shot {mode} pair for row{i}')
                ordered=candidates[np.argsort(dist[row,candidates],kind='stable')]
                chosen=[];seen=set()
                for j in ordered:
                    if shots[j] in seen:continue
                    chosen.append(int(j));seen.add(shots[j])
                    if len(chosen)==4:break
                output[i,:len(chosen)]=chosen;count[i]=len(chosen)
    weight/=weight.sum()
    return dict(positive=positive,negative=negative,positive_count=pn,negative_count=nnn,weight=weight)


def sample_triplets(pairs,rng,n):
    anchor=rng.choice(len(pairs['weight']),n,replace=True,p=pairs['weight'])
    pos=pairs['positive'][anchor,rng.integers(pairs['positive_count'][anchor])]
    neg=pairs['negative'][anchor,rng.integers(pairs['negative_count'][anchor])]
    return np.stack((anchor,pos,neg),-1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(391,64),nn.SiLU(),nn.Linear(64,16))

    def forward(self,x):
        return F.normalize(self.net(x),dim=-1)


def pair_loss(z):
    if z.ndim!=3 or z.shape[1]!=3 or not torch.isfinite(z).all():
        raise ValueError('finite batch x triplet x embedding required')
    a,p,n=z.unbind(1)
    return F.softplus(((a*n).sum(-1)-(a*p).sum(-1))/.1).mean()


def neighbour_score(query,memory,shots,y,*,exclude,k=3):
    if (query.ndim!=2 or memory.ndim!=2 or query.shape[1]!=memory.shape[1]
            or shots.shape!=(len(memory),) or y.shape!=shots.shape or exclude.shape!=(len(query),)
            or not np.isin(y,[0,1]).all() or not np.isfinite(query).all() or not np.isfinite(memory).all()
            or k<1):raise ValueError('finite matching signal embeddings and authorized historical labels required')
    q=query.astype('float64');m=memory.astype('float64');out=[]
    pools={label:[(sid,np.flatnonzero((shots==sid)&(y==label))) for sid in np.unique(shots[y==label])]
           for label in (0,1)}
    for lo in range(0,len(q),128):
        x=q[lo:lo+128]
        dist=np.maximum((x*x).sum(1)[:,None]+(m*m).sum(1)[None,:]-2*x@m.T,0.)
        result=[]
        for label in (0,1):
            if len(pools[label])<k:raise ValueError('insufficient distinct historical shots for declared k')
            ds=np.stack([dist[:,ids].min(1) for _,ids in pools[label]],1)
            for col,(sid,_) in enumerate(pools[label]):ds[exclude[lo:lo+len(x)]==sid,col]=np.inf
            best=np.partition(ds,k-1,axis=1)[:,:k]
            if not np.isfinite(best).all():raise ValueError('insufficient history after whole-shot exclusion')
            result.append(best.mean(1))
        out.append(result[0]-result[1])
    return np.concatenate(out)
