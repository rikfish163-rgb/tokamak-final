"""Historical-supervised loss learning; deployed encoder step accepts signals only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder


class UpdateLoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+69,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,h,context):
        if h.ndim!=2 or context.shape!=(len(h),69):
            raise ValueError('one 69-dimensional observed context per state required')
        return F.softplus(self.net(torch.cat((h,context.detach()),-1))).mean()


@torch.no_grad()
def context_features(x,mask,center,scale):
    if (x.ndim!=3 or x.shape[1:]!=(128,23) or mask.shape!=x.shape
            or mask.dtype!=torch.bool or center.shape!=(23,) or scale.shape!=(23,)
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale<=0).any()):
        raise ValueError('W128/23 observed windows and finite source scaler required')
    valid=mask&torch.isfinite(x)
    # Double arithmetic and finite clipping prevent large raw differences from
    # overflowing; missing channels retain a separate availability coordinate.
    clean=torch.where(valid,x,center).double()
    z=torch.asinh((clean-center.double())/scale.double()).clamp(-10,10)
    def mean(a,b):
        m=valid[:,a:b];n=m.sum(1)
        return (z[:,a:b]*m).sum(1)/n.clamp_min(1),n
    recent,n=mean(-25,None);older,oldn=mean(-50,-25)
    delta=torch.where((n>0)&(oldn>0),recent-older,0.)
    return torch.cat((recent,delta,n/25.),-1).clamp(-10,10).to(x.dtype)


def adapt_window(model,rule,x,mask,context,*,create_graph=False,mode='observed'):
    if (mode not in ('observed','no_context','entropy') or len(x)!=1
            or x.shape!=(1,128,23) or mask.shape!=x.shape or mask.dtype!=torch.bool
            or context.shape!=(1,69) or not torch.isfinite(context).all()):
        raise ValueError('one causal window and declared update mode required')
    fast=initial_encoder(model)
    if not (mask&torch.isfinite(x)).any():
        return fast,dict(gradient_steps=0,delta_norm=0.,reason='no_observations')
    h=model.predict_fast(x,mask,fast,mode='risk_state')['risk_state']
    if mode=='entropy':
        z=model.base.risk_head(h)
        loss=(F.softplus(z)-z*z.sigmoid()).mean()
    else:
        loss=rule(h,torch.zeros_like(context) if mode=='no_context' else context)
    gradients=torch.autograd.grad(loss,tuple(fast.values()),create_graph=create_graph)
    norm=torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in gradients]))
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite learned-loss update')
    # Differentiate the actual clipped operator during meta-training.
    factor=1./norm.clamp_min(1.)
    updated={k:v-.001*factor*g for (k,v),g in zip(fast.items(),gradients)}
    distance=sum((updated[k].detach()-v.detach()).double().square().sum()
                 for k,v in fast.items()).sqrt()
    if not create_graph:
        updated={k:v.detach().requires_grad_(True) for k,v in updated.items()}
    return updated,dict(gradient_steps=1,delta_norm=float(distance),gradient_norm=float(norm.detach()),
                        clip_factor=float(factor.detach()),loss=float(loss.detach()))


def step(model,rule,signal,count,*,device='cuda',create_graph=False,mode='observed'):
    if (set(signal)!={'values','available'} or type(count)!=int or count<128
            or count>len(signal['values']) or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or signal['values'].shape!=signal['available'].shape
            or signal['available'].dtype!=bool or model.cfg.window!=128):
        raise ValueError('signal-only observed W128 prefix required')
    prefix={k:v[:count] for k,v in signal.items()}
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(prefix,count-1,128)]
    c=context_features(x,m,model.base.center,model.base.scale)
    fast,r=adapt_window(model,rule,x,m,c,create_graph=create_graph,mode=mode)
    r.update(last_observation=count-1,first_affected_prediction=count)
    return fast,r


def folds(records):
    d=sorted(r['shot'] for r in records if r['is_disrupt'])
    s=sorted(r['shot'] for r in records if not r['is_disrupt'])
    ids=d+s
    if len(d)!=4 or len(s)!=16 or len(set(ids))!=20:
        raise ValueError('exactly four distinct disruptive and sixteen safe TRAIN shots required')
    return [dict(fold=f,held=sorted([d[f]]+s[f::4]),
                 fit=sorted(set(ids)-set([d[f]]+s[f::4]))) for f in range(4)]


def summarize(rows):
    groups={};per_shot=[]
    for group in ('safe','disruptive_timely','disruptive_early'):
        part=[r for r in rows if r['group']==group]
        shots=[]
        for sid in sorted({r['shot'] for r in part}):
            rs=[r for r in part if r['shot']==sid]
            shot=dict(shot=sid,group=group,n=len(rs),
                      before=float(np.mean([r['before'] for r in rs])),
                      after=float(np.mean([r['after'] for r in rs])))
            shots.append(shot);per_shot.append(shot)
        if shots:groups[group]={key:float(np.mean([r[key] for r in shots])) for key in ('before','after')}
    if not {'safe','disruptive_timely'}<=set(groups):raise ValueError('both classes required')
    main=[groups[g] for g in ('safe','disruptive_timely')]
    d=[r for r in per_shot if r['group']=='disruptive_timely']
    better=sum(r['after']<r['before'] for r in d)
    return dict(groups=groups,per_shot=per_shot,balanced_before=float(np.mean([r['before'] for r in main])),
                balanced_after=float(np.mean([r['after'] for r in main])),improved_disruptive_shots=better,
                passes_screen=bool(len(d)==4 and better>=3 and all(r['after']<=r['before'] for r in main)
                                   and sum(r['after'] for r in main)<sum(r['before'] for r in main)))
