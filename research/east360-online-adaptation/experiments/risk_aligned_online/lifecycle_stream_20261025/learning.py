"""Post-seal supervision and event-covered SOURCE replay for the full stream."""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..causal_input import apply_input_contract
from ..delayed_recurrent_memory_20261025.learning import batch
from ..event_prefix_probe import read_events, DEFAULT_LABELS, LABEL_SHA256
from ..simple_train import sha
from .protocol import sampled_endpoints


@torch.no_grad()
def cache_example(model,signal,ends,labels,*,sid,flag,delta=None):
    hs,vs=[],[]
    for lo in range(0,len(ends),32):
        x,m=batch(signal,ends[lo:lo+32],model.base.center.device)
        h,v=model.tokens(x,m);hs.append(h);vs.append(v)
    if delta is None:delta,_=model.memory_delta(signal)
    device=model.base.center.device
    return dict(h=torch.cat(hs),valid=torch.cat(vs),delta=delta[ends].detach(),
                y=torch.as_tensor(labels,device=device,dtype=torch.float32),
                shot=int(sid),flag=bool(flag),ends=np.asarray(ends,int).tolist())


def target_example(model,shot,scores,signal,delta,*,endpoint_policy='full_window'):
    sampler=sampled_endpoints
    if endpoint_policy=='prefix16':
        from ..short_history_20261031.sampling import sampled_with_prefix
        sampler=sampled_with_prefix
    elif endpoint_policy!='full_window':
        raise ValueError('unknown endpoint policy')
    ends=sampler(shot['times'],shot['y'],shot['valid'],scores,
                 down=shot['down'] if shot['is_disrupt'] else None)
    return cache_example(model,signal,ends,shot['y'][ends],sid=shot['shot'],
                         flag=shot['is_disrupt'],delta=delta)


def source_replay(model):
    if sha(DEFAULT_LABELS)!=LABEL_SHA256:raise ValueError('event CSV changed')
    events,_=read_events(DEFAULT_LABELS)
    train=set(simple_data.source_roles()['train']);advanced=simple_data.advanced_times()
    chosen=[]
    for family in range(6):
        pool=sorted((e for sid,es in events.items() if sid in train for e in es if e.family==family),
                    key=lambda e:(e.shot,e.index))
        if len(pool)<8:raise ValueError('insufficient source TRAIN events for balanced replay')
        chosen.extend(pool[i] for i in np.linspace(0,len(pool)-1,8,dtype=int))
    rows=[];manifest=[]
    for e in chosen:
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('J-TEXT',e.shot)),
                                    'J-TEXT',advanced)
        candidates=np.flatnonzero(s['valid'] & (np.arange(len(s['times']))>=127)
            & (s['times']>=e.start) & (s['times']<e.end))
        if s['is_disrupt']:
            candidates=candidates[np.round((s['down']-s['times'][candidates])*1000,9)>10]
        if not len(candidates):
            manifest.append(dict(shot=e.shot,event=e.index,family=e.family,status='no_valid_past128_timely_window'))
            continue
        ends=np.unique(candidates[np.linspace(0,len(candidates)-1,min(2,len(candidates)),dtype=int)])
        signal={k:s[k] for k in ('values','available')}
        rows.append(cache_example(model,signal,ends,s['y'][ends],sid=e.shot,flag=s['is_disrupt']))
        manifest.append(dict(shot=e.shot,event=e.index,family=e.family,status='included',ends=ends.tolist(),
                             source_horizon_ms=s['horizon']))
    if not rows:raise ValueError('empty SOURCE event replay')
    return rows,manifest


def logits(model,e,fast):
    h,v=e['h'],e['valid'];d=e['delta'] if fast else torch.zeros_like(e['delta'])
    return torch.cat([model.risk(h[i:i+32],v[i:i+32],d[i:i+32]) for i in range(0,len(h),32)])


def window_loss(z,y,disruptive,policy='all_points'):
    if policy not in ('all_points','disruptive_positive_only'):
        raise ValueError('unknown target label policy')
    if policy=='disruptive_positive_only' and disruptive:
        # Controlled ablation: keep the125ms label itself, but do not interpret
        # earlier disruptive-shot windows as certified safe training examples.
        # Safe-shot supervision and SOURCE event replay remain unchanged.
        keep=y>0
        if not keep.any():raise ValueError('no eligible positive target window')
        z,y=z[keep],y[keep]
    losses=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    # Equal risk-state weighting inside a shot; no invented positives when
    # its observed/eligible training windows contain only one risk state.
    groups=[losses[y==label].mean() for label in (0,1) if (y==label).any()]
    return torch.stack(groups).mean()


def example_loss(model,e,fast,policy='all_points'):
    return window_loss(logits(model,e,fast),e['y'],e['flag'],policy)


def population_loss(model,examples,fast,policy='all_points'):
    if not examples:raise ValueError('nonempty authorized history required')
    groups=[]
    for flag in (False,True):
        group=[e for e in examples if e['flag']==flag]
        if group:groups.append(torch.stack([example_loss(model,e,fast,policy) for e in group]).mean())
    return torch.stack(groups).mean()


@torch.no_grad()
def guard_losses(model,replay,source,fast,policy='all_points'):
    values={}
    for flag in (False,True):
        group=[e for e in replay if e['flag']==flag]
        if group:values['historical_D' if flag else 'historical_S']=float(population_loss(model,group,fast,policy))
    values['source_event']=float(population_loss(model,source,fast))
    return values


def update(model,current,replay,source,*,fast,steps,guarded,policy='all_points'):
    before=model.adapted_state()
    old=guard_losses(model,replay,source,fast=True if fast=='mixed' else fast,policy=policy) if guarded else {}
    optimizer=torch.optim.Adam([model.slow.weight,model.memory.output.weight],lr=.001)
    trace=[]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        modes=(False,True) if fast=='mixed' else (fast,)
        total=0.
        for mode in modes:
            target=population_loss(model,current,mode,policy)
            prior=population_loss(model,replay,mode,policy) if replay else target
            src=population_loss(model,source,mode)
            loss=(.5*target+.25*prior+.25*src)/len(modes)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite revealed-history objective')
            loss.backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_([model.slow.weight,model.memory.output.weight],1.,error_if_nonfinite=True))
        optimizer.step();trace.append(dict(loss=total,gradient_norm=norm))
    new=guard_losses(model,replay,source,fast=True if fast=='mixed' else fast,policy=policy) if guarded else {}
    accepted=not guarded or all(new[k]<=old[k]+1e-7 for k in old)
    candidate=model.adapted_state()
    changes={k:float((candidate[k]-before[k]).norm()) for k in before}
    if not accepted:model.restore(before)
    return dict(accepted=accepted,target_label_policy=policy,guard_before=old,guard_after=new,steps=trace,
                candidate_delta=changes,retained_delta=changes if accepted else {k:0. for k in changes},
                guard_is_independent=False,history_ids=[e['shot'] for e in replay])
