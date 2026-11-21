"""Whole-shot adaptation using past-context/later-label historical pairs.

All historical supervision belongs to the fold's preparation shots. Current
signals only choose examples. Temporary encoder state accumulates inside a
shot, resets at the next shot, and never changes the fixed risk head.
"""
import numpy as np
import torch

from .. import simple_data
from ..arrived_teacher_20261023.learning import sample_queries
from ..deepcoral_prefix_20261022.learning import adapt_arrived
from ..history_retrieval_20261023.learning import normalize, query_key, select_memory
from ..history_tangent_20261023.learning import flat, logit, matched_step, apply_step
from ..paired_feedback_20261022.model import initial_encoder
from ..trajectory_rank_20261023.learning import pairwise_loss

ARMS=('static','forecast','now_rank','lag_rank','initial_rank')
RANK_ARMS=ARMS[2:]


class FoldAccess:
    """Per-fold access boundary, separate from known stratification metadata."""
    def __init__(self, plan, source_ids):
        self.fit=set(plan['fit']);self.held=set(plan['held']);self.source=set(source_ids)
        if len(self.fit)!=15 or len(self.held)!=5 or self.fit&self.held:
            raise ValueError('disjoint fifteen-fit / five-held shot identities required')
        self.phase='fit';self.log=[]

    def check(self, kind, device, sid):
        if kind not in ('signal','outcome') or self.phase not in ('fit','predict'):
            raise ValueError('explicit signal/outcome and fit/predict phase required')
        permitted=(sid in self.source if device=='J-TEXT' else
            sid in self.fit or (kind=='signal' and self.phase=='predict' and sid in self.held)
            if device=='EAST' else False)
        if not permitted:raise AssertionError(f'forbidden {self.phase} {device} {kind} {sid}')
        self.log.append(dict(phase=self.phase,kind=kind,device=device,shot=int(sid)))


def make_memory(model, records, *, device='cuda'):
    ids=[r['shot'] for r in records]
    if len(ids)!=15 or len(set(ids))!=15 or any(r['device']!='EAST' for r in records):
        raise ValueError('fifteen distinct authorized EAST preparation shots required')
    windows=[];lag_windows=[];manifest=[]
    for s in records:
        for row in sample_queries(s):
            e=row['query']
            if e-25<127:raise ValueError('complete historical lag context required')
            windows.append(simple_data.window(s,e,128));lag_windows.append(simple_data.window(s,e-25,128))
            manifest.append(dict(shot=s['shot'],end=e,key_end_now=e,key_end_lag=e-25,
                                 label=int(s['y'][e]),group=row['group']))
    if len(manifest)!=480:raise ValueError('exactly32 training windows per preparation shot required')
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    lx,lm=[torch.as_tensor(np.stack(v),device=device) for v in zip(*lag_windows)]
    keys={};origin=initial_encoder(model)
    with torch.no_grad():
        for name,xx,mm in (('keys',x,m),('lag_keys',lx,lm)):
            h=[model.predict_fast(xx[i:i+32],mm[i:i+32],origin,mode='risk_state')['risk_state'].cpu().numpy()
               for i in range(0,len(xx),32)]
            keys[name]=normalize(np.concatenate(h))
    return dict(x=x,mask=m,**keys,shots=np.array([r['shot'] for r in manifest]),
                labels=np.array([r['label'] for r in manifest])),manifest


def rank_updates(model,signal,count,memory,states,*,device):
    key=query_key(model,signal,count,device=device)
    first=query_key(model,signal,128,device=device)
    updated={};receipts={};length=None
    for arm in RANK_ARMS:
        keys=memory['keys'] if arm=='now_rank' else memory['lag_keys']
        selected,distances=select_memory(dict(memory,keys=keys),first if arm=='initial_rank' else key,'nearest')
        x,m=memory['x'][selected],memory['mask'][selected]
        labels=torch.as_tensor(memory['labels'][selected],device=device)
        fast=states[arm]
        z=logit(model,x,m,fast);loss=pairwise_loss(z,labels)
        g=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach();norm=float(g.double().norm())
        if not torch.isfinite(loss) or not torch.isfinite(g).all():raise FloatingPointError('nonfinite ranking step')
        if arm=='now_rank':length=.001*norm/max(1.,norm)
        new=apply_step(fast,matched_step(g,length))
        after=pairwise_loss(logit(model,x,m,new),labels)
        if not torch.isfinite(after):raise FloatingPointError('nonfinite updated ranking loss')
        delta=flat([fast[k].detach()-new[k].detach() for k in fast])
        updated[arm]=new
        receipts[arm]=dict(last_observation=count-1,first_affected_prediction=count,
            selected_indices=selected,selected_shots=memory['shots'][selected].tolist(),
            selected_labels=memory['labels'][selected].tolist(),distances=distances,
            retrieval_end=127 if arm=='initial_rank' else count-1,
            history_key_lag=0 if arm=='now_rank' else 25,gradient_steps=int(bool(delta.any())),
            parameter_delta_norm=float(delta.double().norm()),requested_step_norm=length,
            gradient_norm=norm,loss_before=float(loss.detach()),loss_after=float(after.detach()))
    return updated,receipts


def predict(model,signal,memory,*,device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or signal['available'].shape!=signal['values'].shape
            or signal['available'].dtype!=bool or model.cfg.window!=128 or model.cfg.horizon!=25):
        raise ValueError('W128/H25 signal-only flat23 stream required')
    n=len(signal['values'])
    if n==0:raise ValueError('nonempty stream required')
    states={a:initial_encoder(model) for a in ARMS}
    scores={a:[] for a in ARMS};updates={a:[] for a in ARMS[1:]}
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for a in ARMS:scores[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        # All predictions through hi-1 are sealed before any update. Subsequent
        # windows cannot change the prefix passed to either update rule.
        prefix={k:v[:hi] for k,v in signal.items()}
        states['forecast'],receipt=adapt_arrived(model,prefix,hi,states['forecast'],device=device)
        updates['forecast'].append(receipt)
        new,receipts=rank_updates(model,prefix,hi,memory,states,device=device)
        states.update(new)
        for a in RANK_ARMS:updates[a].append(receipts[a])
    scores={a:np.concatenate(v) for a,v in scores.items()}
    if not all(np.isfinite(v).all() for v in scores.values()):raise FloatingPointError('nonfinite risk trace')
    return scores,updates


def gate(results):
    base=results['static'];main=results['lag_rank']
    if any(r['n_disrupt']!=4 or r['n_safe']!=16 for r in results.values()):
        raise ValueError('complete four-disruptive/sixteen-safe held predictions required')
    return bool(main['tp']>=base['tp'] and main['fp']<=base['fp'] and main['fpr']<=.20
                and main['J']>base['J'] and all(main['J']>results[a]['J'] for a in ARMS if a not in ('static','lag_rank')))
