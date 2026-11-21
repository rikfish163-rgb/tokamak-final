"""Historical-label-assisted local encoder adaptation, not a score correction.

Current unlabelled signals select historical training examples. Their known
labels supply the gradient. The screen excludes the current shot's entire
memory and resets each probe; this is not evidence of continuous-stream gains.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..arrived_teacher_20261023.learning import sample_queries as previous_sampling
from ..history_tangent_20261023.learning import flat, logit, matched_step, apply_step
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('nearest','initial','centroid')


def normalize(values):
    values=np.asarray(values,dtype=np.float64)
    norm=np.linalg.norm(values,axis=-1,keepdims=True)
    if values.ndim not in (1,2) or not np.isfinite(values).all() or not np.isfinite(norm).all() or (norm==0).any():
        raise ValueError('finite nonzero embedding vectors required')
    return values/norm


def select_memory(memory,query,mode,*,excluded_shot=None):
    keys,labels,shots=memory['keys'],memory['labels'],memory['shots']
    if (mode not in ('nearest','centroid') or keys.ndim!=2
            or labels.shape!=shots.shape or len(keys)!=len(labels)
            or not np.isin(labels,[0,1]).all() or not np.isfinite(keys).all()
            or not np.allclose(np.linalg.norm(keys,axis=1),1.,rtol=1e-9,atol=1e-9)):
        raise ValueError('normalized memory with binary labels and shot identities required')
    query=normalize(query)
    if query.shape!=(keys.shape[1],):raise ValueError('query must match memory dimension')
    selected=[];distances=[]
    for label in (0,1):
        eligible=np.flatnonzero((labels==label)&(shots!=excluded_shot if excluded_shot is not None else True))
        if len(set(shots[eligible].tolist()))<2:
            raise ValueError('two distinct authorized history shots per class required')
        center=normalize(keys[eligible].mean(0)) if mode=='centroid' else query
        distances_here=1.-keys[eligible]@center
        order=np.argsort(distances_here,kind='stable');seen=set()
        for local in order:
            index=int(eligible[local]);sid=int(shots[index])
            if sid in seen:continue
            selected.append(index);distances.append(float(distances_here[local]));seen.add(sid)
            if len(seen)==2:break
    return selected,distances


def sample_queries(shot):
    local=dict(shot,valid=shot['valid']&(np.arange(len(shot['times']))>=280))
    return previous_sampling(local)


def query_key(model,signal,count,*,device):
    prefix={k:v[:count] for k,v in signal.items()}
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(prefix,count-1,128)]
    h=model.predict_fast(x,m,initial_encoder(model),mode='risk_state')['risk_state']
    return normalize(h.detach().cpu().numpy()[0])


def step_all(model,signal,count,memory,*,excluded_shot=None,device='cuda'):
    if (set(signal)!={'values','available'} or type(count)!=int or count<256
            or count>len(signal['values']) or signal['values'].ndim!=2 or signal['values'].shape[1]!=23
            or signal['values'].shape!=signal['available'].shape or signal['available'].dtype!=bool
            or model.cfg.window!=128):
        raise ValueError('signal-only W128 observed prefix, at least256 points required')
    key=query_key(model,signal,count,device=device)
    first=query_key(model,signal,128,device=device)
    output={};receipts={};gradients={};length=None;origin=initial_encoder(model)
    for arm in ARMS:
        selected,distances=select_memory(memory,first if arm=='initial' else key,
            'centroid' if arm=='centroid' else 'nearest',excluded_shot=excluded_shot)
        indices=torch.as_tensor(selected,device=device)
        x,m=memory['x'][indices],memory['mask'][indices]
        targets=torch.as_tensor(memory['labels'][selected],device=device,dtype=x.dtype)
        # Reset independently per arm; same autograd path before and after.
        fast={k:v.detach().clone().requires_grad_(True) for k,v in origin.items()}
        loss=F.binary_cross_entropy_with_logits(logit(model,x,m,fast),targets)
        g=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach()
        norm=float(g.double().norm())
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite historical supervised gradient')
        if arm=='nearest':length=.001*norm/max(1.,norm)
        step=matched_step(g,length);updated=apply_step(fast,step)
        delta=flat([fast[k].detach()-updated[k].detach() for k in fast])
        after=F.binary_cross_entropy_with_logits(logit(model,x,m,updated),targets)
        if not torch.isfinite(after):raise FloatingPointError('nonfinite updated history loss')
        receipts[arm]=dict(selected_indices=selected,selected_shots=memory['shots'][selected].tolist(),
            selected_labels=memory['labels'][selected].tolist(),distances=distances,
            excluded_shot=excluded_shot,last_observation=count-1,first_affected_prediction=count,
            retrieval_end=127 if arm=='initial' else count-1 if arm=='nearest' else None,
            gradient_steps=int(bool(delta.any())),gradient_norm=norm,requested_step_norm=length,
            parameter_delta_norm=float(delta.double().norm()),loss_before=float(loss.detach()),loss_after=float(after.detach()))
        output[arm]=updated;gradients[arm]=g
    return output,receipts,gradients
