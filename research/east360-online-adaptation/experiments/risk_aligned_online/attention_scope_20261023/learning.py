"""Change temporal attention matching without changing the first-layer values."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('qk','value_matched','full_matched')
WEIGHT='base.encoder.layers.0.self_attn.in_proj_weight'
PREFIX='base.encoder.layers.0.'


def masks(fast,dim):
    if WEIGHT not in fast or fast[WEIGHT].shape!=(3*dim,dim):
        raise ValueError('first-layer packed Q/K/V matrix must match declared width')
    q={k:torch.zeros_like(v,dtype=torch.bool) for k,v in fast.items()}
    v={k:torch.zeros_like(w,dtype=torch.bool) for k,w in fast.items()}
    q[WEIGHT][:2*dim]=True;v[WEIGHT][2*dim:]=True
    return q,v


def attention_probe(model,x,available,fast):
    """Read-only first-layer tensors on a specified observed input.

    This is an instrument, not a replacement for the actual attention forward.
    Its attention matrix is tested against PyTorch MultiheadAttention.
    """
    with torch.no_grad():
        valid=available&torch.isfinite(x);clean=torch.where(valid,x,model.base.center)
        z=torch.asinh((clean-model.base.center)/model.base.scale)
        z=torch.cat((z,valid.to(z.dtype)),-1)
        z=F.linear(z.reshape(len(x),model.cfg.window//model.cfg.patch,-1),fast['base.embedding.weight'],fast['base.embedding.bias'])
        z=torch.cat((z,fast['base.cls'].expand(len(x),-1,-1)),1)+fast['base.position']
        tv=valid.reshape(len(x),model.cfg.window//model.cfg.patch,-1).any(-1)
        tv=torch.cat((tv,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)
        norm=model.base.encoder.layers[0].norm1
        z=F.layer_norm(z,norm.normalized_shape,fast[PREFIX+'norm1.weight'],fast[PREFIX+'norm1.bias'],norm.eps)
        packed=F.linear(z,fast[WEIGHT],fast[PREFIX+'self_attn.in_proj_bias'])
        q,k,v=[a.reshape(len(x),z.shape[1],model.cfg.heads,-1).transpose(1,2) for a in packed.chunk(3,-1)]
        scores=(q@k.transpose(-2,-1))/math.sqrt(q.shape[-1])
        scores=scores.masked_fill(~tv[:,None,None,:],float('-inf'))
        return dict(normalized=z,valid=tv,attention=scores.softmax(-1),value=v)


def update_all(model,prefix,count,states,*,device):
    if (set(prefix)!={'values','available'} or len(prefix['values'])!=count
            or set(states)!=set(ARMS)):
        raise ValueError('exact signal-only observed prefix and all declared states required')
    x,m,truth,tm,ends=forecast_examples(prefix,count,model.cfg,device)
    valid=target_mask(x,m,truth,tm);observed=int(valid.sum())
    q,v=masks(states['qk'],model.cfg.dim)
    selectors={'qk':q,'value_matched':v,'full_matched':{k:torch.ones_like(t,dtype=torch.bool) for k,t in states['qk'].items()}}
    next_states={};receipts={};requested=None
    for arm in ARMS:
        fast=states[arm]
        r=dict(last_observation=count-1,first_affected_prediction=count,forecast_origins=ends,
            observed_targets=observed,gradient_steps=0,parameter_delta_norm=0.,requested_step_norm=0.,
            attention_change_max=0.,value_change_max=0.)
        if not observed:
            r['reason']='no_observed_target';next_states[arm]=fast;receipts[arm]=r;continue
        loss=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],truth,valid)
        grads=dict(zip(fast,torch.autograd.grad(loss,tuple(fast.values()))))
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError('nonfinite observed-signal feedback')
        direction={k:g*selectors[arm][k] for k,g in grads.items()}
        norm=sum(g.double().square().sum() for g in direction.values()).sqrt()
        if arm=='qk':requested=.001*min(float(norm),1.)
        if norm==0 or requested==0:
            # There is no direction to rescale. Preserve the state and record
            # this explicitly, rather than substituting a different gradient.
            steps={k:torch.zeros_like(g) for k,g in direction.items()}
        else:steps={k:g*(requested/float(norm)) for k,g in direction.items()}
        changed={k:(p-steps[k]).detach().requires_grad_(True) for k,p in fast.items()}
        delta=sum((changed[k].detach()-p.detach()).double().square().sum() for k,p in fast.items()).sqrt()
        for k in fast:
            if not torch.equal(changed[k][~selectors[arm][k]],fast[k][~selectors[arm][k]]):
                raise AssertionError('update escaped declared parameter scope')
        before_probe=attention_probe(model,x,m,fast);after_probe=attention_probe(model,x,m,changed)
        ad=float((after_probe['attention']-before_probe['attention']).abs().max())
        vd=float((after_probe['value']-before_probe['value']).abs().max())
        if arm=='qk' and vd!=0:raise AssertionError('Q/K-only update changed first-layer V')
        if arm=='value_matched' and ad!=0:raise AssertionError('V-only update changed first-layer attention')
        with torch.no_grad():after=model.base.forecast_loss(model.predict_fast(x,m,changed)['forecast'],truth,valid)
        if not torch.isfinite(after):raise FloatingPointError('nonfinite updated prediction')
        r.update(gradient_steps=int(delta>0),parameter_delta_norm=float(delta),requested_step_norm=requested,
            selected_gradient_norm=float(norm),loss_before=float(loss.detach()),loss_after=float(after),
            attention_change_max=ad,value_change_max=vd)
        next_states[arm]=changed;receipts[arm]=r
    return next_states,receipts


def predict(model,signal,*,device='cuda'):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or not len(signal['values'])
            or signal['available'].shape!=signal['values'].shape or signal['available'].dtype!=bool
            or model.cfg.window!=128 or model.cfg.horizon!=25):
        raise ValueError('nonempty flat23 signal-only W128/H25 stream required')
    states={a:initial_encoder(model) for a in ARMS};original=initial_encoder(model)
    scores={a:[] for a in ('static',)+ARMS};updates={a:[] for a in ARMS}
    n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                scores['static'].append(logit(model,x,m,original).sigmoid().cpu().numpy())
                for a in ARMS:scores[a].append(logit(model,x,m,states[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        states,receipts=update_all(model,{k:v[:hi] for k,v in signal.items()},hi,states,device=device)
        for a in ARMS:updates[a].append(receipts[a])
    result={k:np.concatenate(v) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite warning')
    return result,updates
