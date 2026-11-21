"""Minimal direct future-state hypothesis test, not a JEPA reproduction.

The target is the fixed parent's classifier input on later ARRIVED signals,
not physical ground truth. Only an auxiliary residual linear predictor is
fitted offline. Its feedback updates a disposable first-encoder copy; the
classifier head and stored parent remain untouched. One-step probes are not
continuous deployment results.
"""
import numpy as np
import torch

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder


class DirectLatent(torch.nn.Module):
    def __init__(self,mean,scale,coefficient):
        super().__init__()
        mean,scale,coefficient=[np.asarray(v) for v in (mean,scale,coefficient)]
        if (mean.ndim!=1 or len(mean)%2 or len(mean)<2 or scale.shape!=mean.shape
                or coefficient.shape!=(len(mean)+1,len(mean)//2) or not (scale>0).all()
                or not all(np.isfinite(v).all() for v in (mean,scale,coefficient))):
            raise ValueError('finite residual linear predictor and positive training-only scales required')
        for name,v in zip(('mean','scale','coefficient'),(mean,scale,coefficient)):
            self.register_buffer(name,torch.as_tensor(v,dtype=torch.float32))

    def forward(self,current,previous):
        if current.ndim!=2 or current.shape!=previous.shape or current.shape[-1]*2!=len(self.mean):
            raise ValueError('matching current and past full risk states required')
        x=(torch.cat((current,current-previous),-1)-self.mean)/self.scale
        x=torch.cat((x,torch.ones_like(x[:,:1])),-1)
        return current+x@self.coefficient


def fit_predictor(current,previous,future,*,penalty=1e-3):
    h,p,t=[np.asarray(v,dtype=np.float64) for v in (current,previous,future)]
    if (h.ndim!=2 or min(h.shape)<2 or p.shape!=h.shape or t.shape!=h.shape
            or not all(np.isfinite(v).all() for v in (h,p,t))
            or not np.isfinite(penalty) or penalty<=0):
        raise ValueError('matching finite training matrices and positive penalty required')
    x=np.concatenate((h,h-p),axis=1);mean=x.mean(0);std=x.std(0)
    # Numerically constant training dimensions have no learned scale; leave
    # their centered coordinate in native units. This affects no target data.
    scale=np.where(std>1e-8,std,1.)
    x=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    regularizer=np.eye(x.shape[1])*penalty;regularizer[-1,-1]=0.
    coefficient=np.linalg.solve(x.T@x/len(x)+regularizer,x.T@(t-h)/len(x))
    return DirectLatent(mean,scale,coefficient)


def origins(length):
    if type(length) is not int or length<1:raise ValueError('positive integer length required')
    if length<203:return []
    return np.unique(np.linspace(152,length-51,min(128,length-202)).astype(int)).tolist()


def encode_at(model,signal,ends,*,fast=None,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signal-only records required')
    windows=[simple_data.window(signal,int(end),model.cfg.window) for end in ends]
    x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
    if fast is None:fast=initial_encoder(model)
    return model.predict_fast(x,m,fast,mode='risk_state')['risk_state']


def arrived_step(model,predictor,signal,end,*,device='cuda'):
    if (set(signal)!={'values','available'} or type(end) is not int or end<152
            or end+25>=len(signal['values']) or model.cfg.window!=128 or model.cfg.horizon!=25):
        raise ValueError('signal-only full past windows and already arrived H25 target required')
    # Enforce the accessible prefix before any encoding, not merely by promise.
    prefix={k:v[:end+26] for k,v in signal.items()}
    observed=int((prefix['available'][end+1:] & np.isfinite(prefix['values'][end+1:])).sum())
    fast=initial_encoder(model)
    receipt=dict(origin=end,target_end=end+25,first_affected_prediction=end+26,
        observed_new_values=observed,gradient_steps=0,parameter_delta_norm=0.)
    if observed==0:return fast,receipt
    target=encode_at(model,prefix,[end+25],device=device).detach()
    h=encode_at(model,prefix,[end,end-25],fast=fast,device=device)
    loss=(predictor(h[:1],h[1:])-target).square().mean()
    gradients=torch.autograd.grad(loss,tuple(fast.values()),allow_unused=False)
    norm=sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite latent feedback')
    multiplier=(1./norm.clamp_min(1.)).detach()
    updated={k:(v-.001*multiplier*g).detach().requires_grad_(True) for (k,v),g in zip(fast.items(),gradients)}
    delta=sum((updated[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    after=encode_at(model,prefix,[end,end-25],fast=updated,device=device)
    error=(predictor(after[:1],after[1:])-target).square().mean()
    receipt.update(gradient_steps=1,gradient_norm=float(norm),parameter_delta_norm=float(delta),
                   loss_before=float(loss.detach()),loss_after=float(error.detach()))
    return updated,receipt


def risk_summary(records):
    result={}
    for arm in ('static','learned','persistence'):
        c={}
        for label in (0,1):
            logits=np.asarray([r['logits'][arm] for r in records if r['label']==label],dtype=float)
            if not np.isfinite(logits).all():raise FloatingPointError('nonfinite supervised probe')
            c['positive' if label else 'negative']=dict(n=len(logits),
                bce=float((np.logaddexp(0.,logits)-label*logits).mean()) if len(logits) else None)
        both=all(c[k]['n'] for k in ('positive','negative'))
        c['balanced_bce']=sum(c[k]['bce'] for k in ('positive','negative'))/2 if both else None
        result[arm]=c
    b,a=result['static'],result['learned']
    result['passes_direction_gate']=bool(a['balanced_bce'] is not None
        and a['balanced_bce']<b['balanced_bce']
        and all(a[k]['bce']<=b[k]['bce'] for k in ('positive','negative')))
    return result
