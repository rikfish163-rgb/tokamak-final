"""Diagnostic readout and fast-parameter factors; classifier path is untouched."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('full_all','full_observation','patch_all','patch_observation')
SUMMARY={'base.cls','base.position'}


class PatchForecast(nn.Linear):
    """Same linear readout, with its final CLS input block explicitly zeroed.

    Temporal tokens may still contain CLS information through attention. This
    removes only the direct readout, not all influence of CLS on forecasting.
    """
    def __init__(self,linear,cls_dim):
        if type(cls_dim) is not int or not 0<cls_dim<linear.in_features:
            raise ValueError('a nonempty temporal block plus final CLS block required')
        super().__init__(linear.in_features,linear.out_features,bias=linear.bias is not None,
                         device=linear.weight.device,dtype=linear.weight.dtype)
        self.cls_dim=cls_dim
        self.load_state_dict(linear.state_dict())

    def forward(self,x):
        observed=torch.cat((x[...,:-self.cls_dim],torch.zeros_like(x[...,-self.cls_dim:])),dim=-1)
        return F.linear(observed,self.weight,self.bias)


def temporary(model,scope):
    if scope not in ('all','observation'):raise ValueError('explicit fast parameter scope required')
    fast=initial_encoder(model)
    if scope=='observation':
        for k in SUMMARY:fast[k].requires_grad_(False)
    return fast


def adapt(model,signal,count,fast,*,device):
    x,m,target,tm,ends=forecast_examples(signal,count,model.cfg,device)
    observed=int(tm.sum());selected={k:v for k,v in fast.items() if v.requires_grad}
    if not selected:raise ValueError('nonempty fast parameter set required')
    receipt=dict(last_observation=count-1,first_affected_prediction=count,forecast_origins=ends,
                 observed_targets=observed,gradient_steps=0,parameter_delta_norm=0.,
                 trainable_coordinates=sum(v.numel() for v in selected.values()),
                 group_delta_norm=dict(summary=0.,observation=0.))
    if not observed:
        # No available measured target means no signal gradient. Record the
        # non-update; missing diagnostics are never synthesized as true zeros.
        return fast,receipt
    out=model.predict_fast(x,m,fast)['forecast'];loss=model.base.forecast_loss(out,target,tm)
    gradients=torch.autograd.grad(loss,tuple(selected.values()),allow_unused=False)
    norm=sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite arrived feedback')
    factor=(1./norm.clamp_min(1.)).detach();updated=dict(fast)
    for (k,v),g in zip(selected.items(),gradients):
        updated[k]=(v-.001*factor*g).detach().requires_grad_(True)
    delta=sum((updated[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    groups={}
    for name,keys in (('summary',[k for k in fast if k in SUMMARY]),('observation',[k for k in fast if k not in SUMMARY])):
        groups[name]=float(sum((updated[k].detach()-fast[k].detach()).double().square().sum() for k in keys).sqrt())
    with torch.no_grad():after=model.base.forecast_loss(model.predict_fast(x,m,updated)['forecast'],target,tm)
    receipt.update(gradient_steps=1,gradient_norm=float(norm),parameter_delta_norm=float(delta),
                   group_delta_norm=groups,loss_before=float(loss.detach()),loss_after=float(after))
    return updated,receipt


def predict(models,signal,*,device='cuda'):
    if (set(models)!={'full','patch'} or set(signal)!={'values','available'}
            or signal['values'].ndim!=2 or signal['values'].shape[1]!=23
            or signal['available'].shape!=signal['values'].shape or signal['available'].dtype!=bool
            or not len(signal['values'])):
        raise ValueError('two explicit heads and a nonempty signal-only flat23 stream required')
    if any(m.cfg.window!=128 or m.cfg.horizon!=25 for m in models.values()):
        raise ValueError('W128 H25 experiment required')
    fast={a:temporary(models[a.split('_')[0]],a.split('_')[1]) for a in ARMS}
    original={h:temporary(m,'all') for h,m in models.items()}
    scores={a:[] for a in ('static',)+ARMS};updates={a:[] for a in ARMS};n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                baseline=logit(models['full'],x,m,original['full']).sigmoid()
                torch.testing.assert_close(baseline,logit(models['patch'],x,m,original['patch']).sigmoid(),rtol=0,atol=0)
                scores['static'].append(baseline.cpu().numpy())
                for a in ARMS:scores[a].append(logit(models[a.split('_')[0]],x,m,fast[a]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        prefix={k:v[:hi] for k,v in signal.items()}
        for a in ARMS:
            fast[a],receipt=adapt(models[a.split('_')[0]],prefix,hi,fast[a],device=device)
            updates[a].append(receipt)
    result={a:np.concatenate(rows) for a,rows in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise FloatingPointError('nonfinite risk')
    return result,updates
