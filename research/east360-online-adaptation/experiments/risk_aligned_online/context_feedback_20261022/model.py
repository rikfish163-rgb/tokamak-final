"""Learn which observed channel errors may adapt an immutable risk model.

No risk-score branch is added. Source-only outer supervision trains a bounded
channel-weight controller. At deployment the controller only reads a frozen
causal representation, masked completion errors, availability and optionally
INFERRED weak-event probabilities. It never reads an outcome or event label.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import io23
from ..masked_state_20261022.model import masked_views
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder


class ContextFeedback(PairedFeedback):
    def __init__(self,base,level_scale):
        super().__init__(base,level_scale)
        self.mode_channel=io23.RAW23.index('mode_number_n')
        self.controller=nn.Sequential(nn.Linear(2*self.cfg.dim+23+23+6,32),
                                      nn.Tanh(),nn.Linear(32,23))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def freeze_base(self):
        self.requires_grad_(False); self.controller.requires_grad_(True); self.eval()

    def context(self,x,mask,losses,counts,*,use_event):
        with torch.no_grad():
            h,valid=self.encode(x,mask)
            mean=(h*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
            # The chronological first-to-last token difference is a learned
            # trajectory descriptor, NOT an identified physical cause.
            first=valid.long().argmax(1)
            last=valid.shape[1]-1-valid.flip(1).long().argmax(1)
            delta=h[torch.arange(len(h),device=h.device),last]-h[torch.arange(len(h),device=h.device),first]
            event=self.event_head(mean).softmax(-1) if use_event else mean.new_zeros((len(h),6))
            context=torch.cat((F.layer_norm(mean,(self.cfg.dim,)),
                               F.layer_norm(delta,(self.cfg.dim,)),
                               torch.log1p(losses.detach())[None],
                               (counts/self.cfg.window)[None],event),-1)
        return context.detach()


def channel_losses(model,prediction,truth,target):
    valid=target & torch.isfinite(truth)
    valid=valid.clone(); valid[:,:,model.mode_channel]=False
    target_value=torch.where(valid,truth.detach(),model.base.center)
    z=torch.asinh((target_value-model.base.center)/model.base.scale)
    error=(prediction-z)/model.level_scale
    loss=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
    counts=valid.sum((0,1))
    sums=loss.masked_fill(~valid,0.).sum((0,1))
    return sums/counts.clamp_min(1),counts


def update_observed(model,x,available,*,use_event,uniform=False,create_graph=False):
    if len(x)!=1 or x.shape!=available.shape:
        raise ValueError('one observed window required')
    fast=initial_encoder(model)
    origin={k:v.detach().clone() for k,v in fast.items()}
    xx,mm,target=masked_views(x,available)
    target=target.clone(); target[:,:,model.mode_channel]=False
    count=int(target.sum())
    if not count:
        return fast,dict(steps=0,observations=0,delta=0.,weights=[1.]*23)
    weights=None; first_loss=None
    with torch.enable_grad():
        for _ in range(2):
            pred=model.predict_fast(xx,mm,fast,mode='reconstruct')['reconstruction']
            losses,counts=channel_losses(model,pred,xx,target)
            if weights is None:
                if uniform: weights=torch.ones_like(losses)
                else:
                    context=model.context(x,available,losses,counts,use_event=use_event)
                    raw=.5+torch.sigmoid(model.controller(context)[0])
                    # Preserve the total scale of observed-target weighting.
                    # Positive normalized weights lie within[1/3,3]; the
                    # controller cannot succeed by turning all updates off.
                    weights=raw/(raw*counts).sum()*counts.sum()
            loss=(weights*losses*counts).sum()/counts.sum()
            if first_loss is None: first_loss=float(loss.detach())
            gradients=torch.autograd.grad(loss,tuple(fast.values()),
                create_graph=create_graph,retain_graph=create_graph)
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('nonfinite context adaptation')
            fast={k:v-.01*g for (k,v),g in zip(fast.items(),gradients)}
            if not create_graph: fast={k:v.detach().requires_grad_(True) for k,v in fast.items()}
    distance=sum((v.detach()-origin[k]).double().square().sum() for k,v in fast.items()).sqrt()
    return fast,dict(steps=2,observations=count,delta=float(distance),
                     first_loss=first_loss,weights=weights.detach().cpu().tolist())


def operational_loss(logits,times,valid,*,is_disrupt,down,threshold):
    if logits.ndim!=1 or len(logits)!=len(times) or len(valid)!=len(times):
        raise ValueError('one complete shot trace required')
    if not 0<threshold<1: raise ValueError('finite probability threshold required')
    eligible=np.array(valid,dtype=bool,copy=True)
    if is_disrupt:
        eligible &= np.round((down-np.asarray(times))*1000.,9)>10.
    if not eligible.any(): raise ValueError('no eligible operational endpoints')
    peak=logits[torch.as_tensor(eligible,device=logits.device)].max()
    margin=peak-math.log(threshold/(1-threshold))
    return F.softplus(-margin if is_disrupt else margin)
