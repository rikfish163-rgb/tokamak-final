"""Curvature-controlled one-step latent regression, without risk-score edits."""
import torch
from ..internal_fast_window_20261023.model import InnerLayer,WindowAdapter


class CurvatureLayer(InnerLayer):
    def fast_weights(self,h,valid,*,mode):
        raw,info=super().fast_weights(h,valid,mode=mode)
        _,k,v=self.coordinates(h)
        observed=valid.clone();observed[:,-1]=False
        count=observed.sum(-1)
        # Hessian in each output column is C=K'K/N. Since C is PSD,
        # lambda_max(C)<=trace(C). eta/(1+trace(C)) < 1/lambda_max(C)
        # for eta<=0.5, so a real-arithmetic quadratic step cannot overshoot.
        trace=(k.square().sum(-1)*observed).sum(-1)/count.clamp_min(1)
        origin=self.origin.expand(len(h),-1,-1)
        # Off means exactly the old graph, not a numerically redundant
        # normalize-and-cancel expression whose backward can round differently.
        fast=raw if mode=='off' else origin+(raw-origin)/(1+trace[:,None,None])
        if mode=='reverse':
            length=h.shape[1];positions=torch.arange(length,device=h.device).expand(len(h),-1)
            ordered=torch.where(observed,positions,length).sort(-1).values
            ranks=observed.long().cumsum(-1)-1
            indices=ordered.gather(1,(count[:,None]-1-ranks).clamp(0,length-1)).clamp_max(length-1)
            v=v.gather(1,indices[:,:,None].expand_as(v))
        info.update(curvature_trace=trace.detach(),effective_eta=(self.eta/(1+trace)).detach(),
            delta_norm=(fast-origin).detach().square().sum((1,2)).sqrt(),
            loss_after=(((k@fast-v).square()*observed[:,:,None]).sum((1,2))/(2*count.clamp_min(1))).detach())
        return fast,info


class CurvatureAdapter(WindowAdapter):
    def __init__(self,base):
        super().__init__(base)
        old=self.layer;self.layer=CurvatureLayer(base.cfg.dim)
        # Exactly the same initial tensors as the preceding experiment.
        self.layer.load_state_dict(old.state_dict())
