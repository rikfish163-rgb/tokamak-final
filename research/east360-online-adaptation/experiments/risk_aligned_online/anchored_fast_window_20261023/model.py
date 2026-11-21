"""Only a current-window fast-weight change may modify the frozen parent."""
import torch
from ..internal_fast_window_20261023.model import InnerLayer, WindowAdapter


def stable_center(x, observed):
    # Subtract one actually observed value first: a constant record must be
    # exactly zero, not tiny roundoff subsequently amplified by trace division.
    first=observed.long().argmax(1)
    reference=x.gather(1,first[:,None,None].expand(-1,1,x.shape[-1]))
    difference=(x-reference)*observed[...,None]
    mean=difference.sum(1,keepdim=True)/observed.sum(1).clamp_min(1)[:,None,None]
    return (difference-mean)*observed[...,None]


def centered_step(k,v,observed,origin,eta,*,rule,reverse):
    if rule not in ('fixed','normalized') or k.ndim!=3 or k.shape!=v.shape or observed.shape!=k.shape[:2] or observed.dtype!=torch.bool:
        raise ValueError('matching coordinates, observation mask and declared step rule required')
    count=observed.sum(1); n=count.clamp_min(1)
    k=stable_center(k,observed); v=stable_center(v,observed)
    if reverse:
        length=k.shape[1]; positions=torch.arange(length,device=k.device).expand(len(k),-1)
        ordered=torch.where(observed,positions,length).sort(-1).values
        ranks=observed.long().cumsum(-1)-1
        indices=ordered.gather(1,(count[:,None]-1-ranks).clamp(0,length-1)).clamp_max(length-1)
        v=v.gather(1,indices[...,None].expand_as(v))*observed[...,None]
    error=k@origin-v
    gradient=k.transpose(1,2)@error/n[:,None,None]
    trace=k.square().sum((1,2))/n
    updated=(count>=2)&(trace>0)
    denominator=1+trace if rule=='fixed' else torch.where(trace>0,trace,torch.ones_like(trace))
    # Divide the gradient before multiplying eta; avoids an overflowing inverse
    # for small but representable float32 trace. Exact zero trace means no step.
    delta=-eta*(gradient/denominator[:,None,None])*updated[:,None,None]
    info=dict(updated=updated.detach(),observations=count.detach(),curvature_trace=trace.detach(),
        effective_eta=(eta.double()/denominator.double()*updated).detach(),
        delta_norm=delta.detach().square().sum((1,2)).sqrt(),
        loss_before=(error.square().sum((1,2))/(2*n)).detach(),
        loss_after=((k@(origin+delta)-v).square().sum((1,2))/(2*n)).detach())
    return delta,info


class AnchoredLayer(InnerLayer):
    def __init__(self,dim,rank=16,*,rule):
        super().__init__(dim,rank)
        if rule not in ('fixed','normalized'): raise ValueError(rule)
        self.rule=rule

    def forward(self,h,valid,*,mode):
        if mode=='off':
            return h,dict(updated=torch.zeros(len(h),dtype=torch.bool,device=h.device))
        if mode not in ('on','reverse'): raise ValueError(mode)
        q,k,v=self.coordinates(h)
        observed=valid.clone(); observed[:,-1]=False
        delta,info=centered_step(k,v,observed,self.origin,self.eta,rule=self.rule,reverse=mode=='reverse')
        residual=self.output(q@delta)*valid[...,None]
        info['residual_norm']=residual.detach().square().sum((1,2)).sqrt()
        return h+residual,info


class AnchoredAdapter(WindowAdapter):
    def __init__(self,base,*,rule):
        super().__init__(base)
        initial=self.layer.state_dict(); self.layer=AnchoredLayer(base.cfg.dim,rule=rule)
        self.layer.load_state_dict(initial)
