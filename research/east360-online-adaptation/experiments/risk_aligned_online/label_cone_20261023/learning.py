"""GEM-style signed historical tangents, fixed at the preparation parent.

This is NOT original GEM: history Jacobians are not recomputed at each step.
Actual historical BCE changes are measured separately, never used as a veto.
"""
import numpy as np
import torch
from scipy.optimize import nnls
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..entropy_stream_20261023.learning import validate
from ..history_tangent_20261023.learning import flat,logit,matched_step,apply_step
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('signed','norm_only','wrong_labels')


class Cone:
    """Exact small dual QP for full-row-rank, homogeneous constraints H u >= 0."""
    def __init__(self,rows,labels):
        if (rows.ndim!=2 or min(rows.shape)<1 or labels.shape!=(len(rows),)
                or not torch.isfinite(rows).all() or not torch.isin(labels,labels.new_tensor([0,1])).all()
                or (rows.double().norm(dim=1)==0).any()):
            raise ValueError('nonzero finite output gradients and binary labels required')
        h=rows.detach().double()
        self.h=(h/h.norm(dim=1,keepdim=True))*(1-2*labels.to(h))[:,None]
        gram=(self.h@self.h.T).cpu().numpy()
        try:
            self.cholesky=np.linalg.cholesky(gram)
        except np.linalg.LinAlgError as error:
            # The archived twenty anchors are full rank. Dependent geometry
            # needs a different solver; never hide failure by changing updates.
            raise ValueError('full-row-rank historical constraints required') from error
        self.condition=float(np.linalg.cond(gram))

    def project(self,gradient):
        if (gradient.ndim!=1 or gradient.shape[0]!=self.h.shape[1]
                or gradient.device!=self.h.device or not torch.isfinite(gradient).all()):
            raise ValueError('finite matching gradient required')
        g=gradient.detach().double();c=(self.h@g).cpu().numpy()
        if (c>=0).all():
            dual=np.zeros(len(c))
        else:
            # Gram=L L'. The dual is equivalent to ||L' lambda + L^-1 c||².
            dual,_=nnls(self.cholesky.T,-np.linalg.solve(self.cholesky,c),maxiter=2000)
        lam=torch.as_tensor(dual,device=g.device,dtype=torch.float64)
        u=g+self.h.T@lam
        margin=self.h@u;scale=max(1.,float(g.norm()))
        complementary=float((lam*margin).abs().max())
        stationarity=float((u-g-self.h.T@lam).abs().max())
        if (not torch.isfinite(u).all() or float(margin.min()) < -1e-9*scale
                or (dual<0).any() or complementary>1e-8*scale*scale
                or stationarity>1e-9*scale):
            raise FloatingPointError('historical QP failed KKT verification')
        if float(u.norm())<=1e-12*scale:
            u.zero_()  # numerical zero of an exact zero projection, not a raw-gradient substitute
        output=u.to(gradient);cast_margin=self.h@output.double()
        if float(cast_margin.min()) < -2e-6*scale:
            raise FloatingPointError('float-cast projection violated historical constraints')
        return output,dict(min_margin_before=float(c.min()),min_margin_after=float(cast_margin.min()),
            active_constraints=int((dual>1e-10).sum()),kkt_complementarity=complementary,
            kkt_stationarity=stationarity,dual_max=float(dual.max()),raw_norm=float(g.norm()),
            projected_norm=float(output.double().norm()))


def history_bce(model,anchors,fast):
    z=logit(model,anchors['x'],anchors['mask'],fast).detach()
    return F.binary_cross_entropy_with_logits(z,anchors['labels'].to(z),reduction='none')


def update_all(model,signal,count,states,cones,anchors,*,device):
    validate(signal)
    x,m,target,tm,ends=forecast_examples(signal,count,model.cfg,device)
    observed=int((tm&torch.isfinite(target)).sum())
    updated={};receipts={};length=None;parent=dict(model.named_parameters())
    for arm in ARMS:
        fast=states[arm]
        r=dict(last_observation=count-1,first_affected_prediction=count,forecast_origins=ends,
            observed_targets=observed,gradient_steps=0,parameter_delta_norm=0.,requested_step_norm=0.)
        if not observed:
            r['reason']='no_observed_targets';updated[arm]=fast;receipts[arm]=r;continue
        before_bce=history_bce(model,anchors,fast)
        loss=model.base.forecast_loss(model.predict_fast(x,m,fast)['forecast'],target,tm)
        g=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach()
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite arrived feedback')
        if arm=='norm_only':
            direction=g;info={}
        else:
            direction,info=cones[arm].project(g)
        raw=float(g.double().norm());direction_norm=float(direction.double().norm())
        if arm=='signed':length=.001*direction_norm/max(1.,raw)
        step=matched_step(direction,length)
        next_fast=apply_step(fast,step)
        actual=flat([fast[k].detach()-next_fast[k].detach() for k in fast])
        after_bce=history_bce(model,anchors,next_fast)
        with torch.no_grad():
            after=model.base.forecast_loss(model.predict_fast(x,m,next_fast)['forecast'],target,tm)
        if not torch.isfinite(after) or not torch.isfinite(after_bce).all():
            raise FloatingPointError('nonfinite post-update diagnostic')
        r.update(info,gradient_steps=int(bool(step.any())),requested_step_norm=length,
            parameter_delta_norm=float(actual.double().norm()),raw_gradient_norm=raw,
            projected_gradient_norm=direction_norm,loss_before=float(loss.detach()),loss_after=float(after),
            history_bce_before=before_bce.cpu().tolist(),history_bce_change=(after_bce-before_bce).cpu().tolist(),
            true_linear_margin_min=float((cones['signed'].h@step.double()).min()),
            true_actual_margin_min=float((cones['signed'].h@actual.double()).min()),
            parent_distance=float(flat([v.detach()-parent[k].detach() for k,v in next_fast.items()]).double().norm()))
        updated[arm]=next_fast;receipts[arm]=r
    return updated,receipts


def predict(model,signal,cones,anchors,*,device='cuda'):
    validate(signal)
    if model.cfg.window!=128 or model.cfg.horizon!=25 or set(cones)!={'signed','wrong_labels'}:
        raise ValueError('fixed W128 H25 model and declared cones required')
    states={a:initial_encoder(model) for a in ARMS}
    curves={a:[] for a in ARMS};updates={a:[] for a in ARMS};n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,t,128) for t in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for a in ARMS:
                curves[a].append(logit(model,x,m,states[a]).sigmoid().detach().cpu().numpy())
        if hi>=256 and hi<n:
            states,receipts=update_all(model,signal,hi,states,cones,anchors,device=device)
            for a in ARMS:updates[a].append(receipts[a])
    drift={a:(logit(model,anchors['x'],anchors['mask'],states[a]).detach()-anchors['logits']).cpu().tolist() for a in ARMS}
    curves={a:np.concatenate(v) for a,v in curves.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite inference')
    return curves,updates,drift
