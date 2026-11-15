"""Fixed-entropy encoder control: fresh local steps versus within-shot accumulation."""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('local_reset','cumulative')


def validate(signal):
    if (set(signal)!={'values','available'} or signal['values'].ndim!=2
            or signal['values'].shape[1]!=23 or not len(signal['values'])
            or signal['available'].shape!=signal['values'].shape or signal['available'].dtype!=bool):
        raise ValueError('nonempty signal-only 23-column input required')


def adapt(model,signal,count,fast,*,device='cuda'):
    validate(signal)
    if (type(count)!=int or count<128 or count>len(signal['values'])
            or model.cfg.window!=128 or set(fast)!=set(model.encoder_names())):
        raise ValueError('W128 prefix and exact temporary encoder scope required')
    prefix={k:v[:count] for k,v in signal.items()}
    x,m=[torch.as_tensor(v[None],device=device) for v in simple_data.window(prefix,count-1,128)]
    receipt=dict(last_observation=count-1,first_affected_prediction=count,
                 gradient_steps=0,parameter_delta_norm=0.)
    if not (m&torch.isfinite(x)).any():
        receipt['reason']='no_observations';return fast,receipt
    def score(params):
        h=model.predict_fast(x,m,params,mode='risk_state')['risk_state']
        return model.base.risk_head(h)
    z=score(fast);loss=(F.softplus(z)-z*z.sigmoid()).mean()
    grads=torch.autograd.grad(loss,tuple(fast.values()))
    norm=torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite entropy step')
    factor=1./norm.clamp_min(1.)
    updated={k:(v-.001*factor*g).detach().requires_grad_(True) for (k,v),g in zip(fast.items(),grads)}
    delta=sum((updated[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    parent=dict(model.named_parameters())
    total=sum((v.detach()-parent[k].detach()).double().square().sum() for k,v in updated.items()).sqrt()
    after_z=score(updated);after=(F.softplus(after_z)-after_z*after_z.sigmoid()).mean()
    receipt.update(gradient_steps=1,parameter_delta_norm=float(delta),parent_distance=float(total),
                   gradient_norm=float(norm),clip_factor=float(factor),loss_before=float(loss.detach()),
                   loss_after=float(after.detach()),observed_logit_before=float(z.detach().reshape(())),
                   observed_logit_after=float(after_z.detach().reshape(())))
    return updated,receipt


def predict(model,signal,*,device='cuda'):
    validate(signal)
    if model.cfg.window!=128:raise ValueError('fixed W128 model required')
    n=len(signal['values']);fast={a:initial_encoder(model) for a in ARMS}
    curves={a:[] for a in ARMS};updates={a:[] for a in ARMS}
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,t,128) for t in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for a in ARMS:
                h=model.predict_fast(x,m,fast[a],mode='risk_state')['risk_state']
                curves[a].append(model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy())
        if hi>=256 and hi<n:
            for a in ARMS:
                origin=initial_encoder(model) if a=='local_reset' else fast[a]
                fast[a],receipt=adapt(model,signal,hi,origin,device=device)
                updates[a].append(receipt)
    curves={a:np.concatenate(v) for a,v in curves.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):raise FloatingPointError('nonfinite entropy trace')
    return curves,updates
