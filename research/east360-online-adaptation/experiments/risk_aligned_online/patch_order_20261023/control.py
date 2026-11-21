"""Past-patch order sensitivity, not an adaptation or physical intervention.

The newest patch and all simultaneous channel tuples are preserved. Earlier
patch order is reversed only when there is a full record window with a
time-constant availability pattern. Other endpoints remain in the evaluation
with unchanged inputs; coverage is reported, not hidden by excluding shots.
"""
import numpy as np
import torch

from ..paired_feedback_20261022.model import initial_encoder
from ..temporal_dependence_20261023.controls import causal_windows


def reverse_past_patches(values,available,counts,*,patch=8):
    x,m,n=np.asarray(values),np.asarray(available),np.asarray(counts)
    if (x.ndim!=3 or x.shape[-1]!=23 or m.shape!=x.shape or m.dtype!=bool
            or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x[m]).all()
            or n.shape!=(len(x),) or not np.issubdtype(n.dtype,np.integer) or (n<1).any()
            or type(patch) is not int or patch<1 or x.shape[1]%patch or x.shape[1]<3*patch):
        raise ValueError('finite observed B,W,23 windows, boolean masks and positive integer record counts required')
    eligible=(n>=x.shape[1]) & (m==m[:,:1]).all(axis=(1,2)) & m.any(axis=(1,2))
    blocks=x.shape[1]//patch
    permutation=np.concatenate([np.arange(k*patch,(k+1)*patch) for k in range(blocks-2,-1,-1)]
                               +[np.arange((blocks-1)*patch,blocks*patch)])
    result=x.copy();result[eligible]=x[eligible][:,permutation]
    return result,eligible


def predict_reversed(model,signal,*,device='cuda'):
    windows,masks=causal_windows(signal,model.cfg.window)
    fast=initial_encoder(model);parts={k:[] for k in ('probability','eligible','changed')}
    for lo in range(0,len(windows),32):
        x=np.array(windows[lo:lo+32],copy=True);m=np.array(masks[lo:lo+32],copy=True)
        xx,eligible=reverse_past_patches(x,m,np.arange(lo+1,lo+len(x)+1),patch=model.cfg.patch)
        h=model.predict_fast(torch.as_tensor(xx,device=device),torch.as_tensor(m,device=device),
                             fast,mode='risk_state')['risk_state']
        parts['probability'].append(model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().numpy())
        parts['eligible'].append(eligible)
        parts['changed'].append(((xx!=x)&m).any(axis=(1,2)))
    result={k:np.concatenate(v) for k,v in parts.items()}
    if not np.isfinite(result['probability']).all():raise FloatingPointError('nonfinite order-control prediction')
    return result
