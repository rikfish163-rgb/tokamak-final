"""Regularized information probes, not deployed risk correction modules."""
import numpy as np
import torch
from torch.nn import functional as F
from ..arrived_evidence_20261024.run import normalize as fit_normalization
from ..event_prefix_probe import causal_summary
from .. import simple_data


def prefix_statistics(signal, ends, center, scale, *, device):
    ends=np.asarray(ends)
    if (set(signal)!={'values','available'} or ends.ndim!=1 or not len(ends)
            or not np.issubdtype(ends.dtype,np.integer) or ends.min()<127
            or ends.max()>=len(signal['values'])):
        raise ValueError('complete past128 windows in signal-only input required')
    c=torch.as_tensor(center,dtype=torch.float64,device=device)
    s=torch.as_tensor(scale,dtype=torch.float64,device=device)
    result=[]
    for start in range(0,len(ends),128):
        windows=[simple_data.window(signal,int(e),128) for e in ends[start:start+128]]
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        result.append(causal_summary(x.double(),m,c,s).cpu().numpy())
    return np.concatenate(result)


def linear_score(x, coefficient, intercept=True):
    return x@coefficient[:-1]+coefficient[-1] if intercept else x@coefficient


def logistic_fit(x,y,weights,*,offset=None,intercept=True,device,l2=.1):
    """Strict convex full-batch Newton fit with an explicit convergence check."""
    x,y,w=[np.asarray(a,dtype=np.float64) for a in (x,y,weights)]
    off=np.zeros(len(y)) if offset is None else np.asarray(offset,dtype=np.float64)
    if (x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or off.shape!=y.shape
            or set(np.unique(y))!={0.,1.} or (w<0).any() or w.sum()<=0 or l2<=0
            or not all(np.isfinite(v).all() for v in (x,y,w,off))):
        raise ValueError('finite paired inputs, two labels, nonnegative weights, positive L2 required')
    w=w/w.sum()
    if intercept:x=np.column_stack((x,np.ones(len(x))))
    xx,yy,ww,oo=[torch.as_tensor(v,dtype=torch.float64,device=device) for v in (x,y,w,off)]
    b=torch.zeros(x.shape[1],dtype=torch.float64,device=device)
    reg=torch.full_like(b,l2)
    if intercept:reg[-1]=0.
    def objective(candidate):
        z=xx@candidate+oo
        return (ww*(F.softplus(z)-yy*z)).sum()+.5*(reg*candidate.square()).sum()
    trace=[]
    for iteration in range(61):
        z=xx@b+oo;p=z.sigmoid()
        grad=xx.T@(ww*(p-yy))+reg*b
        current=objective(b);maximum=float(grad.abs().max())
        trace.append(dict(iteration=iteration,objective=float(current),gradient_max=maximum))
        if maximum<=1e-8:return b.cpu().numpy(),trace
        if iteration==60:break
        hessian=xx.T@((ww*p*(1-p))[:,None]*xx)+torch.diag(reg)
        direction=torch.linalg.solve(hessian,grad)
        descent=grad@direction
        if not torch.isfinite(direction).all() or descent<=0:
            raise FloatingPointError('invalid Newton direction')
        for shrink in range(32):
            step=2.**(-shrink);candidate=b-step*direction
            if objective(candidate)<=current-1e-4*step*descent:
                b=candidate;trace[-1]['step_size']=step;break
        else:raise RuntimeError('Newton line search did not converge')
    raise RuntimeError(f'Newton fit not stationary: {trace[-1]}')
