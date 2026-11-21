"""Historical cross-task spectral projection of arrived Q/K signal gradients."""
import numpy as np
import torch

from ..attention_basis_20261029.learning import feedback_gradient
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS=('cold','random_projector','spectral','reverse_target','constant_feedback')


def cross_moment(g,h,weights):
    if (g.ndim!=4 or g.shape[1:]!=(2,64,64) or h.shape!=g.shape or weights.shape!=(len(g),)
            or not all(torch.isfinite(v).all() for v in (g,h,weights)) or (weights<0).any()
            or not torch.isclose(weights.sum(),weights.new_tensor(1.))):
        raise ValueError('finite paired Q/K gradients and normalized nonnegative shot weights required')
    g=g.double();h=h.double();weights=weights.double()
    norm=g.flatten(1).norm(dim=1).clamp_min(1e-12)
    raw=h.transpose(-1,-2)@(g/norm[:,None,None,None])
    moment=(raw*weights[:,None,None,None]).sum(0)
    return (moment+moment.transpose(-1,-2))/2


def positive_projector(s):
    if (s.shape!=(2,64,64) or not torch.isfinite(s).all()
            or not torch.allclose(s,s.transpose(-1,-2),atol=1e-12,rtol=1e-8)):
        raise ValueError('two finite symmetric64x64 moments required')
    eigenvalues,u=torch.linalg.eigh(s.double())
    cutoff=eigenvalues.abs().amax(dim=1,keepdim=True)*1e-12
    keep=(eigenvalues[:,-2:]>cutoff).to(u.dtype)
    selected=u[:,:,-2:]
    return (selected*keep[:,None,:])@selected.transpose(-1,-2),eigenvalues


def proposal(g,p,eta):
    if (g.ndim!=4 or g.shape[1:]!=(2,64,64) or p.shape!=(2,64,64)
            or not torch.isfinite(g).all() or not torch.isfinite(p).all() or not 0<=eta<=1):
        raise ValueError('finite Q/K gradient/projector and step in[0,1] required')
    norm=g.flatten(1).norm(dim=1).clamp_min(1e-12)
    return -eta*((g/norm[:,None,None,None])@p.to(g))


def predict(net,signal,slow,policy):
    if set(signal)!={'values','available'}:raise ValueError('no outcomes in online inputs')
    p=torch.as_tensor(policy['projector'],device=net.base.center.device,dtype=net.base.center.dtype)
    eta=float(policy['step']);mean=p.new_tensor(policy['mean_gradient'])
    if (p.shape!=(2,64,64) or not torch.allclose(p,p.transpose(-1,-2),atol=1e-6,rtol=0)
            or not torch.allclose(p@p,p,atol=1e-5,rtol=0)
            or (torch.linalg.eigvalsh(p).gt(1e-5).sum(-1)>2).any()):
        raise ValueError('rank<=2 orthogonal projector per Q/K required')
    a=torch.stack((net.q_basis,net.k_basis));random=a.transpose(-1,-2)@a
    constant=proposal(mean[None],p,eta)[0].flatten()
    states={k:p.new_zeros(8192) for k in ARMS};scores={k:[] for k in ARMS}
    info={k:dict(updates=0,opportunities=0,observed_targets=0,max_matrix_norm=0.,last_observations=[]) for k in ARMS[1:]}
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,m=batch(signal,range(lo,hi),p.device)
        state=torch.cat([states[k].expand(len(x),-1) for k in ARMS])
        with torch.no_grad():
            z=net(x.repeat(len(ARMS),1,1),m.repeat(len(ARMS),1,1),state,slow,'qk',forecast=False)['logit']
            z=z.sigmoid().reshape(len(ARMS),len(x)).cpu().numpy()
        if not np.isfinite(z).all():raise FloatingPointError('invalid online risk')
        for k,v in zip(ARMS,z):scores[k].append(v)
        if hi>=256 and hi%128==0 and hi<len(signal['values']):
            prefix={k:v[:hi] for k,v in signal.items()};g,r=feedback_gradient(net,prefix,slow)
            reverse,_=feedback_gradient(net,prefix,slow,reverse=True)
            new=dict(random_projector=proposal(g[None],random,eta)[0].flatten(),
                spectral=proposal(g[None],p,eta)[0].flatten(),reverse_target=proposal(reverse[None],p,eta)[0].flatten(),
                constant_feedback=constant if r['observed'] else torch.zeros_like(constant))
            for k,v in new.items():
                states[k]=v.detach();q=info[k];q['updates']+=int(v.norm()>0);q['opportunities']+=1
                q['observed_targets']+=r['observed'];q['last_observations'].append(hi-1)
                q['max_matrix_norm']=max(q['max_matrix_norm'],float(v.norm()))
    return {k:np.concatenate(v) for k,v in scores.items()},info
