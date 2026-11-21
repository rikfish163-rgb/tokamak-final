"""Causal Q/K updates with a history-fitted PSD contraction, not score correction."""
import numpy as np
import torch

from ..attention_basis_20261029.learning import feedback_gradient
from ..attention_spectral_20261029.learning import proposal
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS=('cold','random_contraction','protected','reverse_target','constant_feedback')


def validate_matrix(p):
    if (p.shape!=(2,64,64) or not torch.isfinite(p).all()
            or not torch.allclose(p,p.transpose(-1,-2),atol=1e-6,rtol=0)):
        raise ValueError('two finite symmetric64x64 matrices required')
    ev=torch.linalg.eigvalsh(p.double())
    if ev.min() < -1e-6 or ev.max()>1+1e-6 or (ev.sum(1)>2+1e-6).any():
        raise ValueError('PSD, operator norm<=1 and trace<=2 required')


def matched_random(p):
    validate_matrix(p)
    rng=torch.Generator(device='cpu').manual_seed(131)
    q,r=torch.linalg.qr(torch.randn(2,64,64,generator=rng,dtype=torch.float64))
    q=q*r.diagonal(dim1=-2,dim2=-1).sign()[:,None,:]
    eigenvalues=torch.linalg.eigvalsh(p.detach().cpu().double())
    return ((q*eigenvalues[:,None,:])@q.transpose(-1,-2)).to(p)


def accept_history(delta,flags,indices):
    if (delta.shape!=flags.shape or delta.ndim!=1 or flags.dtype!=np.bool_ or not np.isfinite(delta).all()
            or indices.ndim!=1 or len(indices)!=len(set(indices.tolist())) or not len(indices)
            or (indices<0).any() or (indices>=len(delta)).any()):
        raise ValueError('finite paired shot losses and unique fit indices required')
    d=delta[indices];f=flags[indices]
    if not f.any() or f.all():raise ValueError('both fitted classes required')
    return bool(np.all(d<=1e-6) and d[f].mean() < -1e-6)


def predict(net,signal,slow,policy):
    if set(signal)!={'values','available'}:raise ValueError('no outcomes in online inputs')
    p=torch.as_tensor(policy['matrix'],device=net.base.center.device,dtype=net.base.center.dtype)
    random=p.new_tensor(policy['random_matrix']);eta=float(policy['step']);mean=p.new_tensor(policy['mean_gradient'])
    validate_matrix(p);validate_matrix(random)
    if not torch.allclose(torch.linalg.eigvalsh(p.double()),torch.linalg.eigvalsh(random.double()),atol=1e-6,rtol=1e-5):
        raise ValueError('random control must match contraction spectrum')
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
            new=dict(random_contraction=proposal(g[None],random,eta)[0].flatten(),
                protected=proposal(g[None],p,eta)[0].flatten(),reverse_target=proposal(reverse[None],p,eta)[0].flatten(),
                constant_feedback=constant if r['observed'] else torch.zeros_like(constant))
            for k,v in new.items():
                states[k]=v.detach();q=info[k];q['updates']+=int(v.norm()>0);q['opportunities']+=1
                q['observed_targets']+=r['observed'];q['last_observations'].append(hi-1)
                q['max_matrix_norm']=max(q['max_matrix_norm'],float(v.norm()))
    return {k:np.concatenate(v) for k,v in scores.items()},info
