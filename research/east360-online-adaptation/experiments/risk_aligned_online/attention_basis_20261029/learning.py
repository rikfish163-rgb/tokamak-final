"""Historical operational supervision learns a Q/K update subspace, not scores."""
import numpy as np
import torch

from ..attention_lowrank_20261029.model import AttentionAdapter
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS=('cold','random_basis','learned_basis','reverse_target','constant_feedback')


class MatrixAttention(AttentionAdapter):
    """Same frozen forward as LoRA; explicit matrices permit exact cached G."""
    def deltas(self,state,mode):
        if state.ndim!=2 or state.shape[1]!=8192 or mode!='qk' or not torch.isfinite(state).all():
            raise ValueError('finite Bx8192 Q/K matrix updates required')
        d=state.reshape(len(state),2,64,64)
        return dict(q=d[:,0],k=d[:,1])


def project_rows(a):
    if a.shape!=(2,2,64) or not torch.isfinite(a).all():raise ValueError('two finite rank2 bases required')
    q,r=torch.linalg.qr(a.transpose(-1,-2),mode='reduced')
    diag=r.diagonal(dim1=-2,dim2=-1)
    if (diag.abs()<=1e-9).any():raise ValueError('rank-deficient basis')
    return (q*diag.sign()[:,None,:]).transpose(-1,-2)


def proposal(g,a,eta):
    if (g.ndim!=4 or g.shape[1:]!=(2,64,64) or a.shape!=(2,2,64) or eta.numel()!=1
            or not torch.isfinite(g).all() or not torch.isfinite(a).all() or not torch.isfinite(eta)
            or not 0<=float(eta.detach())<=1):raise ValueError('finite Q/K gradients/bases and step in[0,1] required')
    gb=g@a.transpose(-1,-2)
    norm=gb.flatten(1).norm(dim=1).clamp_min(1e-12)
    b=-eta*gb/norm[:,None,None,None]
    return b@a


def feedback_gradient(net,prefix,slow,*,reverse=False):
    if set(prefix)!={'values','available'} or len(prefix['values'])<256:raise ValueError('matured signal-only prefix required')
    count=len(prefix['values']);x,a,truth,mask,origins=forecast_examples(prefix,count,net.base.cfg,net.base.center.device)
    mask=mask&torch.isfinite(truth)&a[:,-1:]&torch.isfinite(x[:,-1:]);mask[:,:,10]=False
    if reverse:truth=truth.flip(1);mask=mask.flip(1)
    state=x.new_zeros(1,8192,requires_grad=True)
    prediction=net(x,a,state.expand(len(x),-1),slow,'qk')['forecast']
    loss=net.base.forecast_loss(prediction,truth,mask);g,=torch.autograd.grad(loss,state)
    if not torch.isfinite(g).all():raise FloatingPointError('invalid matrix feedback gradient')
    return g.detach().reshape(2,64,64),dict(count=count,origins=origins,observed=int(mask.sum()),loss=float(loss.detach()))


def predict(net,signal,slow,policy):
    if set(signal)!={'values','available'}:raise ValueError('no outcomes in online inputs')
    a=torch.as_tensor(policy['basis'],device=net.base.center.device,dtype=net.base.center.dtype)
    eta=a.new_tensor(policy['step']);mean=a.new_tensor(policy['mean_gradient'])
    if not torch.allclose(a@a.transpose(-1,-2),torch.eye(2,device=a.device).expand(2,2,2),atol=1e-5,rtol=0):raise ValueError('policy basis must be orthonormal')
    random=torch.stack((net.q_basis,net.k_basis));constant=proposal(mean[None],a,eta)[0].flatten()
    states={k:a.new_zeros(8192) for k in ARMS};scores={k:[] for k in ARMS}
    info={k:dict(updates=0,opportunities=0,observed_targets=0,max_matrix_norm=0.,last_observations=[]) for k in ARMS[1:]}
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,m=batch(signal,range(lo,hi),a.device)
        state=torch.cat([states[k].expand(len(x),-1) for k in ARMS])
        with torch.no_grad():
            z=net(x.repeat(len(ARMS),1,1),m.repeat(len(ARMS),1,1),state,slow,'qk',forecast=False)['logit']
            z=z.sigmoid().reshape(len(ARMS),len(x)).cpu().numpy()
        if not np.isfinite(z).all():raise FloatingPointError('invalid online risk')
        for k,v in zip(ARMS,z):scores[k].append(v)
        if hi>=256 and hi%128==0 and hi<len(signal['values']):
            prefix={k:v[:hi] for k,v in signal.items()};g,r=feedback_gradient(net,prefix,slow)
            rev,_=feedback_gradient(net,prefix,slow,reverse=True)
            new=dict(random_basis=proposal(g[None],random,eta)[0].flatten(),
                learned_basis=proposal(g[None],a,eta)[0].flatten(),reverse_target=proposal(rev[None],a,eta)[0].flatten(),
                constant_feedback=constant if r['observed'] else torch.zeros_like(constant))
            for k,v in new.items():
                states[k]=v.detach();q=info[k];q['updates']+=int(v.norm()>0);q['opportunities']+=1
                q['observed_targets']+=r['observed'];q['last_observations'].append(hi-1)
                q['max_matrix_norm']=max(q['max_matrix_norm'],float(v.norm()))
    return {k:np.concatenate(v) for k,v in scores.items()},info
