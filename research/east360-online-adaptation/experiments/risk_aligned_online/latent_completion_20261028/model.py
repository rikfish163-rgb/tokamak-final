"""Frozen-source temporal completion; disposable embedding adaptation only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from ..aligned_state_20261024.model import fixed_readout
from ..delayed_recurrent_memory_20261025.learning import batch


class Imputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(17*65,128), nn.SiLU(), nn.Linear(128,17*64))

    def forward(self, h, valid):
        clean = h.masked_fill(~valid[...,None],0.)
        x = torch.cat((clean,valid[...,None].to(h.dtype)),-1)
        return clean+self.net(x.flatten(1)).reshape(-1,17,64)


def tokens(base, values, available, state):
    if values.shape != available.shape or values.shape[1:] != (128,23) or available.dtype != torch.bool:
        raise ValueError('observed128x23 causal windows required')
    if state.shape not in ((128,), (len(values),128)) or not torch.isfinite(state).all():
        raise ValueError('finite embedding-only state required')
    a = available & torch.isfinite(values)
    z = torch.asinh((torch.where(a,values,base.center)-base.center)/base.scale)
    x = base.embedding(torch.cat((z,a.to(z.dtype)),-1).reshape(len(z),16,-1))
    gain,bias = state.expand(len(values),128).chunk(2,-1)
    x = x*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
    x = torch.cat((x,base.cls.expand(len(x),-1,-1)),1)+base.position
    v = a.reshape(len(x),16,-1).any(-1)
    v = torch.cat((v,torch.ones(len(x),1,device=x.device,dtype=torch.bool)),1)
    with sdpa_kernel(SDPBackend.MATH):
        h = base.encoder.layers[0](x,src_key_padding_mask=~v)
    return h,v


def masked(values, available):
    x = values.repeat_interleave(2,0)
    a = (available & torch.isfinite(values)).repeat_interleave(2,0)
    parity = torch.arange(2,device=x.device).repeat(len(values))
    keep = (torch.arange(128,device=x.device)//8)[None] % 2 != parity[:,None]
    return x,a & keep[...,None]


def latent_loss(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('matching latent targets and valid-token mask required')
    return ((prediction-target).square()*valid[...,None]).sum((1,2))/(valid.sum(1)*64).clamp_min(1)


def risk_kl(logit, teacher_logit):
    p = teacher_logit.detach().sigmoid()
    return F.binary_cross_entropy_with_logits(logit,p,reduction='none')-F.binary_cross_entropy_with_logits(teacher_logit.detach(),p,reduction='none')


def step(base, imputer, values, available, state, *, reverse=False):
    if len(values)!=1:
        raise ValueError('one current shot only')
    if not (available & torch.isfinite(values)).any():
        return state,dict(updates=0,gradient_norm=0.,loss=0.)
    fast = state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        h,v = tokens(base,values,available,fast)
        xx,aa = masked(values,available)
        mh,mv = tokens(base,xx,aa,fast)
        if reverse:
            # Reverse only the already observed signal-token features; CLS is
            # retained. This is a deliberately mismatched temporal comparator.
            mh=torch.cat((mh[:,:-1].flip(1),mh[:,-1:]),1)
            mv=torch.cat((mv[:,:-1].flip(1),mv[:,-1:]),1)
        loss=latent_loss(imputer(mh,mv),h.expand(2,-1,-1),v.expand(2,-1)).mean()
        gradient,=torch.autograd.grad(loss,fast)
        norm=gradient.norm()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError('nonfinite completion adaptation')
        proposed=fast-.05*gradient/norm.clamp_min(1e-12)
        proposed=proposed/proposed.norm().clamp_min(1.)
    return proposed.detach(),dict(updates=int(norm>0),gradient_norm=float(norm),loss=float(loss.detach()))


def readout(base,h,v,slow):
    if slow is not None:
        last=h[:,-2]+F.linear(h[:,-2],slow)*v[:,-2,None]
        h=torch.cat((h[:,:-2],last[:,None],h[:,-1:]),1)
    return fixed_readout(base,h,v)


def predict(base, imputers, signal, slow, *, reversed_arm=False):
    if set(signal)!={'values','available'}:
        raise ValueError('prediction accepts observed signals only, no outcome')
    specs={k:(m,False) for k,m in imputers.items()}
    if reversed_arm:
        specs['task_reverse']=(imputers['task'],True)
    states={k:base.center.new_zeros(128) for k in specs}
    result={k:[] for k in ('cold',*specs)}
    info={k:dict(updates=0,gradient_norm_sum=0.,loss_sum=0.,max_state_norm=0.) for k in specs}
    names=list(result)
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']))
        x,a=batch(signal,range(lo,hi),base.center.device)
        fast=torch.cat([base.center.new_zeros(len(x),128)]+[states[k].expand(len(x),-1) for k in specs])
        with torch.no_grad():
            h,v=tokens(base,x.repeat(len(names),1,1),a.repeat(len(names),1,1),fast)
            probabilities=readout(base,h,v,slow).sigmoid().reshape(len(names),len(x)).cpu().numpy()
        if not np.isfinite(probabilities).all():raise FloatingPointError('nonfinite risk')
        for name,p in zip(names,probabilities):result[name].append(p)
        # First score and seal this batch, then learn from its last available
        # complete128-sample window. Updates cannot revise the scored prefix.
        if hi%128==0 and hi<len(signal['values']):
            for name,(imputer,reverse) in specs.items():
                states[name],r=step(base,imputer,x[-1:],a[-1:],states[name],reverse=reverse)
                q=info[name];q['updates']+=r['updates'];q['gradient_norm_sum']+=r['gradient_norm'];q['loss_sum']+=r['loss']
                q['max_state_norm']=max(q['max_state_norm'],float(states[name].norm()))
    return {k:np.concatenate(v) for k,v in result.items()},info
