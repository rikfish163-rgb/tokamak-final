"""Offline matched-site teaching; inner updates never receive query labels."""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples

GROUPS=('safe','early','danger')
STEP=.05


def query_pools(shots):
    pools={k:[] for k in GROUPS};excluded=[]
    for i,s in enumerate(shots):
        if s['device']!='J-TEXT':raise ValueError('offline J-TEXT episodes only')
        q=np.flatnonzero(s['valid']&(np.arange(len(s['valid']))>=256))
        if not len(q):excluded.append(dict(shot=s['shot'],reason='no_valid_query_after256'));continue
        if not s['is_disrupt']:pools['safe'].append((i,q));continue
        lead=np.round((s['down']-s['times'][q])*1000,9)
        early=q[s['y'][q]<=.5];danger=q[(s['y'][q]>.5)&(lead>10)]
        if len(early):pools['early'].append((i,early))
        if len(danger):pools['danger'].append((i,danger))
    return pools,excluded


def assemble(shots,choices,device):
    feedback=[];query=[];labels=[];receipts=[]
    for i,q in choices:
        s=shots[i];q=int(q)
        if q<256 or q>=len(s['values']) or not s['valid'][q]:raise ValueError('valid query after first clock required')
        count=256+128*((q-256)//128)
        signal={k:s[k][:count] for k in ('values','available')}
        # Same four matured H25 targets as the target-domain Q/K implementation.
        from ..simple23 import Config
        x,a,truth,mask,origins=forecast_examples(signal,count,Config(),device)
        if len(origins)!=4:raise ValueError('four complete targets required')
        feedback.append((x,a,truth,mask));query.append(simple_data.window(s,q,128));labels.append(float(s['y'][q]))
        receipts.append(dict(shot=s['shot'],query=q,count=count,origins=origins,
            last_target=count-1,label=float(s['y'][q]),is_disrupt=bool(s['is_disrupt'])))
    return dict(feedback={k:torch.stack(v) for k,v in zip(('x','a','truth','mask'),zip(*feedback))},
        qx=torch.as_tensor(np.stack([q[0] for q in query]),device=device),
        qa=torch.as_tensor(np.stack([q[1] for q in query]),device=device),
        y=torch.tensor(labels,device=device),receipts=receipts)


def forward(net,x,a,state,site,*,forecast):
    if site=='qk':return net(x,a,state,None,'qk',forecast=forecast)
    if site=='tokens':return net.base(x,a,net.base.initial_fast(len(x))+state)
    raise ValueError('qk or tokens required')


def adapt(net,feedback,*,site='qk',mode='true',create_graph=False):
    if set(feedback)!={'x','a','truth','mask'} or mode not in ('true','reverse','hold'):
        raise ValueError('signal-only feedback and declared control required')
    x=feedback['x'];a=feedback['a'];truth=feedback['truth'];mask=feedback['mask']
    if (x.ndim!=4 or x.shape[1:]!=(4,128,23) or a.shape!=x.shape or a.dtype!=torch.bool
            or truth.shape!=(len(x),4,25,23) or mask.shape!=truth.shape or mask.dtype!=torch.bool):
        raise ValueError('paired Bx4 full windows and arrived H25 targets required')
    state=x.new_zeros(len(x),8192 if site=='qk' else 128,requires_grad=True)
    mask=mask&torch.isfinite(truth)&a[:,:,-1:,:]&torch.isfinite(x[:,:,-1:,:])
    mask=mask.clone();mask[:,:,:,10]=False
    if mode=='reverse':truth=truth.flip(2);mask=mask.flip(2)
    if mode=='hold':truth=x[:,:,-1:,:].expand_as(truth)
    clean=torch.where(mask,truth.detach(),net.base.center)
    target=torch.asinh((clean-net.base.center)/net.base.scale)
    prediction=forward(net,x.flatten(0,1),a.flatten(0,1),state[:,None,:].expand(-1,4,-1).reshape(len(x)*4,-1),site,forecast=True)['forecast'].reshape_as(truth)
    loss=F.smooth_l1_loss(prediction,target,reduction='none').masked_fill(~mask,0.)
    loss=loss.sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)
    gradient,=torch.autograd.grad(loss.sum(),state,create_graph=create_graph)
    norm=(gradient.square().sum(1,keepdim=True)+1e-16).sqrt()
    updated=-STEP*gradient/norm
    if not torch.isfinite(updated).all():raise FloatingPointError('nonfinite arrived update')
    return updated,loss,state,gradient


def paired_risk(net,e,*,site='qk',mode='true',create_graph=False):
    updated,signal,initial,g=adapt(net,e['feedback'],site=site,mode=mode,create_graph=create_graph)
    before=forward(net,e['qx'],e['qa'],initial,site,forecast=False)['logit']
    losses=F.binary_cross_entropy_with_logits(before,e['y'],reduction='none')
    h,=torch.autograd.grad(losses.sum(),initial,retain_graph=True)
    h=h.detach();cosine=(g*h).sum(1)/((g.square().sum(1)+1e-16).sqrt()*(h.square().sum(1)+1e-16).sqrt())
    alignment=((1-cosine)*(h.norm(dim=1)>1e-8)).mean()
    after=forward(net,e['qx'],e['qa'],updated,site,forecast=False)['logit']
    after_loss=F.binary_cross_entropy_with_logits(after,e['y'],reduction='none')
    return dict(signal=signal,before=losses,after=after_loss,alignment=alignment,cosine=cosine,
        state_norm=updated.norm(dim=1),probability_change=after.sigmoid()-before.sigmoid())


def train_step(net,optimizer,e,mode):
    if mode not in ('forecast','matched'):raise ValueError('declared source training arm required')
    ids={id(p) for p in net.base.forecast_head.parameters()}
    if ({id(p) for g in optimizer.param_groups for p in g['params']}!=ids
            or any(p.requires_grad and id(p) not in ids for p in net.parameters())):
        raise ValueError('only existing forecast head may be trained')
    net.eval();optimizer.zero_grad(set_to_none=True)
    r=paired_risk(net,e,create_graph=True)
    objective=.2*r['signal'].mean()
    if mode=='matched':objective=objective+.5*r['after'].mean()+.1*r['alignment']
    if not torch.isfinite(objective):raise FloatingPointError('invalid source objective')
    objective.backward()
    norm=torch.nn.utils.clip_grad_norm_(net.base.forecast_head.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(objective=float(objective.detach()),signal=float(r['signal'].mean().detach()),
        risk_before=float(r['before'].mean().detach()),risk_after=float(r['after'].mean().detach()),
        cosine=float(r['cosine'].mean().detach()),head_gradient_norm=float(norm))
