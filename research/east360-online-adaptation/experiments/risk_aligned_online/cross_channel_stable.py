"""Factorial diagnostic: four-view feedback x observed encoding-step control.

Original source classifier and auxiliary head remain fixed. The functional
bound measures only original patch-value encoding on an arrived window; it
does not bound future unseen windows, final risks, or physical state changes.
All arms compute four views, even when selecting only one rotating loss.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .cross_channel_prefix import CrossChannelModel
from .fast_update import normalized_step
from .redesign_20261004.intra import _batch_windows
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .simple_online import _durable_json,_hash

HEAD=Path(__file__).parent/'runs/cross_channel_operational_source_20261021/signal_head.pt'


class FourViewModel(CrossChannelModel):
    def four_losses(self,values,available,fast):
        if len(values)!=1 or self.site!='embedding':
            raise ValueError('one observed window and original embedding site required')
        a=(available & torch.isfinite(values)).expand(4,-1,-1)
        hidden=a & (torch.arange(23,device=values.device)[None,None,:]%4==
                    torch.arange(4,device=values.device)[:,None,None])
        visible=a & ~hidden
        hidden=hidden & visible.flatten(1).any(1)[:,None,None]
        masked=torch.where(visible,values.detach().expand(4,-1,-1),self.center)
        truth=torch.asinh((torch.where(a,values.detach(),self.center)-self.center)/self.scale)
        features=[];hook=self.base.norm.register_forward_hook(lambda mod,inp,out:features.append(out))
        try:self(masked,visible,fast.expand(4,-1))
        finally:hook.remove()
        if len(features)!=1:raise RuntimeError('one normalized original activation required')
        pred=self.decoder(features[0][:,:16]).reshape(4,128,23)
        error=F.smooth_l1_loss(pred,truth.detach(),reduction='none').masked_fill(~hidden,0.)
        counts=hidden.sum((1,2))
        return error.sum((1,2))/counts.clamp_min(1),counts


def embedding_change(model,x,available,delta):
    """RMS of exact linear embedding difference over real patch tokens."""
    if model.site!='embedding' or delta.shape!=(1,64*8*23):raise ValueError('original value-weight delta required')
    a=available & torch.isfinite(x)
    z=torch.asinh((torch.where(a,x.detach(),model.center)-model.center)/model.scale)
    patches=z.reshape(len(x),16,8*23)
    valid=a.reshape(len(x),16,-1).any(-1)
    movement=F.linear(patches,delta.reshape(64,8*23)).double()
    return (movement.square().masked_fill(~valid[:,:,None],0.).sum()/
            (valid.sum().clamp_min(1)*64)).sqrt()


def limited_step(model,x,a,current,gradient,*,cap):
    if cap is not None and (not np.isfinite(cap) or cap<=0):raise ValueError('positive source-only encoding cap required')
    proposed=normalized_step(current,current.new_zeros(current.shape),gradient)
    change=embedding_change(model,x,a,proposed-current)
    scale=1. if cap is None else min(1.,float(cap)/max(float(change),1e-30))
    result=current+scale*(proposed-current)
    actual=float(embedding_change(model,x,a,result-current))
    return result,dict(proposed_encoding_rms=float(change),actual_encoding_rms=actual,
                       scale=scale,limited=bool(scale<1.))


def predict_shot(model,shot,*,intra,feedback='rotate',cap=None,device='cpu',
                 stride=32,step=.05,radius=1.,objective='level',data_api=simple_data):
    if set(shot)&{'y','valid','is_disrupt','down','horizon','IsDisrupt','DownTime'}:
        raise ValueError('outcome fields prohibited before seal')
    if feedback not in ('rotate','joint') or (stride,step,radius,objective)!=(32,.05,1.,'level'):
        raise ValueError('locked rotate/joint,first7,stride32,step.05,radius1 protocol')
    times=np.asarray(shot['times']);values=np.asarray(shot['values']);a=np.asarray(shot['available'])
    if (not len(times) or values.shape!=(len(times),23) or a.shape!=values.shape or a.dtype!=bool
            or not np.isfinite(times).all() or (np.diff(times)<=0).any()):raise ValueError('ordered signal record required')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    fast=model.initial_fast(1).requires_grad_();first=model.summarize_fast(fast)
    raw=dict(values=values,available=a);scores=np.empty(len(times),np.float32);attempts=[];start=0
    ticks=list(range(7,len(times)-1,32)) if intra else []
    for i,end in enumerate([*ticks,len(times)-1]):
        for lo in range(start,end+1,128):
            hi=min(lo+128,end+1);x,mask=_batch_windows(data_api,raw,range(lo,hi),model,device)
            scores[lo:hi]=model(x,mask,fast.expand(hi-lo,-1))['logit'].detach().sigmoid().cpu().numpy()
        start=end+1
        if end not in ticks:continue
        x,mask=_batch_windows(data_api,raw,[end],model,device)
        losses,counts=model.four_losses(x,mask,fast)
        phase=i%4;loss=losses[phase] if feedback=='rotate' else losses.mean()
        count=int(counts[phase] if feedback=='rotate' else counts.sum())
        row=dict(update_index=end,observed_through=end,effective_index=end+1,
            predictions_sealed_through_index=end,phase=phase,feedback=feedback,
            computed_views=4,targets=count,accepted=False,loss_before=None,loss_after=None,
            fast_before=model.summarize_fast(fast),fast_after=model.summarize_fast(fast),
            reason='no_observed_hidden_targets')
        if count:
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite signal loss')
            grad,=torch.autograd.grad(loss,fast)
            candidate,diagnostic=limited_step(model,x,mask,fast.detach(),grad.detach(),cap=cap)
            candidate=candidate.requires_grad_()
            newlosses,_=model.four_losses(x,mask,candidate)
            after=newlosses[phase] if feedback=='rotate' else newlosses.mean()
            if not torch.isfinite(after):raise FloatingPointError('nonfinite candidate loss')
            accept=bool(after<loss.detach())
            row.update(**diagnostic,loss_before=float(loss.detach()),loss_after=float(after.detach()),
                views_before=losses.detach().tolist(),views_after=newlosses.detach().tolist(),
                gradient_norm=float(grad.detach().double().norm()),accepted=accept,
                reason='selected_loss_decreased' if accept else 'selected_loss_not_decreased')
            if accept:fast=candidate;row['fast_after']=model.summarize_fast(fast)
        fast=fast.detach().requires_grad_();attempts.append(row)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    return scores,dict(intra=bool(intra),feedback=feedback,encoding_step_cap=cap,
        attempts=attempts,attempted_updates=len(attempts),accepted_updates=sum(r['accepted'] for r in attempts),
        fast_initial=first,fast_final=model.summarize_fast(fast),fast_discarded=True,
        model_parameter_change_norm=0.,model_buffer_change_norm=0.,outcome_access=False,
        real_time_latency_validated=False,computed_views_per_update=4)


def main():
    p=argparse.ArgumentParser(description='Source-only calibration of the encoding-step bound')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(131)
    roles=simple_data.source_roles();train=list(map(int,roles['train']))
    if len(train)!=1353 or set(train)&(set(roles['val'])|set(roles['test'])):raise ValueError('canonical source TRAIN required')
    ids=sorted(map(int,np.random.default_rng(131).choice(train,120,replace=False)))
    base,_=load_parent(DEFAULT_PARENT);m=FourViewModel(base,site='embedding').cuda().eval()
    head=torch.load(HEAD,map_location='cpu',weights_only=True)
    if head['parent_sha256']!=PARENT_SHA256 or head['branch']!='signal':raise ValueError('head provenance')
    m.decoder.load_state_dict(head['decoder']);m.requires_grad_(False)
    records=[]
    for n,sid in enumerate(ids,1):
        s=simple_data.read_shot('J-TEXT',sid)
        _,audit=predict_shot(m,s,intra=True,feedback='rotate',cap=None,device='cuda')
        _durable_json(args.output/f'{sid}_audit.json',audit)
        records.extend(dict(shot=sid,**r) for r in audit['attempts'] if r['accepted'])
        if n%40==0:print(json.dumps(dict(source_shots=n)),flush=True)
    if not records:raise ValueError('source calibration needs actual accepted nonzero steps')
    values=np.array([r['actual_encoding_rms'] for r in records])
    cap=float(np.quantile(values,.95))
    if not np.isfinite(cap) or cap<=0:raise ValueError('nonpositive measured source cap')
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_stable.py','cross_channel_prefix.py','simple23.py','simple_data.py',
            'io23.py','fast_update.py','redesign_20261004/intra.py'):
            z.write(root/name,name)
    _durable_json(args.output/'calibration.json',dict(scope='SOURCE TRAIN SIGNALS ONLY; NO LABEL READ',
        parent_sha256=PARENT_SHA256,head_sha256=_hash(HEAD),ids=ids,seed=131,
        source_outcomes_read=False,east_data_used=False,quantile=.95,encoding_rms_cap=cap,
        accepted_steps=len(records),rms_quantiles=np.quantile(values,[0,.25,.5,.75,.95,1]).tolist(),
        definition='RMS patch-value embedding change over observed tokens, NOT risk or physical change',
        does_not_bound_unseen_future_inputs=True,code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    print(json.dumps(dict(encoding_rms_cap=cap,accepted_steps=len(records))),flush=True)


if __name__=='__main__':main()
