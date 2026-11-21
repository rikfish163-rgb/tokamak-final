"""Source-prepared encoder copy; observed-signal-only episodic neural updates.

Inspired by TTT-MAE and CTA's source feature alignment, NOT a reproduction of
their image algorithms. Original parent and risk head stay immutable. Source
preparation changes a separate encoder; its static performance must be tested
separately before attributing any benefit to test-time updates.
"""
import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .redesign_20261004.intra import _batch_windows


def encoder_parameters(model):
    return {name:p for name,p in model.base.named_parameters()
            if name.startswith(('embedding.','encoder.','norm.'))}


def encoded(model,x,a):
    features=[]
    hook=model.base.norm.register_forward_hook(lambda mod,inp,out:features.append(out))
    try:out=model(x,a)
    finally:hook.remove()
    if len(features)!=1 or features[0].shape[-1]!=64:
        raise ValueError('one original64D encoder output required, not scalar risk alignment')
    return out,features[0]


def source_step(student,teacher,optimizer,x,a,y,*,phase,objective='joint'):
    # Closed ablation choices; default preserves the historical joint recipe.
    # All terms are measured, but an omitted term contributes no gradient.
    if objective not in ('joint','no_signal','no_supervised'):
        raise ValueError('explicit source objective control required')
    allowed=[*encoder_parameters(student).values(),*student.decoder.parameters()]
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len(allowed) or {id(p) for p in actual}!={id(p) for p in allowed}:
        raise ValueError('optimizer must update only copied encoder and auxiliary decoder')
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():_,target=encoded(teacher,x,a)
    out,state=encoded(student,x,a)
    valid=a.reshape(len(x),16,-1).any(-1)
    valid=torch.cat((valid,torch.ones(len(x),1,dtype=torch.bool,device=x.device)),1)
    alignment=((state-target.detach()).square()*valid[:,:,None]).sum()/(valid.sum()*64)
    supervised=F.binary_cross_entropy_with_logits(out['logit'],y)
    signal=student.reconstruction_loss(x,a,student.initial_fast(len(x)),phase=phase)
    if objective=='joint':loss=supervised+alignment+.20*signal
    elif objective=='no_signal':loss=supervised+alignment
    else:loss=alignment+.20*signal
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite source preparation loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(allowed,1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(loss.detach()),supervised=float(supervised.detach()),
        alignment=float(alignment.detach()),signal=float(signal.detach()),
        gradient_norm=float(norm),alignment_width=64,objective=objective)


def _distance(params,origin):
    return torch.stack([(p.detach()-origin[name]).double().square().sum()
                        for name,p in params.items()]).sum().sqrt()


def predict_shot(model,shot,*,intra,device='cpu',stride=32,step=.05,radius=1.,
                 objective='level',data_api=simple_data,scope='encoder'):
    # Legacy runner arguments are checked, not reinterpreted as learning rates.
    # This experiment locks SGD lr1e-3, clip1, whole-encoder radius0.5 below.
    if set(shot)&{'y','valid','is_disrupt','down','horizon','IsDisrupt','DownTime'}:
        raise ValueError('outcome fields prohibited before prediction seal')
    if (stride,step,radius,objective)!=(32,.05,1.,'level'):
        raise ValueError('explicit fixed runner protocol required')
    if scope not in ('encoder','decoder','both'):
        raise ValueError('explicit encoder/decoder/both update scope required')
    times=np.asarray(shot['times']);x=np.asarray(shot['values']);a=np.asarray(shot['available'])
    if (not len(times) or x.shape!=(len(times),23) or a.shape!=x.shape or a.dtype!=bool
            or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('ordered observed flat23 signal record required')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    flags={k:p.requires_grad for k,p in model.named_parameters()}
    groups={}
    if scope in ('encoder','both'):groups['encoder']=encoder_parameters(model)
    if scope in ('decoder','both'):
        groups['decoder']={f'decoder.{name}':p for name,p in model.decoder.named_parameters()}
    params={name:p for group in groups.values() for name,p in group.items()}
    origin={name:p.detach().clone() for name,p in params.items()}
    scores=np.empty(len(times),np.float32);attempts=[];start=0;final_norm=0.
    final_distances={name:0. for name in groups}
    risk_path_unchanged=False
    ticks=list(range(7,len(times)-1,32)) if intra else []
    raw=dict(values=x,available=a)
    try:
        model.requires_grad_(False)
        for p in params.values():p.requires_grad_(bool(intra))
        for end in [*ticks,len(times)-1]:
            with torch.no_grad():
                for lo in range(start,end+1,128):
                    hi=min(lo+128,end+1)
                    values,mask=_batch_windows(data_api,raw,range(lo,hi),model,device)
                    scores[lo:hi]=model(values,mask)['logit'].sigmoid().cpu().numpy()
            start=end+1
            if end not in ticks:continue
            values,mask=_batch_windows(data_api,raw,[end],model,device)
            losses,counts=model.four_losses(values,mask,model.initial_fast(1))
            loss=losses.mean()
            row=dict(update_index=end,observed_through=end,effective_index=end+1,
                accepted=False,loss_before=None,loss_after=None,targets=int(counts.sum()))
            if counts.sum():
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite observed-signal loss')
                grads=torch.autograd.grad(loss,list(params.values()))
                norm=torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite neural update gradient')
                gradients=dict(zip(params,grads))
                group_norms={name:float(torch.stack([gradients[key].double().square().sum()
                    for key in group]).sum().sqrt()) for name,group in groups.items()}
                before={name:p.detach().clone() for name,p in params.items()}
                with torch.no_grad():
                    # Separate clipping/projection preserves the encoder's step
                    # budget when decoder gradients are added. A joint global
                    # norm would confound decoder freedom with a smaller encoder step.
                    for group_name,group in groups.items():
                        scale=1e-3/max(1.,group_norms[group_name])
                        for name,p in group.items():p.add_(gradients[name],alpha=-scale)
                        distance=float(_distance(group,origin))
                        if distance>.5:
                            for name,p in group.items():
                                p.copy_(origin[name]+(p-origin[name])*(.5/distance))
                    after=model.four_losses(values,mask,model.initial_fast(1))[0].mean()
                if not torch.isfinite(after):raise FloatingPointError('nonfinite proposed-signal loss')
                accepted=bool(after<loss.detach())
                if not accepted:
                    with torch.no_grad():
                        for name,p in params.items():p.copy_(before[name])
                row.update(accepted=accepted,loss_before=float(loss.detach()),loss_after=float(after),
                    gradient_norm=float(norm),parameter_delta_norm=float(_distance(params,origin)),
                    group_gradient_norms=group_norms,
                    group_parameter_distances={name:float(_distance(group,origin))
                                               for name,group in groups.items()})
            attempts.append(row)
        final_norm=float(_distance(params,origin))
        final_distances={name:float(_distance(group,origin)) for name,group in groups.items()}
        allowed={name if name.startswith('decoder.') else f'base.{name}' for name in params}
        for name,value in model.state_dict().items():
            if name not in allowed:
                torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
        risk_path_unchanged=all(torch.equal(value,initial[name])
            for name,value in model.state_dict().items() if not name.startswith('decoder.'))
    finally:
        model.load_state_dict(initial,strict=True)
        for name,p in model.named_parameters():p.requires_grad_(flags[name])
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,initial[name],rtol=0,atol=0)
    return scores,dict(intra=bool(intra),attempts=attempts,attempted_updates=len(attempts),
        accepted_updates=sum(row['accepted'] for row in attempts),fast_discarded=True,
        parameter_delta_norm=final_norm,temporary_parameters=sum(p.numel() for p in params.values()),
        temporary_scope=list(params),update_lr=.001,gradient_clip=1.,parameter_radius=.5,
        parent_copy_restored=True,outcome_access=False,risk_head_updated=False,
        auxiliary_decoder_updated=bool(intra and scope in ('decoder','both') and
                                      final_distances.get('decoder',0)>0),
        update_scope=scope,parameter_group_distances=final_distances,
        risk_path_unchanged_before_reset=risk_path_unchanged,
        clipping_and_radius_are_per_group=True,real_time_latency_validated=False)
