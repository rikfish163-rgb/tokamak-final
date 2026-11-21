"""Arrived direct-H25 feedback for temporary attention matrices only."""
import numpy as np
import torch

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS=('cold','qk_arrived','qk_reverse','v_arrived')


def update(net,prefix,state,slow,arm):
    if set(prefix)!={'values','available'} or arm not in ARMS[1:]:
        raise ValueError('observed signal-only prefix and declared feedback arm required')
    count=len(prefix['values'])
    x,a,truth,mask,ends=forecast_examples(prefix,count,net.base.cfg,state.device)
    mask=mask & torch.isfinite(truth) & a[:,-1:] & torch.isfinite(x[:,-1:])
    # The stored mode index remains a risk input but its continuous distance is
    # not certified across devices; it is not a regression target.
    mask[:,:,10]=False
    if arm=='qk_reverse':
        truth=truth.flip(1);mask=mask.flip(1)
    info=dict(updates=0,observed_targets=int(mask.sum()),last_observation=count-1,
        first_affected_prediction=count,forecast_origins=ends,loss_before=0.,loss_after=0.,gradient_norm=0.)
    if not mask.any():return state,info
    mode='v' if arm=='v_arrived' else 'qk';fast=state.detach().clone().requires_grad_(True)
    prediction=net(x,a,fast.expand(len(x),-1),slow,mode)['forecast']
    loss=net.base.forecast_loss(prediction,truth,mask);g,=torch.autograd.grad(loss,fast);norm=g.norm()
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite attention feedback')
    new=(fast-.05*g/norm.clamp_min(1e-12)).detach();new=new/new.norm().clamp_min(1.)
    with torch.no_grad():
        after=net.base.forecast_loss(net(x,a,new.expand(len(x),-1),slow,mode)['forecast'],truth,mask)
    if not torch.isfinite(after):raise FloatingPointError('nonfinite updated forecast')
    info.update(updates=int(norm>0),loss_before=float(loss.detach()),loss_after=float(after),gradient_norm=float(norm))
    return new,info


def predict(net,signal,slow):
    if set(signal)!={'values','available'}:raise ValueError('no outcomes in online inputs')
    states={k:net.base.center.new_zeros(512) for k in ARMS[1:]}
    curves={k:[] for k in ARMS}
    info={k:dict(updates=0,opportunities=0,loss_before_sum=0.,loss_after_sum=0.,loss_decreased=0,
                 max_state_norm=0.,max_matrix_norm=0.,observed_targets=0,last_observations=[]) for k in states}
    for lo in range(0,len(signal['values']),32):
        hi=min(lo+32,len(signal['values']));x,a=batch(signal,range(lo,hi),net.base.center.device)
        qstates=torch.cat((x.new_zeros(len(x),512),states['qk_arrived'].expand(len(x),-1),states['qk_reverse'].expand(len(x),-1)))
        with torch.no_grad():
            qscore=net(x.repeat(3,1,1),a.repeat(3,1,1),qstates,slow,'qk',forecast=False)['logit'].sigmoid().reshape(3,len(x)).cpu().numpy()
            vscore=net(x,a,states['v_arrived'].expand(len(x),-1),slow,'v',forecast=False)['logit'].sigmoid().cpu().numpy()
        for name,score in zip(ARMS,(*qscore,vscore)):
            if not np.isfinite(score).all():raise FloatingPointError('nonfinite online risk')
            curves[name].append(score)
        if hi%128==0 and hi>=153 and hi<len(signal['values']):
            prefix={k:v[:hi] for k,v in signal.items()}
            for name in states:
                states[name],r=update(net,prefix,states[name],slow,name);q=info[name]
                q['updates']+=r['updates'];q['opportunities']+=1;q['observed_targets']+=r['observed_targets']
                q['loss_before_sum']+=r['loss_before'];q['loss_after_sum']+=r['loss_after']
                q['loss_decreased']+=int(r['loss_after']<r['loss_before'])
                q['last_observations'].append(r['last_observation'])
                q['max_state_norm']=max(q['max_state_norm'],float(states[name].norm()))
                ds=net.deltas(states[name][None],'v' if name=='v_arrived' else 'qk')
                q['max_matrix_norm']=max(q['max_matrix_norm'],float(sum(v.square().sum() for v in ds.values()).sqrt()))
    return {k:np.concatenate(v) for k,v in curves.items()},info
