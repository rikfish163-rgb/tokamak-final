"""Experimental arrived-subset feedback; not a deployed or admitted predictor.

One direct H25 forecast, disjoint level targets h1..5 and h6..25. The latter
receives h5 only as the left endpoint of the (5,6) difference. Callers own real
arrival snapshots and must emit risk before updating; no outcomes accepted.
"""
import torch
from torch.nn import functional as F

from ..fast_update import normalized_step
from ..ffn_intra_20261005.loss import _observed_signals, _masked_mean, weighted_signal_loss


def event_clock(origin,mode):
    """Tuples: target snapshot cutoff, update cutoff, earliest effect, segment."""
    if type(origin) is not int or origin<0 or origin%26 or mode not in ('partial','delayed'):
        raise ValueError('fixed26 origin and declared partial/delayed clock required')
    first=origin+(5 if mode=='partial' else 25)
    return [(origin+5,first,first+1,'early'),(origin+25,origin+25,origin+26,'late')]


def _block(truth,observed,segment):
    if segment not in ('early','late'):
        raise ValueError('declare early or late feedback segment')
    length=5 if segment=='early' else 21
    if (truth.ndim!=3 or truth.shape[1:]!=(length,23) or truth.shape!=observed.shape
            or observed.dtype!=torch.bool):
        raise ValueError('only arrived block required: early h1..5, late h5..25')
    valid=observed & torch.isfinite(truth)
    level=valid.clone()
    if segment=='late':level[:,0]=False
    return valid,level


def segment_components(prediction,truth,observed,center,scale,segment):
    valid,level_mask=_block(truth,observed,segment)
    if prediction.ndim!=3 or prediction.shape!=(len(truth),25,23):
        raise ValueError('unchanged direct H25x23 forecast required')
    prediction=prediction[:,:5] if segment=='early' else prediction[:,4:]
    prediction,target,valid=_observed_signals(prediction,truth,valid,center,scale)
    pairs=valid[:,1:] & valid[:,:-1]
    level=F.smooth_l1_loss(prediction,target,reduction='none')
    difference=F.smooth_l1_loss(prediction[:,1:]-prediction[:,:-1],
                               target[:,1:]-target[:,:-1],reduction='none')
    return dict(level=_masked_mean(level,level_mask),difference=_masked_mean(difference,pairs),
        valid=level_mask.any(1),difference_valid=pairs.any(1),
        level_count=int(level_mask.sum()),difference_count=int(pairs.sum()))


def segment_step(model,delta,x,available,truth,observed,segment,*,create_graph=False):
    _,valid=_block(truth,observed,segment)
    row=dict(segment=segment,level_count=int(valid.sum()),state_changed=False,
             loss=None,gradient_norm=None,reason='no_valid_targets')
    if not valid.any():
        # Empty events consume no update: not even a zero-gradient projection.
        return delta,row
    forecast=model(x,available,delta)['forecast']
    parts=segment_components(forecast,truth,observed,model.center,model.scale,segment)
    loss=weighted_signal_loss(parts).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite partial feedback loss')
    gradient,=torch.autograd.grad(loss,delta,create_graph=create_graph)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('nonfinite partial feedback gradient')
    step=.02 if segment=='early' else .08
    candidate=normalized_step(delta[None],torch.zeros_like(delta)[None],gradient[None],
                              step=step,radius=1.)[0]
    if not torch.isfinite(candidate).all():
        raise FloatingPointError('nonfinite partial feedback state')
    row.update(reason='finite_segment_update',loss=float(loss.detach()),
        difference_count=parts['difference_count'],step=step,
        gradient_norm=float(gradient.detach().double().norm()),
        state_changed=not torch.equal(candidate.detach(),delta.detach()))
    return candidate,row
