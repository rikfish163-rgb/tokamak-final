"""Centered feature covariance, source replay, and arrived H25 feedback.

Fixed-head Deep CORAL variant. No covariance inverse or output correction.
The independent source-supervision and natural-alignment samplers deliberately
do not impose a 50% disruptive-window mixture on the covariance estimate.
"""
import numpy as np
import torch

from .. import simple_data

SIGNAL_KEYS = {'shot', 'device', 'values', 'available'}


def coral_loss(source, target):
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or source.shape[1] < 1 or min(len(source), len(target)) < 2):
        raise ValueError('two feature matrices with matching dimension and >=2 rows required')
    if source.dtype not in (torch.float32, torch.float64) or target.dtype != source.dtype:
        raise ValueError('matching float32/64 covariance arithmetic required')
    a, b = source-source.mean(0), target-target.mean(0)
    cs, ct = a.T@a/(len(a)-1), b.T@b/(len(b)-1)
    return (cs-ct).square().mean()/4.


def validate_prefix(records, order, count):
    if (type(count) is not int or not 0<count<len(order)
            or len(order)!=len(set(order)) or len(records)!=count
            or [r['shot'] for r in records] != list(order[:count])
            or any(set(r)!=SIGNAL_KEYS or r['device']!='EAST' for r in records)):
        raise ValueError('exact ended EAST signal-only prefix required')


def eligible_records(records, cfg, *, future=False):
    """Declared full-window sample population, with explicit exclusion ledger.

TRAIN shot1057650 is shorter than W128 in the actual HDF5. It remains in
supervised training (the established padded sampler), but cannot supply a
full covariance window. H25 training additionally needs 25 arrived points.
This signal-only eligibility does not depend on disruption/event labels.
"""
    if not records or any(set(r)!=SIGNAL_KEYS for r in records):
        raise ValueError('signal-only records required')
    kept,excluded=[],[]
    horizon=cfg.horizon if future else 0
    for s in records:
        length=len(s['values'])
        reason=('too_short_for_full_context_and_target' if length<cfg.window+horizon else
                'no_observed_input' if not s['available'][:length-horizon].any() else None)
        if reason: excluded.append(dict(shot=s['shot'],length=length,reason=reason))
        else: kept.append(s)
    if not kept: raise ValueError('no eligible full-window records')
    return kept,excluded


def natural_batch(records, rng, size, cfg, device, *, future=False):
    if (not records or any(set(r)!=SIGNAL_KEYS for r in records)
            or type(size) is not int or size<1):
        raise ValueError('signal-only equal-shot sampling required')
    examples, receipts = [], []
    for _ in range(size):
        s=records[int(rng.integers(len(records)))]
        last=len(s['values'])-(cfg.horizon if future else 0)
        # Full causal windows, not post-outcome time selection. At least one
        # observation must exist in the input; no all-missing target fitting.
        valid=s['available'].any(1).astype(np.int64)
        cumulative=np.concatenate(([0],np.cumsum(valid)))
        ends=np.arange(cfg.window-1,last)
        ends=ends[(cumulative[ends+1]-cumulative[ends+1-cfg.window])>0]
        if not len(ends): raise ValueError(f'no full observed window: {s["shot"]}')
        end=int(rng.choice(ends))
        x,m=simple_data.window(s,end,cfg.window)
        example=(x,m)
        if future:
            example+=(s['values'][end+1:end+1+cfg.horizon],
                      s['available'][end+1:end+1+cfg.horizon])
        examples.append(example); receipts.append(dict(shot=int(s['shot']),end=end))
    return (*[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)],receipts)


def supervised_batch(shots, pools, rng, size, cfg, device):
    examples, receipts=[],[]
    for _ in range(size):
        label=int(rng.integers(2)); group=pools[label]
        i,ends=group[int(rng.integers(len(group)))]; end=int(rng.choice(ends))
        s=shots[i]; x,m=simple_data.window(s,end,cfg.window)
        examples.append((x,m,np.float32(label)))
        receipts.append(dict(shot=int(s['shot']),end=end,label=label))
    return (*[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)],receipts)


def forecast_examples(signal, observed_count, cfg, device):
    if set(signal)!={'values','available'}:
        raise ValueError('no outcome or event fields in arrived feedback')
    if (type(observed_count) is not int or not cfg.window+cfg.horizon<=observed_count<=len(signal['values'])
            or signal['values'].shape!=signal['available'].shape):
        raise ValueError('full context and actually arrived forecast targets required')
    # Four non-overlapping H25 targets, all ending before observed_count.
    ends=[observed_count-1-cfg.horizon*(j+1) for j in range(4)]
    ends=[e for e in ends if e>=cfg.window-1]
    examples=[]
    for end in ends:
        x,m=simple_data.window(signal,end,cfg.window)
        examples.append((x,m,signal['values'][end+1:end+1+cfg.horizon],
                         signal['available'][end+1:end+1+cfg.horizon]))
    return (*[torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)],ends)


def adapt_arrived(model, signal, observed_count, fast, *, device, learning_rate=1e-3):
    if not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('positive finite learning rate required')
    x,m,target,tm,ends=forecast_examples(signal,observed_count,model.cfg,device)
    count=int(tm.sum())
    receipt=dict(last_observation=observed_count-1, first_affected_prediction=observed_count,
                 forecast_origins=ends, observed_targets=count, gradient_steps=0,
                 parameter_delta_norm=0.)
    if not count:
        # A missing diagnostic is not a zero-valued target. No gradient exists
        # for an entirely unobserved target batch; record the non-update.
        return fast,receipt
    prediction=model.predict_fast(x,m,fast)['forecast']
    loss=model.base.forecast_loss(prediction,target,tm)
    grads=torch.autograd.grad(loss,tuple(fast.values()),allow_unused=False)
    norm=sum(g.square().sum() for g in grads).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite direct-forecast update')
    factor=(1./norm.clamp_min(1.)).detach()
    updated={k:(v-learning_rate*factor*g).detach().requires_grad_(True)
             for (k,v),g in zip(fast.items(),grads)}
    delta=sum((updated[k].detach()-v.detach()).double().square().sum() for k,v in fast.items()).sqrt()
    with torch.no_grad():
        after=model.base.forecast_loss(model.predict_fast(x,m,updated)['forecast'],target,tm)
    receipt.update(gradient_steps=1, gradient_norm=float(norm), parameter_delta_norm=float(delta),
                   loss_before=float(loss.detach()),loss_after=float(after))
    return updated,receipt
