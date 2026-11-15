"""Ended-cold20 caches and fixed risk-guided operator fitting, no future data."""
import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..lifecycle_stream_20261025.protocol import sampled_endpoints
from ..warmstart_alignment_20261022.alignment import transform
from .learning import configure_operator_training, objective, risk_logits

TRAINABLE = {'memory.key.weight', 'memory.key.bias', 'memory.kernel', 'memory.rate'}


def pair_schedule(examples):
    ids = [e['shot'] for e in examples]
    if (len(ids) != 20 or len(set(ids)) != 20
            or any(type(e['flag']) is not bool for e in examples)):
        raise ValueError('exactly20 distinct ended shots with boolean classes required')
    ds = [i for i, e in enumerate(examples) if e['flag']]
    ss = [i for i, e in enumerate(examples) if not e['flag']]
    if len(ds) != 18 or len(ss) != 2:
        raise ValueError('the fixed current-order cold18D/2S contract changed')
    return [dict(epoch=epoch, disruptive_index=i, safe_index=ss[j % 2])
            for epoch in (1, 2) for j, i in enumerate(ds)]


@torch.no_grad()
def prepare_examples(model, pending, mapping):
    if [s['shot'] for s, _ in pending] != mapping['calibration_ids']:
        raise ValueError('only exact ended mapping prefix may prepare the operator')
    examples, manifest = [], []
    device = model.base.center.device
    for s, scores in pending:
        if s['device'] != 'EAST':
            raise ValueError('EAST-only rule fitting')
        signal = transform({k:s[k] for k in ('values', 'available')}, mapping)
        ends = sampled_endpoints(s['times'], s['y'], s['valid'], scores,
                                 down=s['down'] if s['is_disrupt'] else None)
        if (ends < 0).any() or (ends >= len(signal['values'])).any():
            raise ValueError('endpoint outside actual shot, including partial final patches')
        hs, vs = [], []
        for lo in range(0, len(ends), 32):
            x, m = batch(signal, ends[lo:lo+32], device)
            h, v = model.tokens(x, m)
            hs.append(h.detach())
            vs.append(v.detach())
        means, complete = patch_signal(signal, model.base.center, model.base.scale)
        y = torch.as_tensor(s['y'][ends], device=device, dtype=torch.float32)
        examples.append(dict(h=torch.cat(hs), valid=torch.cat(vs), means=means.detach(),
                             complete=complete.detach(), y=y, flag=bool(s['is_disrupt']),
                             ends=ends.tolist(), shot=int(s['shot']),
                             actual_samples=len(signal['values'])))
        used = ends[s['y'][ends] > 0] if s['is_disrupt'] else ends
        manifest.append(dict(shot=s['shot'], flag=bool(s['is_disrupt']), ends=ends.tolist(),
                             labels=s['y'][ends].tolist(), used_by_loss=used.tolist(),
                             actual_samples=len(signal['values']), observed_patches=len(means)))
    pair_schedule(examples)
    return examples, manifest


@torch.no_grad()
def training_evaluation(model, examples):
    rows = []
    for e in examples:
        loss, info = objective(model, e)
        rows.append(dict(shot=e['shot'], flag=e['flag'], loss=float(loss), **info))
    balanced = np.mean([np.mean([r['loss'] for r in rows if r['flag'] == flag])
                        for flag in (False, True)])
    return dict(balanced_loss=float(balanced), per_shot=rows)


def real_preflight(model, examples, pending, mapping):
    """Verify cache/gradient and future-prefix contracts on one real D and S."""
    selected = configure_operator_training(model)
    observations = {s['shot']:s for s, _ in pending}
    receipts = []
    for flag in (True, False):
        e = next(e for e in examples if e['flag'] == flag)
        s = observations[e['shot']]
        signal = transform({k:s[k] for k in ('values', 'available')}, mapping)
        hs, vs = [], []
        with torch.no_grad():
            for lo in range(0, len(e['ends']), 32):
                x, m = batch(signal, e['ends'][lo:lo+32], model.base.center.device)
                h, valid = model.tokens(x, m)
                hs.append(h)
                vs.append(valid)
        fresh = dict(e, h=torch.cat(hs), valid=torch.cat(vs))
        assert torch.equal(e['h'], fresh['h']) and torch.equal(e['valid'], fresh['valid'])
        loss, _ = objective(model, e)
        gradients = torch.autograd.grad(loss, selected)
        other_loss, _ = objective(model, fresh)
        other_gradients = torch.autograd.grad(other_loss, selected)
        assert torch.equal(loss, other_loss)
        assert all(torch.equal(a, b) and torch.isfinite(a).all()
                   for a, b in zip(gradients, other_gradients, strict=True))
        norms = {name:float(g.norm()) for name, g in zip(
            ('key.weight', 'key.bias', 'kernel', 'rate'), gradients, strict=True)}
        # Preserve every signal through index128 and alter only later arrivals.
        changed = {k:np.array(v, copy=True) for k, v in signal.items()}
        changed['values'][129:] += 31.
        changed['available'][129:] = False
        causal = []
        with torch.no_grad():
            for observed in (signal, changed):
                ends = [16, 127, 128]
                x, m = batch(observed, ends, model.base.center.device)
                h, valid = model.tokens(x, m)
                means, complete = patch_signal(observed, model.base.center, model.base.scale)
                causal.append(risk_logits(model, dict(h=h, valid=valid, means=means,
                                                      complete=complete, ends=ends)))
        assert torch.equal(*causal), 'future arrivals changed earlier risk'
        receipts.append(dict(shot=e['shot'], disruptive=flag, cache_logits_loss_gradients_exact=True,
                             gradient_norms=norms, gradients_finite=True,
                             future_after128_does_not_affect=[16, 127, 128]))
    return receipts


def fit_operator(model, examples, *, progress):
    selected = configure_operator_training(model)
    actual = {n for n, p in model.named_parameters() if p.requires_grad}
    if actual != TRAINABLE or sum(p.numel() for p in selected) != 7001:
        raise ValueError('operator-only7001 parameter contract changed')
    before = {n:v.detach().clone() for n, v in model.state_dict().items()}
    with torch.no_grad():
        off_before = [risk_logits(model, e, mode='off').detach().clone() for e in examples]
    initial = training_evaluation(model, examples)
    optimizer = torch.optim.Adam(selected, lr=.001)
    trace = []
    for step, pair in enumerate(pair_schedule(examples), 1):
        optimizer.zero_grad(set_to_none=True)
        losses, details = [], []
        for index in (pair['disruptive_index'], pair['safe_index']):
            e = examples[index]
            # A fresh full causal trace on every use; no detached cached delta.
            loss, detail = objective(model, e)
            (.5 * loss).backward()
            losses.append(float(loss.detach()))
            details.append(dict(shot=e['shot'], flag=e['flag'], **detail))
        gradients = {}
        for name, parameter in model.named_parameters():
            if name in TRAINABLE:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('missing/nonfinite real operator gradient: '+name)
                gradients[name] = float(parameter.grad.norm())
            elif parameter.grad is not None:
                raise AssertionError('frozen parameter received gradient: '+name)
        norm = float(torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True))
        optimizer.step()
        row = dict(step=step, epoch=pair['epoch'], loss=float(np.mean(losses)),
                   shot_losses=losses, shots=details, gradients=gradients,
                   norm_before_clip=norm, clip_applied=norm > 1.,
                   eta=float(model.memory.eta.detach()))
        trace.append(row)
        progress(row)
    after = model.state_dict()
    frozen = [n for n in before if n not in TRAINABLE]
    assert all(torch.equal(before[n], after[n]) for n in frozen), 'a frozen tensor changed'
    for name in TRAINABLE:
        assert any(r['gradients'][name] > 0 for r in trace), name
        assert not torch.equal(before[name], after[name]), name
    with torch.no_grad():
        assert all(torch.equal(old, risk_logits(model, e, mode='off'))
                   for old, e in zip(off_before, examples, strict=True)), 'off risk changed'
    final = training_evaluation(model, examples)
    return dict(optimizer_steps=36, parameters=7001, training=trace,
                before=initial, after=final, frozen_tensors_bitwise=len(frozen),
                zero_feedback_training_logits_bitwise=True,
                parameter_change_norms={n:float((after[n]-before[n]).norm()) for n in TRAINABLE})
