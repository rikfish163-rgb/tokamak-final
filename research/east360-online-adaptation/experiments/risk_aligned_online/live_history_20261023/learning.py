"""Current historical tangents, with an explicit finite-loss control arm.

Only arrived diagnostic targets drive the proposed update. Authorized history
labels constrain its direction; no current-shot outcomes are accepted here.
The bounded guard is an experimental policy, not a nonlinear generalization
guarantee. Rejected proposals retain the previous fast state by design.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..entropy_stream_20261023.learning import validate
from ..history_tangent_20261023.learning import flat, logit, matched_step, apply_step
from ..label_cone_20261023.learning import Cone
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('live', 'guarded', 'norm_only')
HISTORY_TOL = 1e-6
FRACTIONS = (1., .5, .25, .125, .0625, .03125)


def bce(z, labels):
    return F.binary_cross_entropy_with_logits(z.detach().double(), labels.to(z).double(), reduction='none')


def current_geometry(model, anchors, fast):
    z = logit(model, anchors['x'], anchors['mask'], fast)
    # Batched VJP returns one parameter gradient per historical output, not
    # their average. Unit tests compare it with separate scalar backwards.
    gradients = torch.autograd.grad(z, tuple(fast.values()),
        grad_outputs=torch.eye(len(z), device=z.device, dtype=z.dtype), is_grads_batched=True)
    rows = torch.cat([g.reshape(len(z), -1) for g in gradients], dim=1).detach()
    if not torch.isfinite(z).all() or not torch.isfinite(rows).all():
        raise FloatingPointError('nonfinite current historical geometry')
    return rows, z.detach(), bce(z, anchors['labels'])


def history_loss(model, anchors, fast):
    return bce(logit(model, anchors['x'], anchors['mask'], fast), anchors['labels'])


def backtrack(fast, step, evaluate, before, origin):
    if before.shape!=origin.shape or not torch.isfinite(before).all() or not torch.isfinite(origin).all():
        raise ValueError('finite matching historical losses required')
    if float((before-origin).max())>HISTORY_TOL:
        raise AssertionError('guarded state already outside the fixed original envelope')
    trials = []
    for fraction in FRACTIONS:
        candidate = apply_step(fast, fraction*step)
        after = evaluate(candidate)
        if after.shape!=before.shape or not torch.isfinite(after).all():
            raise FloatingPointError('invalid history evaluation during line search')
        delta = float((after-before).max()); absolute = float((after-origin).max())
        accepted = delta<=HISTORY_TOL and absolute<=HISTORY_TOL
        trials.append(dict(fraction=fraction, max_step_bce_increase=delta,
                           max_parent_bce_increase=absolute, accepted=accepted))
        if accepted:
            return candidate, dict(fraction=fraction, trials=trials, reason='history_guard_accepted')
    # A failed finite-step proposal is not replaced by an unprotected update.
    # This explicit reject rule is predeclared and covered by regression tests.
    return fast, dict(fraction=0., trials=trials, reason='history_guard_rejected')


def update_all(model, signal, count, states, anchors, *, device):
    validate(signal)
    x, m, target, tm, ends = forecast_examples(signal, count, model.cfg, device)
    observed = int((tm & torch.isfinite(target)).sum())
    origin = bce(anchors['logits'], anchors['labels'])
    updated = {}; receipts = {}; accepted_length = None; parent = dict(model.named_parameters())
    for arm in ARMS:
        fast = states[arm]
        r = dict(last_observation=count-1, first_affected_prediction=count,
                 forecast_origins=ends, observed_targets=observed, gradient_steps=0,
                 parameter_delta_norm=0., requested_step_norm=0., accepted_step_norm=0.)
        if not observed:
            r['reason']='no_observed_targets'; updated[arm]=fast; receipts[arm]=r; continue
        if arm=='norm_only':
            before = history_loss(model, anchors, fast); cone = None; info = {}
        else:
            rows, _, before = current_geometry(model, anchors, fast)
            cone = Cone(rows, anchors['labels']); info = dict(gram_condition=cone.condition)
        loss = model.base.forecast_loss(model.predict_fast(x, m, fast)['forecast'], target, tm)
        g = flat(torch.autograd.grad(loss, tuple(fast.values()))).detach()
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite arrived diagnostic gradient')
        raw_norm = float(g.double().norm())
        if cone is None:
            direction = g; length = accepted_length
        else:
            direction, solver = cone.project(g); info.update(solver)
            length = .001*float(direction.double().norm())/max(1., raw_norm)
        step = matched_step(direction, length)
        if arm=='guarded':
            next_fast, guard = backtrack(fast, step, lambda q:history_loss(model,anchors,q), before, origin)
            info.update(guard); accepted_length = length*guard['fraction']; applied = guard['fraction']*step
        else:
            next_fast = apply_step(fast, step); applied = step
        actual = flat([fast[k].detach()-next_fast[k].detach() for k in fast])
        after_history = history_loss(model, anchors, next_fast)
        with torch.no_grad():
            after = model.base.forecast_loss(model.predict_fast(x,m,next_fast)['forecast'], target,tm)
        if not torch.isfinite(after) or not torch.isfinite(after_history).all():
            raise FloatingPointError('nonfinite updated diagnostics')
        r.update(info, gradient_steps=int(bool(actual.any())), parameter_delta_norm=float(actual.double().norm()),
            requested_step_norm=length, accepted_step_norm=accepted_length if arm=='guarded' else length,
            raw_gradient_norm=raw_norm, projected_gradient_norm=float(direction.double().norm()),
            loss_before=float(loss.detach()), loss_after=float(after),
            history_bce_before=before.cpu().tolist(), history_bce_change=(after_history-before).cpu().tolist(),
            history_change_from_parent=(after_history-origin).cpu().tolist(),
            parent_distance=float(flat([v.detach()-parent[k].detach() for k,v in next_fast.items()]).double().norm()))
        if cone is not None:
            r['current_linear_margin_min']=float((cone.h@applied.double()).min())
            r['current_actual_margin_min']=float((cone.h@actual.double()).min())
        updated[arm]=next_fast; receipts[arm]=r
    return updated, receipts


def predict(model, signal, anchors, *, device='cuda'):
    validate(signal)
    if model.cfg.window!=128 or model.cfg.horizon!=25:
        raise ValueError('declared W128 H25 model required')
    states={a:initial_encoder(model) for a in ARMS}
    curves={a:[] for a in ARMS}; updates={a:[] for a in ARMS}; n=len(signal['values'])
    for lo in range(0,n,128):
        hi=min(lo+128,n)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,t,128) for t in range(start,min(start+32,hi))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for a in ARMS:
                curves[a].append(logit(model,x,m,states[a]).sigmoid().detach().cpu().numpy())
        if hi>=256 and hi<n:
            states,rs=update_all(model,signal,hi,states,anchors,device=device)
            for a in ARMS:updates[a].append(rs[a])
    drift={a:(logit(model,anchors['x'],anchors['mask'],states[a]).detach()-anchors['logits']).cpu().tolist() for a in ARMS}
    curves={a:np.concatenate(v) for a,v in curves.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite stream predictions')
    return curves,updates,drift
