"""Offline diagnostic, NOT an online policy: future risk gradients are oracles.

Channel contributions retain the total observed-target denominator. No raw
physical invariant, future signal availability, or deployable gain is claimed.
"""
import numpy as np
import torch
from scipy.optimize import linprog
from torch.nn import functional as F

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import flat, logit, apply_step, matched_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder


def solve_weights(a, b):
    """Maximize worst future-risk descent, subject to signal descent per row.

Zero-gradient channels are allowed by the declared simplex. A zero-direction
optimum is explicitly reported by the finite-step test, never counted as a
successful adaptation. This is a first-order LP, not a normalized-step oracle.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) == 0 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching nonempty finite row-by-channel alignments required')
    n, c = a.shape
    inequalities = np.vstack((np.column_stack((-a, np.ones(n))), np.column_stack((-b, np.zeros(n)))))
    equality = np.array([[1.]*c+[0.]])
    objective = np.array([0.]*c+[-1.])
    result = linprog(objective, A_ub=inequalities, b_ub=np.zeros(2*n),
                     A_eq=equality, b_eq=[1.], bounds=[(0., None)]*c+[(None, None)], method='highs')
    if not result.success:
        raise RuntimeError(f'feedback-cone LP unsuccessful: {result.message}')
    w, t = result.x[:-1], float(result.x[-1])
    primal_error = max(float(np.maximum(inequalities@result.x, 0).max()),
                       abs(w.sum()-1.), float(np.maximum(-w, 0).max()))
    # HiGHS returns objective derivatives w.r.t. right-hand sides. Verify
    # stationarity and complementarity independently from solver success.
    stationarity = objective-inequalities.T@result.ineqlin.marginals-equality.T@result.eqlin.marginals-result.lower.marginals
    dual_error = float(np.max(np.abs(stationarity)))
    complementarity = max(float(np.max(np.abs(result.ineqlin.marginals*result.ineqlin.residual))),
                          float(np.max(np.abs(result.lower.marginals[:-1]*w))))
    if max(primal_error, dual_error, complementarity) > 1e-7:
        raise AssertionError('LP KKT audit failed')
    return w, dict(margin=t, minimum_signal_alignment=float((b@w).min()),
                   minimum_risk_alignment=float((a@w).min()), primal_error=primal_error,
                   dual_stationarity_error=dual_error, complementarity_error=complementarity)


def feedback_prefix(signal, query):
    if (set(signal) != {'values', 'available'} or type(query) is not int or query < 280
            or query >= len(signal['values']) or signal['values'].shape != signal['available'].shape
            or signal['values'].ndim != 2 or signal['values'].shape[1] != 23
            or signal['available'].dtype != bool):
        raise ValueError('signal-only 23-channel input and query>=280 required')
    count = query-24  # last actually observed endpoint is query-25.
    return {k:v[:count] for k,v in signal.items()}, count


def collect_geometry(model, signal, query, label, *, device):
    if label not in (0., 1.):
        raise ValueError('binary OFFLINE future risk target required')
    prefix, count = feedback_prefix(signal, query)
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('declared W128 H25 parent required')
    x, mask, target, tm, origins = forecast_examples(prefix, count, model.cfg, device)
    valid = target_mask(x, mask, target, tm)
    observed = int(valid.sum())
    if not observed:
        raise ValueError('no observed forecast targets: unevaluable, not a zero label')
    fast = initial_encoder(model)
    pred = model.predict_fast(x, mask, fast)['forecast']
    signal_loss = model.base.forecast_loss(pred, target, valid)
    g = flat(torch.autograd.grad(signal_loss, tuple(fast.values()), retain_graph=True)).detach()
    parts, losses = [], []
    for c in range(23):
        cm = torch.zeros_like(valid); cm[:, :, c] = valid[:, :, c]
        component = model.base.forecast_loss(pred, target, cm)*cm.sum()/valid.sum()
        parts.append(flat(torch.autograd.grad(component, tuple(fast.values()), retain_graph=True)).detach())
        losses.append(float(component.detach()))
    parts = torch.stack(parts)
    xx, mm = [torch.as_tensor(v[None], device=device) for v in simple_data.window(signal, query, 128)]
    z = logit(model, xx, mm, fast)
    y = torch.full_like(z, label)
    risk_loss = F.binary_cross_entropy_with_logits(z, y)
    r = flat(torch.autograd.grad(risk_loss, tuple(fast.values()))).detach()
    gd, rd, pd = g.double(), r.double(), parts.double()
    if not torch.isfinite(pd).all() or not torch.isfinite(rd).all() or min(gd.norm(), rd.norm()) <= 0:
        raise ValueError('nonfinite or zero total gradient: unevaluable geometry')
    relative = float((pd.sum(0)-gd).norm()/gd.norm())
    if relative >= 1e-5:
        raise AssertionError('channel gradients do not reconstruct total')
    a, b = pd@rd/(gd.norm()*rd.norm()), pd@gd/gd.square().sum()
    record = dict(query=query, last_observation=count-1, forecast_origins=origins,
                  label=float(label), offline_oracle=True, observed_targets=observed,
                  per_channel_observations=valid.sum((0,1)).cpu().tolist(),
                  risk_before=float(risk_loss.detach()), signal_before=float(signal_loss.detach()),
                  probability_before=float(z.detach().sigmoid()[0]),
                  gradient_norm=float(gd.norm()), risk_gradient_norm=float(rd.norm()),
                  gradient_sum_relative_error=relative, channel_loss=losses,
                  risk_alignment=a.cpu().tolist(), signal_alignment=b.cpu().tolist(),
                  channel_gram=(pd@pd.T).cpu().tolist())
    state = dict(parts=parts, risk_gradient=r, signal_gradient=g, fast=fast,
                 x=x, mask=mask, target=target, valid=valid, xx=xx, mm=mm, y=y)
    return record, state


def finite_step(model, state, direction, length):
    if direction.shape != state['signal_gradient'].shape or length not in (.001, .01):
        raise ValueError('matching direction and preregistered step norm required')
    step = matched_step(direction, length)
    changed = apply_step(state['fast'], step)
    # Both pre/post evaluations use the gradient-enabled math path. no_grad
    # would switch fused attention and contaminate tiny loss differences.
    z = logit(model, state['xx'], state['mm'], changed)
    risk = F.binary_cross_entropy_with_logits(z, state['y'])
    prediction = model.predict_fast(state['x'], state['mask'], changed)['forecast']
    loss = model.base.forecast_loss(prediction, state['target'], state['valid'])
    actual = flat([changed[k].detach()-v.detach() for k,v in state['fast'].items()])
    return dict(step_norm=length, actual_step_norm=float(actual.double().norm()),
                zero_direction=bool(direction.double().norm() == 0),
                signal_after=float(loss.detach()), risk_after=float(risk.detach()),
                probability_after=float(z.detach().sigmoid()[0]),
                predicted_risk_delta=float(-state['risk_gradient'].double()@step.double()),
                predicted_signal_delta=float(-state['signal_gradient'].double()@step.double()))
