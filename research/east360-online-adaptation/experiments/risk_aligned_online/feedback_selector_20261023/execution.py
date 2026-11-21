"""Strictly separated prefix-supervised collection and signal-only execution."""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..lag_bias_feedback_20261022.learning import adapt_arrived
from ..paired_feedback_20261022.model import initial_encoder
from .policy import make_context, choose_action, ACTIONS


def check_signal(signal):
    if (set(signal) != {'values', 'available'} or signal['values'].ndim != 2
            or signal['values'].shape != signal['available'].shape
            or signal['values'].shape[1] != 23 or not len(signal['values'])
            or signal['available'].dtype != bool):
        raise ValueError('nonempty signal-only Nx23 record required')


def block_logits(model, signal, start, stop, state, *, device):
    check_signal(signal)
    values = []
    for lo in range(start, stop, 32):
        windows = [simple_data.window(signal, i, 128) for i in range(lo, min(lo + 32, stop))]
        x, m = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
        h = model.predict_fast(x, m, state, mode='risk_state')['risk_state']
        values.append(model.base.risk_head(h).squeeze(-1).detach())
    return torch.cat(values)


def context_at(model, signal, count, state, *, device):
    check_signal(signal)
    if type(count) is not int or count < 256 or count % 128:
        raise ValueError('scheduled arrived context required')
    x, m, target, tm, origins = forecast_examples(signal, count, model.cfg, device)
    order = np.argsort(origins).tolist()
    x, m, target, tm = [v[order] for v in (x, m, target, tm)]
    tm = tm & torch.isfinite(target)
    truth = torch.asinh((torch.where(tm, target, model.base.center) - model.base.center) / model.base.scale)
    prediction = model.predict_fast(x, m, state)['forecast']
    xx, mm = simple_data.window(signal, count - 1, 128)
    xx = torch.as_tensor(xx, device=device); mm = torch.as_tensor(mm, device=device)
    mm = mm & torch.isfinite(xx)
    level = torch.asinh((torch.where(mm, xx, model.base.center) - model.base.center) / model.base.scale)
    z = block_logits(model, signal, count - 1, count, state, device=device)[0]
    return make_context(prediction, truth, tm, level, mm, float(z))


def eligible_query(outcome, start, stop):
    """Historical preparation/evaluation only; never called by predict_policy."""
    valid = outcome['valid'][start:stop].copy()
    if outcome['is_disrupt']:
        lead = np.round((outcome['down'] - outcome['times'][start:stop]) * 1000., 9)
        valid &= lead > 30.
    return valid


def peak_loss(logits, selected, disruptive):
    logits = np.asarray(logits, dtype=np.float64); selected = np.asarray(selected)
    if (logits.ndim != 1 or selected.shape != logits.shape or selected.dtype != bool
            or not selected.any() or not np.isfinite(logits).all()):
        raise ValueError('nonempty finite eligible query required')
    peak = logits[selected].max()
    return float(np.logaddexp(0., -peak if disruptive else peak))


def counterfactual(model, signal, count, state, selected, disruptive, *, device,
                   candidate_order=('recent', 'lag_bias')):
    """Independent finite steps; no candidate may alter its common parent."""
    if set(candidate_order) != {'recent', 'lag_bias'} or len(candidate_order) != 2:
        raise ValueError('exact two distinct candidates required')
    states = {'hold': state}
    for action in candidate_order:
        states[action], _ = adapt_arrived(model, signal, count, state, device=device,
                                         correct=action == 'lag_bias')
    losses = {}; peaks = {}
    for action in ACTIONS:
        z = block_logits(model, signal, count, min(count + 128, len(signal['values'])),
                         states[action], device=device).cpu().numpy()
        losses[action] = peak_loss(z, selected, disruptive)
        peaks[action] = int(np.flatnonzero(selected)[z[selected].argmax()] + count)
    return dict(losses=losses, peak_indices=peaks,
                utility=[losses[a] - losses['hold'] for a in ACTIONS[1:]]), states


def collect_shot(model, signal, outcome, sid, *, device):
    """Called only for a sealed ended prefix shot, on mapped coordinates."""
    rows = []; excluded = []
    for trajectory in ('static', 'recent', 'lag_bias'):
        state = initial_encoder(model)
        for count in range(256, len(signal['values']), 128):
            selected = eligible_query(outcome, count, min(count + 128, len(signal['values'])))
            if not selected.any():
                excluded.append(dict(shot=sid, trajectory=trajectory, count=count,
                                     reason='no_future_timely_or_valid_point'))
                # No future valid point remains under the terminal-time mask;
                # no label is synthesized and no zero-utility row is fitted.
                continue
            context = context_at(model, signal, count, state, device=device)
            target, candidates = counterfactual(model, signal, count, state, selected,
                                                outcome['is_disrupt'], device=device)
            rows.append(dict(shot=sid, is_disrupt=bool(outcome['is_disrupt']),
                             trajectory=trajectory, count=count, context=context.tolist(),
                             eligible_query_points=int(selected.sum()), **target))
            state = candidates['hold' if trajectory == 'static' else trajectory]
    return rows, excluded


def predict_policy(model, signal, policy, *, device='cuda'):
    check_signal(signal)
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('fixed W128/H25 parent required')
    states = {k: initial_encoder(model) for k in ('contextual', 'mean_context')}
    predictions = {k: [] for k in states}; records = {k: [] for k in states}
    n = len(signal['values'])
    for lo in range(0, n, 128):
        hi = min(lo + 128, n)
        for name, state in states.items():
            z = block_logits(model, signal, lo, hi, state, device=device)
            predictions[name].append(z.sigmoid().cpu().numpy())
        if hi < 256 or hi == n:
            continue
        for name, state in states.items():
            x = context_at(model, signal, hi, state, device=device)
            action, utilities = choose_action(policy, x, mean_context=name == 'mean_context')
            receipt = dict(last_observation=hi - 1, first_affected_prediction=hi,
                           gradient_steps=0, parameter_delta_norm=0.)
            if action != 'hold':
                states[name], receipt = adapt_arrived(model, signal, hi, state,
                    device=device, correct=action == 'lag_bias')
            scaled = (x - policy['mean']) / policy['scale']
            records[name].append(dict(receipt, action=action, predicted_utility=utilities,
                context=x.tolist(), scaled_context_abs_max=float(abs(scaled).max()),
                context_components_over5sd=int((abs(scaled) > 5).sum())))
    result = {k: np.concatenate(v) for k, v in predictions.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite risk trace')
    return result, records
