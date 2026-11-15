"""Bounded, per-shot, signal-only Q/K transport with matured H25 feedback."""

import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from .gradients import observed_gradients
from .learning import bounded_step, predict_rule


NAMES = ('cold', 'operational', 'point', 'reverse', 'shuffled', 'constant',
         'risk_axis', 'signal_axis', 'identical')
FIRST_ORIGIN = 127
HORIZON = 25
STATE_SIZE = 8192
MAX_CHUNK = 32
RADIUS = .05


def make_states(signal_gradient, reverse_gradient, risk_gradient, observed, rules):
    """Make replacements, never accumulated states, from one matured query."""
    gradients = (signal_gradient, reverse_gradient, risk_gradient)
    if (any(not isinstance(g, torch.Tensor) or g.shape != (1, STATE_SIZE)
            or not g.is_floating_point() or not torch.isfinite(g).all() for g in gradients)
            or any(g.device != signal_gradient.device or g.dtype != signal_gradient.dtype
                   for g in gradients)
            or not isinstance(observed, int) or isinstance(observed, bool)
            or not 0 <= observed <= HORIZON * 22
            or not isinstance(rules, dict) or set(rules) != {'operational', 'point', 'shuffled'}):
        raise ValueError('three finite same-device Q/K gradients, observation count and rules required')
    g, reverse, risk = (value[0].detach() for value in gradients)
    states = g.new_zeros((len(NAMES), STATE_SIZE))
    gnorm, risknorm = g.norm(), risk.norm()
    if observed and gnorm > 0:
        mapped = {
            'operational': predict_rule(g, rules['operational']),
            'point': predict_rule(g, rules['point']),
            'reverse': predict_rule(reverse, rules['operational']),
            'shuffled': predict_rule(g, rules['shuffled']),
            'constant': rules['operational']['mean'].to(device=g.device, dtype=g.dtype),
        }
        for name, direction in mapped.items():
            states[NAMES.index(name)] = bounded_step(direction, RADIUS)
        operational = states[NAMES.index('operational')]
        norm = operational.norm()
        projection = torch.dot(operational, risk)
        safe_risknorm = torch.where(risknorm > 0, risknorm, torch.ones_like(risknorm))
        states[NAMES.index('risk_axis')] = norm * torch.sign(projection) * risk / safe_risknorm
        states[NAMES.index('signal_axis')] = -norm * g / gnorm
    norms = states.norm(dim=1)
    op = states[NAMES.index('operational')]
    axes = torch.stack((g, risk, op))
    denominator = norms[:, None] * axes.norm(dim=1)[None, :]
    cosines = torch.where(denominator > 0, (states @ axes.T) /
                          denominator.clamp_min(torch.finfo(g.dtype).tiny), 0.)
    stats = torch.cat((norms, cosines.flatten(),
                       torch.stack((gnorm, reverse.norm(), risknorm)))).detach().cpu().numpy()
    norm_values = stats[:len(NAMES)]
    if not np.isfinite(stats).all() or (norm_values > RADIUS + 1e-6).any():
        raise FloatingPointError('Q/K state exceeded its fixed radius')
    cosine_values = stats[len(NAMES):-3].reshape(len(NAMES), 3)
    details = dict(
        observed=observed,
        state_norm={name: float(value) for name, value in zip(NAMES, norm_values)},
        cosine_signal={name: float(row[0]) for name, row in zip(NAMES, cosine_values)},
        cosine_risk={name: float(row[1]) for name, row in zip(NAMES, cosine_values)},
        cosine_operational={name: float(row[2]) for name, row in zip(NAMES, cosine_values)},
        signal_gradient_norm=float(stats[-3]), reverse_gradient_norm=float(stats[-2]),
        risk_gradient_norm=float(stats[-1]), max_bound=float(norm_values.max()))
    return states, details


def _prefix(signal, end):
    return {name: value[:end + 1] for name, value in signal.items()}


def _forward(net, signal, start, stop, states, slow):
    """Same 9 x 32 padded forward geometry as the sealed arrival stream."""
    x, mask = batch(_prefix(signal, stop - 1), range(start, stop), net.base.center.device)
    count = len(x)
    if count < MAX_CHUNK:
        x = torch.cat((x, x[-1:].expand(MAX_CHUNK - count, -1, -1)))
        mask = torch.cat((mask, mask[-1:].expand(MAX_CHUNK - count, -1, -1)))
    expanded = states[:, None, :].expand(-1, MAX_CHUNK, -1).reshape(-1, STATE_SIZE)
    with torch.no_grad():
        output = net(x.repeat(len(NAMES), 1, 1), mask.repeat(len(NAMES), 1, 1),
                     expanded, slow, 'qk', forecast=False)
        logits = output['logit']
        if logits.shape != (len(NAMES) * MAX_CHUNK,) or not torch.isfinite(logits).all():
            raise FloatingPointError('invalid actual neural risk logits')
        rows = logits.reshape(len(NAMES), MAX_CHUNK)
        probabilities = torch.stack([row.sigmoid() for row in rows])
        if not torch.equal(probabilities[0], probabilities[NAMES.index('identical')]):
            raise AssertionError('identical and cold predictions differ')
    return probabilities[:, :count].cpu().numpy()


def predict(net, signal, slow, rules):
    """Score one shot; a new call always starts from zero temporary Q/K."""
    if not isinstance(signal, dict) or set(signal) != {'values', 'available'}:
        raise ValueError('signal-only values and available fields required')
    values, available = np.asarray(signal['values']), np.asarray(signal['available'])
    if (values.ndim != 2 or values.shape[1] != 23 or not len(values)
            or values.dtype.kind not in 'fiu' or available.shape != values.shape
            or available.dtype != np.bool_):
        raise ValueError('nonempty N-by-23 values and boolean availability required')
    signal = dict(values=values, available=available)
    states = net.base.center.new_zeros((len(NAMES), STATE_SIZE))
    scores, receipts, active = None, [], None
    cursor, event = 0, FIRST_ORIGIN
    while cursor < len(values):
        stop = min(cursor + MAX_CHUNK, event + 1, len(values))
        probabilities = _forward(net, signal, cursor, stop, states, slow)
        if scores is None:
            scores = np.empty((len(NAMES), len(values)), dtype=probabilities.dtype)
        scores[:, cursor:stop] = probabilities
        if active is not None:
            active['apply_count'] += stop - cursor
        cursor = stop
        if cursor == event + 1:
            if event >= FIRST_ORIGIN + HORIZON:
                origin = event - HORIZON
                g, reverse, risk, info = observed_gradients(net, _prefix(signal, event), origin, slow)
                states, details = make_states(g, reverse, risk, info['observed'], rules)
                active = dict(origin=origin, maturity=event, first_affected=event + 1,
                              apply_count=0, **{**info, **details})
                receipts.append(active)
            event += HORIZON
    return {name: scores[i] for i, name in enumerate(NAMES)}, receipts
