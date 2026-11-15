"""Signal-only four-candidate replay with explicit matured feedback timing."""

import numpy as np
import torch
from torch.nn import functional as F

from ..operational_transport_20261030.stream import _forward
from .learning import representation_step
from .observation import observation
from .scaling import transform_context
from .teaching import context_controls

VARIANTS = ('raw_fixed', 'scaled_fixed', 'raw_calibrated', 'scaled_calibrated')
ARMS = ('cold', 'real', 'reverse', 'mean_feedback', 'constant', 'risk_axis', 'identical')
NAMES = ('cold', 'identical', *(v+'__'+a for v in VARIANTS for a in ARMS[1:6]))


def new_states(net, prefix, origin, slow, policies):
    if set(policies) != set(VARIANTS) or len(prefix['values']) != origin+26:
        raise ValueError('four frozen policies and exactly matured prefix required')
    observed = observation(net, prefix, origin)
    x, mask = observed['x'], observed['available']
    source_state = x.new_zeros(1, 8192, requires_grad=True)
    risk = net(x, mask, source_state, slow, 'qk', forecast=False)['logit']
    risk_gradient, = torch.autograd.grad(risk.sum(), source_state)
    states, norms, cosines = {}, {'cold': 0., 'identical': 0.}, {'cold': None, 'identical': None}
    for variant in VARIANTS:
        policy = policies[variant]
        context = observed['context']
        if variant.startswith('scaled'):
            context = transform_context(context.cpu(), policy['scale']).to(x.device)
        controls = context_controls(context, policy['mean_context'])
        contexts = torch.cat([controls[a] for a in ARMS[1:5]])
        delta = policy['controller'](contexts, observed['support'].expand(4))
        zero = x.new_zeros(4, 8192, requires_grad=True)
        h, _, _ = net.first(x.expand(4, -1, -1), mask.expand(4, -1, -1), zero, 'qk')
        step = representation_step(h[:, -1], h[:, -1].detach(), delta, zero,
                                   radius=policy['radius']).detach()
        real_norm = step[0].norm()
        for i in range(1, 4):
            magnitude = step[i].norm()
            if magnitude > 0:
                step[i] *= real_norm/magnitude
        axis = (step[0]*risk_gradient[0]).sum().sign()*risk_gradient[0]
        axis *= real_norm/risk_gradient.norm().clamp_min(1e-30)
        matrix = x.new_zeros(9, 8192)
        matrix[1:5], matrix[5] = step, axis
        states[variant] = matrix
        values = matrix[1:6].norm(dim=1)
        if not torch.isfinite(matrix).all() or values.max() > policy['radius']+1e-6:
            raise FloatingPointError('temporary action exceeds frozen radius')
        cosine = F.cosine_similarity(matrix[1:6], matrix[1:2], dim=1)
        for i, arm in enumerate(ARMS[1:6]):
            name = variant+'__'+arm
            norms[name] = float(values[i])
            # Zero directions are undefined, not orthogonal evidence.
            cosines[name] = float(cosine[i]) if values[i] > 0 and real_norm > 0 else None
    return states, dict(state_norm=norms, cos_real=cosines, max_bound=max(norms.values()),
                        feedback_supported=bool(observed['support'].item()))


def predict(net, signal, slow, policies):
    if not isinstance(signal, dict) or set(signal) != {'values', 'available'}:
        raise ValueError('only observed values and availability may enter prediction')
    values, available = np.asarray(signal['values']), np.asarray(signal['available'])
    if (values.ndim != 2 or values.shape[1] != 23 or not len(values)
            or available.shape != values.shape or available.dtype != np.bool_):
        raise ValueError('nonempty N-by-23 signal and boolean mask required')
    states = {v: net.base.center.new_zeros(9, 8192) for v in VARIANTS}
    curves = {name: np.empty(len(values), np.float32) for name in NAMES}
    cursor, event, active, receipts = 0, 127, None, []
    while cursor < len(values):
        stop = min(cursor+32, event+1, len(values))
        baseline = None
        for variant in VARIANTS:
            scores = _forward(net, signal, cursor, stop, states[variant], slow)
            if baseline is None:
                baseline = scores[0]
                curves['cold'][cursor:stop] = baseline
                curves['identical'][cursor:stop] = scores[6]
            if not np.array_equal(scores[0], baseline) or not np.array_equal(scores[0], scores[6]):
                raise AssertionError('fixed padded forward baseline differs')
            for i, arm in enumerate(ARMS[1:6], 1):
                curves[variant+'__'+arm][cursor:stop] = scores[i]
        if active is not None:
            active['apply_count'] += stop-cursor
        cursor = stop
        if cursor == event+1:
            if event >= 152:
                prefix = {k: v[:event+1] for k, v in signal.items()}
                states, details = new_states(net, prefix, event-25, slow, policies)
                # Unsupported feedback deliberately replaces the previous
                # action with zero, as specified in the full-stream protocol.
                active = dict(origin=event-25, maturity=event, first_affected=event+1,
                              apply_count=0, **details)
                receipts.append(active)
            event += 25
    return curves, receipts
