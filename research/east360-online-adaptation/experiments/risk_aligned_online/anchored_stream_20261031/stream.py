"""Signal-only full-shot replay of frozen anchored targets at two gradient clocks."""

from collections.abc import Mapping

import numpy as np
import torch

from ..anchored_feedback_20261030.run import feedback_controls, state_for
from ..context_vector_20261030.observation import observation
from ..context_vector_20261030.scaling import transform_context
from ..delayed_recurrent_memory_20261025.learning import batch
from ..operational_transport_20261030.stream import _forward


VARIANTS = ('origin', 'mature')
ARMS = ('real', 'reverse', 'mean_feedback', 'donor')
NAMES = ('cold', 'identical', *(variant + '__' + arm
                                for variant in VARIANTS for arm in ARMS))
FIRST_ORIGIN = 127
HORIZON = 25
STATE_SIZE = 8192
MAX_CHUNK = 32
_POLICY_FIELDS = {'controller', 'mean_context', 'scale', 'donor_origins',
                  'donor_feedback', 'radius'}


def _validate_policies(policies):
    if not isinstance(policies, dict) or set(policies) != set(VARIANTS):
        raise ValueError('exactly origin and mature frozen policies required')
    for policy in policies.values():
        if (not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS
                or isinstance(policy['radius'], bool) or policy['radius'] != .1
                or not isinstance(policy['scale'], Mapping)):
            raise ValueError('anchored policy fields and fixed radius .1 required')
        mean, origins, feedback = (policy['mean_context'], policy['donor_origins'],
                                   policy['donor_feedback'])
        if (not isinstance(mean, torch.Tensor) or mean.shape != (358,)
                or not isinstance(origins, torch.Tensor) or origins.ndim != 2
                or origins.shape[1] != 128 or len(origins) == 0
                or not isinstance(feedback, torch.Tensor)
                or feedback.shape != (len(origins), 230)):
            raise ValueError('mean 358 and nonempty donor origins Nx128/feedback Nx230 required')
        if any(not v.is_floating_point() or v.device != mean.device or v.dtype != mean.dtype
               or not bool(torch.isfinite(v).all()) for v in (mean, origins, feedback)):
            raise ValueError('policy tensors must be finite and share floating dtype and device')


def new_states(net, prefix, origin, slow, policies):
    """Replace both temporary states after exactly 25 samples have arrived.

    Both policies see the same original ordered context. Only the input to the
    representation gradient changes: origin window versus maturity window.
    """
    _validate_policies(policies)
    if (not isinstance(prefix, dict) or set(prefix) != {'values', 'available'}
            or not isinstance(origin, int) or isinstance(origin, bool)
            or origin < FIRST_ORIGIN):
        raise ValueError('signal-only complete matured prefix and origin required')
    values, available = np.asarray(prefix['values']), np.asarray(prefix['available'])
    if (values.shape != (origin + HORIZON + 1, 23) or values.dtype.kind not in 'fiu'
            or available.shape != values.shape or available.dtype != np.bool_):
        raise ValueError('matured prefix requires N-by-23 values and boolean availability')
    # The origin and maturity windows together touch only these 153 samples.
    start = origin - FIRST_ORIGIN
    if not np.isfinite(values[start:][available[start:]]).all():
        raise ValueError('available observations must be finite')
    prefix = dict(values=values, available=available)
    observed = observation(net, prefix, origin)
    x, mask, support = observed['x'], observed['available'], observed['support']
    mature_x, mature_mask = batch(prefix, [origin + HORIZON], x.device)
    inputs = {'origin': (x, mask), 'mature': (mature_x, mature_mask)}
    states, donor_indices = {}, {}
    norms = {name: 0. for name in NAMES}
    cosines = {name: None for name in NAMES}

    for variant in VARIANTS:
        policy = policies[variant]
        context = transform_context(observed['context'].detach().cpu(), policy['scale']).to(x.device)
        mean, origins, feedback = (policy['mean_context'], policy['donor_origins'],
                                   policy['donor_feedback'])
        if context.device != mean.device or context.dtype != mean.dtype:
            raise ValueError('scaled context and policy tensors must share device and dtype')
        # Double precision matches the historical nearest-origin rule; argmin
        # deterministically selects the first row on a tie. No donor feedback,
        # clocks, shot outcomes, or future signal enter this choice.
        distances = (origins.double() - context[:, :128].double()).square().mean(dim=1)
        if not bool(torch.isfinite(distances).all()):
            raise ValueError('donor origin distances must be finite')
        donor_index = int(distances.argmin().item())
        donor_indices[variant] = donor_index
        donor = torch.cat((context[:, :128], feedback[donor_index:donor_index + 1]), dim=1)
        controls = feedback_controls(context, mean, donor)
        gradient_x, gradient_mask = inputs[variant]
        data = dict(x=gradient_x, mask=gradient_mask, support=support)
        steps = {}
        with torch.enable_grad():
            for arm in ARMS:
                steps[arm] = state_for(net, policy['controller'], 'anchored', data,
                                       controls[arm], mean).detach()
        if any(step.shape != (1, STATE_SIZE) or not bool(torch.isfinite(step).all())
               for step in steps.values()):
            raise FloatingPointError('invalid anchored Q/K state')
        if bool(torch.count_nonzero(steps['mean_feedback'])):
            raise AssertionError('mean feedback must produce an exactly zero anchored state')
        if not bool(support.item()) and any(bool(torch.count_nonzero(s)) for s in steps.values()):
            raise AssertionError('unsupported feedback must reset the temporary state')
        real_norm = steps['real'].norm()
        for arm in ('reverse', 'donor'):
            magnitude = steps[arm].norm()
            steps[arm] = torch.where(magnitude > 0,
                                     steps[arm] * (real_norm / magnitude.clamp_min(1e-30)),
                                     torch.zeros_like(steps[arm]))
        matrix = x.new_zeros(9, STATE_SIZE)
        for index, arm in enumerate(ARMS, 1):
            matrix[index] = steps[arm][0]
        if not bool(torch.isfinite(matrix).all()) or bool((matrix.norm(dim=1) > .1 + 1e-6).any()):
            raise FloatingPointError('temporary action exceeds frozen radius')
        states[variant] = matrix
        reference = matrix[1].double()
        reference_norm = reference.norm()
        for index, arm in enumerate(ARMS, 1):
            name = variant + '__' + arm
            row = matrix[index].double()
            row_norm = row.norm()
            norms[name] = float(row_norm)
            if bool(row_norm > 0) and bool(reference_norm > 0):
                cosines[name] = float(torch.dot(row, reference) / (row_norm * reference_norm))

    return states, dict(state_norm=norms, cos_real=cosines, max_bound=max(norms.values()),
                        feedback_supported=bool(support.item()), donor_index=donor_indices)


def predict(net, signal, slow, policies):
    """Score one shot in order; every call starts with zero temporary states."""
    _validate_policies(policies)
    if not isinstance(signal, dict) or set(signal) != {'values', 'available'}:
        raise ValueError('only observed values and availability may enter prediction')
    values, available = np.asarray(signal['values']), np.asarray(signal['available'])
    if (values.ndim != 2 or values.shape[1] != 23 or not len(values)
            or values.dtype.kind not in 'fiu' or available.shape != values.shape
            or available.dtype != np.bool_):
        raise ValueError('nonempty N-by-23 values and boolean availability required')
    signal = dict(values=values, available=available)
    states = {variant: net.base.center.new_zeros(9, STATE_SIZE) for variant in VARIANTS}
    curves = {name: np.empty(len(values), np.float32) for name in NAMES}
    cursor, event, active, receipts = 0, FIRST_ORIGIN, None, []
    while cursor < len(values):
        stop = min(cursor + MAX_CHUNK, event + 1, len(values))
        # Validate only arrivals in this chunk, not as-yet unseen signal.
        if not np.isfinite(values[cursor:stop][available[cursor:stop]]).all():
            raise ValueError('available observations must be finite')
        score_prefix = {key: value[:stop] for key, value in signal.items()}
        baseline = None
        for variant in VARIANTS:
            scores = _forward(net, score_prefix, cursor, stop, states[variant], slow)
            if baseline is None:
                baseline = scores[0]
                curves['cold'][cursor:stop] = baseline
                curves['identical'][cursor:stop] = scores[8]
            if (not np.array_equal(scores[0], baseline)
                    or not np.array_equal(scores[0], scores[8])
                    or not np.array_equal(scores[0], scores[3])):
                raise AssertionError('cold, identical and mean-feedback baselines must match')
            for index, arm in enumerate(ARMS, 1):
                curves[variant + '__' + arm][cursor:stop] = scores[index]
        if active is not None:
            active['apply_count'] += stop - cursor
        cursor = stop
        if cursor == event + 1:
            if event >= FIRST_ORIGIN + HORIZON:
                prefix = {key: value[:event + 1] for key, value in signal.items()}
                states, details = new_states(net, prefix, event - HORIZON, slow, policies)
                active = dict(origin=event - HORIZON, maturity=event,
                              first_affected=event + 1, apply_count=0, **details)
                receipts.append(active)
            event += HORIZON
    return curves, receipts
