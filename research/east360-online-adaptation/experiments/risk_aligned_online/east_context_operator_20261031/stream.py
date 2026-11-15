"""Signal-only, shot-local replay of the source and learned context rules.

The caller owns persistent learning and supplies two evaluation-mode models
whose frozen state is identical. This module only reads the current shot; it
neither changes model parameters nor carries context across calls.
"""

import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import (
    PATCH, endpoint_corrections, patch_signal,
)


ARMS = ('cold', 'source_rule', 'learned_rule', 'reverse_feedback',
        'negative_shift', 'zero_feedback')

_OPERATOR = frozenset(('memory.key.weight', 'memory.key.bias',
                       'memory.kernel', 'memory.rate'))


def _check_models(reference, learned):
    """Require the common frozen path assumed when reusing its first block."""
    if (type(reference) is not type(learned)
            or type(reference.memory) is not type(learned.memory)
            or reference.cfg != learned.cfg
            or reference.cfg.window != 128 or reference.cfg.patch != PATCH
            or reference.readout != 'full' or learned.readout != 'full'
            or reference.feedback != 'progressive' or learned.feedback != 'progressive'):
        raise ValueError('matching 128-window progressive full-readout adapters required')
    if any(module.training for model in (reference, learned) for module in model.modules()):
        raise ValueError('both adapters must be in evaluation mode')
    left, right = reference.state_dict(), learned.state_dict()
    if left.keys() != right.keys():
        raise ValueError('matching frozen adapter state required')
    for name, value in left.items():
        if name in _OPERATOR:
            continue
        other = right[name]
        if (value.shape != other.shape or value.dtype != other.dtype
                or value.device != other.device or not torch.equal(value, other)):
            raise ValueError(f'frozen adapter state differs: {name}')


def _check_trace(trace, patches):
    if (trace['delta'].shape != (patches, 92) or trace['norm'].shape != (patches,)
            or trace['zero_direction'].shape != (patches,)
            or trace['zero_direction'].dtype != torch.bool
            or trace['forecast_loss'].ndim != 0):
        raise ValueError('invalid context trace shape')
    for key in ('delta', 'norm', 'forecast_loss'):
        if not torch.isfinite(trace[key]).all():
            raise FloatingPointError(f'nonfinite context trace: {key}')


def _receipt(trace, *, trace_source, delta_sign):
    # Forecast loss is a full-shot diagnostic for this trace, never an input
    # to risk. For cold it comes from learned off; for negative_shift it is
    # from the unnegated learned trace, as identified by trace_source.
    return dict(updates=int(trace['updates']), norm=trace['norm'].cpu().tolist(),
                forecast_loss=float(trace['forecast_loss']),
                zero_direction=trace['zero_direction'].cpu().tolist(),
                first_affected_index=16, reset_each_shot=True,
                trace_source=trace_source, delta_sign=delta_sign)


@torch.no_grad()
def predict(reference, learned, signal):
    """Return six per-sample risk-probability curves and trace receipts.

    Only ``values`` and boolean ``available`` from this shot are accepted.
    The declared first affected index (16) is an earliest possibility for
    progressive feedback, not a promise that any nonzero effect occurs.
    Receipts describe the completed shot: ``updates`` and ``forecast_loss``
    are full-shot diagnostics, while ``norm`` and ``zero_direction`` are per
    completed patch. They must not be fed back into earlier predictions.
    ``trace_source`` identifies the underlying diagnostic trace, including
    learned-off for cold and unnegated learned-memory for negative_shift.
    """
    _check_models(reference, learned)
    means, complete = patch_signal(signal, reference.base.center, reference.base.scale)
    source = reference.memory.trace(means, complete, mode='memory')
    _check_trace(source, len(means))
    main = learned.memory.trace(means, complete, mode='memory')
    _check_trace(main, len(means))
    reverse = learned.memory.trace(means, complete, mode='reverse',
                                   reference_norm=main['norm'])
    _check_trace(reverse, len(means))
    off = learned.memory.trace(means, complete, mode='off')
    _check_trace(off, len(means))
    if torch.count_nonzero(off['delta']) or torch.count_nonzero(off['norm']) or off['updates']:
        raise ValueError('off trace must have zero correction, norm and updates')

    traces = dict(cold=off, source_rule=source, learned_rule=main,
                  reverse_feedback=reverse, negative_shift=main, zero_feedback=off)
    info = {
        arm: _receipt(trace, trace_source=source_name, delta_sign=sign)
        for arm, trace, source_name, sign in (
            ('cold', off, 'zero_feedback', 0),
            ('source_rule', source, 'source_rule', 1),
            ('learned_rule', main, 'learned_rule', 1),
            ('reverse_feedback', reverse, 'reverse_feedback', 1),
            ('negative_shift', main, 'learned_rule', -1),
            ('zero_feedback', off, 'zero_feedback', 0),
        )
    }
    curves = {arm: [] for arm in ARMS}
    count = len(signal['values'])
    for lo in range(0, count, 32):
        ends = range(lo, min(lo + 32, count))
        x, available = batch(signal, ends, reference.base.center.device)
        h, valid = reference.tokens(x, available)
        indices = torch.arange(lo, min(lo + 32, count), device=h.device)
        for arm in ARMS:
            delta = endpoint_corrections(traces[arm]['delta'], indices)
            if arm == 'negative_shift':
                delta = -delta
            model = reference if arm in ('cold', 'source_rule') else learned
            logits = model.risk(h, valid, delta)
            if logits.shape != (len(indices),) or not torch.isfinite(logits).all():
                raise FloatingPointError('one finite risk logit per endpoint required')
            curves[arm].append(logits.sigmoid().cpu().numpy())
    return {arm: np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            for arm, parts in curves.items()}, info
