"""CPU-only, train-fold context scaling and experimental radius calibration.

These functions do not modify the original model inputs, model, or fitted
statistics. The caller is responsible for supplying only train-fold contexts
to ``fit_context_scale`` and only historical misclassified eligible queries to
``calibrate_radius``. The radius is a preregistered sensitivity-experiment
bound, not a universal physical safety guarantee.
"""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch


_HIDDEN = 128
_BLOCKS = 5
_CHANNELS = 23
_WIDTH = _HIDDEN + _BLOCKS * 2 * _CHANNELS
_MIN_SCALE = 1e-6


def _context(value, *, nonempty):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or not value.is_floating_point() or value.ndim != 2
            or value.shape[1] != _WIDTH or (nonempty and len(value) == 0)):
        raise ValueError('context must be a CPU floating tensor of shape (B,358)')
    if not bool(torch.isfinite(value).all()):
        raise ValueError('context must be finite, including missing means')
    blocks = value[:, _HIDDEN:].reshape(len(value), _BLOCKS, 2 * _CHANNELS)
    fractions = blocks[:, :, _CHANNELS:]
    if bool(((fractions < 0) | (fractions > 1)).any()):
        raise ValueError('availability fractions must be in [0,1]')
    return blocks


def _array(value, name, *, objects=False):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError(f'{name} must be on CPU')
        value = value.detach().tolist()
    try:
        result = np.asarray(value, dtype=object if objects else None)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be one-dimensional') from exc
    if result.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return result


def _ids(value, size):
    ids = _array(value, 'ids', objects=True)
    if len(ids) != size:
        raise ValueError('ids must have one entry per row')
    result = ids.tolist()
    for sid in result:
        if isinstance(sid, (bool, np.bool_)) or not isinstance(sid, (str, Real)):
            raise ValueError('shot ids must be finite real numbers or strings')
        if isinstance(sid, Real):
            try:
                finite = math.isfinite(sid)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValueError('shot ids must be finite')
    return result


def _shot_weights(ids):
    counts = {}
    for sid in ids:
        counts[sid] = counts.get(sid, 0) + 1
    return torch.tensor([1. / (len(counts) * counts[sid]) for sid in ids],
                        dtype=torch.float64)


@torch.no_grad()
def fit_context_scale(context, ids):
    """Fit a detached scale dictionary using only supplied training-fold rows.

    Each shot has equal weight and its windows share that weight equally.
    Hidden coordinates use weighted population mean and standard deviation.
    For each feedback channel, RMS is pooled over all five chronological
    blocks, dividing by their shot/window-weighted *valid-block count*. A
    fraction greater than zero marks a block valid; its magnitude does not
    weight that block's already-masked mean a second time.

    Hidden standard deviations and feedback RMS values <= 1e-6 are replaced
    by 1. This prevents a constant or unobservable coordinate from acquiring
    an arbitrary large multiplier; it does not remove that coordinate.
    ``change_constant`` means near-zero RMS, not zero variance around a
    nonzero mean; ``change_unobservable`` separately identifies no valid data.
    """
    blocks = _context(context, nonempty=True)
    ids = _ids(ids, len(context))
    weights = _shot_weights(ids)

    # Scale before taking moments to avoid squaring large finite values.
    hidden = context[:, :_HIDDEN].to(torch.float64)
    hidden_max = hidden.abs().amax(dim=0)
    hidden_unit = hidden / torch.where(hidden_max > 0, hidden_max, 1.)
    # Center relative to an actual row so exactly constant coordinates stay
    # exactly constant even when floating-point weights do not sum to one.
    unit_center = hidden_unit[0] + ((hidden_unit - hidden_unit[0]) * weights[:, None]).sum(dim=0)
    unit_var = ((hidden_unit - unit_center).square() * weights[:, None]).sum(dim=0)
    center = unit_center * hidden_max
    deviation = unit_var.sqrt() * hidden_max
    h_constant = deviation <= _MIN_SCALE
    hscale = torch.where(h_constant, 1., deviation)

    means = blocks[:, :, :_CHANNELS].to(torch.float64)
    valid = blocks[:, :, _CHANNELS:] > 0
    means = torch.where(valid, means, 0.)
    mean_max = means.abs().amax(dim=(0, 1))
    mean_unit = means / torch.where(mean_max > 0, mean_max, 1.)
    valid_weight = (valid.to(torch.float64) * weights[:, None, None]).sum(dim=(0, 1))
    unit_squares = (mean_unit.square() * weights[:, None, None]).sum(dim=(0, 1))
    rms = mean_max * (unit_squares / torch.where(valid_weight > 0, valid_weight, 1.)).sqrt()
    change_constant = rms <= _MIN_SCALE
    change_scale = torch.where(change_constant, 1., rms)

    result = dict(hcenter=center.to(context.dtype), hscale=hscale.to(context.dtype),
                  h_constant=h_constant, change_scale=change_scale.to(context.dtype),
                  change_constant=change_constant, change_unobservable=valid_weight == 0)
    if any(not bool(torch.isfinite(result[name]).all())
           for name in ('hcenter', 'hscale', 'change_scale')):
        raise ValueError('fitted statistics must be representable and finite')
    return result


def _scale_tensor(stats, name, size, dtype, *, positive=False):
    value = stats.get(name)
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or not value.is_floating_point() or value.shape != (size,)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be a finite CPU floating tensor of shape ({size},)')
    value = value.to(dtype=dtype)
    if not bool(torch.isfinite(value).all()) or (positive and bool((value <= 0).any())):
        raise ValueError(f'{name} must be representable and positive if a scale')
    return value


def transform_context(context, stats):
    """Return a new context in the fitted frame; never change inputs or stats.

    Feedback means are divided by their shared channel RMS. Availability
    fractions are unchanged and means for missing blocks are exactly zero.
    """
    blocks = _context(context, nonempty=False)
    if not isinstance(stats, Mapping):
        raise ValueError('stats must be a fitted scale dictionary')
    center = _scale_tensor(stats, 'hcenter', _HIDDEN, context.dtype)
    hscale = _scale_tensor(stats, 'hscale', _HIDDEN, context.dtype, positive=True)
    change_scale = _scale_tensor(stats, 'change_scale', _CHANNELS, context.dtype, positive=True)
    hidden = (context[:, :_HIDDEN] - center) / hscale
    fractions = blocks[:, :, _CHANNELS:]
    means = torch.where(fractions > 0, blocks[:, :, :_CHANNELS] / change_scale, 0.)
    feedback = torch.cat((means, fractions), dim=2).reshape(len(context), _WIDTH - _HIDDEN)
    result = torch.cat((hidden, feedback), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('transformed context must be finite')
    return result


def _median_positive(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    lower, upper = float(ordered[middle - 1]), float(ordered[middle])
    return lower + (upper - lower) / 2.


def calibrate_radius(distances, ids, eligible, source_qk_norm):
    """Calibrate from historical misclassified eligible queries, by shot.

    ``eligible`` must already combine query eligibility and historical
    misclassification. All distances must be finite real values, including
    unselected rows; selected distances must additionally be positive. First
    take the ordinary median within each represented shot, then the ordinary
    median of those shot medians. Clip to [0.1, max(0.1, 0.1*source_qk_norm)].
    This is a sensitivity-experiment bound, not a physical safety claim.
    """
    if (isinstance(source_qk_norm, (bool, np.bool_))
            or not isinstance(source_qk_norm, Real)
            or not math.isfinite(source_qk_norm) or source_qk_norm < 0):
        raise ValueError('source_qk_norm must be a finite nonnegative real scalar')
    distance_array = _array(distances, 'distances')
    if distance_array.dtype.kind not in 'fiu':
        raise ValueError('distances must be finite real values, not None')
    with np.errstate(over='ignore', invalid='ignore'):
        distance_array = distance_array.astype(np.float64)
    if not np.isfinite(distance_array).all():
        raise ValueError('distances must be finite')
    shot_ids = _ids(ids, len(distance_array))
    mask = _array(eligible, 'eligible')
    if len(mask) != len(distance_array) or (len(mask) and mask.dtype.kind != 'b'):
        raise ValueError('eligible must be a boolean vector with one entry per row')
    mask = mask.astype(bool)
    if np.any(distance_array[mask] <= 0):
        raise ValueError('selected distances must be positive')

    by_shot = {}
    for distance, sid, selected in zip(distance_array, shot_ids, mask):
        if selected:
            by_shot.setdefault(sid, []).append(float(distance))
    if not by_shot:
        return dict(radius=.1, raw_median=None, contributing_shots=0, clipped=False,
                    status='no_historical_misclassified_query')
    raw = _median_positive([_median_positive(values) for values in by_shot.values()])
    upper = max(.1, .1 * float(source_qk_norm))
    radius = min(max(raw, .1), upper)
    return dict(radius=radius, raw_median=raw, contributing_shots=len(by_shot),
                clipped=radius != raw, status='ok')
