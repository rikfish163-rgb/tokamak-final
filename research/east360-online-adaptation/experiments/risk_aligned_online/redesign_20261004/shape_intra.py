"""Bounded, signal-only H25 residual-shape hypothesis; no predictor copy."""
from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F


def _device_matches(requested, actual):
    # An unindexed CUDA request means the current device, not a mismatched
    # tensor. Comparing device metadata does not initialize or occupy CUDA.
    return (requested.type == actual.type and
            (requested.type == 'cpu' or requested.index is None or requested.index == actual.index))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _shape(forecast, target, available):
    counts = available.sum(1, keepdim=True)
    valid = available & (counts >= 2)
    residual = (forecast - target).masked_fill(~valid, 0.)
    mean = residual.sum(1, keepdim=True) / counts.clamp_min(1)
    centered = (residual - mean).masked_fill(~valid, 0.)
    loss = F.smooth_l1_loss(centered, torch.zeros_like(centered), reduction='none')
    loss = loss.sum() / valid.sum().clamp_min(1)
    components = dict(shape=float(loss.detach()), level=None,
                      future_first_difference=None, origin_first_interval=None,
                      valid_future_pairs=0, valid_origin_intervals=0,
                      valid_shape_times=int(valid.sum()),
                      valid_shape_channels=int((counts >= 2).sum()),
                      residual_time_mean=mean.detach().reshape(-1).cpu().tolist())
    return loss, components


def attempt(model, pending, previous, signals, times, fast, step, radius, device):
    """Try one new-gradient step after two nonoverlapping matured blocks.

    Caller seals through new origin+25 before calling, supplies the previous
    fully matured pending tuple, and owns pair/fast resets at shot boundaries.
    ``pair_reset`` requests clearing invalid evidence; ``needs_previous`` is
    a valid warmup block. Neither input pending tuple nor caller fast is changed.
    """
    # Lazy imports allow intra._attempt to dispatch here without an import cycle.
    from .intra import _finite, _norm, _vector
    from ..fast_update import normalized_step

    source_record = pending[0] if isinstance(pending, tuple) and len(pending) == 4 else {}
    record = dict(source_record) if isinstance(source_record, dict) else {}
    safe_fast = isinstance(fast, torch.Tensor)
    vector = _vector(fast) if safe_fast else None
    norm = _norm(fast) if safe_fast else None
    record.update(objective='shape', accepted=False, reason='invalid_evidence',
                  effective_index=None, effective_time=None,
                  predictions_sealed_through_index=None,
                  valid_targets=0, valid_innovations=0, valid_origin_intervals=0,
                  loss_before=None, loss_after=None, issued_loss=None,
                  loss_components_before=None, loss_components_after=None,
                  loss_gain=None, gradient=None, grad_norm=None, gradient_finite=None,
                  fast_before=vector, proposed_fast=None, fast_after=vector,
                  fast_change_norm=0., candidate_change_norm=0.,
                  fast_norm_before=norm, fast_norm_after=norm,
                  pair_origins=[], shape_components={}, pair_losses_before=None,
                  pair_losses_after=None, pair_loss_gains=None, pair_tolerances=None,
                  pair_gradients=None, pair_grad_norms=None, cos=None,
                  pair_reset=False, source_asinh_units=True,
                  loss_units='fixed source-asinh residual; masked time-mean removed',
                  fast_update_rule='source fast_update.normalized_step',
                  fast_update_norm_epsilon=1e-12)

    def reject(reason, invalid=True):
        record.update(reason=reason, pair_reset=invalid)
        return fast, _json_safe(record)

    if (not isinstance(step, (int, float, np.number)) or isinstance(step, bool)
            or not isinstance(radius, (int, float, np.number)) or isinstance(radius, bool)
            or step != .05 or radius != 1.):
        return reject('invalid_configuration')
    if (not safe_fast or fast.shape != (1, 128) or not fast.is_floating_point()
            or not _finite(fast)):
        return reject('invalid_fast')
    if (model.center.shape != (23,) or model.scale.shape != (23,)
            or not _finite(model.center) or not _finite(model.scale)
            or not bool((model.scale > 0).all())
            or fast.dtype != model.center.dtype or fast.device != model.center.device
            or not _device_matches(torch.device(device), fast.device)):
        return reject('invalid_source_normalization')

    def block(item):
        if not isinstance(item, tuple) or len(item) != 4 or not isinstance(item[0], dict):
            return None, 'invalid_pending'
        rec, values, observed, issued = item
        origin, maturity = rec.get('origin_index'), rec.get('maturity_index')
        if (not isinstance(origin, Integral) or isinstance(origin, bool) or origin < 0
                or not isinstance(maturity, Integral) or isinstance(maturity, bool)
                or maturity != origin + 25):
            return None, 'invalid_maturity'
        if maturity >= len(times):
            return None, 'not_fully_matured'
        if (not isinstance(values, torch.Tensor) or values.shape != (1, 128, 23)
                or not isinstance(observed, torch.Tensor) or observed.shape != values.shape
                or observed.dtype != torch.bool or values.dtype != fast.dtype
                or values.device != fast.device or observed.device != fast.device
                or not isinstance(issued, torch.Tensor) or issued.shape != (1, 25, 23)
                or issued.dtype != fast.dtype or issued.device != fast.device):
            return None, 'invalid_pending_shape'
        if not _finite(issued):
            return None, 'nonfinite_forecast'
        if not bool(torch.isfinite(values)[observed].all()):
            return None, 'nonfinite_observed_input'
        return (int(origin), int(maturity), values, observed, issued), None

    try:
        new, error = block(pending)
        if error:
            return reject(error)
        old, error = block(previous) if previous is not None else (None, None)
        if error:
            return reject(error)
        origin, maturity = new[:2]
        record['predictions_sealed_through_index'] = maturity
        record['pair_origins'] = [old[0], origin] if old is not None else [origin]
        if old is not None and old[1] >= origin:
            return reject('overlapping_blocks')
        prefix_times = np.asarray(times[:maturity + 1])
        if (prefix_times.shape != (maturity + 1,) or not np.isfinite(prefix_times).all()
                or bool(np.any(np.diff(prefix_times) <= 0))):
            return reject('invalid_times')
        for item, parsed in ((pending, new), (previous, old)):
            if parsed is not None:
                o, m = parsed[:2]
                if (item[0].get('origin_time') != float(times[o])
                        or item[0].get('maturity_time') != float(times[m])):
                    return reject('mismatched_record_times')
        raw, masks = signals['values'], signals['available']
        if (raw.ndim != 2 or raw.shape[1] != 23 or masks.shape != raw.shape
                or masks.dtype != np.dtype(bool) or len(raw) <= maturity):
            return reject('invalid_signals')
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return reject('invalid_evidence')

    from .intra import _output

    with torch.inference_mode(False), torch.enable_grad():
        current = fast.detach().clone().requires_grad_(True)
        blocks = [('old', old), ('new', new)] if old is not None else [('new', new)]
        evidence, losses, gradients, norms = [], [], [], []
        for name, (o, m, values, observed, issued) in blocks:
            # Exactly this block's own arrived truth. No origin, intervening
            # gaps, future diagnostics, outcome fields or risk targets enter.
            truth = torch.as_tensor(raw[o + 1:o + 26], dtype=fast.dtype,
                                    device=device)[None].detach()
            available = torch.as_tensor(masks[o + 1:o + 26], device=device)[None]
            available = available & torch.isfinite(truth)
            if truth.shape != (1, 25, 23):
                return reject('not_fully_matured')
            truth = torch.where(available, truth, model.center).detach()
            target = torch.asinh((truth - model.center.detach()) / model.scale.detach()).detach()
            if not _finite(target):
                return reject('nonfinite_target')
            if name == 'new':
                record['valid_targets'] = int(available.sum())
            if not bool((available.sum(1) >= 2).any()):
                return reject('no_shape_evidence')
            issued_loss, _ = _shape(issued.detach(), target, available)
            if not _finite(issued_loss):
                return reject('nonfinite_loss')
            if name == 'new':
                record['issued_loss'] = float(issued_loss.detach())
            # Saved issued forecasts/fasts are audit only, never gradient inputs.
            try:
                out = _output(model, values, observed, current)
            except (ValueError, KeyError, TypeError, AttributeError):
                return reject('invalid_model_output')
            if not _finite(out['logit']):
                return reject('nonfinite_logit')  # Numerical integrity, not risk gating.
            if not _finite(out['forecast']):
                return reject('nonfinite_forecast')
            loss, components = _shape(out['forecast'], target, available)
            record['shape_components'][name] = components
            if name == 'new':
                record.update(loss_before=float(loss.detach()), loss_components_before=components)
            if not _finite(loss):
                return reject('nonfinite_loss')
            if not loss.requires_grad:
                return reject('no_fast_gradient')
            gradient, = torch.autograd.grad(loss, current, allow_unused=True)
            if gradient is None:
                return reject('no_fast_gradient')
            finite, length = _finite(gradient), _norm(gradient)
            if name == 'new':
                record.update(gradient_finite=finite, gradient=_vector(gradient) if finite else None,
                              grad_norm=length)
            if not finite or not np.isfinite(length):
                return reject('nonfinite_gradient')
            if length == 0.:
                return reject('zero_gradient')
            evidence.append((values, observed, target, available))
            losses.append(loss)
            gradients.append(gradient.detach())
            norms.append(length)
        if old is None:
            return reject('needs_previous', invalid=False)
        record.update(pair_losses_before=[float(loss.detach()) for loss in losses],
                      pair_gradients=[_vector(g) for g in gradients], pair_grad_norms=norms)
        cos = float(((gradients[0].double() / norms[0]) *
                     (gradients[1].double() / norms[1])).sum())
        record['cos'] = cos
        if not np.isfinite(cos):
            return reject('nonfinite_cosine')
        if cos <= 0.:
            return reject('nonpositive_cosine', invalid=False)
        proposed = normalized_step(current.detach(), torch.zeros_like(current), gradients[1],
                                   step=step, radius=radius).detach().requires_grad_(True)
        if not _finite(proposed) or not np.isfinite(_norm(proposed)):
            return reject('nonfinite_fast')
        record.update(proposed_fast=_vector(proposed),
                      candidate_change_norm=_norm(proposed - current.detach()))
        if record['candidate_change_norm'] == 0.:
            return reject('zero_step')
        after_losses = []
        for name, (values, observed, target, available) in zip(('old', 'new'), evidence):
            try:
                out = _output(model, values, observed, proposed)
            except (ValueError, KeyError, TypeError, AttributeError):
                return reject('invalid_model_output')
            if not _finite(out['logit']):
                return reject('nonfinite_candidate_logit')
            if not _finite(out['forecast']):
                return reject('nonfinite_candidate_forecast')
            loss, components = _shape(out['forecast'], target, available)
            record['shape_components'][name + '_after'] = components
            if not _finite(loss):
                return reject('nonfinite_candidate_loss')
            after_losses.append(float(loss.detach()))
            if name == 'new':
                record.update(loss_after=after_losses[-1], loss_components_after=components)
        gains = [before - after for before, after in zip(record['pair_losses_before'], after_losses)]
        tolerances = [8 * torch.finfo(loss.dtype).eps * max(abs(before), abs(after), 1e-12)
                      for loss, before, after in zip(losses, record['pair_losses_before'], after_losses)]
        record.update(pair_losses_after=after_losses, pair_loss_gains=gains,
                      pair_tolerances=tolerances, loss_gain=gains[1])
        if not all(gain > tolerance for gain, tolerance in zip(gains, tolerances)):
            return reject('loss_not_improved', invalid=False)
        accepted = proposed.detach().clone().requires_grad_(True)
        effective = maturity + 1
        record.update(accepted=True, reason='loss_improved', fast_after=_vector(accepted),
                      fast_change_norm=record['candidate_change_norm'], fast_norm_after=_norm(accepted),
                      effective_index=effective if effective < len(times) else None,
                      effective_time=float(times[effective]) if effective < len(times) else None)
        return accepted, _json_safe(record)
