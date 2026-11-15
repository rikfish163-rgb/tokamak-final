"""Causal H25 signal supervision of a disposable internal model delta.

Independent redesign, not V3C. No outcome fields, model-parameter optimizer,
target normalization fit, synthetic future inputs, or risk correction head.
"""
from collections import deque
from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..fast_update import normalized_step


def _finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _norm(tensor):
    # Float64 accumulation avoids overflowing a finite float32 gradient.
    return float(tensor.detach().double().norm())


def _vector(tensor):
    return tensor.detach().reshape(-1).cpu().tolist()


def _fast_receipt(model, tensor):
    summarize = getattr(model, 'summarize_fast', None)
    return _vector(tensor) if summarize is None else summarize(tensor)


def _window(data_api, signals, end, model, device):
    return _batch_windows(data_api,signals,[end],model,device)


def _batch_windows(data_api, signals, endpoints, model, device):
    # Keep the original per-endpoint causal provider boundary, but transfer a
    # complete batch to CUDA once instead of two tiny transfers per endpoint.
    # Array order, padding, dtype and missingness remain unchanged.
    windows=[]
    for end in endpoints:
        prefix={key:value[:end+1] for key,value in signals.items()}
        x,a=data_api.window(prefix,end,model.cfg.window)
        x,a=np.asarray(x),np.asarray(a)
        if x.shape != (128,23) or a.shape != x.shape or a.dtype != bool:
            raise ValueError('window must return [128, 23] values and boolean availability')
        windows.append((x,a))
    if not windows:
        raise ValueError('one or more causal endpoints required')
    values=torch.as_tensor(np.stack([x for x,a in windows]),dtype=model.center.dtype,device=device)
    available=torch.as_tensor(np.stack([a for x,a in windows]),device=device)
    return values, available & torch.isfinite(values)


def _output(model, values, available, fast):
    # All forwards, including both acceptance candidates, have enabled autograd
    # and a gradient-carrying fast tensor. Detaching results is not no_grad.
    out = model(values, available, fast)
    batch = len(values)
    if out['logit'].shape != (batch,) or out['forecast'].shape != (batch, 25, 23):
        raise ValueError('model must return logit [B] and direct forecast [B, 25, 23]')
    return out


def _origin_anchor(model, values, available):
    """Last genuinely observed past value per channel, in source-asinh units."""
    positions = torch.arange(128, device=values.device)[None, :, None]
    last = positions.masked_fill(~available, -1).amax(1)
    seen = last >= 0
    raw = values.gather(1, last.clamp_min(0)[:, None, :]).squeeze(1)
    raw = torch.where(seen, raw, model.center)
    return torch.asinh((raw - model.center) / model.scale).detach(), seen


def _loss(model, forecast, truth, available, objective, anchor, anchor_seen, signal_loss=None):
    # The level term is the source meta-training objective across ALL valid 23.
    # Source-learned INTRA weights do not alter persistent replay's loss.
    if signal_loss is None:
        signal_loss = getattr(model, 'intra_forecast_loss', model.forecast_loss)
    level = signal_loss(forecast, truth, available)
    components = dict(level=float(level.detach()) if _finite(level) else None,
                      future_first_difference=None, origin_first_interval=None,
                      valid_future_pairs=0, valid_origin_intervals=0)
    if objective in ('level','forward_holdout'):
        return level, components
    target = torch.asinh((truth.detach() - model.center) / model.scale)
    # Subtracting the SAME origin from both sides is just level error. The
    # primary innovation objective instead contrasts ADJACENT future points.
    pairs = available[:, 1:] & available[:, :-1]
    increment = F.smooth_l1_loss(torch.diff(forecast, dim=1), torch.diff(target, dim=1),
                                reduction='none')
    increment = increment.masked_fill(~pairs, 0.).sum() / pairs.sum().clamp_min(1)
    first_valid = available[:, 0] & anchor_seen
    first_interval = F.smooth_l1_loss(forecast[:, 0] - anchor, target[:, 0] - anchor,
                                     reduction='none')
    first_interval = first_interval.masked_fill(~first_valid, 0.).sum() / first_valid.sum().clamp_min(1)
    # This small anchor term is algebraically first-point LEVEL error; report
    # it separately and do not pretend it is the new difference objective.
    components.update(future_first_difference=float(increment.detach()) if _finite(increment) else None,
                      origin_first_interval=float(first_interval.detach()) if _finite(first_interval) else None,
                      valid_future_pairs=int(pairs.sum()), valid_origin_intervals=int(first_valid.sum()))
    return increment + .1 * first_interval, components


def _record(origin, times, fast, model=None):
    """Keep the original three-argument vector receipt for diagnostic callers.

    The online runner explicitly supplies its model for compact FFN receipts;
    the existing normalized-step diagnostic uses the full-vector contract.
    """
    maturity = origin + 25
    return dict(origin_index=origin, origin_time=float(times[origin]),
                maturity_index=maturity,
                maturity_time=float(times[maturity]) if maturity < len(times) else None,
                effective_index=None, effective_time=None,
                predictions_sealed_through_index=None, accepted=False, reason='unmatured',
                fast_at_origin=_fast_receipt(model, fast))


def _attempt(model, pending, signals, times, fast, step, radius, objective, device,
             previous=None):
    if objective == 'shape':
        from .shape_intra import attempt
        return attempt(model, pending, previous, signals, times, fast, step, radius, device)
    record, values, observed, issued = pending
    origin, maturity = record['origin_index'], record['maturity_index']
    truth = torch.as_tensor(signals['values'][origin + 1:maturity + 1],
                            dtype=model.center.dtype, device=device)[None].detach()
    available = torch.as_tensor(signals['available'][origin + 1:maturity + 1],
                                device=device)[None] & torch.isfinite(truth)
    truth = torch.where(available, truth, model.center).detach()
    anchor, anchor_seen = _origin_anchor(model, values, observed)
    record.update(predictions_sealed_through_index=maturity, valid_targets=int(available.sum()),
                  valid_innovations=0, valid_origin_intervals=0,
                  loss_before=None, loss_after=None, issued_loss=None,
                  loss_components_before=None, loss_components_after=None,
                  loss_gain=None, gradient=None, grad_norm=None, gradient_finite=None,
                  fast_before=_fast_receipt(model, fast), proposed_fast=None,
                  fast_after=_fast_receipt(model, fast),
                  fast_change_norm=0., candidate_change_norm=0., fast_norm_before=_norm(fast),
                  fast_norm_after=_norm(fast))

    def reject(reason):
        record['reason'] = reason
        return fast, record

    if not record['valid_targets']:
        return reject('no_valid_targets')
    fit_available=available
    holdout_available=None
    if objective=='forward_holdout':
        # Fixed chronological half split, not a split selected by risk labels.
        # All 25 points have arrived before either fitting or validation. Only
        # the first 12 enter the gradient; the last 13 accept/reject that step.
        # Adjacent observations remain correlated: this is a signal-transfer
        # check, NOT statistically independent validation or a risk guarantee.
        fit_available=available.clone();fit_available[:,12:]=False
        holdout_available=available.clone();holdout_available[:,:12]=False
        record.update(training_targets=int(fit_available.sum()),
                      holdout_targets=int(holdout_available.sum()),
                      holdout_before=None,holdout_after=None,holdout_gain=None)
        if not record['training_targets']:return reject('no_training_targets')
        if not record['holdout_targets']:return reject('no_holdout_targets')
    if not _finite(issued):
        return reject('nonfinite_forecast')
    # A contextual loss must use ONE detached context for both gradient and
    # candidate acceptance. Recomputing weights from the candidate forecast
    # changes the objective being compared. Archived models lack this hook and
    # retain exactly their original loss and numerical path.
    prepare_loss = getattr(model, 'prepare_intra_forecast_loss', None)
    prepare_matured = getattr(model, 'prepare_matured_forecast_loss', None)
    if prepare_matured is not None:
        if objective != 'level':
            raise ValueError('matured state feedback is a separate full-H25 objective')
        prepared_loss, receipt = prepare_matured(issued, truth, fit_available,
                                                values=values, observed=observed)
        record['matured_objective'] = receipt
    else:
        prepared_loss = None if prepare_loss is None else prepare_loss(issued, truth, fit_available)
    issued_loss, _ = _loss(model, issued, truth, fit_available, objective, anchor, anchor_seen,
                           signal_loss=prepared_loss)
    if not _finite(issued_loss):
        return reject('nonfinite_loss')
    record['issued_loss'] = float(issued_loss.detach())
    current = fast.detach().clone().requires_grad_(True)
    before = _output(model, values, observed, current)
    if not _finite(before['logit']):
        return reject('nonfinite_logit')
    if not _finite(before['forecast']):
        return reject('nonfinite_forecast')
    loss, components = _loss(model, before['forecast'], truth, fit_available, objective, anchor, anchor_seen,
                             signal_loss=prepared_loss)
    record['loss_components_before'] = components
    record['valid_innovations'] = components['valid_future_pairs']
    record['valid_origin_intervals'] = components['valid_origin_intervals']
    if not _finite(loss):
        return reject('nonfinite_loss')
    record['loss_before'] = float(loss.detach())
    if holdout_available is not None:
        signal_loss=getattr(model,'intra_forecast_loss',model.forecast_loss)
        holdout_before=signal_loss(before['forecast'].detach(),truth,holdout_available)
        if not _finite(holdout_before):return reject('nonfinite_holdout_loss')
        record['holdout_before']=float(holdout_before)
    if not loss.requires_grad:
        return reject('no_fast_gradient')
    gradient, = torch.autograd.grad(loss, current, allow_unused=True)
    if gradient is None:
        return reject('no_fast_gradient')
    record['gradient_finite'] = _finite(gradient)
    if not record['gradient_finite']:
        return reject('nonfinite_gradient')
    record['gradient'] = _fast_receipt(model, gradient)
    norm = _norm(gradient)
    if not np.isfinite(norm):
        return reject('nonfinite_gradient')
    record['grad_norm'] = norm
    if norm == 0.:
        return reject('zero_gradient')
    # Preserve the source meta-training's smooth normalization AND projection,
    # including its 1e-12 denominators in the original tensor dtype.
    transform = getattr(model, 'intra_gradient_transform', None)
    contextual_projection = getattr(model, 'project_intra_gradient', None)
    if contextual_projection is not None:
        if transform is not None:
            raise ValueError('one declared gradient transformation, not stacked transformations')
        # The target window is fully mature. No outcome or later signal is
        # passed. Models without this optional hook retain their exact path.
        used_gradient, projection_receipt = contextual_projection(gradient.detach(),
            values=values, observed=observed, truth=truth, available=available,
            fast=current.detach(), step=step, radius=radius)
        record['gradient_projection'] = projection_receipt
    else:
        used_gradient = gradient.detach() if transform is None else transform(gradient.detach())
    if used_gradient.shape != gradient.shape or not _finite(used_gradient):
        raise ValueError('finite matching transformed fast gradient required')
    if transform is not None or contextual_projection is not None:
        record['transformed_gradient'] = _fast_receipt(model, used_gradient)
        record['transformed_gradient_norm'] = _norm(used_gradient)
    proposed = normalized_step(current.detach(), torch.zeros_like(current), used_gradient,
                               step=step, radius=radius)
    offset_norm = _norm(proposed)
    if not np.isfinite(offset_norm) or not _finite(proposed):
        return reject('nonfinite_fast')
    proposed = proposed.detach().requires_grad_(True)
    record['proposed_fast'] = _fast_receipt(model, proposed)
    record['candidate_change_norm'] = _norm(proposed - current.detach())
    if record['candidate_change_norm'] == 0.:
        return reject('zero_step')
    after = _output(model, values, observed, proposed)
    if not _finite(after['logit']):
        return reject('nonfinite_candidate_logit')
    if not _finite(after['forecast']):
        return reject('nonfinite_candidate_forecast')
    candidate_loss, after_components = _loss(model, after['forecast'], truth, fit_available, objective,
                                             anchor, anchor_seen, signal_loss=prepared_loss)
    record['loss_components_after'] = after_components
    if not _finite(candidate_loss):
        return reject('nonfinite_candidate_loss')
    record['loss_after'] = float(candidate_loss.detach())
    gain = record['loss_before'] - record['loss_after']
    record['loss_gain'] = gain
    # Do not accept roundoff-sized changes as useful measured loss improvement.
    tolerance = 8 * torch.finfo(loss.dtype).eps * max(abs(record['loss_before']),
                                                    abs(record['loss_after']), 1e-12)
    if gain <= tolerance:
        return reject('loss_not_improved')
    if holdout_available is not None:
        signal_loss=getattr(model,'intra_forecast_loss',model.forecast_loss)
        holdout_after=signal_loss(after['forecast'].detach(),truth,holdout_available)
        if not _finite(holdout_after):return reject('nonfinite_candidate_holdout_loss')
        record['holdout_after']=float(holdout_after)
        held_gain=record['holdout_before']-record['holdout_after']
        held_tolerance=8*torch.finfo(loss.dtype).eps*max(
            abs(record['holdout_before']),abs(record['holdout_after']),1e-12)
        record.update(holdout_gain=held_gain,holdout_tolerance=held_tolerance)
        if held_gain<=held_tolerance:return reject('holdout_not_improved')
    accepted = proposed.detach().clone().requires_grad_(True)
    effective = maturity + 1
    record.update(accepted=True, reason='loss_improved', fast_after=_fast_receipt(model, accepted),
                  fast_change_norm=record['candidate_change_norm'], fast_norm_after=_norm(accepted),
                  effective_index=effective if effective < len(times) else None,
                  effective_time=float(times[effective]) if effective < len(times) else None)
    return accepted, record


def _state_changes(model, parameters, buffers):
    changes = []
    for current, previous in ((dict(model.named_parameters()), parameters),
                              (dict(model.named_buffers()), buffers)):
        changed = [name for name, old in previous.items() if not torch.equal(current[name].detach(), old)]
        norm = sum(_norm(current[name].detach() - previous[name]) ** 2 for name in changed) ** .5
        changes.append((norm, changed))
    return changes


def predict_shot(model, shot, intra, device='cpu', stride=100, step=.05,
                 radius=1., objective='level', data_api=simple_data):
    """Public EAST-only replay API; offline source work cannot enter this lane."""
    return _predict_shot(model, shot, intra, device=device, stride=stride,
                         step=step, radius=radius, objective=objective, data_api=data_api)


def predict_source_diagnostic(model, shot, *, intra, device='cpu', data_api=simple_data):
    """Offline SOURCE whole-shot validation at fixed stride100/.05/radius1.

    Not a J-TEXT online deployment branch. Uses the same causal signal-only
    numerical core with an explicit source identity, never a relabelled EAST
    record or an exception bypass.
    """
    return _predict_shot(model, shot, intra, device=device, data_api=data_api,
                         offline_source=True)


def _predict_shot(model, shot, intra, device='cpu', stride=100, step=.05,
                  radius=1., objective='level', data_api=simple_data, offline_source=False):
    """Return all original endpoint probabilities and a JSON-safe causal audit.

    Origins are 0, stride, 2*stride, ...; each direct H25 target is supervised
    once, after sealing the maturity endpoint's prediction. Only indices AFTER
    maturity may see an accepted fast delta. Registered model tensors stay
    unchanged, and the temporary delta is discarded at return. Durable
    whole-shot sealing and later outcome reveal belong to the caller.
    Explicit offline_source is for SOURCE validation of the adaptation rule,
    not a J-TEXT deployment branch; normal online callers remain EAST-only.
    """
    if (not isinstance(stride, Integral) or isinstance(stride, bool) or stride < 1
            or not np.isfinite(step) or step <= 0 or not np.isfinite(radius) or radius <= 0):
        raise ValueError('positive finite step/radius and integer stride required')
    if objective not in ('level', 'innovation', 'shape','forward_holdout'):
        raise ValueError('objective must be level, innovation, shape or forward_holdout')
    if objective == 'shape' and (stride != 100 or step != .05 or radius != 1.):
        raise ValueError('shape hypothesis requires stride100, step.05 and radius1')
    if type(offline_source) is not bool:
        raise ValueError('offline_source must be a boolean')
    if offline_source and shot.get('device') != 'J-TEXT':
        raise ValueError('offline source diagnostic requires explicit J-TEXT identity')
    if not offline_source and shot.get('device', 'EAST') != 'EAST':
        raise ValueError('online INTRA supports EAST only; source data are offline only')
    if model.cfg.window != 128 or model.cfg.horizon != 25:
        raise ValueError('the fixed contract requires W128 and direct H25')
    if getattr(model.cfg, 'fast_site', 'tokens') != 'tokens':
        raise ValueError('the matched adaptation site is the original source tokens')
    dimension=getattr(model,'temporary_dim',128)
    if type(dimension) is not int or dimension<1:
        raise ValueError('positive declared temporary dimension required')
    if (model.center.shape != (23,) or model.scale.shape != (23,)
            or not _finite(model.center) or not _finite(model.scale) or not bool((model.scale > 0).all())):
        raise ValueError('finite positive source normalization for 23 channels required')
    # Whitelist reads, not keys: test/runner dictionaries may contain outcomes.
    signals = dict(values=np.asarray(shot['values']), available=np.asarray(shot['available']))
    times = np.asarray(shot['times'])
    if signals['values'].ndim != 2 or signals['values'].shape[1] != 23:
        raise ValueError('flat23 diagnostic signals required')
    size = len(signals['values'])
    if signals['available'].shape != signals['values'].shape or signals['available'].dtype != bool:
        raise ValueError('matching boolean flat23 availability required')
    if times.shape != (size,) or not np.isfinite(times).all() or bool(np.any(np.diff(times) <= 0)):
        raise ValueError('finite strictly increasing original endpoint times required')
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    scores = np.empty(size, np.float32)
    audit = dict(intra=bool(intra), objective=objective, stride=int(stride), step=float(step),
                 radius=float(radius), horizon=25, score_count=size, attempts=[], pending=[],
                 attempted_updates=0, accepted_updates=0, fast_discarded=True,
                 fast_update_rule='source fast_update.normalized_step', fast_update_norm_epsilon=1e-12,
                 loss_weights=dict(level=1. if objective in ('level','forward_holdout') else 0.,
                                   future_first_difference=1. if objective == 'innovation' else 0.,
                                   origin_first_interval=.1 if objective == 'innovation' else 0.),
                 numerical_path='eval; frozen model; autograd enabled; fast requires_grad')
    audit['scope'] = ('OFFLINE_SOURCE_DIAGNOSTIC_NOT_JTEXT_ONLINE' if offline_source
                      else 'EAST_CAUSAL_REPLAY')
    audit.update(fast_parameter_count=dimension,
                 fast_site=getattr(model,'adaptation_site','source tokens before Transformer'))
    if hasattr(model, 'fast_gradient_transform'):
        audit['fast_gradient_transform'] = model.fast_gradient_transform
    if hasattr(model, 'intra_objective_name'):
        audit['intra_objective_name'] = model.intra_objective_name
    if hasattr(model,'signal_weights'):
        audit['channel_signal_weights']=_vector(model.signal_weights)
        audit['channel_weights_used_for_persistent_loss']=False
    if objective == 'shape':
        audit['loss_weights']['masked_residual_shape'] = 1.
    if objective=='forward_holdout':
        audit.update(forecast_fit_steps=[1,12],forecast_holdout_steps=[13,25],
                     holdout_used_for_gradient=False,holdout_is_independent_risk_validation=False)
    try:
        model.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        # Also work inside caller no_grad/inference_mode without inheriting a
        # non-gradient numerical path. Do not move or mutate the shared model.
        with torch.inference_mode(False), torch.enable_grad():
            initial = model.initial_fast(1).detach().clone().to(device)
            if initial.shape != (1, dimension) or not _finite(initial) or bool(initial.any()):
                raise ValueError('initial_fast must return the declared finite zero temporary dimension')
            fast = initial.requires_grad_(True)
            pending = deque()
            previous_matured = None  # Same-shot signal-only evidence; never persistent.
            start, next_origin = 0, 0
            while start < size:
                stop = min(size, start + 256)
                if intra and next_origin < size:
                    stop = min(stop, next_origin + 1)
                if pending:
                    stop = min(stop, pending[0][0]['maturity_index'] + 1)
                values,observed=_batch_windows(data_api,signals,range(start,stop),model,device)
                out = _output(model, values, observed, fast.expand(stop - start, -1))
                if not _finite(out['logit']):
                    raise FloatingPointError(f'nonfinite risk logit in endpoints {start}:{stop}')
                scores[start:stop] = out['logit'].detach().sigmoid().cpu().numpy()
                end = stop - 1  # Scores up to and including this index are sealed.
                if intra and end == next_origin:
                    pending.append((_record(end, times, fast, model), values[-1:].detach().clone(),
                                    observed[-1:].detach().clone(), out['forecast'][-1:].detach().clone()))
                    next_origin += int(stride)
                if pending and pending[0][0]['maturity_index'] == end:
                    matured = pending.popleft()
                    fast, record = _attempt(model, matured, signals, times, fast,
                                            step, radius, objective, device,
                                            previous=previous_matured)
                    if objective == 'shape':
                        previous_matured = None if record['pair_reset'] else matured
                    audit['attempts'].append(record)
                start = stop
            audit['pending'] = [item[0] for item in pending]
        audit['attempted_updates'] = len(audit['attempts'])
        audit['accepted_updates'] = sum(record['accepted'] for record in audit['attempts'])
        (parameter_norm, changed_parameters), (buffer_norm, changed_buffers) = _state_changes(
            model, parameters, buffers)
        audit.update(model_parameter_change_norm=parameter_norm, model_buffer_change_norm=buffer_norm)
        if changed_parameters or changed_buffers:
            raise RuntimeError(f'model state changed during INTRA: {changed_parameters + changed_buffers}')
        return scores, audit
    finally:
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        # Restore heterogeneous child modes without recursive train() overrides.
        for module, training in modes:
            module.training = training
