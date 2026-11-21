"""Arrived-signal feedback in the source task's internal coordinates.

The frozen task readout compares two trajectories at the SAME endpoint: the
directly forecast trajectory and the now-observed trajectory. It supplies a
task metric, not a disruption label or a guarantee of correct risk ordering.
"""
import torch
from torch.nn import functional as F

from ..fast_update import normalized_step
from ..ffn_intra_20261005.loss import weighted_signal_loss
from ..partial_intra_20261005.feedback import segment_components


def _valid(truth, observed, segment):
    length = {'early': 5, 'late': 25}.get(segment)
    if (length is None or truth.ndim != 3 or truth.shape[1:] != (length, 23)
            or truth.shape != observed.shape or observed.dtype != torch.bool):
        raise ValueError('early requires arrived h1..5; late requires arrived h1..25')
    valid = observed & torch.isfinite(truth)
    return valid, valid if segment == 'early' else valid[:, 5:]


def _coordinates(model, values, valid):
    clean = torch.where(valid, values.detach(), model.center)
    return torch.asinh((clean - model.center) / model.scale).detach()


def feedback_loss(model, prediction, x, available, truth, observed, segment, *, use_task):
    valid, new_valid = _valid(truth, observed, segment)
    if x.shape != (len(truth), 128, 23) or available.shape != x.shape or available.dtype != torch.bool:
        raise ValueError('origin must be an observed 128x23 window with a boolean mask')
    block, mask = (truth, valid) if segment == 'early' else (truth[:, 4:], valid[:, 4:])
    parts = segment_components(prediction, block, mask, model.center, model.scale, segment)
    signal = weighted_signal_loss(parts).mean()
    task = signal * 0.
    if use_task and new_valid.any():
        origin_mask = available & torch.isfinite(x)
        origin_z = _coordinates(model, x, origin_mask)
        arrived_z = _coordinates(model, truth, valid)

        def error(h):
            future_mask = valid[:, :h]
            full_mask = torch.cat((origin_mask[:, h:], future_mask), dim=1)
            forecast = torch.where(future_mask, prediction[:, :h], 0.)
            predicted = torch.cat((origin_z[:, h:], forecast), dim=1)
            actual = torch.cat((origin_z[:, h:], arrived_z[:, :h]), dim=1)
            predicted_risk = model.risk_from_source_coordinates(predicted, full_mask)
            actual_risk = model.risk_from_source_coordinates(actual, full_mask).detach()
            return predicted_risk - actual_risk

        e5 = error(5)
        active = new_valid.flatten(1).any(1)
        if segment == 'early':
            terms = F.smooth_l1_loss(e5, torch.zeros_like(e5), reduction='none')
        else:
            e25 = error(25)
            early_active = valid[:, :5].flatten(1).any(1)
            endpoint = F.smooth_l1_loss(e25, torch.zeros_like(e25), reduction='none')
            change = F.smooth_l1_loss(e25 - e5, torch.zeros_like(e25), reduction='none')
            terms = .5 * (endpoint + change.masked_fill(~early_active, 0.))
        task = terms.masked_fill(~active, 0.).sum() / active.sum().clamp_min(1)
    loss = signal + task
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite signal/task feedback')
    return loss, dict(signal_loss=float(signal.detach()), task_loss=float(task.detach()),
                      level_count=parts['level_count'], difference_count=parts['difference_count'])


def segment_step(model, delta, x, available, truth, observed, segment, *, use_task, create_graph=False):
    _, new_valid = _valid(truth, observed, segment)
    row = dict(segment=segment, state_changed=False, reason='no_valid_targets',
               level_count=int(new_valid.sum()), loss=None, gradient_norm=None)
    if not new_valid.any():
        return delta, row
    prediction = model(x, available, delta)['forecast']
    loss, details = feedback_loss(model, prediction, x, available, truth, observed, segment,
                                  use_task=use_task)
    gradient, = torch.autograd.grad(loss, delta, create_graph=create_graph)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('nonfinite internal-state gradient')
    step = .02 if segment == 'early' else .08
    candidate = normalized_step(delta[None], torch.zeros_like(delta)[None], gradient[None],
                                step=step, radius=1.)[0]
    if not torch.isfinite(candidate).all():
        raise FloatingPointError('nonfinite internal-state update')
    row.update(details, reason='finite_segment_update', loss=float(loss.detach()), step=step,
               gradient_norm=float(gradient.detach().double().norm()),
               state_changed=not torch.equal(candidate.detach(), delta.detach()))
    return candidate, row
