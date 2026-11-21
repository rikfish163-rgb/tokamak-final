"""Learn the geometry of arrived-state feedback, never an extra risk score.

Only a source-trained 5x5 loss matrix changes offline. Its Frobenius norm is
fixed at sqrt(5), so training cannot win by scaling the entire auxiliary loss
to zero. Identity is exactly the previously tested event-state loss. The
source model, direct H25 head, and state teacher remain immutable.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .event_dynamics import EventDynamicsFfnModel, arrived_windows, forecast_windows, HORIZONS
from .fast_update import normalized_step


def metric_loss(prediction, target, metric):
    if (prediction.ndim != 2 or prediction.shape[-1] != 5 or target.shape != prediction.shape
            or metric.shape != (5, 5) or not all(torch.isfinite(t).all() for t in (prediction, target, metric))):
        raise ValueError('finite aligned five-dimensional states and 5x5 metric required')
    transformed = (prediction - target) @ metric.T
    return F.smooth_l1_loss(transformed, torch.zeros_like(transformed))


class EventMetricModel(EventDynamicsFfnModel):
    intra_objective_name = 'source-learned five-state forecast error geometry; frozen during EAST'

    def __init__(self, base, directions, feature_mean, feature_scale):
        super().__init__(base, directions, feature_mean, feature_scale, mode='event')
        self.requires_grad_(False)
        self.metric = nn.Parameter(torch.eye(5, device=self.center.device, dtype=self.center.dtype))

    @torch.no_grad()
    def project_metric(self):
        norm = self.metric.norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('nonzero finite metric required; no identity substitution')
        self.metric.mul_(math.sqrt(5.) / norm)

    def prepare_matured_forecast_loss(self, issued, truth, available, *, values, observed):
        values, observed = values.detach().clone(), observed.detach().clone()
        truth, available = truth.detach().clone(), available.detach().clone()
        actual, masks, common = arrived_windows(values, observed, truth, available, self.center)
        target = self.coordinates(actual, masks).detach()
        if not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite arrived state target')
        receipt = dict(mode='event_metric', horizons=list(HORIZONS), state_dimensions=5,
            valid_common_targets=int(common.sum()), target_detached=True, outcome_access=False,
            predicted_signals_used_for_risk=False, hard_event_pseudo_labels=False,
            teacher='immutable source; zero temporary and persistent FFN',
            loss_geometry='fixed source-trained 5x5 transform; Frobenius norm sqrt(5)')

        def loss(prediction, _truth, _available):
            if not common.any():
                # No observed intersection supplies no supervision, just as
                # in the original state objective. No fabricated target.
                return prediction.sum() * 0.
            virtual, vm = forecast_windows(values, observed, prediction, common, self.center, self.scale)
            return metric_loss(self.coordinates(virtual, vm), target, self.metric)

        return loss, receipt


def source_step(model, episode, fast, *, create_graph):
    """Same numerical step/acceptance as runtime; source query is NOT input."""
    output = model(episode['x'], episode['a'], fast)
    objective, _ = model.prepare_matured_forecast_loss(output['forecast'].detach(),
        episode['truth'], episode['mask'], values=episode['x'], observed=episode['a'])
    before = objective(output['forecast'], episode['truth'], episode['mask'])
    gradient, = torch.autograd.grad(before, fast, create_graph=create_graph)
    candidate = normalized_step(fast, torch.zeros_like(fast), gradient, step=.05, radius=1.)
    after = objective(model(episode['x'], episode['a'], candidate)['forecast'],
                      episode['truth'], episode['mask'])
    if not torch.isfinite(before) or not torch.isfinite(after) or not torch.isfinite(candidate).all():
        raise FloatingPointError('invalid state metric inner step')
    tolerance = 8 * torch.finfo(before.dtype).eps * max(abs(float(before.detach())),
                                                       abs(float(after.detach())), 1e-12)
    accepted = float(before.detach() - after.detach()) > tolerance
    current = candidate if accepted else fast
    return current, dict(accepted=accepted, loss_before=float(before.detach()),
        loss_after=float(after.detach()), maturity_offset=25, earliest_query_offset=26), (after if accepted else before)
