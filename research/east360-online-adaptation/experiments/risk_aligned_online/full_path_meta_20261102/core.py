"""Training-only extension: preserve earlier state dependence when requested.

This duplicates the small fixed metric step explicitly rather than changing
the immutable innovation_metric trial's imported code while it is running.
Numeric equality with that deployed step is an experiment prerequisite.
"""

import torch

from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..risk_bridge_forecaster_20261102.core import auxiliary


def advance(model, pack, state, truth, mask, coefficient, *, create_graph=False):
    if mask.shape != truth.shape or mask.dtype != torch.bool:
        raise ValueError("matching matured diagnostic target and mask required")
    current = state.clone() if create_graph else state.detach().clone()
    if not mask.any():
        return current, dict(
            accepted=False, reason="no_target", gradient_norm=0.0, step_norm=0.0
        )
    with torch.enable_grad():
        variable = current.requires_grad_(True)
        prediction = auxiliary(model, pack, variable, coefficient)
        objective = (
            innovation_loss(model, prediction, truth, mask)
            + 0.01 * variable.square().mean()
        )
        (gradient,) = torch.autograd.grad(
            objective, variable, create_graph=create_graph
        )
        if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
            raise FloatingPointError("nonfinite complete-path innovation gradient")
        delta = metric_direction(pack, gradient)
        proposal = variable + delta
    with torch.no_grad():
        predicted = auxiliary(model, pack, proposal, coefficient)
        proposed_loss = (
            innovation_loss(model, predicted, truth, mask)
            + 0.01 * proposal.square().mean()
        )
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite complete-path proposal")
        accepted = bool(proposed_loss < objective.detach())
    after = proposal if accepted else variable
    if not create_graph:
        after = after.detach()
    return after, dict(
        accepted=accepted,
        reason="improved" if accepted else "not_improved",
        loss_before=float(objective.detach()),
        loss_proposal=float(proposed_loss),
        gradient_norm=float(gradient.detach().double().norm()),
        step_norm=float((after.detach() - current.detach()).double().norm()),
        gradient256=gradient.detach().flatten().cpu().tolist(),
        state256_after=after.detach().flatten().cpu().tolist(),
    )
