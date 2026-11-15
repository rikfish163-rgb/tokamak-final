"""Same bounded32 internal gates, forecast horizon and optimizer; centered loss."""

import math

import torch

from ..dual_history_ttt_20261101.model import latent_forward, project_gates
from ..gate_loss_audit_20261101.core import centered_loss


def loss(model, prediction, raw_truth, mask):
    clean = torch.where(mask, raw_truth.detach(), model.base.center)
    target = torch.asinh((clean - model.base.center) / model.base.scale)
    return centered_loss(prediction, target, mask)


def step(model, pack, beta, truth, mask, *, step_size=0.1, anchor=0.01):
    if (
        not math.isfinite(step_size)
        or step_size < 0
        or not math.isfinite(anchor)
        or anchor < 0
    ):
        raise ValueError("finite nonnegative step and anchor required")
    if (
        truth.shape != (len(pack["h"]), 25, 23)
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matching full25x23 target and availability required")
    if not torch.isfinite(truth[mask]).all():
        raise ValueError("nonfinite observed target")
    current = beta.detach().clone()
    observed = int(mask.sum())
    if not observed:
        return current, dict(
            loss_before=None,
            loss_proposal=None,
            loss_retained=None,
            gradient_norm=0.0,
            step_norm=0.0,
            observed=0,
            accepted=False,
            reason="no_target",
            gradient_computed=False,
            gradient32=[0.0] * 32,
            beta32_after=current.flatten().cpu().tolist(),
        )
    with torch.enable_grad():
        variable = current.detach().requires_grad_(True)
        predicted = latent_forward(model, pack, variable, forecast=True)["forecast"]
        objective = (
            loss(model, predicted, truth, mask) + anchor * variable.square().mean()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite centered objective")
        gradient = torch.autograd.grad(objective, variable)[0].detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite centered gate gradient")
    magnitude = gradient.double().norm()
    direction = (
        (gradient.double() / magnitude).to(current.dtype)
        if magnitude
        else torch.zeros_like(current)
    )
    with torch.no_grad():
        proposal = project_gates(current - step_size * direction)
        predicted = latent_forward(model, pack, proposal, forecast=True)["forecast"]
        proposed_objective = (
            loss(model, predicted, truth, mask) + anchor * proposal.square().mean()
        )
        if not torch.isfinite(proposed_objective):
            raise FloatingPointError("nonfinite centered proposed objective")
        accepted = bool(proposed_objective < objective.detach())
        after = proposal if accepted else current
        retained = proposed_objective if accepted else objective.detach()
    return after.detach(), dict(
        loss_before=float(objective.detach()),
        loss_proposal=float(proposed_objective),
        loss_retained=float(retained),
        gradient_norm=float(magnitude),
        step_norm=float((after - current).double().norm()),
        observed=observed,
        accepted=accepted,
        reason="improved" if accepted else "not_improved",
        gradient_computed=True,
        gradient32=gradient.flatten().cpu().tolist(),
        beta32_after=after.flatten().cpu().tolist(),
    )
