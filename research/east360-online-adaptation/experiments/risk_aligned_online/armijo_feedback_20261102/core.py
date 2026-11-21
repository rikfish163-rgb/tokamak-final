"""Matured-data Armijo step with explicit finite online computation budget."""

import torch

from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..risk_bridge_forecaster_20261102.core import auxiliary


def select_step(current, gradient, direction, before, evaluate):
    if (
        current.shape != gradient.shape
        or current.shape != direction.shape
        or before.numel() != 1
    ):
        raise ValueError(
            "matching state, gradient, direction and scalar objective required"
        )
    if not all(torch.isfinite(x).all() for x in (current, gradient, direction, before)):
        raise FloatingPointError("nonfinite line-search input")
    old = float(before.detach())
    slope = float((gradient.detach().double() * direction.detach().double()).sum())
    receipt = dict(
        accepted=False,
        alpha=0.0,
        evaluations=0,
        loss_before=old,
        loss_after=old,
        armijo_bound=None,
        trials=[],
    )
    if not gradient.detach().count_nonzero():
        if direction.detach().count_nonzero():
            raise ValueError(
                "zero gradient must not create a nonzero candidate direction"
            )
        return current, dict(receipt, reason="stationary")
    if not direction.detach().count_nonzero():
        return current, dict(receipt, reason="no_effective_direction")
    if slope >= 0:
        raise ValueError("a strict descent direction is required")
    for index in range(11):
        alpha = 2.0 ** (-index)
        # Construct outside no_grad so the chosen candidate retains the whole
        # meta-training path; the discrete decision only inspects scalar values.
        proposal = current + alpha * direction
        with torch.no_grad():
            loss = evaluate(proposal)
        if loss.numel() != 1 or not torch.isfinite(loss).all():
            raise FloatingPointError("nonfinite or nonscalar proposed objective")
        value = float(loss)
        bound = old + 1e-4 * alpha * slope
        receipt["trials"].append(dict(alpha=alpha, loss=value, armijo_bound=bound))
        receipt["evaluations"] += 1
        if value < old and value <= bound:
            return proposal, dict(
                receipt,
                accepted=True,
                reason="armijo",
                alpha=alpha,
                loss_after=value,
                armijo_bound=bound,
            )
    # This is the defined termination of a bounded online solver, not proof of
    # stationarity. Keep the current graph/state and expose the exhausted budget.
    return current, dict(receipt, reason="budget_exhausted")


def propose(model, pack, state, truth, mask, coefficient, *, create_graph=False):
    if mask.shape != truth.shape or mask.dtype != torch.bool:
        raise ValueError("matching matured diagnostic target and mask required")
    current = state.clone() if create_graph else state.detach().clone()
    if not mask.any():
        return current, dict(
            accepted=False,
            reason="no_target",
            gradient_norm=0.0,
            step_norm=0.0,
            alpha=0.0,
            evaluations=0,
            trials=[],
        )

    def evaluate(candidate):
        return (
            innovation_loss(
                model, auxiliary(model, pack, candidate, coefficient), truth, mask
            )
            + 0.01 * candidate.square().mean()
        )

    with torch.enable_grad():
        variable = current.requires_grad_(True)
        before = evaluate(variable)
        (gradient,) = torch.autograd.grad(before, variable, create_graph=create_graph)
        direction = metric_direction(pack, gradient)
        after, receipt = select_step(variable, gradient, direction, before, evaluate)
    if not create_graph:
        after = after.detach()
    return after, dict(
        receipt,
        gradient_norm=float(gradient.detach().double().norm()),
        step_norm=float((after.detach() - current.detach()).double().norm()),
        gradient256=gradient.detach().flatten().cpu().tolist(),
        state256_after=after.detach().flatten().cpu().tolist(),
    )
