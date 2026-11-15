"""No outcome access: only matured diagnostic error changes shot-local W."""

import torch

from ..centered_gate_stream_20261101.model import loss as innovation_loss
from ..risk_bridge_forecaster_20261102.core import auxiliary


def metric_direction(pack, gradient):
    """Damped pullback of the uncapped first-FFN hidden RMS quadratic form.

    Damping makes this a well-defined trust metric, not an exact invariant
    natural gradient. The per-step bound is evaluated on this observed pack.
    The unchanged forward path separately caps the total shift at each query.
    """
    phi, valid = pack["phi"], pack["valid"][:, :16]
    if (
        gradient.shape != (4, 64)
        or phi.shape != (*valid.shape, 4)
        or valid.dtype != torch.bool
    ):
        raise ValueError("matching 4x64 gradient, Bx16x4 phi and token mask required")
    if not torch.isfinite(gradient).all() or not torch.isfinite(phi[valid]).all():
        raise FloatingPointError("nonfinite observed metric input")
    if not valid.any():
        return gradient * 0
    clean = phi.double().masked_fill(~valid[:, :, None], 0)
    count = valid.sum().clamp_min(1) * 64
    flat = clean.reshape(-1, 4)
    covariance = flat.T @ flat / count
    ridge = 0.01 * covariance.trace() / 4 + 1e-8
    metric = covariance + ridge * torch.eye(4, dtype=clean.dtype, device=clean.device)
    delta = -torch.linalg.solve(metric, gradient.double())
    squared_rms = (clean @ delta).square().sum() / count
    # Clamp before sqrt: its unused derivative at exactly zero would otherwise
    # produce 0*infinity during second-order meta training.
    delta = delta * (0.05 / squared_rms.clamp_min(0.05**2).sqrt())
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite metric step")
    return delta.to(gradient.dtype)


def propose(
    model, pack, state, truth, hold, mask, coefficient, *, geometry, create_graph=False
):
    if geometry not in ("unit", "metric"):
        raise ValueError("known update geometry required")
    if (
        truth.shape != hold.shape
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matching observed diagnostic targets required")
    current = state.detach().clone()
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
            raise FloatingPointError("nonfinite matured innovation gradient")
        if geometry == "unit":
            norm = (gradient.double().square().sum() + 1e-24).sqrt()
            delta = (-0.1 * gradient.double() / norm).to(current.dtype)
        else:
            delta = metric_direction(pack, gradient)
        proposal = variable + delta
    with torch.no_grad():
        predicted = auxiliary(model, pack, proposal, coefficient)
        proposed_loss = (
            innovation_loss(model, predicted, truth, mask)
            + 0.01 * proposal.square().mean()
        )
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite proposed innovation objective")
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
