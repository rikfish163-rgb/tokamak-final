"""Bounded Armijo acceptance on the currently observed auxiliary loss."""

import math

import torch


_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
_ARMIJO = 1e-4


def _finite_scalar(value, name):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().item()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} must be finite")
    return value


def _norm(value):
    """Calculate a stable real-valued norm, including for subnormal inputs."""
    value = value.to(dtype=torch.float64)
    if not value.numel():
        return 0.0
    scale = value.abs().max().item()
    if scale == 0.0:
        return 0.0
    result = scale * torch.linalg.vector_norm(value / scale).item()
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite norm")
    return result


def _dot(gradient, displacement):
    result = (gradient * displacement.to(dtype=torch.float64)).sum().item()
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite directional dot product")
    return result


@torch.no_grad()
def search(objective, proposal, gradient, loss_before):
    """Return the first acceptable displacement and an auditable receipt.

    ``objective`` receives the trial displacement, not a scaled loss or an
    accumulated state.  A rejected search returns a detached exact-zero tensor;
    loss_after then denotes the unchanged baseline loss.  Trial losses are
    recorded in factor order and nonfinite values are errors, never abstentions.
    """
    if not isinstance(proposal, torch.Tensor) or not isinstance(gradient, torch.Tensor):
        raise TypeError("proposal and gradient must be tensors")
    if proposal.shape != gradient.shape:
        raise ValueError("proposal and gradient must have the same shape")
    if proposal.is_complex() or gradient.is_complex():
        raise ValueError("proposal and gradient must be real")
    proposal = proposal.detach()
    gradient = gradient.detach().to(device=proposal.device)
    if not torch.isfinite(proposal).all().item():
        raise FloatingPointError("proposal must be finite")
    if not torch.isfinite(gradient).all().item():
        raise FloatingPointError("gradient must be finite")
    before = _finite_scalar(loss_before, "loss_before")
    gradient64 = gradient.to(dtype=torch.float64)
    proposed_dot = _dot(gradient64, proposal)
    receipt = {
        "trials": 0,
        "factor": 0.0,
        "reason": "no_acceptable_step",
        "loss_before": before,
        "loss_after": before,
        "proposed_norm": _norm(proposal),
        "accepted_norm": 0.0,
        "directional_dot": 0.0,
        "gradient_norm": _norm(gradient),
        "trial_losses": [],
        "proposed_directional_dot": proposed_dot,
    }
    zero = torch.zeros_like(proposal)
    if (
        not torch.count_nonzero(proposal).item()
        or not torch.count_nonzero(gradient).item()
    ):
        receipt["reason"] = "zero_direction"
        return zero, receipt
    if proposed_dot >= 0.0:
        receipt["reason"] = "non_descent"
        return zero, receipt

    for factor in _FACTORS:
        trial = proposal * factor
        directional_dot = _dot(gradient64, trial)
        loss = _finite_scalar(objective(trial), "trial loss")
        receipt["trials"] += 1
        receipt["trial_losses"].append(loss)
        if (
            directional_dot < 0.0
            and loss < before
            and loss <= before + _ARMIJO * directional_dot
        ):
            receipt.update(
                factor=factor,
                reason="accepted",
                loss_after=loss,
                accepted_norm=_norm(trial),
                directional_dot=directional_dot,
            )
            return trial.detach(), receipt
    return zero, receipt
