"""Only the diagnostic predictor sees B; the classifier remains unchanged."""

import math

import torch

from ..channel_feedback_20261102.loss import channel_losses
from ..centered_gate_stream_20261101.model import loss as shape_loss
from ..lora_stream_20261102.core import forward


def augment(prediction, logit, coefficient, bias, scale):
    if (
        prediction.ndim != 3
        or coefficient.shape != prediction.shape[1:]
        or logit.shape != prediction.shape[:1]
        or not math.isfinite(float(bias))
        or not math.isfinite(float(scale))
        or scale <= 0
        or not all(torch.isfinite(v).all() for v in (prediction, logit, coefficient))
    ):
        raise ValueError(
            "finite matching forecast, logit, table and fixed axis required"
        )
    coordinate = (logit - bias) / scale
    return prediction + coordinate[:, None, None] * coefficient[None]


def auxiliary(model, pack, state, coefficient):
    output = forward(model, pack, state, forecast=True)
    axis = model.base.risk_head
    prediction = augment(
        output["forecast"],
        output["logit"],
        coefficient,
        float(axis.bias.detach()[0]),
        float(axis.weight.detach().double().norm()),
    )
    return prediction


def propose(model, pack, state, truth, hold, mask, coefficient, *, create_graph=False):
    if (
        truth.shape != hold.shape
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matching observed targets required")
    current = state.detach().clone()
    if not mask.any():
        return current, dict(
            accepted=False, reason="no_target", gradient_norm=0.0, step_norm=0.0
        )
    with torch.enable_grad():
        variable = current.requires_grad_(True)
        prediction = auxiliary(model, pack, variable, coefficient)
        objective = (
            channel_losses(model, prediction, truth, hold, mask).sum()
            + 0.01 * variable.square().mean()
        )
        (gradient,) = torch.autograd.grad(
            objective, variable, create_graph=create_graph
        )
        # Smooth only the parameter-gradient norm for finite meta derivatives
        # at a zero update. 1e-24 is not a signal-unit/diagnostic noise scale.
        magnitude = (gradient.double().square().sum() + 1e-24).sqrt()
        proposal = variable - 0.1 * (gradient.double() / magnitude).to(current.dtype)
    if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
        raise FloatingPointError("nonfinite auxiliary inner update")
    with torch.no_grad():
        proposed = auxiliary(model, pack, proposal, coefficient)
        proposed_loss = (
            channel_losses(model, proposed, truth, hold, mask).sum()
            + 0.01 * proposal.square().mean()
        )
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite auxiliary proposal")
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


def outer_objective(model, origin, later, truth, hold, mask, label, coefficient):
    """One post-history supervised episode, not a current-shot label access."""
    zero = truth.new_zeros(4, 64)
    after, receipt = propose(
        model, origin, zero, truth, hold, mask, coefficient, create_graph=True
    )
    updated = forward(model, later, after)["logit"]
    with torch.no_grad():
        before = forward(model, later, zero)["logit"]
    signed_after = (1 - 2 * label) * updated.double()
    signed_before = (1 - 2 * label) * before.double()
    risk_after = torch.logaddexp(torch.zeros_like(signed_after), signed_after).mean()
    risk_before = torch.logaddexp(torch.zeros_like(signed_before), signed_before).mean()
    prediction = auxiliary(model, origin, zero, coefficient)
    signal = shape_loss(model, prediction, truth, mask)
    regularizer = 1e-4 * coefficient.square().mean()
    return dict(
        meta=(risk_after - risk_before) / 0.1 + 0.20 * signal + regularizer,
        forecast=signal + regularizer,
        risk_before=risk_before,
        risk_after=risk_after,
        state=after,
        receipt=receipt,
    )
