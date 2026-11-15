"""External direct diagnostic decoder; no risk-logit bridge or source mutation."""

from copy import deepcopy

import torch

from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..lora_stream_20261102.core import forward


def clone_head(model):
    """Independent trainable copy, preserving source weights, dtype and device."""
    head = model.base.forecast_head
    if not isinstance(head, torch.nn.Linear) or (
        head.in_features,
        head.out_features,
    ) != (1088, 575):
        raise ValueError("the source Linear1088->575 forecast head is required")
    return deepcopy(head).requires_grad_(True)


def prediction(model, pack, state, decoder):
    """Decode the actual final normalized/masked token input, without detaching."""
    captured = []

    def capture(_module, args):
        captured.append(args[0])

    hook = model.base.forecast_head.register_forward_pre_hook(capture)
    try:
        forward(model, pack, state, forecast=True)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("exactly one frozen forecast-head input required")
    result = decoder(captured[0]).reshape(len(captured[0]), 25, 23)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite direct diagnostic prediction")
    return result


def advance(model, pack, state, truth, mask, decoder, *, create_graph=False):
    """The fixed full_path metric step, replacing only auxiliary prediction."""
    if mask.shape != truth.shape or mask.dtype != torch.bool:
        raise ValueError("matching matured diagnostic target and mask required")
    current = state.clone() if create_graph else state.detach().clone()
    if not mask.any():
        return current, dict(
            accepted=False,
            reason="no_target",
            loss_before=None,
            loss_proposal=None,
            gradient_norm=0.0,
            step_norm=0.0,
            gradient256=[0.0] * current.numel(),
            state256_after=current.detach().flatten().cpu().tolist(),
        )
    with torch.enable_grad():
        variable = current.requires_grad_(True)
        predicted = prediction(model, pack, variable, decoder)
        objective = (
            innovation_loss(model, predicted, truth, mask)
            + 0.01 * variable.square().mean()
        )
        (gradient,) = torch.autograd.grad(
            objective, variable, create_graph=create_graph
        )
        if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
            raise FloatingPointError("nonfinite direct decoder innovation gradient")
        proposal = variable + metric_direction(pack, gradient)
    with torch.no_grad():
        predicted = prediction(model, pack, proposal, decoder)
        proposed_loss = (
            innovation_loss(model, predicted, truth, mask)
            + 0.01 * proposal.square().mean()
        )
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite direct decoder proposal")
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
