"""Subtract the same forecast's hold-target loss, not the final risk score."""

import torch

from ..centered_gate_stream_20261101.model import loss as shape_loss
from ..lora_stream_20261102.core import forward


def components(model, prediction, state, truth, hold, mask):
    signal = shape_loss(model, prediction, truth, mask)
    control = shape_loss(model, prediction, hold, mask)
    regularizer = 0.01 * state.square().mean()
    return dict(
        signal=signal,
        hold=control,
        regularizer=regularizer,
        objective=signal - control + regularizer,
    )


def numbers(terms):
    return {key: float(value.detach()) for key, value in terms.items()}


def propose(model, pack, state, truth, hold, mask):
    if (
        truth.shape != (len(pack["h"]), 25, 23)
        or hold.shape != truth.shape
        or mask.shape != truth.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matching25x23 signal/hold targets and boolean mask required")
    if not torch.isfinite(truth[mask]).all() or not torch.isfinite(hold[mask]).all():
        raise ValueError("finite observed targets required")
    current = state.detach().clone()
    observed = int(mask.sum())
    if not observed:
        return current, dict(
            loss_before=None,
            loss_proposal=None,
            loss_retained=None,
            terms_before=None,
            terms_proposal=None,
            terms_retained=None,
            gradient_norm=0.0,
            step_norm=0.0,
            observed=0,
            accepted=False,
            reason="no_target",
            gradient_computed=False,
            gradient256=[0.0] * 256,
            state256_after=current.flatten().cpu().tolist(),
            max_action_rms=None,
        )
    with torch.enable_grad():
        variable = current.detach().requires_grad_(True)
        before = forward(model, pack, variable, forecast=True)
        terms = components(model, before["forecast"], variable, truth, hold, mask)
        objective = terms["objective"]
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite difference objective")
        gradient = torch.autograd.grad(objective, variable)[0].detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("nonfinite difference gradient")
    magnitude = gradient.double().norm()
    direction = (
        (gradient.double() / magnitude).to(current.dtype)
        if magnitude > 0
        else torch.zeros_like(current)
    )
    with torch.no_grad():
        proposal = current - 0.1 * direction
        proposed = forward(model, pack, proposal, forecast=True)
        proposed_terms = components(
            model, proposed["forecast"], proposal, truth, hold, mask
        )
        proposed_loss = proposed_terms["objective"]
        if not torch.isfinite(proposed_loss):
            raise FloatingPointError("nonfinite proposed difference objective")
        accepted = bool(proposed_loss < objective.detach())
        after = proposal if accepted else current
        retained = proposed_loss if accepted else objective.detach()
        rms = proposed["rms"] if accepted else before["rms"].detach()
    return after.detach(), dict(
        loss_before=float(objective.detach()),
        loss_proposal=float(proposed_loss),
        loss_retained=float(retained),
        terms_before=numbers(terms),
        terms_proposal=numbers(proposed_terms),
        terms_retained=numbers(proposed_terms if accepted else terms),
        gradient_norm=float(magnitude),
        step_norm=float((after - current).double().norm()),
        observed=observed,
        accepted=accepted,
        reason="improved"
        if accepted
        else ("zero_gradient" if magnitude == 0 else "not_improved"),
        gradient_computed=True,
        gradient256=gradient.flatten().cpu().tolist(),
        state256_after=after.flatten().cpu().tolist(),
        max_action_rms=float(rms.max()),
    )
