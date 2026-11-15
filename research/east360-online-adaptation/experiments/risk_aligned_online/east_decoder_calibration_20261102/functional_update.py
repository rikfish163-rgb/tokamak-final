"""One controlled replacement of parameter protection by observed FFN protection."""

import math

import torch

from ..direct_decoder_meta_20261102.core import prediction
from ..innovation_metric_20261102.core import innovation_loss, metric_direction

FRACTIONS = (1.0, 0.25, 0.0625, 0.015625)


def protection(pack, state, kind):
    if kind == "parameter":
        return 0.01 * state.square().mean()
    if kind != "hidden":
        raise ValueError("known protection coordinate required")
    valid = pack["valid"][:, :16]
    raw = (pack["phi"] @ state).masked_fill(~valid[:, :, None], 0)
    return 0.01 * raw.double().square().sum() / (valid.sum().clamp_min(1) * 64)


def first_improvement(before, values):
    if not all(math.isfinite(v) for v in (before, *values)):
        raise FloatingPointError("finite proxy losses required; no silent rejection")
    return next((i for i, value in enumerate(values) if value < before), None)


def advance(model, pack, state, truth, mask, decoder, *, kind):
    if kind not in ("parameter", "hidden"):
        raise ValueError("known protection coordinate required")
    if truth.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("matching matured diagnostic target/mask required")
    current = state.detach().clone()
    if not mask.any():
        return current, dict(
            accepted=False,
            reason="no_target",
            loss_before=None,
            loss_proposal=None,
            gradient_norm=0.0,
            step_norm=0.0,
            gradient256=[0.0] * 256,
            state256_after=current.flatten().cpu().tolist(),
            tested_steps=[],
            chosen_fraction=None,
        )
    with torch.enable_grad():
        variable = current.requires_grad_(True)
        forecast = innovation_loss(
            model, prediction(model, pack, variable, decoder), truth, mask
        )
        objective = forecast + protection(pack, variable, kind)
        (gradient,) = torch.autograd.grad(objective, variable)
    if not torch.isfinite(gradient).all() or not torch.isfinite(objective):
        raise FloatingPointError("nonfinite protected forecast gradient")
    direction = metric_direction(pack, gradient).detach()
    trials, accepted, after, chosen = [], False, current.detach(), None
    with torch.no_grad():
        for fraction in FRACTIONS:
            candidate = current.detach() + fraction * direction
            diagnostic = innovation_loss(
                model, prediction(model, pack, candidate, decoder), truth, mask
            )
            anchor = protection(pack, candidate, kind)
            loss = diagnostic + anchor
            trials.append(
                dict(
                    fraction=fraction,
                    diagnostic_loss=float(diagnostic),
                    protection_loss=float(anchor),
                    objective=float(loss),
                )
            )
            if first_improvement(float(objective.detach()), [float(loss)]) == 0:
                accepted, after, chosen = True, candidate, fraction
                break
    return after.detach(), dict(
        accepted=accepted,
        reason="improved" if accepted else "not_improved",
        loss_before=float(objective.detach()),
        loss_proposal=trials[-1]["objective"],
        gradient_norm=float(gradient.detach().double().norm()),
        step_norm=float((after.detach() - current.detach()).double().norm()),
        gradient256=gradient.detach().flatten().cpu().tolist(),
        state256_after=after.detach().flatten().cpu().tolist(),
        tested_steps=trials,
        chosen_fraction=chosen,
    )
