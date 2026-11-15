"""A bounded mixture of actual diagnostic gradients in the first attention block."""

import math

import torch
from torch import nn

import paired_diagnostic_step_20261106 as old


FEATURES, DIRECTIONS, HIDDEN, COEF_LIMIT = 35, 7, 24, 2.0


def split_bank(gradients, anchor, reference):
    """Six diagnostic descents tangent to mature risk, one observed normal mode."""
    if (
        gradients.ndim != 3
        or gradients.shape[1:] != (6, old.WIDTH)
        or anchor.shape != gradients[:, 0].shape
        or not math.isfinite(reference)
        or reference <= 0
        or not torch.isfinite(gradients).all()
        or not torch.isfinite(anchor).all()
    ):
        raise ValueError("finite family gradients, anchor and positive scale required")
    directions = (
        -gradients / (reference**2 + gradients.square().sum(-1, keepdim=True)).sqrt()
    )
    norm = anchor.norm(dim=1, keepdim=True)
    normal = anchor / torch.where(norm > old.NORM_FLOOR, norm, 1.0)
    normal = normal.masked_fill(norm <= old.NORM_FLOOR, 0.0)
    along = (directions * normal[:, None]).sum(-1, keepdim=True)
    tangent = directions - along * normal[:, None]
    observable = along.square().mean(1).sqrt()
    return torch.cat((tangent, (normal * observable)[:, None]), 1)


def describe(target, forecast, valid, origin_risk, maturity_risk):
    """Only two completed observations, their forecast error, and sealed risks."""
    if (
        target.ndim != 2
        or target.shape[1] != old.TARGETS
        or forecast.shape != target.shape
        or valid.shape != target.shape
        or valid.dtype != torch.bool
        or origin_risk.shape != (len(target),)
        or maturity_risk.shape != origin_risk.shape
    ):
        raise ValueError("aligned reached transition, mask and risk context required")
    reached = target.masked_fill(~valid, 0.0)
    error = (forecast - target).masked_fill(~valid, 0.0).clamp(-8, 8) / 4
    return torch.cat(
        (
            reached / 4,
            error,
            valid.to(target),
            origin_risk.clamp(-6, 6)[:, None] / 6,
            (maturity_risk - origin_risk).clamp(-4, 4)[:, None] / 4,
        ),
        1,
    )


def compose(basis, coefficients):
    v = torch.bmm(coefficients[:, None], basis).squeeze(1)
    return old.RADIUS * v / (1 + v.square().sum(1, keepdim=True)).sqrt()


class Combiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(FEATURES, HIDDEN)
        self.head = nn.Linear(HIDDEN, DIRECTIONS)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def coefficients(self, descriptor):
        return (
            COEF_LIMIT
            * self.head(torch.nn.functional.silu(self.hidden(descriptor))).tanh()
        )

    def forward(self, descriptor, basis):
        return compose(basis, self.coefficients(descriptor))


def reached_bank(base, decoder, mixer, ox, oa, mx, ma, target, valid, g_ref):
    """Real diagnostic VJPs, exact old-support control and mature risk derivative.

    No future query window is an input. The original four directions define
    identical eligibility for old and new arms; the new rule can choose zero.
    """
    with torch.enable_grad():
        origin = ox.new_zeros((len(ox), old.WIDTH), requires_grad=True)
        hidden, z0 = old.native(base, ox, oa, origin)
        p = decoder(hidden)
        signed, present = old.family_loss(p, target, valid)
        absolute, _ = old.family_loss(p, target.abs(), valid)
        reversed_target, _ = old.family_loss(p, -target, valid)
        zero_target, _ = old.family_loss(p, torch.zeros_like(target), valid)
        w = mixer(old.mix_inputs(target, valid), present).detach()
        objectives = [
            *signed.unbind(1),
            *absolute.unbind(1),
            (w * reversed_target).sum(1),
            (w * zero_target).sum(1),
        ]
        vjp = torch.stack(
            [
                torch.autograd.grad(y.sum(), origin, retain_graph=i < 7)[0]
                for i, y in enumerate(objectives)
            ],
            1,
        ).detach()
        mature = torch.zeros_like(origin, requires_grad=True)
        zm = old.risk(base, mx, ma, mature)
        normal = torch.autograd.grad(zm.sum(), mature)[0]
    original = (
        (vjp[:, :3] * w[:, :, None]).sum(1),
        vjp[:, 6],
        (vjp[:, 3:6] * w[:, :, None]).sum(1),
        vjp[:, 7],
    )
    (old_update, *_), ok = old.actions(original, present.sum(1) >= 2, g_ref)
    desc = describe(target, p.detach(), valid, z0.detach(), zm.detach())
    return desc, split_bank(vjp[:, :6], normal, g_ref), old_update, ok
