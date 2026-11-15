"""Temporary first-layer descent on an observed signed diagnostic transition."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy

WIDTH, TARGETS, BLOCK = 128, 11, 32
GROUP_SLICES = (slice(0, 3), slice(3, 7), slice(7, 11))
RADIUS, NORM_FLOOR = 0.5, 1e-8


class Decoder(nn.Linear):
    def __init__(self):
        super().__init__(64, TARGETS)


class Mixer(nn.Linear):
    """Learn bounded family reliability from observed magnitudes and masks."""

    def __init__(self):
        super().__init__(6, 3)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features, present):
        w = (0.5 + 1.5 * super().forward(features).softmax(1)) * present
        # An unavailable row has no inner objective; it carries no update.
        return w / w.sum(1, keepdim=True).clamp_min(0.5)


def mix_inputs(y, valid):
    magnitude, fraction = [], []
    for g in GROUP_SLICES:
        count = valid[:, g].sum(1)
        magnitude.append(
            y[:, g].abs().masked_fill(~valid[:, g], 0.0).sum(1) / count.clamp_min(1)
        )
        fraction.append(valid[:, g].float().mean(1))
    return torch.stack((*magnitude, *fraction), 1)


def family_loss(pred, target, valid):
    if (
        pred.ndim != 2
        or pred.shape[1] != TARGETS
        or target.shape != pred.shape
        or valid.shape != pred.shape
        or valid.dtype != torch.bool
    ):
        raise ValueError("aligned signed transition and 11-channel support required")
    raw = F.smooth_l1_loss(pred, target, reduction="none").masked_fill(~valid, 0.0)
    count = torch.stack([valid[:, g].sum(1) for g in GROUP_SLICES], 1)
    loss = torch.stack([raw[:, g].sum(1) for g in GROUP_SLICES], 1)
    return loss / count.clamp_min(1), count > 0


def risk(base, x, available, u=None):
    if u is None:
        return base(x, available)["logit"]
    if u.shape != (len(x), WIDTH):
        raise ValueError("one temporary first-layer affine vector per row required")
    delta = torch.cat((u, torch.zeros_like(u)), 1)
    return policy.routed_pair(base, x, available, latent.routes_all(available), delta)


def native(base, x, available, u=None):
    seen = []

    def capture(_, args):
        seen.append(args[0])

    handle = base.risk_head.register_forward_pre_hook(capture)
    try:
        z = risk(base, x, available, u)
    finally:
        handle.remove()
    if len(seen) != 1 or seen[0].shape != (len(x), 64):
        raise RuntimeError("native CLS input to the frozen risk head was not captured")
    return seen[0], z


def bank(base, decoder, x, available, target, valid):
    """Exactly cache the three family VJPs at zero temporary state."""
    with torch.enable_grad():
        u = x.new_zeros((len(x), WIDTH), requires_grad=True)
        h, _ = native(base, x, available, u)
        p = decoder(h)
        loss, present = family_loss(p, target, valid)
        grads = [
            torch.autograd.grad(loss[:, i].sum(), u, retain_graph=i < 2)[0]
            for i in range(3)
        ]
    return p.detach(), torch.stack(grads, 1).detach(), present


def effects(gradient, reference):
    """Gradient descent with smooth norm bound, set by TRAIN's gradient scale."""
    return (
        -RADIUS
        * gradient
        / (reference**2 + gradient.square().sum(1, keepdim=True)).sqrt()
    )


def actions(gradients, support, reference):
    if len(gradients) != 4 or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("four finite true/control gradients required")
    norms = torch.stack([g.norm(dim=1) for g in gradients], 1)
    common = support & (norms > NORM_FLOOR).all(1)
    real = effects(gradients[0], reference).masked_fill(~common[:, None], 0.0)
    magnitude = real.norm(dim=1, keepdim=True)
    # All arms are disabled on a row without a defined equal-norm direction.
    controls = tuple(
        -g * (magnitude / norms[:, i : i + 1].clamp_min(NORM_FLOOR))
        for i, g in enumerate(gradients[1:], 1)
    )
    return (real, *controls), common


def interventions(base, decoder, mixer, x, available, target, valid, reference):
    """Only the reached signal pair is used, with identical origin for controls."""
    with torch.enable_grad():
        u = x.new_zeros((len(x), WIDTH), requires_grad=True)
        hidden, _ = native(base, x, available, u)
        p = decoder(hidden)
        gradients = []
        for i, q in enumerate(
            (target, -target, target.abs(), torch.zeros_like(target))
        ):
            losses, present = family_loss(p, q, valid)
            if i == 0:
                w = mixer(mix_inputs(target, valid), present).detach()
            gradients.append(
                torch.autograd.grad((losses * w).sum(), u, retain_graph=i < 3)[0]
            )
    values, common = actions(gradients, present.sum(1) >= 2, reference)
    return values, common, p.detach(), w


def write_slots(length, maturity):
    expected = np.arange(2 * BLOCK - 1, length, BLOCK, dtype=np.int64)
    if length <= 0 or not np.array_equal(maturity, expected):
        raise ValueError("all complete consecutive block maturities required")
    return np.searchsorted(maturity, np.arange(length), side="left") - 1


def training_queries(times, native, maturity, support, disruptive, down, horizon):
    """Offline TRAIN mining only: one native peak per supported role/next block."""
    slots = write_slots(len(times), maturity)
    if (
        native.shape != times.shape
        or support.shape != maturity.shape
        or support.dtype != bool
    ):
        raise ValueError(
            "one training risk value per endpoint and causal support per block"
        )
    if disruptive:
        lead = np.round((down - times) * 1000, 9)
        classes = (
            (0, (lead > 10.0) & (lead <= horizon + 1e-6)),
            (2, lead > horizon + 1e-6),
        )
    else:
        classes = ((1, np.ones(len(times), dtype=bool)),)
    triples = []
    for j in np.flatnonzero(support):
        for role, eligible in classes:
            ix = np.flatnonzero((slots == j) & eligible)
            if len(ix):
                triples.append((j, ix[np.argmax(native[ix])], role))
    arr = np.array(triples, dtype=np.int64).reshape(-1, 3)
    return tuple(arr.T)
