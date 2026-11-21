"""Real gradients from matured observations; apply bounded, shot-local writes."""

import torch

import layer0_arrival_motion_20261106 as motion
from layer0_arrival_step_20261106 import bounded_step
import latent_trajectory_adapt_20261105 as latent
import trajectory_policy_20261105 as policy


ARMS = ("arrival", "shape", "lag", "reverse")


def arrival_indices(t, length):
    if (
        t.ndim != 1
        or t.dtype != torch.long
        or (t < motion.H).any()
        or (t >= length).any()
    ):
        raise ValueError(
            "maturity must be an observed endpoint with a full preceding arrival"
        )
    origin = t - motion.H
    j = origin[:, None] + torch.arange(1, motion.H + 1, device=t.device)
    return origin, j


def matched_descent(g, norm):
    if (
        g.ndim != 2
        or norm.shape != (len(g),)
        or (norm < 0).any()
        or not torch.isfinite(g).all()
        or not torch.isfinite(norm).all()
    ):
        raise ValueError("finite directional control and its norm required")
    size = g.double().norm(dim=1)
    active = norm > 0
    if (active & (size == 0)).any():
        raise ValueError("cannot match a nonzero action with an undefined direction")
    a = torch.zeros_like(g)
    a[active] = (
        -g[active].double() * (norm[active].double() / size[active])[:, None]
    ).to(g.dtype)
    if not torch.allclose(a.norm(dim=1), norm, rtol=0, atol=1e-6):
        raise ValueError("direction control failed norm equality")
    return a


def episode(base, windows, mask, values, available, t):
    o, j = arrival_indices(t, len(values))
    x, a = windows[o].contiguous(), mask[o].contiguous()
    hold, _, local, last = motion.origin_terms(x, a, base.center, base.scale)
    target, valid = motion.target_terms(
        values[j], available[j], last, base.center, base.scale
    )
    return x, a, hold, local, target, valid


def evaluate_motion(base, head, ep, action, rows=None):
    x, a, hold, local, target, valid = (
        ep if rows is None else tuple(v[rows] for v in ep)
    )
    ctx, _ = motion.features(base, x, a, action)
    return motion.row_loss(head(hold, local, ctx), target, valid)


def moved_logits(base, x, a, u):
    return policy.routed_pair(
        base, x, a, latent.routes_all(a), torch.cat((u, torch.zeros_like(u)), dim=1)
    )


def step_batch(base, head, windows, mask, values, available, anchors, radius):
    """No target from later than an anchor reaches this function's losses."""
    ep = episode(base, windows, mask, values, available, anchors)
    lag = episode(base, windows, mask, values, available, anchors - policy.STRIDE)
    # Same arrival and shared observation mask for true-vs-reversed losses.
    y, observed = ep[-2:]
    common = observed & observed.flip(1)
    direct = (*ep[:-1], common)
    x, a = windows[anchors].contiguous(), mask[anchors].contiguous()
    u0 = x.new_zeros((len(x), motion.WIDTH))
    with torch.enable_grad():
        u = u0.clone().requires_grad_(True)
        ctx, _ = motion.features(base, ep[0], ep[1], u)
        p = head(ep[2], ep[3], ctx)
        actual_loss = motion.row_loss(p, y, common)
        rev_loss = motion.row_loss(p, y.flip(1), common)
        (g_true,) = torch.autograd.grad(
            actual_loss["combined"].sum(), u, retain_graph=True
        )
        (g_reverse,) = torch.autograd.grad(rev_loss["combined"].sum(), u)
        v = u0.clone().requires_grad_(True)
        lag_loss = evaluate_motion(base, head, lag, v)
        (g_lag,) = torch.autograd.grad(lag_loss["combined"].sum(), v)
        z = u0.clone().requires_grad_(True)
        native = moved_logits(base, x, a, z)
        (g_anchor,) = torch.autograd.grad(native.sum(), z)
    if not all(
        torch.isfinite(v).all() for v in (g_true, g_reverse, g_lag, g_anchor, native)
    ):
        raise FloatingPointError("invalid arrival feedback gradient")
    support = (
        actual_loss["supported"]
        & lag_loss["supported"]
        & (g_true.norm(dim=1) > 0)
        & (g_reverse.norm(dim=1) > 0)
        & (g_lag.norm(dim=1) > 0)
    )
    baseline = actual_loss["combined"].detach()

    def measured(i, d):
        return evaluate_motion(base, head, direct, d, rows=i)["combined"]

    def z_at(i, d):
        return moved_logits(base, x[i], a[i], d)

    raw, lraw, ok, _ = bounded_step(g_true, radius, baseline, measured, support=support)
    shape, lshape, ok_shape, fz = bounded_step(
        g_true,
        radius,
        baseline,
        measured,
        support=support,
        anchor_gradient=g_anchor,
        anchor_zero=native.detach(),
        anchor_eval=z_at,
    )
    norm = raw.norm(dim=1)
    actions = dict(
        arrival=raw,
        shape=shape,
        lag=matched_descent(g_lag, norm),
        reverse=matched_descent(g_reverse, norm),
    )
    with torch.no_grad():
        l_lag = measured(slice(None), actions["lag"])
        l_reverse = measured(slice(None), actions["reverse"])
        rz = moved_logits(base, x, a, raw)
    return actions, dict(
        support=support.detach(),
        accepted=ok,
        accepted_shape=ok_shape,
        radius=radius.detach(),
        norms=torch.stack([v.norm(dim=1) for v in actions.values()], dim=1),
        losses=torch.stack((baseline, lraw, lshape, l_lag, l_reverse), dim=1),
        logits=torch.stack((native.detach(), rz, fz.to(native)), dim=1),
    )


def actions_at(base, head, ctx, values, available, anchors, radius):
    if (
        any(p.requires_grad for p in base.parameters())
        or any(p.requires_grad for p in head.parameters())
        or base.training
        or head.training
        or len(anchors) != len(radius)
        or not len(anchors)
    ):
        raise ValueError(
            "frozen parent and head with one nonempty causal anchor schedule required"
        )
    a, info = [], []
    for i in range(0, len(anchors), 128):
        j = slice(i, i + 128)
        acted, note = step_batch(
            base, head, ctx[0], ctx[1], values, available, anchors[j], radius[j]
        )
        a.append(acted)
        info.append(note)
    return (
        {k: torch.cat([v[k] for v in a]) for k in ARMS},
        {k: torch.cat([v[k] for v in info]) for k in info[0]},
    )
