"""Actual frozen-model interventions at two first-layer temporal placements."""

import numpy as np
import torch

import latent_trajectory_adapt_20261105 as latent
import paired_diagnostic_step_20261106 as old
import trajectory_policy_20261105 as policy
import whole_block_update_20261106 as rule


SITES = ("all", "recent4")
SITE_NORM = 0.35
DIRECTION_FLOOR = 1e-6
AXES = (
    ("signed_mir", (0,)),
    ("signed_sxr", (1,)),
    ("signed_xuv", (2,)),
    ("unsigned_mir", (3,)),
    ("unsigned_sxr", (4,)),
    ("unsigned_xuv", (5,)),
    ("anchor_normal", (6,)),
    ("signed_all", (0, 1, 2)),
    ("unsigned_all", (3, 4, 5)),
)
ARMS = tuple(f"{name}_{s}" for name, _ in AXES for s in ("plus", "minus"))


def episode_windows(times, logits, m, disruptive, down, horizon, threshold):
    """TRAIN-only teaching masks in fixed offsets m+1 through m+32.

    A previously correct source discharge can still provide a positive
    teaching block; only protection and raw new-crossing accounting consult
    the actual frozen first crossing. Safe episodes need all 32 endpoints.
    """
    if (
        times.ndim != 1
        or logits.shape != times.shape
        or not 63 <= m < len(times)
        or (m - 63) % 32
        or not np.isfinite(times).all()
        or not np.isfinite(logits).all()
        or (np.diff(times) <= 0).any()
        or not np.isfinite(threshold)
    ):
        raise ValueError("a real maturity on a finite aligned clock is required")
    k = np.arange(m + 1, min(m + 33, len(times)))
    if not disruptive:
        if len(k) != 32:
            return None
        return np.ones(32, dtype=bool), np.zeros(32, dtype=bool), False
    if not np.isfinite([down, horizon]).all() or horizon <= 10:
        raise ValueError("canonical ended-source horizon and disruption time required")
    raw_lead = (down - times) * 1000
    lead = raw_lead.round(9)
    if not np.isfinite(lead).all():
        raise ValueError("finite ended-shot lead required")
    eligible = (lead[k] > 10) & (lead[k] <= horizon + 1e-6)
    if not eligible.any():
        return None
    hits = np.flatnonzero(
        (raw_lead >= -1e-6) & (logits.astype(np.float64) >= threshold)
    )
    first = int(hits[0]) if hits.size else None
    timely = bool(first is not None and lead[first] > 10)
    protect = np.zeros(32, dtype=bool)
    if timely and m < first <= m + len(k):
        protect[: first - m] = True
    return np.pad(eligible, (0, 32 - len(k))), protect, timely


def routes(mask, site):
    valid = latent.routes_all(mask)
    if site == SITES[0]:
        return valid
    if site != SITES[1]:
        raise ValueError("unknown token placement")
    recent = torch.zeros_like(valid)
    # The final index is CLS; the last four *data* patches end at index 15.
    recent[..., 12:16] = valid[..., 12:16]
    return recent


def risk(base, x, a, u, site):
    if site == SITES[0]:
        return old.risk(base, x, a, u)
    return policy.routed_pair(
        base, x, a, routes(a, site), torch.cat((u, torch.zeros_like(u)), 1)
    )


def localized_bank(base, decoder, ox, oa, mx, ma, target, valid, scale):
    """Signed and unsigned family VJPs at the alternative, recent-token site."""
    captured = []
    with torch.enable_grad():
        u = ox.new_zeros((len(ox), old.WIDTH), requires_grad=True)
        handle = base.risk_head.register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0])
        )
        try:
            risk(base, ox, oa, u, SITES[1])
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (len(ox), 64):
            raise ValueError("the frozen risk-head input must be captured exactly once")
        prediction = decoder(captured[0])
        signed, _ = old.family_loss(prediction, target, valid)
        unsigned, _ = old.family_loss(prediction, target.abs(), valid)
        terms = (*signed.unbind(1), *unsigned.unbind(1))
        grad = torch.stack(
            [
                torch.autograd.grad(loss.sum(), u, retain_graph=i < 5)[0]
                for i, loss in enumerate(terms)
            ],
            dim=1,
        ).detach()
        m = torch.zeros_like(u, requires_grad=True)
        (normal,) = torch.autograd.grad(risk(base, mx, ma, m, SITES[1]).sum(), m)
    return rule.split_bank(grad, normal, scale)


def fixed_states(full, local):
    """Exact matched norms for plus/minus and both actual insertion sites."""
    if (
        full.ndim != 3
        or full.shape[1:] != (7, old.WIDTH)
        or full.shape != local.shape
        or not torch.isfinite(full).all()
        or not torch.isfinite(local).all()
    ):
        raise ValueError("aligned finite seven-direction banks required")
    basis = torch.stack((full, local), 1)
    v = torch.stack([basis[:, :, axes].sum(2) for _, axes in AXES], 2)
    norm = v.norm(dim=-1)
    eligible = (norm > DIRECTION_FLOOR).all(1)
    u = (SITE_NORM * v / norm.clamp_min(DIRECTION_FLOOR)[..., None]).masked_fill(
        ~eligible[:, None, :, None], 0
    )
    return torch.stack((u, -u), dim=3).flatten(2, 3), eligible.repeat_interleave(
        2, dim=1
    )
