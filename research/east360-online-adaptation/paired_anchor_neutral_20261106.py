"""Same-norm finite interventions with a measured, neutral maturity logit."""

import math

import torch


TOL, DIRECTION_FLOOR, BISECTIONS = 1e-4, 1e-8, 18


@torch.no_grad()
def neutralize(original, gradient, baseline, score_at):
    """Rotate the normalized tangent in its plane until the anchor is neutral.

    `score_at(indices, actions)` must perform the actual frozen forward at
    those rows' already observed anchors, not a linear surrogate. The declared
    construction only admits rows with a nonzero normal and tangent and a
    directly neutral tangent or a bracket within [-pi/2, pi/2]. An unsupported
    row returns zero and false rather than being passed off as a matched write.
    """
    if (
        original.ndim != 2
        or gradient.shape != original.shape
        or baseline.shape != (len(original),)
        or any(
            v.dtype != original.dtype
            or v.device != original.device
            or not torch.isfinite(v).all()
            for v in (original, gradient, baseline)
        )
        or not original.is_floating_point()
    ):
        raise ValueError(
            "aligned finite floating-point proposal and anchor normal required"
        )
    action = torch.zeros_like(original)
    accepted = torch.zeros_like(baseline, dtype=torch.bool)
    residual, angles = torch.zeros_like(baseline), torch.zeros_like(baseline)
    magnitude, normal_length = original.norm(dim=1), gradient.norm(dim=1)
    n = gradient / normal_length[:, None].clamp_min(DIRECTION_FLOOR)
    tangent = original - (original * n).sum(1, keepdim=True) * n
    size = tangent.norm(dim=1)
    (ids,) = torch.nonzero(
        (magnitude > DIRECTION_FLOOR)
        & (normal_length > DIRECTION_FLOOR)
        & (size > DIRECTION_FLOOR),
        as_tuple=True,
    )
    if not len(ids):
        return action, accepted, residual, angles
    v = tangent / size[:, None].clamp_min(DIRECTION_FLOOR)

    def evaluate(ix, angle):
        u = magnitude[ix, None] * (
            v[ix] * angle.cos()[:, None] + n[ix] * angle.sin()[:, None]
        )
        score = score_at(ix, u)
        if score.shape != (len(ix),) or not torch.isfinite(score).all():
            raise ValueError("anchor evaluator must return one finite score per row")
        return u, score - baseline[ix]

    flat = torch.zeros_like(baseline[ids])
    ut, dt = evaluate(ids, flat)
    straight = dt.abs() <= TOL
    good = ids[straight]
    action[good], residual[good], accepted[good] = ut[straight], dt[straight], True
    ids = ids[~straight]
    if not len(ids):
        return action, accepted, residual, angles
    hi = torch.full_like(baseline[ids], math.pi / 2)
    lo = -hi
    ul, sl = evaluate(ids, lo)
    uh, sh = evaluate(ids, hi)
    hit_l = sl.abs() <= TOL
    hit_h = (~hit_l) & (sh.abs() <= TOL)
    for hit, u, d, angle in ((hit_l, ul, sl, lo), (hit_h, uh, sh, hi)):
        k = ids[hit]
        action[k], residual[k], angles[k], accepted[k] = (
            u[hit],
            d[hit],
            angle[hit],
            True,
        )
    bracket = (~(hit_l | hit_h)) & ((sl > 0) != (sh > 0))
    ids, lo, hi, sl = ids[bracket], lo[bracket], hi[bracket], sl[bracket]
    if not len(ids):
        return action, accepted, residual, angles
    for _ in range(BISECTIONS):
        mid = (lo + hi) / 2
        _, sm = evaluate(ids, mid)
        exact = sm == 0
        move_lo = (sm > 0) == (sl > 0)
        lo = torch.where(move_lo | exact, mid, lo)
        hi = torch.where(~move_lo | exact, mid, hi)
        sl = torch.where(move_lo | exact, sm, sl)
    mid = (lo + hi) / 2
    um, dm = evaluate(ids, mid)
    satisfied = dm.abs() <= TOL
    k = ids[satisfied]
    action[k], residual[k], angles[k], accepted[k] = (
        um[satisfied],
        dm[satisfied],
        mid[satisfied],
        True,
    )
    return action, accepted, residual, angles
