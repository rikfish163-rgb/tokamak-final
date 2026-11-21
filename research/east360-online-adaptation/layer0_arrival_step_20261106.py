"""A bounded, measured update for an already-arrived motion loss."""

import torch


def _vector(value, name, size, device, *, boolean=False):
    if not isinstance(value, torch.Tensor) or value.shape != (size,):
        raise ValueError(f"{name} must be a tensor of shape ({size},)")
    if value.device != device:
        raise ValueError(f"{name} must be on the gradient's device")
    if boolean:
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
    elif not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value.detach()


def _scores(callback, indices, actions, name):
    scores = callback(indices, actions)
    if not isinstance(scores, torch.Tensor) or scores.shape != indices.shape:
        raise ValueError(f"{name} must return one tensor value per index")
    if scores.device != actions.device or not scores.is_floating_point():
        raise ValueError(f"{name} must return floating-point scores on the action device")
    return scores.detach()


def _norm(values):
    """Rowwise double norms without avoidable squaring overflow/underflow."""
    largest = values.abs().amax(dim=1)
    scale = torch.where(torch.isfinite(largest) & (largest > 0), largest, 1.0)
    return torch.linalg.vector_norm(values / scale[:, None], dim=1) * scale


def _bound(actions, radius, dtype):
    """Rescale to the original cap, including rounding after the dtype cast."""
    norms = _norm(actions)
    finite = torch.isfinite(norms)
    scale = torch.ones_like(norms)
    over = finite & (norms > radius)
    scale[over] = radius[over] / norms[over]
    actions = (actions * scale[:, None]).to(dtype)

    # Casting and norm rounding can put a boundary vector just outside the cap.
    action64 = actions.to(torch.float64)
    norms = _norm(action64)
    direct_norms = torch.linalg.vector_norm(action64, dim=1)
    norms = torch.maximum(norms, torch.where(torch.isfinite(direct_norms), direct_norms, norms))
    near = (actions != 0).sum(dim=1) > 1
    near &= norms >= radius * (1 - 4 * torch.finfo(torch.float64).eps)
    rescale = torch.isfinite(norms) & (norms > 0) & ((norms > radius) | near)
    if rescale.any():
        scale = torch.clamp(radius[rescale] / norms[rescale], max=1)
        finfo = torch.finfo(dtype)
        # Subnormal coordinates have coarser relative precision.
        slack = torch.where(
            radius[rescale] < finfo.tiny,
            torch.full_like(scale, 16 * finfo.eps),
            torch.full_like(scale, 2 * finfo.eps),
        )
        scale *= 1 - slack
        actions[rescale] = (action64[rescale] * scale[:, None]).to(dtype)
    return actions


@torch.no_grad()
def bounded_step(
    gradient, radius, zero_loss, evaluate, *, support,
    anchor_gradient=None, anchor_zero=None, anchor_eval=None,
):
    """Return the first measured, strictly improving bounded step for each row.

    ``gradient`` and optional ``anchor_gradient`` are B x 128 tensors. The
    callbacks receive original row indices and an action of matching batch size
    and must return one actual loss or risk logit per row. All inputs must be on
    one device, and actions retain the gradient dtype. Unsupported, zero-loss,
    zero-radius and zero-direction rows are returned unchanged without scoring.

    When all three anchor arguments are provided, each trial starts along the
    double-precision projected gradient. Up to two corrections use the fixed
    gradient at zero; every correction is evaluated at the actual finite action.
    A trial is eligible only if its actual risk is within 1e-4 of anchor_zero.
    """
    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2 or gradient.shape[1] != 128:
        raise ValueError("gradient must be a B x 128 tensor")
    if not gradient.is_floating_point():
        raise TypeError("gradient must be floating point")
    gradient = gradient.detach()
    size, device = gradient.shape[0], gradient.device
    radius = _vector(radius, "radius", size, device)
    zero_loss = _vector(zero_loss, "zero_loss", size, device)
    support = _vector(support, "support", size, device, boolean=True)

    anchor_parts = (anchor_gradient, anchor_zero, anchor_eval)
    anchored = all(part is not None for part in anchor_parts)
    if any(part is not None for part in anchor_parts) and not anchored:
        raise ValueError("anchor_gradient, anchor_zero and anchor_eval must be supplied together")
    if anchored:
        if (
            not isinstance(anchor_gradient, torch.Tensor)
            or anchor_gradient.shape != gradient.shape
            or anchor_gradient.device != device
            or not anchor_gradient.is_floating_point()
        ):
            raise ValueError("anchor_gradient must be a floating-point tensor matching gradient")
        anchor_gradient = anchor_gradient.detach()
        anchor_zero = _vector(anchor_zero, "anchor_zero", size, device)

    actions = torch.zeros_like(gradient)
    after_loss = zero_loss.clone()
    accepted = torch.zeros(size, dtype=torch.bool, device=device)
    anchor_final = anchor_zero.clone() if anchored else None
    radius64 = radius.to(torch.float64)
    loss64 = zero_loss.to(torch.float64)

    if (support & (~torch.isfinite(radius64) | (radius64 < 0))).any():
        raise ValueError("supported radii must be finite and nonnegative")
    eligible = support & (radius64 > 0)
    if (eligible & ~torch.isfinite(loss64)).any():
        raise ValueError("eligible zero losses must be finite")
    eligible &= loss64 != 0
    if (eligible & ~torch.isfinite(gradient).all(dim=1)).any():
        raise ValueError("eligible gradients must be finite")

    gradient64 = gradient.to(torch.float64)
    gradient_scale = gradient64.abs().amax(dim=1)
    gradient_scale = torch.where(torch.isfinite(gradient_scale) & (gradient_scale > 0), gradient_scale, 1.0)
    scaled_gradient = gradient64 / gradient_scale[:, None]
    original_norm = _norm(scaled_gradient)
    direction = scaled_gradient
    if anchored:
        anchor64 = anchor_gradient.to(torch.float64)
        anchor_zero64 = anchor_zero.to(torch.float64)
        if (eligible & (~torch.isfinite(anchor64).all(dim=1) | ~torch.isfinite(anchor_zero64))).any():
            raise ValueError("eligible anchor inputs must be finite")
        anchor_scale = anchor64.abs().amax(dim=1)
        anchor_scale = torch.where(torch.isfinite(anchor_scale) & (anchor_scale > 0), anchor_scale, 1.0)
        scaled_anchor = anchor64 / anchor_scale[:, None]
        anchor_norm = _norm(scaled_anchor)
        anchor_unit = torch.zeros_like(anchor64)
        nonzero_anchor = anchor_norm > 0
        anchor_unit[nonzero_anchor] = scaled_anchor[nonzero_anchor] / anchor_norm[nonzero_anchor, None]
        direction = scaled_gradient - (scaled_gradient * anchor_unit).sum(dim=1, keepdim=True) * anchor_unit

    direction_norm = _norm(direction)
    eligible &= torch.isfinite(original_norm) & torch.isfinite(direction_norm) & (direction_norm > 0)
    if anchored:
        eligible &= direction_norm >= 1e-10 * original_norm
    unit_direction = torch.zeros_like(direction)
    unit_direction[eligible] = -direction[eligible] / direction_norm[eligible, None]
    pending = eligible.clone()

    for fraction in (1.0, 0.5, 0.25, 0.125):
        indices = torch.nonzero(pending, as_tuple=True)[0]
        if indices.numel() == 0:
            break
        candidate = _bound(unit_direction[indices] * (radius64[indices] * fraction)[:, None], radius64[indices], gradient.dtype)
        usable = (
            torch.isfinite(candidate).all(dim=1)
            & (candidate != 0).any(dim=1)
            & (_norm(candidate.to(torch.float64)) <= radius64[indices])
        )
        indices, candidate = indices[usable], candidate[usable]
        if indices.numel() == 0:
            continue

        if anchored:
            score = _scores(anchor_eval, indices, candidate, "anchor_eval").to(torch.float64).clone()
            for _ in range(2):
                correct = (
                    torch.isfinite(score)
                    & ((score - anchor_zero64[indices]).abs() > 1e-4)
                    & (anchor_norm[indices] > 0)
                )
                positions = torch.nonzero(correct, as_tuple=True)[0]
                if positions.numel() == 0:
                    break
                rows = indices[positions]
                amount = (anchor_zero64[rows] - score[positions]) / anchor_norm[rows] / anchor_scale[rows]
                corrected = candidate[positions].to(torch.float64) + amount[:, None] * anchor_unit[rows]
                finite = torch.isfinite(corrected).all(dim=1) & torch.isfinite(amount)
                score[positions[~finite]] = float("nan")
                positions, rows, corrected = positions[finite], rows[finite], corrected[finite]
                if positions.numel() == 0:
                    continue
                corrected = _bound(corrected, radius64[rows], gradient.dtype)
                finite = (
                    torch.isfinite(corrected).all(dim=1)
                    & (corrected != 0).any(dim=1)
                    & (_norm(corrected.to(torch.float64)) <= radius64[rows])
                )
                score[positions[~finite]] = float("nan")
                positions, rows, corrected = positions[finite], rows[finite], corrected[finite]
                if positions.numel() == 0:
                    continue
                candidate[positions] = corrected
                score[positions] = _scores(anchor_eval, rows, corrected, "anchor_eval").to(torch.float64)

            feasible = (
                torch.isfinite(score)
                & ((score - anchor_zero64[indices]).abs() <= 1e-4)
                & (_norm(candidate.to(torch.float64)) <= radius64[indices])
            )
            indices, candidate, score = indices[feasible], candidate[feasible], score[feasible]
            if indices.numel() == 0:
                continue

        measured = _scores(evaluate, indices, candidate, "evaluate")
        tolerance = torch.maximum(torch.full_like(loss64[indices], 1e-12), 1e-4 * loss64[indices])
        improved = torch.isfinite(measured) & ((loss64[indices] - measured.to(torch.float64)) > tolerance)
        rows = indices[improved]
        if rows.numel() == 0:
            continue
        actions[rows] = candidate[improved]
        after_loss = after_loss.to(torch.promote_types(after_loss.dtype, measured.dtype))
        after_loss[rows] = measured[improved].to(after_loss.dtype)
        accepted[rows] = True
        pending[rows] = False
        if anchored:
            anchor_final = anchor_final.to(torch.promote_types(anchor_final.dtype, score.dtype))
            anchor_final[rows] = score[improved]

    return actions.detach(), after_loss.detach(), accepted.detach(), None if anchor_final is None else anchor_final.detach()
