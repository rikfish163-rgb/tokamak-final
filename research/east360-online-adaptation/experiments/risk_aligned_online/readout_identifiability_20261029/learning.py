"""Frozen readout probes; no risk fitting or online decision changes."""

import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch

VIEWS = ("all_tokens", "last_token", "raw8")


def ridge(x, y, ids, penalty, *, device="cuda"):
    x, y, ids = np.asarray(x), np.asarray(y), np.asarray(ids)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or ids.shape != (len(x),)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError("finite fitting rows and positive ridge penalty required")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    weight /= weight.sum()
    xx = torch.as_tensor(x, device=device, dtype=torch.float64)
    yy = torch.as_tensor(y, device=device, dtype=torch.float64)
    w = torch.as_tensor(weight, device=device)
    mean, target_mean = (w[:, None] * xx).sum(0), (w[:, None] * yy).sum(0)
    constant = xx.max(0).values == xx.min(0).values
    # Exact constant columns have no fitted slope. No near-constant cutoff.
    mean = torch.where(constant, xx[0], mean)
    scale = ((xx - mean).square() * w[:, None]).sum(0).sqrt()
    scale = torch.where(constant, torch.ones_like(scale), scale)
    if (scale <= 0).any():
        raise FloatingPointError("nonconstant column has zero numerical variance")
    design = (xx - mean) / scale * w.sqrt()[:, None]
    target = (yy - target_mean) * w.sqrt()[:, None]
    if len(x) >= x.shape[1]:
        system = design.T @ design + penalty * torch.eye(
            x.shape[1], device=device, dtype=xx.dtype
        )
        coef = torch.linalg.solve(system, design.T @ target)
        solver = "primal"
    else:
        system = design @ design.T + penalty * torch.eye(
            len(x), device=device, dtype=xx.dtype
        )
        coef = design.T @ torch.linalg.solve(system, target)
        solver = "dual"
    rhs = design.T @ target
    residual = design.T @ (design @ coef - target) + penalty * coef
    relative = float(
        residual.norm() / (rhs.norm() + penalty * coef.norm()).clamp_min(1e-30)
    )
    if not torch.isfinite(coef).all() or relative > 1e-8:
        raise FloatingPointError("ridge normal equations not solved accurately")
    return dict(
        mean=mean.cpu().numpy(),
        scale=scale.cpu().numpy(),
        target_mean=target_mean.cpu().numpy(),
        coef=coef.cpu().numpy(),
        solver=solver,
        equation_relative_residual=relative,
        penalty=penalty,
        rows=len(x),
        shots=len(counts),
    )


def predict(fit, x):
    return (np.asarray(x) - fit["mean"]) / fit["scale"] @ fit["coef"] + fit[
        "target_mean"
    ]


@torch.no_grad()
def examples(net, signal, clocks, slow):
    if (
        set(signal) != {"values", "available"}
        or not len(clocks)
        or any(c < 256 or c + 25 > len(signal["values"]) for c in clocks)
    ):
        raise ValueError(
            "signal-only complete causal windows and future targets required"
        )
    x, a = batch(signal, [c - 1 for c in clocks], net.base.center.device)
    observed = a & torch.isfinite(x)
    raw = torch.asinh(
        (torch.where(observed, x, net.base.center) - net.base.center) / net.base.scale
    )
    captured = []
    handle = net.base.forecast_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        result = net(x, a, x.new_zeros(len(x), 8192), slow, "qk")
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("exactly one actual forecast readout expected")
    h = captured[0]
    truth = torch.as_tensor(
        np.stack([signal["values"][c : c + 25] for c in clocks]), device=x.device
    )
    valid = torch.as_tensor(
        np.stack([signal["available"][c : c + 25] for c in clocks]), device=x.device
    )
    valid = valid & torch.isfinite(truth)
    mask = observed[:, -8:].all(1) & observed[:, -9] & valid.all(1)
    mask[:, 10] = False
    future = torch.asinh(
        (torch.where(valid, truth, net.base.center) - net.base.center) / net.base.scale
    )
    last = raw[:, -1]
    trend = (
        last[:, None]
        + (raw[:, -1] - raw[:, -9])[:, None]
        / 8
        * torch.arange(1, 26, device=x.device)[None, :, None]
    )
    return {
        k: v.cpu().numpy()
        for k, v in dict(
            all_tokens=h,
            last_token=h.reshape(len(h), 17, 64)[:, -2],
            raw8=torch.cat((raw[:, -8:], observed[:, -8:].to(raw.dtype)), -1).flatten(
                1
            ),
            current=last,
            future=future,
            mask=mask,
            trend=trend,
            original_forecast=result["forecast"],
            risk=result["logit"],
        ).items()
    }


def fit_probes(data, *, device="cuda"):
    fits = {k: [] for k in VIEWS}
    for channel in range(23):
        take = data["mask"][:, channel]
        if not take.any():
            for view in VIEWS:
                fits[view].append(
                    None
                )  # Explicitly unestimable, never a zero prediction.
            continue
        current = data["current"][take, channel : channel + 1]
        future = data["future"][take, :, channel]
        targets = np.concatenate((current, future, future - current), axis=1)
        for view in VIEWS:
            fits[view].append(
                ridge(data[view][take], targets, data["ids"][take], 0.01, device=device)
            )
    return fits


def predictions(fits, data):
    result = {}
    for view in VIEWS:
        out = np.full((len(data["ids"]), 51, 23), np.nan)
        for c, fit in enumerate(fits[view]):
            if fit is not None:
                out[:, :, c] = predict(fit, data[view])
        result[view + "_current"] = out[:, :1]
        result[view + "_absolute"] = out[:, 1:26]
        result[view + "_increment"] = out[:, 26:] + data["current"][:, None]
    return result


def scores(preds, data, support):
    mask = data["mask"] & np.asarray(support)[None]
    result = {}
    for name, values in preds.items():
        target = (
            data["current"][:, None] if name.endswith("_current") else data["future"]
        )
        available = np.broadcast_to(mask[:, None], target.shape)
        if values.shape != target.shape or not np.isfinite(values[available]).all():
            raise ValueError("prediction missing in explicitly supported channel")
        # Neural heads emit float32 and closed-form probes emit float64.
        # Score both in float64 so long channel sums have identical precision.
        err = np.where(
            available, values.astype(np.float64) - target.astype(np.float64), 0.0
        )
        absolute = np.abs(err)
        losses = dict(
            mse=err**2, smooth_l1=np.where(absolute < 1, 0.5 * err**2, absolute - 0.5)
        )
        channel_sums = {
            k: np.where(available, v, 0.0).sum((0, 1)).tolist()
            for k, v in losses.items()
        }
        per_shot = []
        for sid in np.unique(data["ids"]):
            take = data["ids"] == sid
            n = available[take].sum((1, 2))
            usable = n > 0
            if not usable.any():
                continue
            row = dict(shot=int(sid), windows=int(usable.sum()), elements=int(n.sum()))
            for key, loss in losses.items():
                row[key] = float(np.mean(loss[take].sum((1, 2))[usable] / n[usable]))
            per_shot.append(row)
        if not per_shot:
            raise ValueError("no estimable held-out signal elements")
        result[name] = dict(
            shots=len(per_shot),
            windows=sum(v["windows"] for v in per_shot),
            per_shot=per_shot,
            channel_sums=channel_sums,
            channel_counts=available.sum((0, 1)).tolist(),
            **{key: float(np.mean([r[key] for r in per_shot])) for key in losses},
        )
    return result
