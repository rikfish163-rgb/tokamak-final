"""Fit older auxiliary observations, then use disjoint newer neural feedback."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS = ("cold", "frozen_readout", "local_readout", "local_reverse", "local_hold")
MODES = dict(
    frozen_readout="true",
    local_readout="true",
    local_reverse="reverse",
    local_hold="hold",
)


@dataclass
class Head:
    mean: torch.Tensor
    scale: torch.Tensor
    coef: torch.Tensor
    active: torch.Tensor

    def features(self, h):
        x = (h.to(self.mean.dtype)[:, None, :] - self.mean[None]) / self.scale[None]
        return torch.cat((x, x.new_ones(len(x), 23, 1)), -1)

    def predict(self, h):
        return torch.einsum("bcd,cdh->bhc", self.features(h), self.coef)


def load_head(path, device):
    mean, scale, coef = np.zeros((23, 64)), np.ones((23, 64)), np.zeros((23, 65, 25))
    active = np.zeros(23, bool)
    with np.load(path, allow_pickle=False) as z:
        for c in range(23):
            key = f"last_token_{c}_"
            if key + "coef" not in z:
                continue  # Explicitly unestimable; excluded from every loss.
            active[c] = True
            mean[c], scale[c] = z[key + "mean"], z[key + "scale"]
            coef[c, :64] = z[key + "coef"][:, 26:]
            coef[c, 64] = z[key + "target_mean"][26:]
    if active[2] or active[10] or not active.any() or (scale <= 0).any():
        raise ValueError("cold20 last-token increment prior required")
    return Head(
        *[torch.as_tensor(v, device=device) for v in (mean, scale, coef, active)]
    )


class History:
    def __init__(self):
        self.rows = deque(maxlen=16)

    def add(self, h, y, mask, origin):
        if (
            h.shape != (64,)
            or y.shape != (25, 23)
            or mask.shape != (23,)
            or mask.dtype != torch.bool
        ):
            raise ValueError(
                "one finite feature and complete-channel target row required"
            )
        if not torch.isfinite(h).all() or not torch.isfinite(y[:, mask]).all():
            raise ValueError("nonfinite observed row")
        if self.rows and origin <= self.rows[-1][3]:
            raise ValueError("strict chronological rows; duplicate replay forbidden")
        self.rows.append(
            (h.detach().clone(), y.detach().clone(), mask.detach().clone(), int(origin))
        )


@torch.no_grad()
def solve(prior, history, mode):
    if mode not in ("true", "reverse", "hold"):
        raise ValueError("explicit target control required")
    if not history.rows:
        return prior, [
            0
        ] * 23  # No observations: exact supplied prior, not fake zero labels.
    h = torch.stack([r[0] for r in history.rows])
    y = torch.stack([r[1] for r in history.rows]).to(prior.coef.dtype)
    m = torch.stack([r[2] for r in history.rows]) & prior.active[None]
    if mode == "reverse":
        y = y.flip(1)
    if mode == "hold":
        y = torch.zeros_like(y)
    x = prior.features(h).permute(1, 0, 2) * m.T[:, :, None]
    target = y.permute(2, 0, 1).masked_fill(~m.T[:, :, None], 0)
    rhs = target - x @ prior.coef
    system = (
        x @ x.transpose(1, 2) + torch.eye(len(h), device=x.device, dtype=x.dtype)[None]
    )
    delta = x.transpose(1, 2) @ torch.linalg.solve(system, rhs)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite local readout")
    return Head(
        prior.mean, prior.scale, (prior.coef + delta).detach(), prior.active
    ), m.sum(0).tolist()


def forward_features(net, x, a, state, slow):
    captured = []
    handle = net.base.forecast_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        result = net(x, a, state, slow, "qk")
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("one actual feature capture required")
    return result["logit"], captured[0].reshape(len(x), 17, 64)[:, -2]


def standardized(net, x, a):
    good = a & torch.isfinite(x)
    return torch.asinh(
        (torch.where(good, x, net.base.center) - net.base.center) / net.base.scale
    )


def feedback(net, prefix, slow, prior, history):
    if set(prefix) != {"values", "available"} or len(prefix["values"]) < 256:
        raise ValueError("only arrived signal prefix allowed")
    count = len(prefix["values"])
    x, a, target, mask, origins = forecast_examples(
        prefix, count, net.base.cfg, net.base.center.device
    )
    order = np.argsort(origins).tolist()
    origins = [origins[i] for i in order]
    x, a, target, mask = [v[order] for v in (x, a, target, mask)]
    mask = mask & torch.isfinite(target)
    valid = mask.all(1) & a[:, -1] & torch.isfinite(x[:, -1]) & prior.active[None]
    current = standardized(net, x[:, -1], a[:, -1]).detach()
    truth = standardized(net, target, mask).detach()
    increments = truth - current[:, None]
    with torch.no_grad():
        _, h = forward_features(net, x, a, x.new_zeros(len(x), 8192), slow)
    for i in range(2):
        history.add(h[i], increments[i], valid[i], origins[i])
    last_calibration = history.rows[-1][3] + 25
    if last_calibration >= origins[2] + 1:
        raise AssertionError("fit/gradient target overlap")
    local, counts = solve(prior, history, "true")
    reverse, _ = solve(prior, history, "reverse")
    hold, _ = solve(prior, history, "hold")
    heads = dict(
        frozen_readout=prior,
        local_readout=local,
        local_reverse=reverse,
        local_hold=hold,
    )
    states, details = {}, {}
    v = valid[2:, None].expand(2, 25, 23)
    for name, mode in MODES.items():
        wanted = increments[2:].to(prior.coef.dtype)
        if mode == "reverse":
            wanted = wanted.flip(1)
        if mode == "hold":
            wanted = torch.zeros_like(wanted)
        state = x.new_zeros(1, 8192, requires_grad=True)
        if not v.any():
            states[name] = state[0].detach()
            details[name] = dict(
                observed=0,
                norm=0.0,
                loss_before=None,
                loss_after=None,
                reason="no_observed_recent_channel",
            )
            continue
        _, features = forward_features(net, x[2:], a[2:], state.expand(2, -1), slow)
        loss = F.smooth_l1_loss(heads[name].predict(features)[v], wanted[v])
        (g,) = torch.autograd.grad(loss, state)
        delta = -0.05 * g / (g.square().sum() + 1e-16).sqrt()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite Q/K feedback")
        with torch.no_grad():
            _, after_h = forward_features(net, x[2:], a[2:], delta.expand(2, -1), slow)
            after = F.smooth_l1_loss(heads[name].predict(after_h)[v], wanted[v])
        states[name] = delta[0].detach()
        details[name] = dict(
            observed=int(v.sum()),
            norm=float(delta.norm()),
            loss_before=float(loss.detach()),
            loss_after=float(after),
        )
    before_rows = len(history.rows)
    for i in range(2, 4):
        history.add(h[i], increments[i], valid[i], origins[i])
    receipt = dict(
        clock=count,
        origins=origins,
        calibration_last_target=last_calibration,
        gradient_first_target=origins[2] + 1,
        gradient_last_target=origins[-1] + 25,
        history_fit_rows=before_rows,
        fit_counts_max=max(counts),
        history_rows_after=len(history.rows),
        readout_delta_norm=float((local.coef - prior.coef).norm()),
        arms=details,
    )
    return states, receipt, heads


def predict(net, signal, slow, prior):
    if set(signal) != {"values", "available"}:
        raise ValueError("signal-only prediction required")
    history = History()
    states = {k: net.base.center.new_zeros(8192) for k in ARMS}
    curves = {k: [] for k in ARMS}
    info = {
        k: dict(
            opportunities=0,
            updates=0,
            decreased=0,
            observed=0,
            max_norm=0.0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
        )
        for k in MODES
    }
    info["feedback"] = []
    aux = []
    eligible = list(range(256, len(signal["values"]) - 24, 128))
    sampled = (
        set(
            np.array(eligible)[
                np.linspace(0, len(eligible) - 1, min(4, len(eligible))).astype(int)
            ].tolist()
        )
        if eligible
        else set()
    )
    for lo in range(0, len(signal["values"]), 32):
        hi = min(lo + 32, len(signal["values"]))
        x, a = batch(signal, range(lo, hi), net.base.center.device)
        with torch.no_grad():
            p = (
                net(
                    x.repeat(len(ARMS), 1, 1),
                    a.repeat(len(ARMS), 1, 1),
                    torch.cat([states[k].expand(len(x), -1) for k in ARMS]),
                    slow,
                    "qk",
                    forecast=False,
                )["logit"]
                .sigmoid()
                .reshape(len(ARMS), len(x))
                .cpu()
                .numpy()
            )
        if not np.isfinite(p).all():
            raise FloatingPointError("nonfinite risk")
        for name, v in zip(ARMS, p):
            curves[name].append(v)
        if hi >= 256 and hi % 128 == 0 and hi < len(signal["values"]):
            deltas, r, heads = feedback(
                net, {k: v[:hi] for k, v in signal.items()}, slow, prior, history
            )
            states.update(deltas)
            info["feedback"].append({k: v for k, v in r.items() if k != "arms"})
            for name, d in r["arms"].items():
                q = info[name]
                q["opportunities"] += 1
                q["updates"] += int(d["norm"] > 0)
                q["observed"] += d["observed"]
                q["max_norm"] = max(q["max_norm"], d["norm"])
                if d["observed"]:
                    q["loss_before_sum"] += d["loss_before"]
                    q["loss_after_sum"] += d["loss_after"]
                    q["decreased"] += int(d["loss_after"] < d["loss_before"])
            if hi in sampled:
                cx, ca = batch(signal, [hi - 1], net.base.center.device)
                with torch.no_grad():
                    _, before = forward_features(
                        net, cx, ca, cx.new_zeros(1, 8192), slow
                    )
                    last = standardized(net, cx[:, -1], ca[:, -1])[0]
                    row = dict(
                        clock=hi,
                        last=last.cpu().numpy(),
                        available=(ca[0, -1] & torch.isfinite(cx[0, -1]) & prior.active)
                        .cpu()
                        .numpy(),
                    )
                    for name in MODES:
                        _, after = forward_features(
                            net, cx, ca, states[name][None], slow
                        )
                        row[name + "_before"] = (
                            (heads[name].predict(before)[0] + last).cpu().numpy()
                        )
                        row[name + "_after"] = (
                            (heads[name].predict(after)[0] + last).cpu().numpy()
                        )
                aux.append(row)  # Future targets have not been read here.
    return {k: np.concatenate(v) for k, v in curves.items()}, info, aux
