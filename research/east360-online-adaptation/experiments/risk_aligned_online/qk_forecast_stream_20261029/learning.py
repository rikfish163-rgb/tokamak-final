"""Full-matrix Q/K updates driven only by arrived diagnostic targets."""

import numpy as np
import torch

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch

ARMS = (
    "cold",
    "original_qk",
    "source_forecast",
    "source_reverse",
    "source_hold",
    "cold_forecast",
    "cold_reverse",
    "cold_hold",
)
UPDATES = dict(
    original_qk=("original", "true"),
    source_forecast=("source", "true"),
    source_reverse=("source", "reverse"),
    source_hold=("source", "hold"),
    cold_forecast=("cold", "true"),
    cold_reverse=("cold", "reverse"),
    cold_hold=("cold", "hold"),
)
STEP = 0.05


def arrived(net, prefix, mode):
    if set(prefix) != {"values", "available"} or len(prefix["values"]) < 256:
        raise ValueError("signal-only matured prefix required")
    if mode not in ("true", "reverse", "hold"):
        raise ValueError("declared feedback control required")
    count = len(prefix["values"])
    x, a, target, mask, origins = forecast_examples(
        prefix, count, net.base.cfg, net.base.center.device
    )
    mask = mask & torch.isfinite(target) & a[:, -1:] & torch.isfinite(x[:, -1:])
    mask[:, :, 10] = False
    if mode == "reverse":
        target, mask = target.flip(1), mask.flip(1)
    elif mode == "hold":
        target = x[:, -1:].expand_as(target)
    return x, a, target, mask, origins


def feedback_step(net, prefix, slow, mode):
    x, a, target, mask, origins = arrived(net, prefix, mode)
    state = x.new_zeros(1, 8192, requires_grad=True)
    pred = net(x, a, state.expand(len(x), -1), slow, "qk")["forecast"]
    loss = net.base.forecast_loss(pred, target, mask)
    (gradient,) = torch.autograd.grad(loss, state)
    delta = -STEP * gradient / (gradient.square().sum() + 1e-16).sqrt()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("nonfinite Q/K proposal")
    with torch.no_grad():
        after = net.base.forecast_loss(
            net(x, a, delta.expand(len(x), -1), slow, "qk")["forecast"], target, mask
        )
    return delta[0].detach(), dict(
        count=len(prefix["values"]),
        last_target=len(prefix["values"]) - 1,
        origins=origins,
        observed=int(mask.sum()),
        norm=float(delta.detach().norm()),
        loss_before=float(loss.detach()),
        loss_after=float(after),
    )


def predict(nets, signal, slow):
    if set(nets) != {"original", "source", "cold"} or set(signal) != {
        "values",
        "available",
    }:
        raise ValueError("three fixed auxiliary heads and signal-only shot required")
    net = nets["original"]
    device = net.base.center.device
    states = {k: net.base.center.new_zeros(8192) for k in ARMS}
    scores = {k: [] for k in ARMS}
    info = {
        k: dict(
            updates=0,
            opportunities=0,
            observed_targets=0,
            decreased=0,
            max_matrix_norm=0.0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
            last_observations=[],
        )
        for k in UPDATES
    }
    for lo in range(0, len(signal["values"]), 32):
        hi = min(lo + 32, len(signal["values"]))
        x, a = batch(signal, range(lo, hi), device)
        with torch.no_grad():
            z = net(
                x.repeat(len(ARMS), 1, 1),
                a.repeat(len(ARMS), 1, 1),
                torch.cat([states[k].expand(len(x), -1) for k in ARMS]),
                slow,
                "qk",
                forecast=False,
            )["logit"]
            p = z.sigmoid().reshape(len(ARMS), len(x)).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("nonfinite online risk")
        for k, v in zip(ARMS, p):
            scores[k].append(v)
        if hi >= 256 and hi % 128 == 0 and hi < len(signal["values"]):
            prefix = {k: v[:hi] for k, v in signal.items()}
            for arm, (name, mode) in UPDATES.items():
                states[arm], r = feedback_step(nets[name], prefix, slow, mode)
                q = info[arm]
                q["updates"] += int(r["norm"] > 0)
                q["opportunities"] += 1
                q["observed_targets"] += r["observed"]
                q["decreased"] += int(r["loss_after"] < r["loss_before"])
                q["max_matrix_norm"] = max(q["max_matrix_norm"], r["norm"])
                q["loss_before_sum"] += r["loss_before"]
                q["loss_after_sum"] += r["loss_after"]
                q["last_observations"].append(r["last_target"])
    return {k: np.concatenate(v) for k, v in scores.items()}, info


def fit_head(net, records, slow, *, steps=256):
    if (
        not records
        or type(steps) is not int
        or steps < 1
        or len({s["shot"] for s in records}) != len(records)
        or any(
            set(s) != {"shot", "device", "values", "available"} or s["device"] != "EAST"
            for s in records
        )
    ):
        raise ValueError("distinct ended EAST signal-only records required")
    rng = np.random.default_rng(131)
    pools = []
    excluded = []
    for i, s in enumerate(records):
        clocks = list(range(256, len(s["values"]) + 1, 128))
        clocks = [c for c in clocks if s["available"][c - 100 : c].any()]
        if clocks:
            pools.append((i, clocks))
        else:
            excluded.append(s["shot"])
    if not pools:
        raise ValueError("no complete observed feedback for head fitting")
    frozen = {
        k: v.cpu().clone()
        for k, v in net.state_dict().items()
        if "forecast_head." not in k
    }
    net.requires_grad_(False)
    net.base.forecast_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        net.base.forecast_head.parameters(), lr=1e-4, weight_decay=0
    )
    history = []
    receipts = []
    for step in range(steps):
        examples = []
        for _ in range(4):
            i, clocks = pools[int(rng.integers(len(pools)))]
            c = int(rng.choice(clocks))
            s = records[i]
            x, a, target, mask, origins = arrived(
                net, {k: s[k][:c] for k in ("values", "available")}, "true"
            )
            examples.append((x, a, target, mask))
            receipts.append(
                dict(shot=s["shot"], count=c, last_target=c - 1, origins=origins)
            )
        x, a, target, mask = [torch.cat(v) for v in zip(*examples)]
        optimizer.zero_grad(set_to_none=True)
        pred = net(x, a, x.new_zeros(len(x), 8192), slow, "qk")["forecast"]
        loss = net.base.forecast_loss(pred, target, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite cold-head loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            net.base.forecast_head.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm))
        )
    net.requires_grad_(False)
    if any(
        not torch.equal(v.cpu(), frozen[k])
        for k, v in net.state_dict().items()
        if k in frozen
    ):
        raise ValueError("head fitting changed classifier")
    return dict(
        history=history,
        receipts=receipts,
        fit_ids=[s["shot"] for s in records],
        excluded=excluded,
        eligible_clocks={str(records[i]["shot"]): c for i, c in pools},
        steps=steps,
    )
