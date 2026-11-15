"""Separate history-fit targets, recent targets, and bounded optimization."""

import numpy as np
import torch
from torch.nn import functional as F

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch
from ..local_readout_20261029.learning import (
    History,
    forward_features,
    solve,
    standardized,
)
from .acceptance import search

KINDS = ("true", "reverse", "hold")
MODES = {"prior_true": ("prior", "true")} | {
    f"{history}_{target}": (history, target) for history in KINDS for target in KINDS
}
ARMS = ("cold", *MODES)
CANDIDATE = "true_true"
COMPARATORS = (
    "cold",
    "prior_true",
    "true_reverse",
    "true_hold",
    "reverse_true",
    "hold_true",
)


def prepare(net, prefix, slow, prior, history):
    if set(prefix) != {"values", "available"} or len(prefix["values"]) < 256:
        raise ValueError("arrived signal-only prefix required")
    c = len(prefix["values"])
    x, a, y, mask, origins = forecast_examples(
        prefix, c, net.base.cfg, net.base.center.device
    )
    order = np.argsort(origins).tolist()
    origins = [origins[i] for i in order]
    x, a, y, mask = [v[order] for v in (x, a, y, mask)]
    mask = mask & torch.isfinite(y)
    valid = mask.all(1) & a[:, -1] & torch.isfinite(x[:, -1]) & prior.active[None]
    wanted = (
        standardized(net, y, mask) - standardized(net, x[:, -1], a[:, -1])[:, None]
    ).detach()
    with torch.no_grad():
        _, h = forward_features(net, x, a, x.new_zeros(len(x), 8192), slow)
    for i in range(2):
        history.add(h[i], wanted[i], valid[i], origins[i])
    heads = {"prior": prior}
    for kind in KINDS:
        heads[kind], counts = solve(prior, history, kind)
    receipt = dict(
        clock=c,
        origins=origins,
        calibration_last_target=history.rows[-1][3] + 25,
        gradient_first_target=origins[2] + 1,
        gradient_last_target=origins[-1] + 25,
        history_fit_rows=len(history.rows),
        fit_counts_max=max(counts),
        readout_delta_norm=float((heads["true"].coef - prior.coef).norm()),
    )
    if receipt["calibration_last_target"] >= receipt["gradient_first_target"]:
        raise ValueError("fit/gradient targets overlap")
    return dict(
        x=x[2:],
        a=a[2:],
        wanted=wanted[2:].to(prior.coef.dtype),
        valid=valid[2:, None].expand(2, 25, 23),
        heads=heads,
        receipt=receipt,
        pending_rows=[(h[i], wanted[i], valid[i], origins[i]) for i in range(2, 4)],
    )


def target(values, kind):
    if kind == "true":
        return values
    if kind == "reverse":
        return values.flip(1)
    if kind == "hold":
        return torch.zeros_like(values)
    raise ValueError("unknown factorial target")


def feedback(net, prefix, slow, prior, history):
    d = prepare(net, prefix, slow, prior, history)
    x, a, mask = d["x"], d["a"], d["valid"]
    heads = {arm: d["heads"][fit] for arm, (fit, _) in MODES.items()}
    wanted = {arm: target(d["wanted"], kind) for arm, (_, kind) in MODES.items()}
    states, records = {}, {}
    if not mask.any():
        for arm in MODES:
            states[arm] = x.new_zeros(8192)
            records[arm] = dict(
                observed=0,
                trials=0,
                factor=0.0,
                reason="no_observed_target",
                loss_before=None,
                loss_after=None,
                proposed_norm=0.0,
                accepted_norm=0.0,
                gradient_norm=0.0,
                directional_dot=0.0,
                proposed_directional_dot=0.0,
                trial_losses=[],
            )
    else:
        # All ten gradients share one forward/backward, but distinct state rows.
        # Summing independent arm losses does not average or mix their gradients.
        zero = x.new_zeros(len(MODES), 8192, requires_grad=True)
        _, h = forward_features(
            net,
            x.repeat(len(MODES), 1, 1),
            a.repeat(len(MODES), 1, 1),
            zero.repeat_interleave(2, dim=0),
            slow,
        )
        h = h.reshape(len(MODES), 2, 64)
        losses = torch.stack(
            [
                F.smooth_l1_loss(heads[arm].predict(h[i])[mask], wanted[arm][mask])
                for i, arm in enumerate(MODES)
            ]
        )
        (gradients,) = torch.autograd.grad(losses.sum(), zero)
        for i, arm in enumerate(MODES):
            g = gradients[i : i + 1].detach()
            proposal = -0.05 * g / (g.square().sum() + 1e-16).sqrt()

            def objective(state):
                _, features = forward_features(net, x, a, state.expand(2, -1), slow)
                return F.smooth_l1_loss(
                    heads[arm].predict(features)[mask], wanted[arm][mask]
                )

            # Re-evaluate zero in the same two-row numerical forward as trials.
            # Batch-shape differences must not masquerade as a tiny decrease.
            with torch.no_grad():
                loss_zero = objective(torch.zeros_like(proposal))
            state, r = search(objective, proposal, g, loss_zero)
            r["observed"] = int(mask.sum())
            states[arm] = state[0]
            records[arm] = r
    for row in d["pending_rows"]:
        history.add(*row)
    receipt = dict(d["receipt"], history_rows_after=len(history.rows), arms=records)
    return states, receipt, heads


def predict(net, signal, slow, prior):
    if set(signal) != {"values", "available"}:
        raise ValueError("signal-only prediction required")
    states = {arm: net.base.center.new_zeros(8192) for arm in ARMS}
    curves = {arm: [] for arm in ARMS}
    info = {
        arm: dict(
            opportunities=0,
            updates=0,
            decreased=0,
            observed=0,
            max_norm=0.0,
            trials=0,
            loss_before_sum=0.0,
            loss_after_sum=0.0,
            accepted_norm_sum=0.0,
        )
        for arm in MODES
    }
    info["feedback"] = []
    aux, history = [], History()
    eligible = np.arange(256, len(signal["values"]) - 24, 128)
    sampled = (
        set(
            eligible[
                np.linspace(0, len(eligible) - 1, min(4, len(eligible))).astype(int)
            ].tolist()
        )
        if len(eligible)
        else set()
    )
    for lo in range(0, len(signal["values"]), 32):
        hi = min(lo + 32, len(signal["values"]))
        x, a = batch(signal, range(lo, hi), net.base.center.device)
        with torch.no_grad():
            p = net(
                x.repeat(len(ARMS), 1, 1),
                a.repeat(len(ARMS), 1, 1),
                torch.cat([states[k].expand(len(x), -1) for k in ARMS]),
                slow,
                "qk",
                forecast=False,
            )["logit"]
            p = p.sigmoid().reshape(len(ARMS), len(x)).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("nonfinite risk")
        for arm, v in zip(ARMS, p):
            curves[arm].append(v)
        if hi >= 256 and hi % 128 == 0 and hi < len(signal["values"]):
            deltas, r, heads = feedback(
                net, {k: v[:hi] for k, v in signal.items()}, slow, prior, history
            )
            states.update(deltas)
            info["feedback"].append(r)
            for arm, d in r["arms"].items():
                q = info[arm]
                q["opportunities"] += 1
                q["updates"] += int(d["accepted_norm"] > 0)
                q["observed"] += d["observed"]
                q["trials"] += d["trials"]
                q["max_norm"] = max(q["max_norm"], d["accepted_norm"])
                q["accepted_norm_sum"] += d["accepted_norm"]
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
                    for arm in MODES:
                        _, after = forward_features(
                            net, cx, ca, states[arm][None], slow
                        )
                        row[arm + "_before"] = (
                            (heads[arm].predict(before)[0] + last).cpu().numpy()
                        )
                        row[arm + "_after"] = (
                            (heads[arm].predict(after)[0] + last).cpu().numpy()
                        )
                aux.append(row)
    return {k: np.concatenate(v) for k, v in curves.items()}, info, aux
