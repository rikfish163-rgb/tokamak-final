"""Paired feedback clocks and group objectives; no deploy-time group labels."""

import torch

from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..qk_matched_head_20261029.learning import (
    assemble as blocked_assemble,
    paired_risk,
)
from ..simple23 import Config


def assemble(shots, choices, device, clock):
    if clock not in ("blocked", "fresh"):
        raise ValueError("declared feedback clock required")
    e = blocked_assemble(shots, choices, device)
    if clock == "blocked":
        return e
    feedback = []
    for (i, q), receipt in zip(choices, e["receipts"]):
        q = int(q)
        signal = {k: shots[i][k][:q] for k in ("values", "available")}
        x, a, truth, mask, origins = forecast_examples(signal, q, Config(), device)
        if len(origins) != 4:
            raise ValueError("four matured targets required")
        feedback.append((x, a, truth, mask))
        receipt.update(count=q, origins=origins, last_target=q - 1)
    e["feedback"] = {
        k: torch.stack(v) for k, v in zip(("x", "a", "truth", "mask"), zip(*feedback))
    }
    return e


def aggregate_objective(signal, delta, alignment, mode):
    if mode not in ("mean", "worst") or any(
        x.shape != (3,) for x in (signal, delta, alignment)
    ):
        raise ValueError("three fixed group objectives and declared reduction required")
    components = 0.5 * delta + 0.1 * alignment
    return 0.2 * signal.mean() + (
        components.mean() if mode == "mean" else components.max()
    )


def train_step(net, optimizer, e, mode):
    ids = {id(p) for p in net.base.forecast_head.parameters()}
    if (
        len(e["y"]) != 12
        or {id(p) for g in optimizer.param_groups for p in g["params"]} != ids
        or any(p.requires_grad and id(p) not in ids for p in net.parameters())
    ):
        raise ValueError("head-only optimizer and four episodes per group required")
    net.eval()
    optimizer.zero_grad(set_to_none=True)
    results = []
    for lo in (0, 4, 8):
        sub = {k: e[k][lo : lo + 4] for k in ("qx", "qa", "y")}
        sub["feedback"] = {k: v[lo : lo + 4] for k, v in e["feedback"].items()}
        results.append(paired_risk(net, sub, create_graph=True))
    signal = torch.stack([r["signal"].mean() for r in results])
    delta = torch.stack([(r["after"] - r["before"]).mean() for r in results])
    alignment = torch.stack([r["alignment"] for r in results])
    objective = aggregate_objective(signal, delta, alignment, mode)
    if not torch.isfinite(objective):
        raise FloatingPointError("invalid source objective")
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        net.base.forecast_head.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return dict(
        objective=float(objective.detach()),
        signal=signal.detach().cpu().tolist(),
        delta=delta.detach().cpu().tolist(),
        alignment=alignment.detach().cpu().tolist(),
        head_gradient_norm=float(norm),
    )
