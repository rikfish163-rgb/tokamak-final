"""Ended-shot auxiliary-output attribution, not input importance or causality."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23
from ..critical_shot_meta_20261102.objectives import margin_loss
from ..direct_decoder_meta_20261102.core import prediction
from ..gate_loss_audit_20261101.core import observed
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss, metric_direction
from ..lora_stream_20261102.core import forward
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .direction_audit import macro
from .functional_stream import load
from .run import LOCAL, collect

PLAN = io23.ROOT / "EAST_CHANNEL_DIRECTION_PLAN_20261102.md"


def channel_losses(model, predicted, raw_truth, mask):
    clean = torch.where(mask, raw_truth.detach(), model.base.center)
    target = torch.asinh((clean - model.base.center) / model.base.scale)
    p, y = observed(predicted, target, mask)
    n = mask.sum(1, keepdim=True).clamp_min(1)
    p = p - p.sum(1, keepdim=True) / n
    y = y - y.sum(1, keepdim=True) / n
    loss = F.smooth_l1_loss(p, y, reduction="none").masked_fill(~mask, 0)
    return loss.sum((0, 1)) / mask.sum().clamp_min(1)


def risk_contributions(pack, gradients, risk):
    if gradients.shape != (23, 4, 64) or risk.shape != (4, 64):
        raise ValueError(
            "23 auxiliary channel gradients and one first-FFN risk gradient required"
        )
    valid = pack["valid"][:, :16]
    clean = pack["phi"].double().masked_fill(~valid[:, :, None], 0)
    count = valid.sum().clamp_min(1) * 64
    covariance = clean.reshape(-1, 4).T @ clean.reshape(-1, 4) / count
    ridge = 0.01 * covariance.trace() / 4 + 1e-8
    metric = covariance + ridge * torch.eye(4, device=clean.device, dtype=clean.dtype)
    individual = -torch.linalg.solve(metric, gradients.double())
    total = -torch.linalg.solve(metric, gradients.sum(0).double())
    squared = (clean @ total).square().sum() / count
    factor = 0.05 / squared.clamp_min(0.05**2).sqrt()
    return (individual * risk.double()).sum((1, 2)) * factor


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    model, head, basis, cold, previous, _, _ = load()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    local = read(LOCAL / "protocol.json")
    examples, _ = collect(model, basis, local)
    root.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    write(
        root / "protocol.json",
        dict(
            scope="POSTHOC_AUXILIARY_OUTPUT_ATTRIBUTION_NOT_CAUSAL",
            ids=local["ids"],
            input_hashes=local["input_hashes"],
            threshold=previous["threshold"],
            channels=list(io23.RAW23),
            cold_sha256=sha(cold),
            bound_code_sha256=bound,
            suffix_raw_reads=0,
            optimizer_steps=0,
            same20_previously_used_for_cold_and_C=True,
        ),
    )
    rows = []
    max_gradient_error, max_slope_error = 0.0, 0.0
    for i, e in enumerate(examples):
        variable = model.base.center.new_zeros(4, 64).requires_grad_(True)
        z = forward(model, e["later_pack"], variable)["logit"]
        risk = margin_loss(z, previous["threshold"], int(e["disruptive"]))
        (gr,) = torch.autograd.grad(risk, variable)
        for mode in ("real", "reverse", "hold"):
            variable = model.base.center.new_zeros(4, 64).requires_grad_(True)
            pred = prediction(model, e["origin_pack"], variable, head)
            values = channel_losses(model, pred, e["truth"][mode], e["mask"])
            actual_loss = innovation_loss(model, pred, e["truth"][mode], e["mask"])
            torch.testing.assert_close(values.sum(), actual_loss, atol=1e-7, rtol=1e-6)
            (actual_gradient,) = torch.autograd.grad(
                actual_loss, variable, retain_graph=True
            )
            components = torch.stack(
                [
                    torch.autograd.grad(values[c], variable, retain_graph=c < 22)[0]
                    for c in range(23)
                ]
            )
            error = float((components.sum(0) - actual_gradient).abs().max())
            torch.testing.assert_close(
                components.sum(0), actual_gradient, atol=2e-8, rtol=5e-4
            )
            contributions = risk_contributions(e["origin_pack"], components, gr)
            direction = metric_direction(e["origin_pack"], actual_gradient)
            actual_slope = float((gr.double() * direction.double()).sum())
            slope_error = abs(float(contributions.sum()) - actual_slope)
            if slope_error > 1e-6 + 5e-4 * abs(actual_slope):
                raise AssertionError(
                    "channel contributions fail to reconstruct actual metric-step risk slope"
                )
            max_gradient_error, max_slope_error = (
                max(max_gradient_error, error),
                max(max_slope_error, slope_error),
            )
            rows.append(
                dict(
                    **{
                        k: e[k]
                        for k in (
                            "shot",
                            "query",
                            "label",
                            "disruptive",
                            "actual_lead_ms",
                        )
                    },
                    group="safe"
                    if not e["disruptive"]
                    else "danger30_125"
                    if e["actual_lead_ms"] <= 125
                    else "disruptive_early",
                    mode=mode,
                    actual_slope=actual_slope,
                    contributions23=contributions.detach().cpu().tolist(),
                    diagnostic_loss=float(actual_loss.detach()),
                    gradient_reconstruction_error=error,
                    slope_reconstruction_error=slope_error,
                )
            )
        if (i + 1) % 10 == 0:
            print(
                dict(
                    episodes=i + 1, rows=len(rows), seconds=time.monotonic() - started
                ),
                flush=True,
            )
    summary = {}
    for mode in ("real", "reverse", "hold"):
        summary[mode] = {}
        for group in ("safe", "danger30_125", "disruptive_early"):
            local_rows = [r for r in rows if r["mode"] == mode and r["group"] == group]
            values = [
                macro(local_rows, lambda r, c=c: r["contributions23"][c])
                for c in range(23)
            ]
            summary[mode][group] = dict(
                shots=len({r["shot"] for r in local_rows}),
                episodes=len(local_rows),
                channel_shot_macro_contributions=dict(
                    zip(io23.RAW23, values, strict=True)
                ),
                sum_contributions=float(np.sum(values)),
                shot_macro_actual_slope=macro(local_rows, lambda r: r["actual_slope"]),
            )
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
    ) or any(
        not torch.equal(v, head.state_dict()[k].cpu()) for k, v in head_before.items()
    ):
        raise AssertionError("source/cold/head changed")
    if any(sha(Path(path)) != d for path, d in bound.items()):
        raise AssertionError("frozen code changed")
    write(root / "records.json", rows)
    write(root / "summary.json", summary)
    done = dict(
        status="COMPLETED_POSTHOC_DIAGNOSTIC",
        records=len(rows),
        channels=23,
        max_gradient_reconstruction_error=max_gradient_error,
        max_risk_slope_reconstruction_error=max_slope_error,
        source_unchanged=True,
        cold_unchanged=True,
        head_unchanged=True,
        suffix_raw_reads=0,
        optimizer_steps=0,
        online_gain_measured=False,
        seconds=time.monotonic() - started,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
