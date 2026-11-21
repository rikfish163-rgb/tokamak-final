"""Fit two matching readout copies on the exact twenty ended EAST shots."""

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..shot_objective_alignment_20261023.learning import candidate_ends
from ..simple_online import _durable_json
from .model import token_basis, context_at, context_trace, risk_with_shifts

ROUTES = ("aligned_last", "per_token")


@torch.no_grad()
def prepare_cache(model, targets):
    cache = {}
    direction_means = []
    for shot in targets:
        signal = {k: shot[k] for k in ("values", "available")}
        means, complete = patch_signal(signal, model.base.center, model.base.scale)
        trace = context_trace(model.memory, means, complete)
        direction_means.append(trace.mean(0))
        ends = candidate_ends(shot)
        hs, vs, ds, supports = [], [], [], []
        for lo in range(0, len(ends), 32):
            e = ends[lo : lo + 32]
            x, a = batch(signal, e, model.base.center.device)
            h, valid = model.tokens(x, a)
            basis, supported = token_basis(
                model.memory, x, a, model.base.center, model.base.scale
            )
            delta = torch.einsum("btdc,bc->btd", basis, context_at(trace, e))
            hs.append(h)
            vs.append(valid)
            ds.append(delta)
            supports.append(int(supported.sum()))
        cache[shot["shot"]] = dict(
            h=torch.cat(hs),
            valid=torch.cat(vs),
            delta=torch.cat(ds),
            ends=ends.tolist(),
            disruptive=shot["is_disrupt"],
            supported_tokens=sum(supports),
        )
    direction = torch.stack(direction_means).mean(0)
    norm = direction.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("ended-prefix constant context direction is undefined")
    return cache, direction / norm


def training_logits(model, record, weight, route, lo):
    h, v, d = (record[k][lo : lo + 32] for k in ("h", "valid", "delta"))
    shift = F.linear(d, weight) * v[:, :-1, None]
    if route == "aligned_last":
        shift = torch.cat((torch.zeros_like(shift[:, :-1]), shift[:, -1:]), 1)
    elif route != "per_token":
        raise ValueError(route)
    return risk_with_shifts(model, h, v, shift)


def bag_backward(model, record, weight, route, boundary):
    best, best_lo, best_offset = -math.inf, None, None
    for lo in range(0, len(record["h"]), 32):
        # Identical gradient-enabled path and batch shape for peak scan/replay.
        z = training_logits(model, record, weight, route, lo)
        offset = int(z.detach().argmax())
        value = float(z[offset].detach())
        if value > best:
            best, best_lo, best_offset = value, lo, offset
        del z
    z = training_logits(model, record, weight, route, best_lo)
    assert float(z[best_offset].detach()) == best, "peak scan/replay must agree exactly"
    loss = F.softplus(
        boundary - z[best_offset] if record["disruptive"] else z[best_offset] - boundary
    )
    (0.5 * loss).backward()
    return dict(
        loss=float(loss.detach()),
        peak_logit=best,
        peak_end=record["ends"][best_lo + best_offset],
        candidates=len(record["ends"]),
    )


def prepare(model, targets, threshold, root, *, steps=256):
    if root.exists():
        raise FileExistsError(root)
    if (
        len(targets) != 20
        or len({s["shot"] for s in targets}) != 20
        or any(s["device"] != "EAST" for s in targets)
    ):
        raise ValueError("exact20 ended EAST shots required")
    root.mkdir(parents=True)
    (root / "states").mkdir()
    start = time.monotonic()
    model.requires_grad_(False).eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cache, constant = prepare_cache(model, targets)
    _durable_json(
        root / "cache_manifest.json",
        {
            str(s): {k: v for k, v in r.items() if k not in ("h", "valid", "delta")}
            for s, r in cache.items()
        },
    )
    torch.save(constant.cpu(), root / "constant_direction.pt")
    groups = {
        d: [sid for sid, r in cache.items() if r["disruptive"] == d]
        for d in (False, True)
    }
    if not all(groups.values()):
        raise ValueError("both ended classes required")
    rng = np.random.default_rng(2131)
    pairs = [
        [groups[d][int(rng.integers(len(groups[d])))] for d in (False, True)]
        for _ in range(steps)
    ]
    _durable_json(root / "pairs.json", pairs)
    initial = model.memory.output.weight.detach().clone()
    weights = {r: initial.clone().requires_grad_(True) for r in ROUTES}
    optimizers = {r: torch.optim.Adam([w], lr=0.001) for r, w in weights.items()}
    boundary = math.log(threshold / (1 - threshold))
    records = []
    for step, pair in enumerate(pairs):
        if step in (0, 127, 255):
            torch.save(
                {
                    r: dict(
                        weight=w.detach().cpu(), optimizer=optimizers[r].state_dict()
                    )
                    for r, w in weights.items()
                },
                root / "states" / f"{step:04d}.pt",
            )
        record = dict(step=step, pair=pair)
        for route, weight in weights.items():
            optimizers[route].zero_grad(set_to_none=True)
            bags = [
                dict(
                    shot=sid, **bag_backward(model, cache[sid], weight, route, boundary)
                )
                for sid in pair
            ]
            penalty = 1e-4 * (weight - initial).square().sum()
            penalty.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_([weight], 1.0, error_if_nonfinite=True)
            )
            optimizers[route].step()
            if not torch.isfinite(weight).all():
                raise FloatingPointError("nonfinite readout parameter")
            record[route] = dict(
                bags=bags,
                penalty=float(penalty.detach()),
                gradient_norm=norm,
                retained_change_norm=float((weight - initial).norm().detach()),
            )
        records.append(record)
        if (step + 1) % 8 == 0 or step + 1 == steps:
            progress = dict(
                stage="preparation",
                step=step + 1,
                total=steps,
                seconds=time.monotonic() - start,
                last=record,
            )
            _durable_json(root / "progress.json", progress)
            print({k: v for k, v in progress.items() if k != "last"}, flush=True)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    _durable_json(root / "training.json", records)
    result = {r: w.detach().clone() for r, w in weights.items()}
    torch.save({r: w.cpu() for r, w in result.items()}, root / "readouts.pt")
    _durable_json(
        root / "completed.json",
        dict(
            status="COMPLETED",
            steps=steps,
            target_ids=[s["shot"] for s in targets],
            source_and_context_rule_unchanged=True,
            only_output_copies_trained=True,
            seconds=time.monotonic() - start,
        ),
    )
    return result, constant
