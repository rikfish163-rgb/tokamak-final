"""Frozen-network input fitting, after the exact twenty ended stream shots."""

import copy
import math
import time

import numpy as np
import torch

from ..shot_objective_alignment_20261023.learning import candidate_ends
from ..simple_online import _durable_json
from .core import GROUPS, gain_basis, effective_gain, backward_bag

ARMS = ("density_mean", "density_peak", "structured_mean", "structured_peak")


def export_map(reference, theta, basis, scope):
    prior = torch.as_tensor(reference["gain"], device=theta.device, dtype=torch.float64)
    gains = effective_gain(theta, prior, basis).detach().cpu().tolist()
    mapping = copy.deepcopy(reference)
    mapping.update(
        policy="label_fitted_positive_gain",
        gain=gains,
        scope=scope,
        input_parameters=len(theta),
        theta=theta.detach().cpu().tolist(),
        parameter_names=[
            name for name, _ in (GROUPS[:1] if scope == "density" else GROUPS)
        ],
        basis=basis.cpu().tolist(),
        physical_calibration_certified=False,
        neural_parameter_updates=0,
    )
    for c, row in enumerate(mapping["channels"]):
        row["gain"] = gains[c]
        if basis[:, c].any():
            row["status"] = "label_fitted_positive_gain"
    return mapping


def prepare(base, target, mapping, threshold, output, *, steps=256):
    if output.exists():
        raise FileExistsError(output)
    if [s["shot"] for s in target] != mapping["calibration_ids"] or len(target) != 20:
        raise ValueError("exact ended20 in declared order required")
    if any(s["device"] != "EAST" for s in target):
        raise ValueError("EAST-only input adaptation")
    output.mkdir(parents=True)
    (output / "optimizer_states").mkdir()
    initial = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base.requires_grad_(False)
    base.eval()
    by_class = {
        flag: [s for s in target if bool(s["is_disrupt"]) == flag]
        for flag in (False, True)
    }
    if not all(by_class.values()):
        raise ValueError("both ended outcome classes required")
    candidates = {s["shot"]: candidate_ends(s) for s in target}
    _durable_json(
        output / "candidate_endpoints.json",
        {str(k): v.tolist() for k, v in candidates.items()},
    )
    rng = np.random.default_rng(2131)
    pairs = [
        [
            by_class[flag][int(rng.integers(len(by_class[flag])))]["shot"]
            for flag in (False, True)
        ]
        for _ in range(steps)
    ]
    _durable_json(output / "pairs.json", pairs)
    shots = {s["shot"]: s for s in target}
    device = base.center.device
    prior = torch.as_tensor(mapping["gain"], dtype=torch.float64, device=device)
    bases = {name: gain_basis(name.split("_")[0]).to(device) for name in ARMS}
    theta = {
        name: torch.zeros(
            len(bases[name]), dtype=torch.float64, device=device, requires_grad=True
        )
        for name in ARMS
    }
    opts = {name: torch.optim.Adam([theta[name]], lr=0.03) for name in ARMS}
    boundary = math.log(threshold / (1 - threshold))
    trace = []
    start = time.monotonic()
    for step, pair in enumerate(pairs):
        if step in (0, 63, 127, 255):
            torch.save(
                {
                    name: dict(
                        theta=theta[name].detach().cpu(),
                        optimizer=opts[name].state_dict(),
                    )
                    for name in ARMS
                },
                output / "optimizer_states" / f"{step:04d}.pt",
            )
        record = dict(step=step, pair=pair)
        for name in ARMS:
            before = theta[name].detach().cpu().tolist()
            opts[name].zero_grad(set_to_none=True)
            bags = []
            mode = name.split("_")[1]
            for sid in pair:
                s = shots[sid]
                receipt = backward_bag(
                    base,
                    {k: s[k] for k in ("values", "available")},
                    candidates[sid],
                    s["is_disrupt"],
                    boundary,
                    mode,
                    theta[name],
                    prior,
                    bases[name],
                    weight=0.5,
                    device=device,
                )
                bags.append(dict(shot=sid, disruptive=s["is_disrupt"], **receipt))
            penalty = 0.01 * theta[name].square().sum()
            penalty.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [theta[name]], 1.0, error_if_nonfinite=True
                )
            )
            opts[name].step()
            effective_gain(theta[name], prior, bases[name])
            record[name] = dict(
                theta_before=before,
                theta_after=theta[name].detach().cpu().tolist(),
                bags=bags,
                penalty=float(penalty.detach()),
                gradient_norm=norm,
            )
        trace.append(record)
        if (step + 1) % 8 == 0 or step + 1 == steps:
            status = dict(
                step=step + 1,
                total=steps,
                seconds=time.monotonic() - start,
                losses={
                    a: sum(b["loss"] * 0.5 for b in record[a]["bags"])
                    + record[a]["penalty"]
                    for a in ARMS
                },
            )
            _durable_json(output / "progress.json", status)
            print(status, flush=True)
    assert all(torch.equal(v, base.state_dict()[k].cpu()) for k, v in initial.items())
    assert all(p.grad is None for p in base.parameters())
    _durable_json(output / "training.json", trace)
    maps = {
        name: export_map(mapping, theta[name], bases[name], name.split("_")[0])
        for name in ARMS
    }
    for name, value in maps.items():
        _durable_json(output / f"{name}.json", value)
    _durable_json(
        output / "completed.json",
        dict(
            status="COMPLETED",
            steps=steps,
            seconds=time.monotonic() - start,
            source_network_bitwise_unchanged=True,
            source_network_gradients_absent=True,
            target_ids=[s["shot"] for s in target],
            class_counts={str(k): len(v) for k, v in by_class.items()},
        ),
    )
    return maps
