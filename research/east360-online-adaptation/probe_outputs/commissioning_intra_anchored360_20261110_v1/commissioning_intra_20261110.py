"""Compare matured raw and risk-weighted motion feedback on the frozen EAST head."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .commissioning_transfer_20261108 import STARTUP, capacity, contrast, predict
from .fast_update import normalized_step
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .paired_motion_20261110 import motion_views
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import FAST_RADIUS, FAST_STEP
from .source_anticipation_20261104.static_stream import timing_decomposition


ARMS = ("raw_h25", "motion_h25", "motion_reversed")
ANCHORED_ARMS = ("anchored_motion", "anchored_reversed")
CAPS = ("0.1", "0.15", "0.2")
HORIZON, START, STRIDE = 25, 127, 32
PARENT_ARM = "motion_target_only"
PARENT_RUN = io23.ROOT / "probe_outputs/commissioning_factorial360_20261108_v1"
CALIBRATION = io23.ROOT / "probe_outputs/commissioning_factorial_calibration360_20261108_v1"


def feedback_blocks(n):
    """(forecast origin, first usable prediction, stop); arrival itself is sealed."""
    for origin in range(START, n - HORIZON - 1, STRIDE):
        first = origin + HORIZON + 1
        yield origin, first, min(n, first + STRIDE)


def _motion_loss(predicted, measured, valid, weights):
    weight = valid * weights
    numerator = (F.smooth_l1_loss(predicted, measured, reduction="none") * weight).sum((1, 2))
    count = weight.sum((1, 2))
    return numerator / count.clamp_min(1e-12), count


def replay_shot(base, coef, signal, static, *, anchored=False):
    """Only signal arrays enter; temporary token gain/bias resets on every call.

    GPU batching is an offline replay optimization. The lookback windows end
    at each prediction endpoint. A forecast at q is recomputed with its own
    prefix and state when the loss matures, which is the same state because
    STRIDE exceeds HORIZON and no update has happened in between.
    """
    arms = ANCHORED_ARMS if anchored else ARMS
    n, a = len(static), len(arms)
    device = coef.device
    if base.cfg.window != START + 1 or base.cfg.horizon != HORIZON or base.cfg.fast_site != "tokens":
        raise ValueError("the locked token-state and H25 model are required")
    x = torch.as_tensor(signal["values"], device=device)
    mask = torch.as_tensor(signal["available"], device=device)
    if len(x) != n or mask.shape != x.shape:
        raise ValueError("matched shot signal and frozen curve required")
    motion = torch.as_tensor(contrast(signal["values"], signal["available"]), device=device)
    windows, masks = [
        F.pad(v, (0, 0, START, 0)).unfold(0, START + 1, 1).transpose(1, 2)
        for v in (x, mask)
    ]
    initial = base.initial_fast(a).detach()
    state = initial.clone()
    curves = torch.as_tensor(static, device=device).expand(a, -1).clone()
    counts = torch.zeros(a, device=device)
    max_norm = torch.zeros_like(counts)
    accumulated = torch.zeros(3, device=device)  # natural forecast, persistence, weight mass
    weights = coef[base.cfg.dim:-1].square()
    steps = 0
    hidden = None

    def capture(_module, args):
        nonlocal hidden
        hidden = args[0]

    hook = base.risk_head.register_forward_pre_hook(capture)
    try:
        for origin, first, stop in feedback_blocks(n):
            candidate = state.detach().requires_grad_(True)
            # The extra branch holds the same origin's frozen forecast.
            forecast_state = torch.cat((candidate, initial[:1])) if anchored else candidate
            b = len(forecast_state)
            forecast = base(windows[origin:origin + 1].expand(b, -1, -1),
                            masks[origin:origin + 1].expand(b, -1, -1),
                            forecast_state)["forecast"]
            truth, observed = x[origin + 1:first], mask[origin + 1:first]
            y = torch.stack((truth, truth.flip(0)))
            available = torch.stack((observed, observed.flip(0)))
            past, past_mask = x[origin - 62:origin + 1], mask[origin - 62:origin + 1]
            p, t, hold, valid = motion_views(
                forecast[a - 2:a], y, available, past.expand(2, -1, -1),
                past_mask.expand(2, -1, -1), base.center, base.scale,
                frozen_forecast=forecast[a:] if anchored else None,
            )
            motion_loss, weight_mass = _motion_loss(p, t, valid, weights)
            hold_loss, _ = _motion_loss(hold, t, valid, weights)
            loss = motion_loss.sum()
            if not anchored:
                loss = base.forecast_loss(forecast[:1], truth[None], observed[None]) + loss
            (gradient,) = torch.autograd.grad(loss, candidate)
            with torch.no_grad():
                # Same state bound and step rule for all feedback objectives.
                new_state = normalized_step(state, initial, gradient, step=FAST_STEP, radius=FAST_RADIUS)
                counts += (new_state != state).any(-1)
                state = new_state
                max_norm = torch.maximum(max_norm, (state - initial).norm(dim=-1))
                accumulated += torch.stack((motion_loss[0] * weight_mass[0],
                                            hold_loss[0] * weight_mass[0], weight_mass[0]))
                length = stop - first
                base(windows[first:stop].repeat(a, 1, 1), masks[first:stop].repeat(a, 1, 1),
                     state.repeat_interleave(length, dim=0))
                u = torch.cat((hidden, motion[first:stop].repeat(a, 1)), dim=-1)
                curves[:, first:stop] = (u @ coef[:-1] + coef[-1]).sigmoid().reshape(a, length)
            steps += 1
    finally:
        hook.remove()
    if not torch.isfinite(curves).all():
        raise FloatingPointError("nonfinite risk curve after temporary updates")
    measured, held, mass = accumulated.detach().cpu().tolist()
    details = {
        arm: dict(changed_steps=int(count), max_state_norm=float(norm), eligible_steps=steps)
        for arm, count, norm in zip(arms, counts.cpu().tolist(), max_norm.cpu().tolist(), strict=True)
    }
    details[arms[-2]].update(forecast_error_sum=measured, hold_error_sum=held, error_weight=mass)
    return dict(zip(arms, curves.cpu().numpy(), strict=True)), details


def load_frozen():
    inputs = load_inputs()
    producer = read(PARENT_RUN / "protocol.json")
    threshold_file = CALIBRATION / "thresholds.json"
    coefs_file = PARENT_RUN / "readouts.npz"
    calibrated = read(threshold_file)
    if (producer["order"] != inputs["ids"] or producer["fit_budget"] != inputs["ids"][:STARTUP]
            or producer["source_sha256"] != PARENT_SHA256
            or calibrated["original"] != producer["threshold"]
            or calibrated["readouts_sha256"] != sha(coefs_file)):
        raise ValueError("the existing paired commissioning model and thresholds are required")
    with np.load(coefs_file, allow_pickle=False) as weights:
        coef = torch.from_numpy(weights[PARENT_ARM].copy()).cuda()
    original = float(producer["threshold"])
    cutoffs = {cap: float(calibrated["thresholds"][PARENT_ARM][cap]) for cap in CAPS}
    if not torch.isfinite(coef).all() or not all(0 < t < 1 for t in (original, *cutoffs.values())):
        raise ValueError("frozen readout and alarm thresholds must be finite")
    bound = dict(readouts_sha256=sha(coefs_file), thresholds_sha256=sha(threshold_file))
    return inputs, coef, original, cutoffs, bound


def run(output, *, anchored=False):
    started = time.monotonic()
    configure()
    inputs, coef, original, cutoffs, bound = load_frozen()
    base, checkpoint = load_parent(DEFAULT_PARENT)
    base.cuda()
    initial = state_digest(base)
    if (coef.shape != (base.cfg.dim + 24,) or checkpoint["threshold"] != original
            or initial != inputs["baseline_protocol"]["source_state_sha256"]):
        raise ValueError("qualified source configuration changed")
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    ids = inputs["ids"]
    arms = ANCHORED_ARMS if anchored else ARMS
    names = ("static", *arms)
    write(output / "protocol.json", dict(
        scope="EAST360_DEVELOPMENT_TEMPORARY_TOKEN_UPDATES_ONLY",
        source_sha256=PARENT_SHA256, frozen_parent_arm=PARENT_ARM, parent_run=str(PARENT_RUN),
        **bound, order=ids, startup=STARTUP, startup_threshold=original, suffix_thresholds=cutoffs,
        modes=names, first_origin=START, full_target_points=HORIZON, update_stride=STRIDE,
        updates_after_arrival=True, step=FAST_STEP, radius=FAST_RADIUS,
        update_weights="squared frozen motion-readout coefficients",
        forecast_coordinates=("persistence_plus_adapted_minus_frozen_origin"
                              if anchored else "native_forecast"),
        reverse_control="only reverse already-arrived future signal and mask within each H25",
        fit_labels=0, persistent_updates=0, producer_sha256=sha(Path(__file__)),
        core_sha256=sha(Path(__file__).with_name("paired_motion_20261110.py")),
    ))
    details, seals = [], {}
    with patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome before all curves sealed")):
        for index, sid in enumerate(ids, 1):
            raw_path = io23.path_for("EAST", sid)
            if sha(raw_path) != inputs["hashes"][str(sid)]:
                raise ValueError(f"EAST input file changed: {sid}")
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            signal = prepare_actual(raw, index, inputs["mapping"])
            prediction = predict(base, signal, None if index <= STARTUP else {PARENT_ARM: coef}, "cuda")
            static = prediction["native" if index <= STARTUP else PARENT_ARM]
            prefix = f"{index:04d}_{sid}"
            with np.load(PARENT_RUN / "shots" / (prefix + ".npz"), allow_pickle=False) as parent:
                np.testing.assert_array_equal(parent["times"], raw["times"])
                np.testing.assert_array_equal(parent[PARENT_ARM], static)
            curves, audit = (replay_shot(base, coef, signal, static, anchored=anchored)
                             if index > STARTUP else ({arm: static for arm in arms}, {}))
            out = output / "shots" / (prefix + ".npz")
            with out.open("xb") as f:
                np.savez_compressed(f, times=raw["times"], static=static, **curves)
                f.flush()
                os.fsync(f.fileno())
            seals[out.name] = sha(out)
            details.append(dict(shot=sid, index=index, arms=audit))
            if index % 20 == 0:
                print(dict(stage="sealed", shots=index, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != initial or any(p.grad is not None for p in base.parameters()):
        raise AssertionError("only local temporary state may change")
    write(output / "seal.json", dict(curve_sha256=seals, shot_updates=details))

    per = {cap: {arm: [] for arm in names} for cap in CAPS}
    maxima = {arm: [] for arm in names}
    d = []
    for index, sid in enumerate(ids, 1):
        name = f"{index:04d}_{sid}.npz"
        path = output / "shots" / name
        if sha(path) != seals[name]:
            raise ValueError("a curve changed after sealing")
        flag, down = io23.read_outcome("EAST", sid)
        d.append(flag)
        with np.load(path, allow_pickle=False) as trace:
            times = trace["times"]
            eligible = (np.round((down - times) * 1000., 9) > 30.
                        if flag else np.ones(len(times), bool))
            for arm in names:
                score = trace[arm]
                maxima[arm].append(float(score[eligible].max()) if eligible.any() else -np.inf)
                for cap in CAPS:
                    threshold = original if index <= STARTUP else cutoffs[cap]
                    per[cap][arm].append(dict(shot=sid, disruptive=flag, down_s=down,
                                               **score_curve(times, score, flag, down, threshold)))
    metrics = {cap: {arm: timing_decomposition(rows) for arm, rows in group.items()}
               for cap, group in per.items()}
    reference = read(CALIBRATION / "per_shot.json")
    if sum(d) != 180 or any(per[c]["static"] != reference[c][PARENT_ARM] for c in CAPS):
        raise AssertionError("frozen control must reproduce all calibrated first alarms")
    paired = {cap: {arm: flips(group["static"], group[arm]) for arm in arms}
              for cap, group in per.items()}
    updates = {arm: sum(r["arms"][arm]["changed_steps"] for r in details[STARTUP:]) for arm in arms}
    motion_sums = {key: sum(r["arms"][arms[-2]][key] for r in details[STARTUP:])
                   for key in ("forecast_error_sum", "hold_error_sum", "error_weight")}
    write(output / "per_shot.json", per)
    write(output / "result.json", dict(
        status="COMPLETED_INTRA_COMMISSIONING_DEVELOPMENT", metrics=metrics, paired=paired,
        posthoc_capacity_not_deployable={k: capacity(v, d, original, startup=STARTUP)
                                         for k, v in maxima.items()},
        updates=updates, motion_sums=motion_sums, seconds=time.monotonic() - started,
    ))
    print({cap: {arm: (m["tp"], m["fp"]) for arm, m in a.items()}
           for cap, a in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--persistence-anchor", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), anchored=args.persistence_anchor)
