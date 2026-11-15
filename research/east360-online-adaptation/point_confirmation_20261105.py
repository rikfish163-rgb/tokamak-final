"""Causal confirmation replay over previously verified shadow curves."""

import argparse
from pathlib import Path
import time

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import V5_METADATA_SHA, read, sha, write_new
from verify_point_attribution_20261105 import pair
from verify_trajectory_replay_20261105 import FRESH, _recount, _score

ROOT = io23.ROOT / "probe_outputs"
POINT = ROOT / "trajectory_point_attribution_20261105_v1"
DONOR = ROOT / "point_donor_controls_20261105_v1"
POINT_AUDIT = ROOT / "trajectory_point_attribution_verification_20261105_v1.json"
DONOR_AUDIT = ROOT / "point_donor_controls_verification_20261105_v1.json"
PLAN = io23.ROOT / "POINT_CONFIRMATION_PLAN_20261105.md"
RULES = {
    "intra_confirm_now": ("point_lagged", 0),
    "intra_confirm_past32": ("point_lagged", 32),
    "donor_confirm_now": ("previous_clock", 0),
    "donor_confirm_past32": ("previous_clock", 32),
}


def confirm(shadow, threshold, lookback):
    if type(lookback) is not int or lookback < 0:
        raise ValueError("nonnegative integer lookback required")
    shadow = np.asarray(shadow)
    if shadow.ndim != 1 or not np.isfinite(shadow).all() or not 0 < threshold < 1:
        raise ValueError(
            "finite one-dimensional score and fixed probability threshold required"
        )
    idx = np.arange(len(shadow))
    last = np.maximum.accumulate(np.where(shadow >= threshold, idx, -1 - lookback))
    return idx - last <= lookback


def gated(base, active, shadow, threshold, lookback):
    base, active, shadow = (np.asarray(x) for x in (base, active, shadow))
    if base.ndim != 1 or not (base.shape == active.shape == shadow.shape):
        raise ValueError("aligned one-dimensional length required")
    if any(
        not np.isfinite(x).all() or ((x < 0) | (x > 1)).any()
        for x in (base, active, shadow)
    ):
        raise ValueError("valid probabilities required")
    accept = (
        (base < threshold)
        & (active >= threshold)
        & confirm(np.where(base < threshold, shadow, 0), threshold, lookback)
    )
    return np.where(accept, active, base)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    a, b = read(POINT / "completed.json"), read(DONOR / "completed.json")
    audit_a, audit_b = read(POINT_AUDIT), read(DONOR_AUDIT)
    if (
        audit_a["status"] != "PASS_RECOUNT_LOCKED_POINT_ATTRIBUTION"
        or audit_b["status"] != "PASS_RECOUNT_POINT_DONOR_CONTROLS"
        or audit_a["producer_completion_sha256"] != sha(POINT / "completed.json")
        or audit_b["producer_completion_sha256"] != sha(DONOR / "completed.json")
        or sha(FRESH) != V5_METADATA_SHA
    ):
        raise ValueError(
            "both source replays and HDF inventory must be independently verified"
        )
    cfg = read(POINT / "protocol.json")
    cfg_b = read(DONOR / "protocol.json")
    order, tau = cfg["order"], cfg["threshold"]
    if cfg_b["order"] != order or cfg_b["threshold"] != tau or len(order) != 360:
        raise ValueError("same order and fixed threshold required")
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    inputs = {
        str(f): sha(f)
        for f in (
            Path(__file__).resolve(),
            PLAN,
            POINT_AUDIT,
            DONOR_AUDIT,
            POINT / "completed.json",
            DONOR / "completed.json",
            FRESH,
        )
    }
    write_new(
        output / "protocol.json",
        dict(
            scope="POSTRUN_CAUSAL_SHADOW_CONFIRMATION_DEVELOPMENT",
            threshold=tau,
            order=order,
            rules=RULES,
            uses_future_samples=False,
            fit_or_tune=False,
            independent_validation=False,
            input_sha256=inputs,
        ),
    )
    judged = {k: [] for k in ("static", "point_real", *RULES)}
    receipts = []
    for i, sid in enumerate(order[100:], 101):
        name = f"{i:04d}_{sid}"
        pa, pb = [r / "shots" / f"{name}.npz" for r in (POINT, DONOR)]
        ba, bb = a["sealed"][i - 101], b["sealed"][i - 101]
        if (
            ba["shot"] != sid
            or bb["shot"] != sid
            or sha(pa) != ba["curve_sha256"]
            or sha(pb) != bb["curve_sha256"]
        ):
            raise ValueError("exact parent curve seals required")
        with np.load(pa, allow_pickle=False) as p, np.load(pb, allow_pickle=False) as q:
            t, base, real, lag = [
                p[k].copy() for k in ("times", "static", "point_real", "point_lagged")
            ]
            if not all(
                np.array_equal(p[k], q[k]) for k in ("times", "static", "point_real")
            ):
                raise ValueError("baseline and real must be identical across parents")
            shadows = dict(point_lagged=lag, previous_clock=q["previous_clock"].copy())
        traces = {
            k: gated(base, real, shadows[shadow], tau, age)
            for k, (shadow, age) in RULES.items()
        }
        curve = output / "shots" / f"{name}.npz"
        with curve.open("xb") as f:
            np.savez_compressed(f, times=t, **traces)
        binding = sha(curve)
        seal = curve.with_name(f"{name}_seal.json")
        write_new(
            seal,
            dict(
                rank=i,
                shot=sid,
                curve_sha256=binding,
                source_curve_hashes=[ba["curve_sha256"], bb["curve_sha256"]],
                current_outcome_used_in_gate=False,
            ),
        )
        flag, down = io23.read_outcome("EAST", sid)
        truth = dict(shot=sid, disruptive=flag, down=down)
        with np.load(curve, allow_pickle=False) as reopened:
            if not np.array_equal(reopened["times"], t):
                raise ValueError("unchanged original time grid required")
            for k, (source, age) in RULES.items():
                shadow = (base < tau) & (shadows[source] >= tau)
                # Independent audit of the trailing-window logic, including
                # starts and delays, on the persisted trace.
                past = (
                    np.convolve(
                        shadow.astype(np.int32),
                        np.ones(age + 1, dtype=np.int32),
                        mode="full",
                    )[: len(t)]
                    > 0
                )
                if not np.array_equal(
                    reopened[k] >= tau, (base >= tau) | ((real >= tau) & past)
                ):
                    raise ValueError(
                        "saved guard differs from independent past-only rule"
                    )
                judged[k].append(_score(t, reopened[k], truth, tau))
        judged["static"].append(_score(t, base, truth, tau))
        judged["point_real"].append(_score(t, real, truth, tau))
        reveal = curve.with_name(f"{name}_revealed.json")
        write_new(
            reveal,
            dict(
                shot=sid,
                after_seal_sha256=sha(seal),
                rows={k: v[-1] for k, v in judged.items()},
            ),
        )
        receipts.append(
            dict(
                shot=sid,
                curve_sha256=binding,
                seal_sha256=sha(seal),
                revealed_sha256=sha(reveal),
            )
        )
    old = read(POINT / "per_shot.json")
    first = old["static"][:100]
    results = {k: first + v for k, v in judged.items()}
    if any(results[k] != old[k] for k in ("static", "point_real")):
        raise ValueError("original scores changed")
    report = dict(
        full360={k: _recount(v) for k, v in results.items()},
        suffix260={k: _recount(v) for k, v in judged.items()},
        paired_to_static={
            k: pair(v, results["static"]) for k, v in results.items() if k != "static"
        },
        paired_to_real={
            k: pair(v, results["point_real"])
            for k, v in results.items()
            if k != "point_real"
        },
    )
    if any(sha(Path(f)) != h for f, h in inputs.items()):
        raise ValueError("sealed dependencies changed")
    write_new(output / "summary.json", report)
    write_new(output / "per_shot.json", results)
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_CONFIRMATION_DEVELOPMENT_DIAGNOSTIC",
            seconds=time.monotonic() - started,
            suffix_shots=len(receipts),
            guard_curves=len(receipts) * len(RULES),
            independently_checked_guard_crossings=True,
            sealed=receipts,
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    run(p.parse_args().output)
