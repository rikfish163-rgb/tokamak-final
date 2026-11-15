"""Replay the locked point update from raw, causal signal windows on local CUDA."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.east_context_operator_20261031.run import configure
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_hindsight_capacity_20261104 import (
    SOURCE,
    THRESHOLD,
    read,
    sha,
    timeline,
    write_new,
)
from local_affine_reach_20261105 import COLD
import point_confirmation_20261105 as archived
import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import (
    BATCH,
    FIT_AFTER,
    baseline_for,
    features_at_arrival,
    make_views,
    reference,
    routes_for,
)
from verify_point_attribution_20261105 import pair

ROOT = io23.ROOT
PROBE = ROOT / "probe_outputs"
PLAN = ROOT / "CONFIRMATION_RAW_REPLAY_PLAN_20261105.md"
LOCK = ROOT / "POINT_CONFIRMATION_CANDIDATE_LOCK_20261105.md"
FIT = PROBE / "trajectory_teaching360_20261105_v1/fit.npz"
ROSTER = PROBE / "conditional_confirmation_roster_20261105_v1.json"
ARCHIVE_GUARD = PROBE / "point_confirmation_20261105_v1"
FIT_SHA = "246f0d935f4118a6b0b81f773158e9bec8d5d8dccac24a8007eb58abf3f62690"
ROSTER_SHA = "e3a9401c4dab46f4b77aa71faea8ebfb7f94a6ec8d0a13ef3c3208609a3d5866"
GUARD_SHA = "921721f6966d48eda22e69c29f51d5bbd4c421caf1c59b8b102e59bc563d6f1e"
LOOKBACK, ATOL = 32, 4e-6
DIRECTIONS = ("point_real", "point_lagged", "previous_clock", "first_full")
CONFIRM = {
    "intra_confirm_past32": "point_lagged",
    "donor_confirm_past32": "previous_clock",
}
ARMS = ("static", *DIRECTIONS, *CONFIRM)


class CausalConfirm:
    """Append chronological samples; keep only the last qualifying shadow index."""

    def __init__(self, threshold, lookback):
        if not 0 < threshold < 1 or type(lookback) is not int or lookback < 0:
            raise ValueError("probability threshold and nonnegative lookback required")
        self.threshold, self.lookback = threshold, lookback
        self.next_index, self.last_support = 0, -1 - lookback

    def push(self, base, active, shadow):
        base, active, shadow = (np.asarray(v) for v in (base, active, shadow))
        if base.ndim != 1 or not base.shape == active.shape == shadow.shape:
            raise ValueError("aligned one-dimensional curves required")
        if any(
            not np.isfinite(v).all() or ((v < 0) | (v > 1)).any()
            for v in (base, active, shadow)
        ):
            raise ValueError("valid probabilities required")
        if not len(base):
            return base.copy()
        q = self.next_index + np.arange(len(base))
        tau = self.threshold
        novel = base < tau
        last = np.maximum.accumulate(
            np.where(novel & (shadow >= tau), q, self.last_support)
        )
        accept = novel & (active >= tau) & (q - last <= self.lookback)
        self.next_index += len(base)
        self.last_support = int(last[-1])
        return np.where(accept, active, base)


def infer_raw(
    base,
    signal,
    point,
    threshold,
    *,
    enabled=True,
    previous=None,
    collect_donor=False,
    batch_size=BATCH,
):
    """One shot, with no outcome argument or dependency on archived predictions.

    Vectorizing windows does not share data across endpoints: the native risk at
    m, contrast at m and token routing at q use only their own causal windows.
    Confirmation consumes blocks in order and never revisits emitted samples.
    """
    if type(batch_size) is not int or batch_size < 1 or not 0 < threshold < 1:
        raise ValueError("positive batch size and probability threshold required")
    x, a = make_views(signal, base.center.device)
    p = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        initial = base.initial_fast(1).detach()
        for lo in range(0, len(x), batch_size):
            hi = min(lo + batch_size, len(x))
            out = base(
                x[lo:hi].contiguous(),
                a[lo:hi].contiguous(),
                initial.expand(hi - lo, -1),
            )
            p[lo:hi] = out["logit"].sigmoid().cpu().numpy()
        if enabled or collect_donor:
            tau = float(np.log(threshold / (1 - threshold)))
            feats = features_at_arrival(base, x, a, p, tau)
            direction = policy.action(feats["real"], *point)
        else:
            direction = x.new_empty((0, 256))
        targets = {}
        held = 0
        if enabled and len(direction):
            targets = dict(
                point_real=direction,
                first_full=direction[:1].clone(),
                point_lagged=policy.match_norm(
                    policy.action(feats["lagged"], *point), direction
                ),
            )
            if previous is not None and len(previous):
                idx = torch.arange(len(direction), device=x.device).clamp_max(
                    len(previous) - 1
                )
                targets["previous_clock"] = policy.match_norm(previous[idx], direction)
                held = max(0, len(direction) - len(previous))
    with torch.enable_grad():
        fast = {name: policy.write_fast(value) for name, value in targets.items()}
    norms = {k: v.norm(dim=1).cpu().numpy() for k, v in fast.items()}
    if any(
        not np.isfinite(v).all() or (v > policy.RADIUS + 1e-6).any()
        for v in norms.values()
    ):
        raise FloatingPointError("bounded finite temporary states required")
    curves = {name: p.copy() for name in ARMS}
    if fast:
        fast["first_full"] = fast["first_full"].expand_as(direction)
        with torch.no_grad():
            for lo in range(policy.FIRST, len(x), batch_size):
                hi = min(lo + batch_size, len(x))
                v, mask = x[lo:hi], a[lo:hi]
                q = np.arange(lo, hi, dtype=np.int64)
                weights = routes_for(mask, q)
                idx = torch.as_tensor(
                    (q - policy.FIRST) // policy.STRIDE, device=x.device
                )
                for name, state in fast.items():
                    curves[name][lo:hi] = (
                        policy.routed_pair(base, v, mask, weights, state[idx])
                        .sigmoid()
                        .cpu()
                        .numpy()
                    )
    for name, shadow in CONFIRM.items():
        gate = CausalConfirm(threshold, LOOKBACK)
        for lo in range(0, len(p), batch_size):
            sl = slice(lo, lo + batch_size)
            curves[name][sl] = gate.push(
                p[sl], curves["point_real"][sl], curves[shadow][sl]
            )
    return (
        curves,
        direction,
        dict(
            n_arrivals=len(direction),
            donor_held_arrivals=held,
            optimizer_writes={
                k: int((norms[k] > 0).sum()) if k in norms else 0 for k in DIRECTIONS
            },
            norm_mismatches={
                k: int((abs(norms[k] - norms["point_real"]) > 1e-6).sum())
                for k in ("point_lagged", "previous_clock")
                if k in norms
            },
        ),
    )


def fixed_parity(rank, sid, times, curves, threshold, source_done, done):
    def compare(a, b, name):
        if a.shape != b.shape:
            raise ValueError(f"curve shape differs for {sid}/{name}")
        error = float(np.max(abs(a - b)))
        if error > ATOL or not np.array_equal(a >= threshold, b >= threshold):
            raise ValueError(
                f"archived crossing parity failed for {sid}/{name}, max error {error:g}"
            )
        return error

    t, base = baseline_for(rank, sid, source_done)
    if not np.array_equal(times, t):
        raise ValueError(f"original HDF clock differs for {sid}")
    if rank <= FIT_AFTER:
        return {k: compare(v, base, k) for k, v in curves.items()}
    sources = {
        archived.POINT: ("point_real", "point_lagged"),
        archived.DONOR: ("previous_clock", "first_full"),
        ARCHIVE_GUARD: tuple(CONFIRM),
    }
    errors = {"static": compare(curves["static"], base, "static")}
    for root, keys in sources.items():
        path = root / "shots" / f"{rank:04d}_{sid}.npz"
        seal = done[root]["sealed"][rank - FIT_AFTER - 1]
        if seal["shot"] != sid or sha(path) != seal["curve_sha256"]:
            raise ValueError(f"archived forecast changed: {path}")
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(times, z["times"]):
                raise ValueError(f"archived forecast clock changed: {path}")
            errors.update({k: compare(curves[k], z[k], k) for k in keys})
    return errors


def metrics(rows):
    d = sum(r["disruptive"] for r in rows)
    s = len(rows) - d
    tp = sum(r["status"] == "tp" for r in rows)
    fp = sum(r["status"] == "fp" for r in rows)
    early = sum(r["status"] == "tp" and r["lead_ms"] > 3000 for r in rows)
    return dict(
        n_disruptive=d,
        n_safe=s,
        tp=tp,
        fp=fp,
        tpr=tp / d if d else None,
        fpr=fp / s if s else None,
        j=tp / d - fp / s if d and s else None,
        early_over3000=early,
        tp_30_to_3000=tp - early,
    )


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def run(mode, output, fixed_run=None):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    configure()
    locked_order, cfg, source_done = reference()
    guard_path = Path(archived.__file__)
    if (
        sha(FIT) != FIT_SHA
        or sha(guard_path) != GUARD_SHA
        or cfg["threshold"] != THRESHOLD
    ):
        raise ValueError(
            "locked fit, confirmation rule and operating threshold required"
        )
    map_path = COLD / "input_map.json"
    mapping = read(map_path)
    if (
        mapping["calibration_ids"] != locked_order[:20]
        or sha(map_path) != cfg["input_sha256"][str(map_path)]
    ):
        raise ValueError("locked numerical mapping required")
    base, parent = load_parent(DEFAULT_PARENT)
    digest = state_digest(base)
    if digest != cfg["source_state_sha256"] or parent["threshold"] != THRESHOLD:
        raise ValueError("locked source state and threshold required")
    base.cuda()
    with np.load(FIT, allow_pickle=False) as bank:
        if bank["ids"].tolist() != locked_order[20:100]:
            raise ValueError("historical point fit population changed")
        point = tuple(
            torch.as_tensor(bank["point_real_" + k], device="cuda", dtype=torch.float32)
            for k in ("scale", "matrix")
        )
    dependencies = [
        Path(__file__),
        PLAN,
        LOCK,
        FIT,
        guard_path,
        map_path,
        DEFAULT_PARENT,
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
        Path(policy.__file__),
        Path(features_at_arrival.__code__.co_filename),
        Path(prepare_actual.__code__.co_filename),
        Path(io23.__file__),
        Path(simple_data.__file__),
    ]
    later, archives = [], {}
    if mode == "fixed":
        order, raw_hashes, prefix = locked_order, cfg["raw_sha256"], FIT_AFTER
        for root in (archived.POINT, archived.DONOR, ARCHIVE_GUARD):
            path = root / "completed.json"
            entry = read(path)
            if len(entry["sealed"]) != 260 or any(
                sha(root / k) != h for k, h in entry["artifact_sha256"].items()
            ):
                raise ValueError(
                    f"complete unchanged historical comparison required: {root}"
                )
            if read(root / "protocol.json")["order"] != order:
                raise ValueError("fixed reference order changed")
            archives[root] = entry
            dependencies.append(path)
    else:
        if fixed_run is None:
            raise ValueError(
                "conditional replay requires the completed fixed parity run"
            )
        fixed = read(fixed_run / "completed.json")
        check = read(fixed_run / "protocol.json")
        if (
            fixed["status"] != "COMPLETE_RAW_CONFIRMATION_REPLAY"
            or fixed["n_shots"] != 360
            or not fixed["every_fixed_crossing_exact"]
            or check["mode"] != "fixed"
            or check["input_sha256"][str(Path(__file__).resolve())]
            != sha(Path(__file__))
            or any(sha(fixed_run / k) != h for k, h in fixed["artifact_sha256"].items())
        ):
            raise ValueError(
                "must complete fixed parity with the same producer before conditional"
            )
        if sha(ROSTER) != ROSTER_SHA:
            raise ValueError("locked signal-only roster changed")
        roster = read(ROSTER)
        order, later, raw_hashes, prefix = (
            roster["ids"],
            roster["later_proxy_ids"],
            roster["raw_sha256"],
            0,
        )
        if (
            roster["status"] != "LOCKED_CONDITIONAL_SIGNAL_ONLY_ROSTER"
            or roster["independent_validation"]
            or len(order) != 835
            or order != sorted(set(order))
            or set(order) & set(locked_order)
            or set(raw_hashes) != {str(s) for s in order}
            or later != [s for s in order if s > max(locked_order[:100])]
            or any(sha(p) != h for p, h in roster["input_sha256"].items())
        ):
            raise ValueError("unexpected conditional roster or input provenance")
        dependencies.extend([ROSTER, fixed_run / "completed.json"])
    hashes = {str(p.resolve()): sha(p) for p in dependencies}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    proto = dict(
        mode=mode,
        order=order,
        threshold=THRESHOLD,
        arms=ARMS,
        input_sha256=hashes,
        raw_sha256=raw_hashes,
        fixed_prefix_shots=prefix,
        later_proxy_ids=later,
        independent_validation=False,
        first_anchor=policy.FIRST,
        stride=policy.STRIDE,
        lookback=LOOKBACK,
        radius=policy.RADIUS,
        parity_atol=ATOL,
        model_refitted=False,
        mapped_from_rank=21 if mode == "fixed" else 1,
        past_fit_ranks=[21, 100],
        ordering="fixed development order"
        if mode == "fixed"
        else "numeric shot ID, not verified chronology",
        hdf5_outcome_after_seal=True,
        outcome_authority="HDF5 meta/IsDisrupt and meta/DownTime",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )
    write_new(output / "protocol.json", proto)
    rows = {k: [] for k in ARMS}
    sealed, errors = [], {k: 0.0 for k in ARMS}
    previous, donor_id = None, None
    print("started", mode, len(order), flush=True)
    for rank, sid in enumerate(order, 1):
        raw_path = io23.path_for("EAST", sid)
        before = signature(raw_path)
        raw_hash = raw_hashes[str(sid)]
        enabled = rank > prefix
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcome unavailable before seal"),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            times = raw["times"]
            if mode == "conditional" and len(times) != roster["samples"][str(sid)]:
                raise ValueError("conditional signal grid length changed")
            signal = prepare_actual(raw, rank if mode == "fixed" else 101, mapping)
            curves, direction, stats = infer_raw(
                base,
                signal,
                point,
                THRESHOLD,
                enabled=enabled,
                previous=previous,
                collect_donor=rank == prefix,
            )
        parity = (
            fixed_parity(rank, sid, times, curves, THRESHOLD, source_done, archives)
            if mode == "fixed"
            else None
        )
        if parity is not None:
            errors = {k: max(errors[k], parity[k]) for k in ARMS}
        path = output / "shots" / f"{rank:04d}_{sid}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=times, **curves)
            f.flush()
            os.fsync(f.fileno())
        curve_hash = sha(path)
        seal = path.with_name(path.stem + "_seal.json")
        write_new(
            seal,
            dict(
                rank=rank,
                shot=sid,
                curve_sha256=curve_hash,
                raw_sha256=raw_hash,
                current_outcome_used_in_prediction=False,
                donor=donor_id if enabled else None,
                parity=parity,
                stats=stats,
            ),
        )
        if signature(raw_path) != before or sha(raw_path) != raw_hash:
            raise ValueError(f"raw shot changed: {sid}")
        flag, down = io23.read_outcome("EAST", sid)
        if signature(raw_path) != before:
            raise ValueError(f"raw shot changed during reveal: {sid}")
        judged = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(times, v, flag, down, THRESHOLD),
            )
            for k, v in curves.items()
        }
        for name, row in judged.items():
            rows[name].append(row)
        reveal = path.with_name(path.stem + "_revealed.json")
        seal_hash = sha(seal)
        write_new(reveal, dict(shot=sid, after_seal_sha256=seal_hash, rows=judged))
        sealed.append(
            dict(
                shot=sid,
                curve_sha256=curve_hash,
                seal_sha256=seal_hash,
                revealed_sha256=sha(reveal),
            )
        )
        previous, donor_id = (direction, sid) if len(direction) else (None, None)
        if rank % 25 == 0 or rank == len(order):
            print("sealed/revealed", mode, rank, len(order), flush=True)
    if state_digest(base) != digest or any(
        p.grad is not None for p in base.parameters()
    ):
        raise AssertionError("the qualified source model changed")
    if any(sha(p) != h for p, h in hashes.items()):
        raise ValueError("bound artifact changed during the run")
    proxy = set(later)
    summary = dict(
        full={k: metrics(v) for k, v in rows.items()},
        later_proxy={
            k: metrics([r for r in v if r["shot"] in proxy]) for k, v in rows.items()
        }
        if later
        else None,
        paired_to_static={
            k: pair(v, rows["static"]) for k, v in rows.items() if k != "static"
        },
        paired_capped_to_static={
            k: pair(v, rows["static"], capped=True)
            for k, v in rows.items()
            if k != "static"
        },
    )
    write_new(output / "per_shot.json", rows)
    write_new(output / "summary.json", summary)
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_RAW_CONFIRMATION_REPLAY",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            every_fixed_crossing_exact=mode == "fixed",
            fixed_max_probability_error=errors if mode == "fixed" else None,
            source_state_unchanged=True,
            seconds=time.monotonic() - start,
            sealed=sealed,
            artifact_sha256={
                k: sha(output / k)
                for k in ("protocol.json", "per_shot.json", "summary.json")
            },
        ),
    )
    print("complete", mode, summary["full"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fixed", "conditional"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fixed-run", type=Path)
    args = ap.parse_args()
    run(args.mode, args.output, args.fixed_run)
