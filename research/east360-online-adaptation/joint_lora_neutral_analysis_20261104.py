"""Verify saved risk-neutral EAST curves and estimate matched feedback effects."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import (
    read,
    sha,
)
from experiments.risk_aligned_online.joint_lora_analysis_20261104 import forecast_error
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    BASELINE,
    BASELINE_DONE_SHA,
    check_reference,
)
from experiments.risk_aligned_online.source_anticipation_20261104.static_stream import (
    timing_decomposition,
)

ARMS = ("native", "static", "intra", "hold", "reverse")
FEEDBACK = ARMS[1:]
INVENTORY = io23.ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v5"
END_STATUS = "COMPLETED_RISK_NEUTRAL_LORA_INTRA_DIAGNOSTIC"


def scan_curves(times, curves, flag, down, threshold, *, frozen):
    if set(curves) != set(ARMS) or any(
        a.shape != times.shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1))
        for a in curves.values()
    ):
        raise ValueError("five finite matching probability curves required")
    scores = {
        a: score_curve(times, v, flag, down, threshold) for a, v in curves.items()
    }
    gap = float(np.max(np.abs(curves["native"] - curves["static"])))
    if gap > 3e-6 or scores["native"] != scores["static"]:
        raise ValueError("the native and static controls must be risk-neutral")
    n = len(times) if frozen else min(135, len(times))
    ref = "native" if frozen else "static"
    if any(
        not np.array_equal(v[:n], curves[ref][:n])
        for a, v in curves.items()
        if frozen or a in ("intra", "hold", "reverse")
    ):
        raise ValueError("risk curve changed before its first possible feedback")
    return scores, gap


def paired_transitions(candidate, reference):
    groups = {}

    def capped(row):
        return row["timely"] and row["lead_ms"] <= 3000

    def early(row):
        return row["timely"] and row["lead_ms"] > 3000

    for name, indicator in (
        ("timely", lambda r: r["timely"]),
        ("capped", capped),
        ("early", early),
        ("false_alarms", lambda r: not r["disruptive"] and r["alarm"]),
    ):
        gain, loss = [], []
        for a, b in zip(candidate, reference, strict=True):
            if (a["shot"], a["disruptive"]) != (b["shot"], b["disruptive"]):
                raise ValueError("paired shots and truth must match")
            x, y = indicator(a), indicator(b)
            if x and not y:
                gain.append(a["shot"])
            if y and not x:
                loss.append(a["shot"])
        keys = (
            ("gained", "lost") if name in ("timely", "capped") else ("added", "removed")
        )
        groups[name + "_" + keys[0]], groups[name + "_" + keys[1]] = gain, loss
    return groups


def trace_accounting(data, history, length):
    keys = ("origin", "released_at", "target", "mask", "prediction")
    if set(data) != {a + "__" + k for a in FEEDBACK for k in keys} or set(
        history
    ) != set(FEEDBACK):
        raise ValueError("complete matched traces and updates required")
    origin = np.arange(127, length, 32, dtype=np.int64)
    released = np.where(origin + 8 < length, origin + 8, -1)
    stats = {}
    for arm in FEEDBACK:
        cur = {k: data[arm + "__" + k] for k in keys}
        if not np.array_equal(origin, cur["origin"]) or not np.array_equal(
            released, cur["released_at"]
        ):
            raise ValueError("released clocks differ from actual arrival")
        if any(
            not np.array_equal(cur[k], data["static__" + k], equal_nan=True)
            for k in ("target", "mask")
        ):
            raise ValueError("observed targets and masks must match controls")
        if cur["prediction"].shape != (len(origin), 168):
            raise ValueError("each issued forecast must have 168 coordinates")
        error = forecast_error(cur["prediction"], cur["target"], cur["mask"], released)
        mature = np.flatnonzero(released >= 0)
        count = 0
        for i, row in zip(mature, history[arm], strict=True):
            if row["origin"] != origin[i] or row["arrival"] != released[i]:
                raise ValueError("updates must use exactly the mature origin")
            support = int(cur["mask"][i].sum())
            update = row["updated"]
            grad = row["gradient_norm"]
            distance = row["parameter_displacement"]
            if row["supported"] != support or (update and not support):
                raise ValueError("wrong support or unsupported update")
            if (
                type(update) is not bool
                or not 0 <= distance <= 0.5
                or (grad is not None and (not np.isfinite(grad) or grad < 0))
                or (update and (grad is None or distance == 0))
                or (arm == "static" and (update or distance != 0 or grad is not None))
            ):
                raise ValueError("unexpected parameter/optimizer accounting")
            count += update
        stats[arm] = dict(**error, updates=count, matured=len(mature))
    return stats


def verify_hashes(root, hashes):
    for name, digest in hashes.items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or sha(root / name) != digest
        ):
            raise ValueError(f"unverified artifact: {name}")


def analyze(run, *, prefix=None):
    terminal = prefix is None
    n = 360 if terminal else prefix
    if not 21 <= n <= (360 if terminal else 359):
        raise ValueError(
            "specify a consecutive sealed prefix of 21–359 or a complete run"
        )
    status = None
    if terminal:
        status = read(run / "completed.json")
        if (status["status"], status["scored_shots"], status["inter_updates"]) != (
            END_STATUS,
            360,
            0,
        ):
            raise ValueError("terminal fixed-parent 360-shot receipt required")
        verify_hashes(run, status["artifact_sha256"])
    protocol = read(run / "protocol.json")
    ids = [row["shot"] for row in protocol["order"]]
    if (
        len(ids) != 360
        or set(ids) != set(io23.cohort_ids("EAST"))
        or tuple(protocol["arms"]) != ARMS
        or (protocol["decoder_enabled_after"], protocol["fixed_inter_updates"])
        != (20, 0)
    ):
        raise ValueError("locked order, five arms and zero inter updates required")
    threshold = protocol["threshold"]
    parent = read(run / "parent_activation.json")
    if parent["after_sealed_revealed_shots"] != ids[:20]:
        raise ValueError(
            "decoder activation must follow the first twenty sealed outcomes"
        )
    verify_hashes(Path(parent["preparation"]), parent["artifact_sha256"])
    inv = read(INVENTORY / "completed.json")
    if inv["status"] != "PASS":
        raise ValueError("passed raw-HDF inventory required")
    verify_hashes(INVENTORY, inv["artifact_sha256"])
    east = {r["shot"]: r for r in read(INVENTORY / "hdf5_metadata.json")["EAST"]}
    if sha(BASELINE / "completed.json") != BASELINE_DONE_SHA:
        raise ValueError("reference completion receipt changed")
    reference = read(BASELINE / "completed.json")
    baseline_p = read(BASELINE / "protocol.json")
    if (
        sha(BASELINE / "protocol.json") != reference["artifact_sha256"]["protocol.json"]
        or baseline_p["order"] != protocol["order"]
        or baseline_p["threshold"] != threshold
    ):
        raise ValueError("fixed baseline order and threshold required")
    rows, forecast, seals = {a: [] for a in ARMS}, {a: [] for a in FEEDBACK}, {}
    gaps, movement = [], {a: [] for a in ARMS[2:]}
    for i, sid in enumerate(ids[:n], 1):
        stem = f"{i:04d}_{sid}"
        shot = run / "shots" / stem
        seal_path = shot.with_name(stem + "_seal.json")
        seal = read(seal_path)
        seals[seal_path.name] = sha(seal_path)
        fixed_state = None if i <= 20 else parent["parent_sha256"]
        if (
            seal["shot"] != sid
            or seal["current_outcome_used_in_prediction"] is not False
            or seal["starting_fixed_state_sha256"] != fixed_state
        ):
            raise ValueError(f"bad prediction seal or changed parent: {sid}")
        curve, logs = shot.with_suffix(".npz"), shot.with_name(stem + "_updates.json")
        if sha(curve) != seal["curve_sha256"] or sha(logs) != seal["updates_sha256"]:
            raise ValueError(f"curve or update log hash mismatch: {sid}")
        truth = east[sid]
        stat = Path(truth["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (truth["size"], truth["mtime_ns"]):
            raise ValueError(f"raw HDF file changed since the audit: {sid}")
        flag, down = truth["disruptive"], truth["down"]
        if io23.read_outcome("EAST", sid) != (flag, down):
            raise ValueError(f"live HDF truth changed: {sid}")
        with np.load(curve, allow_pickle=False) as f:
            times = truth["start"] + np.arange(truth["samples"]) / truth["rate"]
            if not np.array_equal(times, f["times"]) or set(f.files) != {
                "times",
                *ARMS,
            }:
                raise ValueError(f"sealed clocks or arm schema changed: {sid}")
            curves = {a: f[a] for a in ARMS}
        ref = BASELINE / "shots" / stem
        check_reference(
            ref.with_suffix(".npz"),
            ref.with_name(stem + "_seal.json"),
            reference["seals"][i - 1],
            times,
            curves["native"],
        )
        scored, gap = scan_curves(times, curves, flag, down, threshold, frozen=i <= 20)
        if gap != seal["max_native_static_gap"]:
            raise ValueError("reported numerical gap differs")
        gaps.append(gap)
        for arm in ARMS:
            rows[arm].append(
                dict(shot=sid, disruptive=flag, down_s=down, **scored[arm])
            )
        history = read(logs)
        if i <= 20:
            if history or seal["forecast_sha256"] is not None:
                raise ValueError("startup must not forecast or update")
        else:
            fp = shot.with_name(stem + "_forecasts.npz")
            if sha(fp) != seal["forecast_sha256"]:
                raise ValueError(f"forecast hash mismatch: {sid}")
            with np.load(fp, allow_pickle=False) as data:
                for arm, stat in trace_accounting(data, history, len(times)).items():
                    forecast[arm].append(stat)
            for arm in movement:
                movement[arm].append(
                    float(np.max(np.abs(curves[arm] - curves["static"])))
                )
    if terminal and rows != read(run / "per_shot.json"):
        raise ValueError(
            "stored shot outcomes/alarms differ from independent re-scoring"
        )
    summary = {
        a: {
            k: timing_decomposition(r[block])
            for k, block in (
                ("full360" if terminal else "prefix", slice(None)),
                ("startup20", slice(20)),
                ("suffix340" if terminal else "prefix_after20", slice(20, None)),
            )
        }
        for a, r in rows.items()
    }
    if terminal and (
        summary != read(run / "summary.json")
        or max(gaps) != status["max_native_static_gap"]
    ):
        raise ValueError("stored aggregate does not match independent re-scoring")
    return dict(
        status="VERIFIED_TERMINAL_DEVELOPMENT"
        if terminal
        else "VERIFIED_PREFIX_ONLY_NOT_FINAL",
        scored_shots=n,
        verified_curves=n * len(ARMS),
        summary=summary,
        paired={
            a + "_vs_" + b: paired_transitions(rows[a], rows[b])
            for a, b in (
                ("intra", "static"),
                ("hold", "static"),
                ("reverse", "static"),
                ("intra", "hold"),
                ("intra", "reverse"),
            )
        },
        traces={
            a: dict(
                issued_shot_equal_mse=float(
                    np.mean([r["mse"] for r in rs if r["mse"] is not None])
                )
                if any(r["mse"] is not None for r in rs)
                else None,
                **{
                    k: sum(r[k] for r in rs)
                    for k in ("observed_elements", "pending", "matured", "updates")
                },
            )
            for a, rs in forecast.items()
        },
        max_native_static_gap=max(gaps),
        median_shot_max_abs_probability_change={
            a: float(np.median(v)) for a, v in movement.items()
        },
        protocol_sha256=sha(run / "protocol.json"),
        seal_sha256=seals,
        inventory_sha256=sha(INVENTORY / "completed.json"),
        script_sha256=sha(Path(__file__)),
        independent_validation=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", type=int)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    result = analyze(options.run, prefix=options.prefix)
    with options.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        {k: result[k] for k in ("status", "scored_shots", "verified_curves", "paired")}
    )
