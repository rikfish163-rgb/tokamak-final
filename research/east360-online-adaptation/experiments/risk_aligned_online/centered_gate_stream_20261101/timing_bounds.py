"""Post-hoc clock-only bounds, not a controller or neural-capacity upper bound.

Assume an impossible perfect oracle may set arbitrary future scores from160,
but cannot revise prior scores or the20 source-only preparation shots. These
bounds distinguish clock-locked errors from merely potentially changeable ones.
"""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..east_context_operator_20261031.run import read
from ..simple_online import _durable_json
from ..simple_train import sha


def bound_row(times, curve, disruptive, down, threshold, *, fixed):
    times, curve = np.asarray(times), np.asarray(curve)
    valid = (
        np.ones(len(times), bool) if not disruptive else (down - times) * 1000 >= -1e-6
    )
    hit = np.flatnonzero(valid & (curve >= threshold))
    first = int(hit[0]) if len(hit) else None
    lead = (
        float(round((down - times[first]) * 1000, 9))
        if disruptive and first is not None
        else None
    )
    actual_TP = bool(lead is not None and lead > 30)
    if not disruptive:
        return dict(
            first_alarm=first,
            actual_TP=False,
            actual_FP=bool(len(hit)),
            locked_FP=bool(len(hit))
            if fixed
            else bool(first is not None and first < 160),
            locked_TP=False,
            can_be_TP=False,
            future_timely_endpoints=0,
        )
    future = (np.arange(len(times)) >= 160) & (np.round((down - times) * 1000, 9) > 30)
    locked_TP = actual_TP and (fixed or first < 160)
    can_be_TP = actual_TP if fixed else bool(locked_TP or future.any())
    return dict(
        first_alarm=first,
        actual_TP=actual_TP,
        actual_FP=False,
        locked_FP=False,
        locked_TP=locked_TP,
        can_be_TP=can_be_TP,
        future_timely_endpoints=int(future.sum()),
    )


def main(root):
    output = root / "timing_bounds.json"
    if output.exists():
        raise FileExistsError(output)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert (
        done["status"] == "COMPLETED"
        and read(root / "verification.json")["status"] == "PASS"
    )
    rows = []
    for i, record in enumerate(p["order"]):
        sid, prefix = record["shot"], f"{i + 1:04d}_{record['shot']}"
        flag, down = io23.read_outcome(
            "EAST", sid
        )  # Finished stream diagnosis, never an online action.
        with np.load(root / "shots" / f"{prefix}.npz", allow_pickle=False) as data:
            rows.append(
                dict(
                    shot=sid,
                    rank=i + 1,
                    disruptive=flag,
                    fixed_warmup=i < 20,
                    **bound_row(
                        data["times"],
                        data["cold"],
                        flag,
                        down,
                        p["threshold"],
                        fixed=i < 20,
                    ),
                )
            )
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    assert nd == ns == 180
    assert sum(r["actual_TP"] for r in rows) == done["metrics"]["cold"]["full360"]["tp"]
    assert sum(r["actual_FP"] for r in rows) == done["metrics"]["cold"]["full360"]["fp"]
    upper_tp, lower_fp = (
        sum(r["can_be_TP"] for r in rows),
        sum(r["locked_FP"] for r in rows),
    )
    result = dict(
        scope="POSTHOC_COMPLETE_STREAM_CLOCK_ONLY_PERFECT_FUTURE_ORACLE_RELAXATION",
        not_an_online_policy=True,
        not_neural_capacity_bound=True,
        earliest_mutable_index=160,
        warmup20_immutable=True,
        outcome_authority="HDF5",
        max_possible_TP_from_clock_only=upper_tp,
        min_possible_FP_from_clock_only=lower_fp,
        clock_only_TPR_ceiling=upper_tp / nd,
        clock_only_FPR_floor=lower_fp / ns,
        clock_only_J_ceiling=upper_tp / nd - lower_fp / ns,
        potentially_changeable_missed_D=[
            r["shot"]
            for r in rows
            if r["disruptive"] and not r["actual_TP"] and r["can_be_TP"]
        ],
        potentially_changeable_FP=[
            r["shot"] for r in rows if r["actual_FP"] and not r["locked_FP"]
        ],
        immutable_missed_D=[
            r["shot"]
            for r in rows
            if r["disruptive"] and not r["actual_TP"] and not r["can_be_TP"]
        ],
        immutable_FP=[r["shot"] for r in rows if r["locked_FP"]],
        rows=rows,
        no_training=True,
        no_gpu=True,
        independent_confirmation=False,
        completed_sha256=sha(root / "completed.json"),
        script_sha256=sha(Path(__file__)),
    )
    _durable_json(output, result)
    print({k: v for k, v in result.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
