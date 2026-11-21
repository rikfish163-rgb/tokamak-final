"""Recount causal support at actual first alarms without writing model state."""

import argparse
from pathlib import Path
import time

import numpy as np

import alarm_order_conditional_20261106 as conditional
from analyze_deep_miss_history_20261106 import read_bound
import moving_peak_replay_20261105 as r
from prealarm_history_support_20261106 import available_next, risk_roles
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes, score


def block_clock(support, size):
    """At endpoint i use availability at the latest completed block release."""
    if support.ndim != 1 or support.dtype != bool or size <= 0:
        raise ValueError(
            "one causal availability mask and positive block size required"
        )
    return support[np.arange(len(support)) // size * size]


def opportunities(t, p, mask, flag, down, threshold):
    if (
        t.ndim != 1
        or p.shape != t.shape
        or len(mask) != len(t)
        or (np.diff(t) <= 0).any()
        or not np.isfinite(t).all()
        or not np.isfinite(p).all()
        or not ((0 <= p) & (p <= 1)).all()
    ):
        raise ValueError(
            "matching chronological signals and native probability required"
        )
    support = available_next(mask)
    for b in (16, 32):
        support[f"pair{b}_block"] = block_clock(support[f"pair2x{b}"], b)
    native = score(t, p, dict(shot=None, disruptive=flag, down=down), threshold)
    first = native["first_index"]
    positive = risk_roles(t, flag, down, 125, 30)["timely_positive"]
    lead = np.round((down - t) * 1000, 9) if flag else np.zeros_like(t)
    reasonable = (lead > 30) & (lead <= 3000)
    return dict(
        status=native["status"],
        first_index=first,
        first_lead_ms=native["lead_ms"],
        geometry={
            k: dict(
                timely_positive=bool((v & positive).any()),
                reasonable=bool((v & reasonable).any()),
                at_original_first=first is not None and bool(v[first]),
            )
            for k, v in support.items()
        },
    )


def summary(rows):
    statuses = ("tp", "fp", "tardy", "miss", "tn")
    counts = {k: sum(q["status"] == k for q in rows) for k in statuses}
    grouped = {}
    for arm in rows[0]["geometry"]:
        groups = {"tp": 0, "fp": 0, "tardy": 0, "miss": 0, "tn": 0}
        deep_pos = deep_reasonable = all_pos = 0
        for q in rows:
            z = q["geometry"][arm]
            groups[q["status"]] += z["at_original_first"]
            all_pos += z["timely_positive"]
            if q["status"] in ("miss", "tardy"):
                deep_pos += z["timely_positive"]
                deep_reasonable += z["reasonable"]
        grouped[arm] = dict(
            at_original_first=groups,
            deep_with_positive=deep_pos,
            deep_with_reasonable=deep_reasonable,
            disrupted_with_positive=all_pos,
        )
    return dict(static=counts, eligible=grouped)


def run(root, out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    protocol, done, _ = read_bound(root, h, r.simple_data.source_roles())
    order, fixed, prior = reference()
    extra_order, extra, extra_prior, mpath, mapping = conditional.references(h)
    if (
        order != protocol["fixed"]
        or extra_order != protocol["conditional"]
        or protocol["threshold"] != fixed["threshold"]
        or protocol["threshold"] != r.THRESHOLD
    ):
        raise ValueError("exact static comparison contract required")
    inputs = {
        str(p.resolve()): h(p)
        for p in (
            root / "completed.json",
            root / "protocol.json",
            Path(__file__),
            mpath,
            Path(available_next.__code__.co_filename),
            Path(read_bound.__code__.co_filename),
            Path(r.prepare_actual.__code__.co_filename),
            Path(score.__code__.co_filename),
            Path(baseline_for.__code__.co_filename),
            Path(conditional.__file__),
        )
    }
    groups = {}
    for name in ("fixed", "conditional"):
        rows, cfg = [], fixed if name == "fixed" else extra
        for q in done["rows"][name]:
            sid, rank = q["shot"], q["rank"]
            path = r.io23.path_for("EAST", sid)
            if h(path) != q["raw_sha256"] or h(path) != cfg["raw_sha256"][str(sid)]:
                raise ValueError("shot bytes changed from the blind holdout seal")
            raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
            t, p = (
                baseline_for(rank, sid, prior)
                if name == "fixed"
                else conditional.baseline(h, rank, sid, extra_prior, q["raw_sha256"])
            )
            if not np.array_equal(raw["times"], t):
                raise ValueError("raw and frozen alarm clocks differ")
            mask = r.prepare_actual(raw, rank if name == "fixed" else 21, mapping)[
                "available"
            ]
            flag, down = r.io23.read_outcome("EAST", sid)
            rows.append(
                dict(
                    shot=sid,
                    rank=rank,
                    sha256=h(path),
                    **opportunities(t, p, mask, flag, down, r.THRESHOLD),
                )
            )
        totals = summary(rows)
        if (totals["static"]["tp"], totals["static"]["fp"]) != (
            (116, 29) if name == "fixed" else (257, 68)
        ):
            raise ValueError("reference first alarm counts changed")
        groups[name] = dict(**totals, rows=rows)
        print(name, totals["static"], totals["eligible"], flush=True)
    h.check()
    r.write_new(
        out,
        dict(
            status="SEALED_DEVELOPMENT_COHORT_CAUSAL_SUPPORT_ONLY",
            groups=groups,
            input_sha256=inputs,
            model_updates=0,
            new_predictions=0,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.root, a.out)
