"""Measure the mismatch between point-supervision and first-alarm credit."""

from collections import Counter
from pathlib import Path
import time

import numpy as np

from experiments.risk_aligned_online import io23
from joint_lora_hindsight_capacity_20261104 import (
    verify_inputs,
    sealed_curve,
    write_new,
    sha,
    V5_METADATA_SHA,
)

OUT = io23.ROOT / "probe_outputs/alarm_supervision_mismatch_20261105_v1.json"
FRESH = (
    io23.ROOT / "probe_outputs/hdf5_outcome_authority_20261105_v1/hdf5_metadata.json"
)


def lead_accounting(lead, scores, threshold):
    lead, scores = np.asarray(lead), np.asarray(scores)
    if (
        lead.ndim != 1
        or scores.shape != lead.shape
        or not len(lead)
        or not np.isfinite(lead).all()
        or not np.isfinite(scores).all()
        or not (np.diff(lead) <= 0).all()
    ):
        raise ValueError("finite, time-ordered probability/lead series required")
    hits = np.flatnonzero((scores >= threshold) & (lead >= -1e-6))
    first = int(hits[0]) if len(hits) else None
    first_lead = float(lead[first]) if first is not None else None
    timely = first_lead is not None and first_lead > 30
    short = int(((lead[hits] > 30) & (lead[hits] <= 125)).sum())
    capped = int(((lead[hits] > 30) & (lead[hits] <= 3000)).sum())
    if first is None:
        band = "miss"
    elif not timely:
        band = "tardy"
    elif first_lead <= 125:
        band = "30_to_125"
    elif first_lead <= 3000:
        band = "125_to_3000"
    else:
        band = "over3000"
    return dict(
        first=first,
        first_lead=first_lead,
        bin=band,
        timely=timely,
        window_crossings=short,
        beyond_window_only=timely and short == 0,
        capped_crossings=capped,
    )


def summarize(rows):
    disrupted = [r for r in rows if r["disruptive"]]
    safe = [r for r in rows if not r["disruptive"]]
    n = len(disrupted)
    timely = sum(r["timely"] for r in disrupted)
    return dict(
        shots=len(rows),
        disruptive=n,
        safe=len(safe),
        fp=sum(r["fp"] for r in safe),
        timely=timely,
        first_lead_bin=dict(Counter(r["bin"] for r in disrupted)),
        timely_with_no_30_to_125_crossing=sum(
            r["beyond_window_only"] for r in disrupted
        ),
        capped_timely_with_no_30_to_125_crossing=sum(
            r["bin"] == "125_to_3000" and r["beyond_window_only"] for r in disrupted
        ),
        timely_if_all_earlier_than_125ms_removed=sum(
            bool(r["window_crossings"]) for r in disrupted
        ),
    )


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    protocol, done, ids, meta = verify_inputs()
    if sha(FRESH) != V5_METADATA_SHA:
        raise ValueError("sealed metadata and current HDF authority differ")
    threshold = protocol["threshold"]
    rows = []
    for rank, sid in enumerate(ids, 1):
        times, probs, seal = sealed_curve(rank, sid, done, meta)
        flag, down = meta[sid]["disruptive"], meta[sid]["down"]
        if flag:
            lead = np.round((down - times) * 1000, 9)
            detail = lead_accounting(lead, probs, threshold)
        else:
            detail = dict(fp=bool(np.any(probs >= threshold)))
        rows.append(
            dict(
                shot=sid,
                rank=rank,
                disruptive=flag,
                curve_sha256=seal["curve_sha256"],
                **detail,
            )
        )
    results = {
        name: summarize(subset)
        for name, subset in (
            ("full360", rows),
            ("startup20", rows[:20]),
            ("suffix340", rows[20:]),
        )
    }
    if (results["full360"]["timely"], results["full360"]["fp"]) != (116, 29):
        raise ValueError(
            "sealed comparator does not reproduce its fixed-threshold counts"
        )
    write_new(
        OUT,
        dict(
            status="COMPLETE_POSTHOC_WINDOW_VS_FIRST_ALARM",
            threshold=threshold,
            independent_validation=False,
            model_parameters_changed=False,
            label_contract="HDF5 hard outcome and DownTime; EAST point-positive <=125ms",
            negative_label_outside_window_does_not_imply_physical_safety=True,
            masking_counterfactual_is_not_deployable=True,
            script_sha256=sha(Path(__file__)),
            hdf5_metadata_sha256=sha(FRESH),
            seconds=time.monotonic() - started,
            results=results,
            rows=rows,
        ),
    )
    print(results, flush=True)


if __name__ == "__main__":
    run()
