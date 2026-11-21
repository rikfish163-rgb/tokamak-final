"""Posthoc accepted-feedback timing on the completed innovation trial only."""

import json
from pathlib import Path

import numpy as np

from ..innovation_metric_20261102.verify import read_json, read_outcome, sha_file

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "probe_outputs/innovation_metric360_v2"
ARMS = ("contrast_unit", "innovation_unit", "innovation_metric", "metric_forecast")


def timing(times, events, disruptive, down, query):
    t = np.asarray(times, dtype=np.float64)
    if (
        t.ndim != 1
        or not len(t)
        or not np.isfinite(t).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("finite increasing recorded times required")
    if query is not None and not 0 <= query < len(t):
        raise ValueError("recorded query or explicit unsupported None required")
    accepted = [e for e in events if e["accepted"]]
    effective = [e for e in accepted if e["first_affected"] < len(t)]
    for e in accepted:
        if not 0 <= e["maturity"] < len(t) or e["first_affected"] != e["maturity"] + 1:
            raise ValueError("accepted event outside its causal clock")
    before = [e for e in effective if query is not None and e["maturity"] < query]
    last = before[-1]["maturity"] if before else None
    offsets = [
        float(round((t[e["first_affected"]] - t[0]) * 1000, 9)) for e in effective
    ]
    lead = (
        [float(round((down - t[e["first_affected"]]) * 1000, 9)) for e in effective]
        if disruptive
        else None
    )
    return dict(
        accepted_receipts=len(accepted),
        accepted_with_future_record=len(effective),
        accepted_with_timely_future=sum(x > 30 for x in lead)
        if lead is not None
        else None,
        accepted_first_effect_in_30_125_ms=sum(30 < x <= 125 + 1e-6 for x in lead)
        if lead is not None
        else None,
        first_effect_offset_ms=offsets[0] if offsets else None,
        last_effect_offset_ms=offsets[-1] if offsets else None,
        record_duration_ms=float(round((t[-1] - t[0]) * 1000, 9)),
        effective_update_after_record1000ms=any(x >= 1000 for x in offsets),
        last_maturity_before_query=last,
        feedback_age_at_query_ms=float(round((t[query] - t[last]) * 1000, 9))
        if last is not None
        else None,
    )


def summarize(rows):
    ages = [
        r["feedback_age_at_query_ms"]
        for r in rows
        if r["feedback_age_at_query_ms"] is not None
    ]
    long = [r for r in rows if r["record_duration_ms"] >= 1000]
    return dict(
        shots=len(rows),
        accepted_receipts=sum(r["accepted_receipts"] for r in rows),
        accepted_with_future_record=sum(r["accepted_with_future_record"] for r in rows),
        zero_effective_updates=sum(r["accepted_with_future_record"] == 0 for r in rows),
        exactly_one_effective_update=sum(
            r["accepted_with_future_record"] == 1 for r in rows
        ),
        query_supported=sum(r["query"] is not None for r in rows),
        query_has_prior_accepted_feedback=len(ages),
        median_feedback_age_ms=float(np.median(ages)) if ages else None,
        records_at_least1000ms=len(long),
        long_records_with_update_after1000ms=sum(
            r["effective_update_after_record1000ms"] for r in long
        ),
    )


def main():
    report = REFERENCE / "accepted_feedback_timing.json"
    if report.exists():
        raise FileExistsError(report)
    done, protocol = (
        read_json(REFERENCE / "completed.json"),
        read_json(REFERENCE / "protocol.json"),
    )
    verified = read_json(REFERENCE / "verification_records.json")
    assert done["status"] == "COMPLETED" and verified["status"] == "PASS"
    assert (
        sha_file(REFERENCE / "completed.json") == verified["hashes"]["completed_sha256"]
    )
    threshold = protocol["threshold"]
    rows, unsupported = {a: [] for a in ARMS}, []
    for index, entry in enumerate(protocol["order"], 1):
        if index <= 20:
            continue
        sid = entry["shot"]
        stem = f"{index:04d}_{sid}"
        seal = read_json(REFERENCE / "shots" / (stem + "_seal.json"))
        curve = REFERENCE / "shots" / (stem + ".npz")
        meta = REFERENCE / "shots" / (stem + "_updates.json")
        assert (
            sha_file(curve) == seal["sha256"]
            and sha_file(meta) == seal["updates_sha256"]
        )
        flag, down = read_outcome(sid)
        with np.load(curve, allow_pickle=False) as saved:
            times, cold = saved["times"], saved["cold"]
        if flag:
            lead = np.round((down - times) * 1000, 9)
            eligible = np.flatnonzero((lead > 30) & (lead <= 125 + 1e-6))
            query = int(eligible[np.argmax(cold[eligible])]) if len(eligible) else None
            if query is None:
                unsupported.append(sid)
        else:
            crossed = np.flatnonzero(cold >= threshold)
            query = int(crossed[0]) if len(crossed) else int(np.argmax(cold))
        info = read_json(meta)
        for arm in ARMS:
            rows[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    query=query,
                    **timing(times, info["updates"][arm], flag, down, query),
                )
            )
    summaries = {
        a: {
            name: summarize(group)
            for name, group in (
                ("suffix340", rows[a]),
                ("D172", [r for r in rows[a] if r["disruptive"]]),
                ("S168", [r for r in rows[a] if not r["disruptive"]]),
            )
        }
        for a in ARMS
    }
    result = dict(
        status="COMPLETED_POSTHOC_DIAGNOSTIC",
        summaries=summaries,
        per_shot=rows,
        disruptive_records_without_30_125_query=unsupported,
        completed_sha256=sha_file(REFERENCE / "completed.json"),
        auditor_sha256=sha_file(Path(__file__)),
        outcome_metadata_reads=340,
        raw_signal_reads=0,
        neural_updates=0,
        query_definition="D: cold maximum within strict30<lead<=125ms. S: first cold alarm, otherwise cold maximum. Same query for every arm; posthoc only.",
        clock_definition="Milliseconds since saved HDF record start, not plasma ignition. Feedback age is query time minus the latest accepted observation maturity strictly before it.",
        limitations=[
            "No late parameter update does not mean later predictions are constant: current inputs and token basis still change.",
            "Updates in the 30-125ms interval are not necessary for earlier accepted updates to affect that interval.",
            "Uses revealed outcomes after a completed stream for diagnosis, never a deployable endpoint selector.",
            "Neither a new performance result nor evidence of a physical mechanism.",
        ],
    )
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(summaries=summaries, unsupported=unsupported)), flush=True)


if __name__ == "__main__":
    main()
