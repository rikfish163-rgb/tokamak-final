"""Offline interval-based pair selection, never an EAST online event oracle."""

import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..event_prefix_probe import DEFAULT_LABELS, assign_events
from ..hdf5_authority_audit_20261103 import metadata
from ..history_gradient_metric_20261102.run import sha, write
from ..source_event_tasks_20261102.data import load_events


def select_pairs(times, events, sid, disruptive, down, horizon, event_ids):
    """Return matched rows and all selected within-event rows (before matching)."""
    times = np.asarray(times, dtype=np.float64)
    if not isinstance(disruptive, bool) or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("HDF boolean outcome and positive source window required")
    if disruptive and (down is None or not np.isfinite(down)):
        raise ValueError("HDF disruption time required")
    if any(e.shot != sid for e in events):
        raise ValueError("events from one declared source shot required")
    _, assigned = assign_events(times, events)
    lead = np.round((down - times) * 1000, 9) if disruptive else None

    def group(q):
        if not disruptive:
            return "safe"
        return "danger" if lead[q] <= horizon + 1e-6 else "early"

    def legal(o):
        return (
            127 <= o and o + 26 < len(times) and (not disruptive or lead[o + 26] > 10)
        )

    selected, matched = [], []
    for event in events:
        eligible = [
            o
            for o in range(127, len(times) - 26)
            if legal(o)
            and assigned[o] == event.index
            and assigned[o + 25] == event.index
        ]
        if not eligible:
            continue
        positions = sorted(
            {min(k * len(eligible) // 5, len(eligible) - 1) for k in range(1, 5)}
        )
        for position in positions:
            o = eligible[position]
            row = dict(
                shot=sid,
                origin=o,
                maturity=o + 25,
                query=o + 26,
                event_id=event_ids[event.index],
                event_index=event.index,
                family=event.family,
                disruptive=disruptive,
                group=group(o + 26),
                query_lead_ms=float(lead[o + 26]) if disruptive else None,
            )
            selected.append(row)
            others = [
                j
                for j in range(max(127, o - 50), min(len(times) - 26, o + 51))
                if legal(j)
                and group(j + 26) == row["group"]
                and not (assigned[j] >= 1 and assigned[j] == assigned[j + 25])
            ]
            if others:
                j = min(others, key=lambda value: (abs(value - o), value))
                matched.append(
                    dict(row, other_origin=j, other_maturity=j + 25, other_query=j + 26)
                )
    return matched, selected


def counts(rows):
    def one(local):
        return dict(
            pairs=len(local),
            events=len({r["event_id"] for r in local}),
            shots=len({r["shot"] for r in local}),
        )

    return dict(
        **one(rows),
        groups={
            g: one([r for r in rows if r["group"] == g])
            for g in ("safe", "danger", "early")
        },
    )


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    events, _ = load_events()
    roles, advanced = io23.source_roles(), io23.advanced_times()
    namespace = {
        key: index
        for index, key in enumerate(
            sorted((sid, e.index) for sid, local in events.items() for e in local), 1
        )
    }
    pairs, selected, records = {}, {}, {}
    for role in ("train", "val"):
        pairs[role], selected[role] = [], []
        for sid in sorted(set(roles[role]) & events.keys()):
            record = metadata(io23.path_for("J-TEXT", sid), "J-TEXT")
            if io23.read_outcome("J-TEXT", sid) != (
                record["disruptive"],
                record["down"],
            ):
                raise ValueError("HDF readers disagree")
            times = record["start"] + np.arange(record["samples"]) / record["rate"]
            local, all_rows = select_pairs(
                times,
                events[sid],
                sid,
                record["disruptive"],
                record["down"],
                io23.effective_horizon_ms(sid, advanced)[0],
                {e.index: namespace[sid, e.index] for e in events[sid]},
            )
            pairs[role].extend(local)
            selected[role].extend(all_rows)
            records[str(sid)] = record
    census = {
        r: dict(matched=counts(pairs[r]), before_matching=counts(selected[r]))
        for r in pairs
    }
    output.mkdir()
    write(output / "pairs.json", pairs)
    write(output / "selected_before_matching.json", selected)
    write(output / "census.json", census)
    write(output / "hdf_metadata.json", records)
    write(
        output / "protocol.json",
        dict(
            scope="JTEXT_TRAIN_VAL_OFFLINE_EVENT_PAIR_SELECTION_NOT_ONLINE",
            event_csv_sha256=sha(DEFAULT_LABELS),
            hard_outcome_authority="HDF5",
            source_index_sha256=sha(io23.JTEXT_INDEX),
            advanced_times_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
            source_window="max(25ms, AdvancedTime)",
            strict_query_lead_ms=10,
            plan_sha256=sha(io23.ROOT / "EVENT_TEMPORAL_TEACHER_PLAN_20261103.md"),
            code_sha256=sha(Path(__file__)),
            train_ids=roles["train"],
            val_ids=roles["val"],
            waveform_reads=0,
            test_hdf_reads=0,
            east_hdf_reads=0,
        ),
    )
    write(
        output / "completed.json",
        dict(
            status="COMPLETED_METADATA_SELECTION",
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(census, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
