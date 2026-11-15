"""Fresh, read-only disk metadata reconciliation; no model or signal training."""

import argparse
from collections import Counter
from pathlib import Path
import time

import h5py
import numpy as np

from . import io23
from .event_prefix_probe import DEFAULT_LABELS, read_events
from .history_gradient_metric_20261102.run import read, sha, write


def metadata(path, device):
    """Independent strict HDF parser for reconciliation against the live reader."""
    stat = path.stat()
    with h5py.File(path, "r") as f:
        flag = np.asarray(f["meta/IsDisrupt"][()])
        down = np.asarray(f["meta/DownTime"][()])
        if flag.size != 1 or flag.dtype.kind not in "biuf":
            raise ValueError("IsDisrupt must be one numeric binary value")
        value = float(flag.item())
        if value != 0.0 and value != 1.0:
            raise ValueError("IsDisrupt must be 0 or 1")
        if (
            down.size != 1
            or down.dtype.kind not in "biuf"
            or not np.isfinite(down.item())
        ):
            raise ValueError("recorded DownTime must be one finite numeric value")
        terminal = float(down.item())
        ip = f["data/ip"]
        n = len(ip)
        start = float(np.asarray(ip.attrs["StartTime"]).item())
        rate = float(np.asarray(ip.attrs["SampleRate"]).item())
        if n < 1 or not np.isfinite([start, rate]).all() or rate <= 0:
            raise ValueError("nonempty record with finite positive clock required")
        end = start + (n - 1) / rate
    if (
        path.stat().st_size != stat.st_size
        or path.stat().st_mtime_ns != stat.st_mtime_ns
    ):
        raise RuntimeError("raw file changed while reading")
    return dict(
        device=device,
        shot=int(path.stem),
        path=str(path.resolve()),
        disruptive=value == 1.0,
        down=terminal if value == 1.0 else None,
        recorded_down_time=terminal,
        flag_shape=list(flag.shape),
        down_shape=list(down.shape),
        samples=n,
        start=start,
        rate=rate,
        end=end,
        disruption_within_record=(start <= terminal <= end + 1 / rate)
        if value == 1.0
        else None,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def census(records, ids):
    ids = list(ids)
    found = [records[sid] for sid in ids if sid in records]
    return dict(
        requested=len(ids),
        distinct=len(set(ids)),
        available=len(found),
        disruptive=sum(r["disruptive"] for r in found),
        safe=sum(not r["disruptive"] for r in found),
        missing=sorted(set(ids) - records.keys()),
    )


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    records, issues, inputs = {}, [], {}
    reader_mismatches = []
    for device in ("J-TEXT", "EAST"):
        records[device] = {}
        paths = sorted((io23.DATA_ROOT / device / "unified_hdf5").glob("*/*.hdf5"))
        if not paths:
            raise FileNotFoundError(device)
        for path in paths:
            try:
                r = metadata(path, device)
                sid = r["shot"]
                if (
                    sid in records[device]
                    or io23.path_for(device, sid).resolve() != path.resolve()
                ):
                    raise ValueError("duplicate shot or noncanonical file location")
                records[device][sid] = r
                actual = io23.read_outcome(device, sid)
                if actual != (r["disruptive"], r["down"]):
                    reader_mismatches.append(
                        dict(device=device, shot=sid, actual=list(actual))
                    )
            except (ValueError, KeyError, OSError) as exc:
                issues.append(dict(device=device, path=str(path), error=str(exc)))
        print(
            dict(
                device=device,
                files=len(paths),
                parsed=len(records[device]),
                issues=len(issues),
            ),
            flush=True,
        )
    lists = {}
    for device, folder, suffix in (
        ("J-TEXT", "J-TEXT", "J-TEXT"),
        ("EAST", "EAST", "EAST_TV"),
        ("EAST", "TEST", "EAST_Test"),
    ):
        classes = {}
        for label, name in ((True, "Disruption"), (False, "Non-disruption")):
            path = io23.ROOT / "shot_list" / folder / f"{name}_{suffix}.json"
            inputs[str(path)] = sha(path)
            ids = read(path)
            classes[label] = ids
        contradictions = [
            dict(
                shot=sid,
                list_disruptive=label,
                hdf5_disruptive=records[device][sid]["disruptive"],
                hdf5_down=records[device][sid]["down"],
            )
            for label, ids in classes.items()
            for sid in ids
            if sid in records[device] and records[device][sid]["disruptive"] != label
        ]
        lists[folder] = dict(
            device=device,
            declared_disruptive=census(records[device], classes[True]),
            declared_safe=census(records[device], classes[False]),
            overlapping_ids=sorted(set(classes[True]) & set(classes[False])),
            duplicate_ids={
                str(label): sorted(sid for sid, n in Counter(ids).items() if n > 1)
                for label, ids in classes.items()
            },
            conflicts=contradictions,
        )
    inputs[str(io23.JTEXT_INDEX)] = sha(io23.JTEXT_INDEX)
    inputs[str(io23.EAST_COHORT)] = sha(io23.EAST_COHORT)
    roles = {
        name: census(records["J-TEXT"], ids)
        for name, ids in io23.source_roles().items()
    }
    manifest = read(io23.EAST_COHORT)["test"]
    ids = manifest["disruption_shots"] + manifest["non_disruption_shots"]
    cohort = census(records["EAST"], ids)
    cohort["manifest_class_mismatches"] = [
        sid
        for key, label in (("disruption_shots", True), ("non_disruption_shots", False))
        for sid in manifest[key]
        if sid in records["EAST"] and records["EAST"][sid]["disruptive"] != label
    ]
    cohort["shotlist_conflicts"] = [
        r for r in lists["TEST"]["conflicts"] if r["shot"] in ids
    ]
    events, event_meta = read_events(DEFAULT_LABELS)
    inputs[str(DEFAULT_LABELS)] = sha(DEFAULT_LABELS)
    event_audit = dict(
        csv_shots=len(event_meta),
        event_intervals=sum(map(len, events.values())),
        census=census(records["J-TEXT"], event_meta),
        conflicts=[
            dict(
                shot=sid,
                csv_disruptive=bool(meta["outcome"]),
                hdf5_disruptive=records["J-TEXT"][sid]["disruptive"],
            )
            for sid, meta in event_meta.items()
            if sid in records["J-TEXT"]
            and bool(meta["outcome"]) != records["J-TEXT"][sid]["disruptive"]
        ],
        usage="interval targets only; CSV hard outcome never overrides HDF5",
    )
    run_audits = {}
    for name in (
        "east_decoder_complete360_v1",
        "east_functional_protection360_v1",
        "east_parameter_feedback_controls360_v1",
    ):
        run = io23.ROOT / "probe_outputs" / name
        p, per = read(run / "protocol.json"), read(run / "per_shot.json")
        inputs[str(run / "protocol.json")] = sha(run / "protocol.json")
        inputs[str(run / "per_shot.json")] = sha(run / "per_shot.json")
        order = [r["shot"] for r in p["order"]]
        arm_counts, mismatches = {}, []
        for arm, rows in per.items():
            if [r["shot"] for r in rows] != order:
                raise ValueError("saved run shot membership/order changed")
            for row in rows:
                r = records["EAST"][row["shot"]]
                if row["disruptive"] != r["disruptive"]:
                    mismatches.append(dict(arm=arm, shot=row["shot"]))
            nd = sum(records["EAST"][r["shot"]]["disruptive"] for r in rows)
            tp = sum(
                r["alarm"]
                and r["lead_ms"] is not None
                and r["lead_ms"] > 30
                and records["EAST"][r["shot"]]["disruptive"]
                for r in rows
            )
            fp = sum(
                r["alarm"] and not records["EAST"][r["shot"]]["disruptive"]
                for r in rows
            )
            arm_counts[arm] = dict(
                tp=tp,
                fp=fp,
                n_disruptive=nd,
                n_safe=len(rows) - nd,
                tpr=tp / nd,
                fpr=fp / (len(rows) - nd),
                j=tp / nd - fp / (len(rows) - nd),
            )
        run_audits[name] = dict(
            full=census(records["EAST"], order),
            startup20=census(records["EAST"], order[:20]),
            suffix340=census(records["EAST"], order[20:]),
            hard_outcome_mismatches=mismatches,
            metrics_from_saved_alarm_and_lead=arm_counts,
            lead_scope="saved lead only; this check does not recompute raw probability curves",
        )
    summary = dict(
        status="PASS"
        if not issues
        and not reader_mismatches
        and not cohort["manifest_class_mismatches"]
        and not any(r["hard_outcome_mismatches"] for r in run_audits.values())
        else "FAIL",
        authority="HDF5 meta/IsDisrupt + meta/DownTime; no outcome fallback",
        disk={d: census(rs, rs) for d, rs in records.items()},
        invalid_metadata=issues,
        live_reader_mismatches=reader_mismatches,
        source_roles=roles,
        cohort360=cohort,
        event_labels=event_audit,
        shotlists=lists,
        saved_runs=run_audits,
        input_sha256=inputs,
        raw_files_modified=0,
        signal_arrays_read=0,
        training_steps=0,
        metadata_fingerprint_scope="fresh metadata+stat only, not whole-HDF content hashes",
        seconds=time.monotonic() - started,
    )
    root.mkdir()
    write(
        root / "hdf5_metadata.json", {d: list(rs.values()) for d, rs in records.items()}
    )
    write(root / "audit.json", summary)
    write(
        root / "completed.json",
        dict(
            status=summary["status"],
            seconds=summary["seconds"],
            artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
        ),
    )
    print(
        {
            k: summary[k]
            for k in (
                "status",
                "disk",
                "source_roles",
                "cohort360",
                "event_labels",
                "seconds",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
