"""Recount a completed LoRA stream from sealed records, without neural replay."""

import argparse
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from ..adaptivity_capacity_20261102.verify_records import (
    finite_number,
    read_bytes,
    read_json,
    read_outcome,
    require,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_COMPLETED_SHA256 = (
    "2b28ac02039b9106627c5aca865d23bcdd340ca2eda5d55d328aea46850f0d42"
)
SOURCE_SHA256 = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
ARMS = ("cold", "real", "reverse", "hold")
LEARNERS = ARMS[1:]
PLAN = "LORA_STREAM_PLAN_20261102.md"
MODULE = "experiments/risk_aligned_online/lora_stream_20261102"


def evaluate_curve(times, scores, disruptive, down, threshold):
    """Independently find the first preterminal alarm; 30 ms is not timely."""
    times, scores = np.asarray(times), np.asarray(scores)
    require(
        times.ndim == 1 and len(times) > 0 and scores.shape == times.shape,
        "curve shape",
    )
    require(np.isfinite(times).all() and np.isfinite(scores).all(), "nonfinite curve")
    require(
        np.all(np.diff(times) > 0) and np.all((scores >= 0) & (scores <= 1)),
        "invalid curve",
    )
    threshold = finite_number(threshold, "threshold")
    require(
        0 < threshold < 1 and type(disruptive) is bool, "invalid outcome or threshold"
    )
    if disruptive:
        down = finite_number(down, "DownTime")
        eligible = (down - times) * 1000 >= -1e-6
    else:
        eligible = np.ones(len(times), dtype=bool)
    crossings = np.flatnonzero(eligible & (scores >= threshold))
    lead = (
        float(np.round((down - times[crossings[0]]) * 1000, 9))
        if disruptive and len(crossings)
        else None
    )
    return dict(
        alarm=bool(len(crossings)),
        lead_ms=lead,
        timely=bool(lead is not None and lead > 30),
    )


def aggregate(rows):
    disruptive = sum(row["disruptive"] for row in rows)
    safe = len(rows) - disruptive
    tp = sum(row["timely"] for row in rows)
    fp = sum(not row["disruptive"] and row["alarm"] for row in rows)
    tpr = tp / disruptive if disruptive else None
    fpr = fp / safe if safe else None
    return dict(
        n=len(rows),
        n_disrupt=disruptive,
        n_safe=safe,
        tp=tp,
        fp=fp,
        tpr=tpr,
        fpr=fpr,
        J=tpr - fpr if disruptive and safe else None,
        early_over3000=sum(row["timely"] and row["lead_ms"] > 3000 for row in rows),
    )


def compare_flips(cold, candidate):
    require(len(cold) == len(candidate), "flip row count mismatch")
    result = {key: [] for key in ("rescued_D", "lost_D", "removed_FP", "added_FP")}
    for baseline, changed in zip(cold, candidate, strict=True):
        require(
            baseline["shot"] == changed["shot"]
            and baseline["disruptive"] == changed["disruptive"],
            "flip identity mismatch",
        )
        if baseline["disruptive"]:
            if changed["timely"] and not baseline["timely"]:
                result["rescued_D"].append(baseline["shot"])
            if baseline["timely"] and not changed["timely"]:
                result["lost_D"].append(baseline["shot"])
        else:
            if baseline["alarm"] and not changed["alarm"]:
                result["removed_FP"].append(baseline["shot"])
            if changed["alarm"] and not baseline["alarm"]:
                result["added_FP"].append(baseline["shot"])
    return result


def check_snapshot(output, protocol):
    declared = protocol["code_sha256"]
    required = {PLAN, f"{MODULE}/core.py", f"{MODULE}/run.py"}
    require(
        isinstance(declared, dict) and required <= set(declared),
        "missing code snapshot entries",
    )
    require(declared[PLAN] == protocol["plan_sha256"], "plan hash mismatch")
    content = read_bytes(output / "code_snapshot.zip")
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        entries = source.infolist()
        names = [entry.filename for entry in entries]
        require(
            len(names) == len(set(names)) and set(names) == set(declared),
            "code snapshot entries mismatch",
        )
        require(
            sum(entry.file_size for entry in entries) <= 512 * 1024 * 1024,
            "oversized code snapshot",
        )
        for entry in entries:
            path = PurePosixPath(entry.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in entry.filename,
                "invalid code path",
            )
            require(
                not entry.is_dir() and entry.file_size <= 64 * 1024 * 1024,
                "invalid code entry",
            )
            expected = declared[entry.filename]
            require(
                sha256_bytes(source.read(entry)) == expected,
                "snapshot code hash mismatch",
            )
            require(
                sha256_file(ROOT / entry.filename) == expected,
                "current code hash mismatch",
            )
    return sha256_bytes(content)


def check_log(info, length, *, warmup):
    """Check only recorded state, gradients, clocks and per-endpoint action summaries."""
    require(
        info.get("reset_each_shot") is True and info.get("temporary_parameters") == 256,
        "invalid reset or parameter count",
    )
    updates, actions = info["updates"], info["action_statistics"]
    require(
        set(updates) == set(LEARNERS) and set(actions) == set(ARMS), "invalid log arms"
    )
    expected_ends = [] if warmup else list(range(159, length, 128))
    totals = {}
    for arm in LEARNERS:
        history = updates[arm]
        require(
            isinstance(history, list)
            and [r["target_last"] for r in history] == expected_ends,
            "update clock mismatch",
        )
        previous = np.zeros(256, dtype=np.float64)
        totals[arm] = dict(
            events=0, accepted=0, nonzero_gradients=0, observed_scalars=0
        )
        for record, end in zip(history, expected_ends, strict=True):
            origin = end - 25
            require(
                record["origin"] == origin
                and record["target_first"] == origin + 1
                and record["risk_sealed_through"] == end
                and record["first_affected"] == end + 1,
                "update clock mismatch",
            )
            for key in ("forecast_sha256", "mask_sha256", "target_sha256"):
                require(
                    isinstance(record.get(key), str)
                    and re.fullmatch(r"[0-9a-f]{64}", record[key]) is not None,
                    f"invalid {key}",
                )
            before, after, gradient = (
                np.asarray(record[k], dtype=np.float64)
                for k in ("state256_before", "state256_after", "gradient256")
            )
            require(
                all(
                    v.shape == (256,) and np.isfinite(v).all()
                    for v in (before, after, gradient)
                ),
                "invalid state or gradient",
            )
            require(np.array_equal(before, previous), "state continuity mismatch")
            gradient_norm = finite_number(record["gradient_norm"], "gradient norm")
            step_norm = finite_number(record["step_norm"], "step norm")
            actual_gradient = float(np.linalg.norm(gradient))
            # The saved step was subtracted in float32 before the norm was taken.
            actual_step = float(
                np.linalg.norm(
                    (after.astype(np.float32) - before.astype(np.float32)).astype(
                        np.float64
                    )
                )
            )
            require(
                gradient_norm >= 0
                and math.isclose(
                    gradient_norm, actual_gradient, rel_tol=1e-10, abs_tol=1e-14
                ),
                "gradient norm mismatch",
            )
            require(
                step_norm >= 0
                and math.isclose(step_norm, actual_step, rel_tol=1e-10, abs_tol=1e-14),
                "step norm mismatch",
            )
            require(step_norm <= 0.100001, "step norm exceeds recorded protocol")
            observed, accepted = record["observed"], record["accepted"]
            require(
                type(observed) is int
                and 0 <= observed <= 25 * 22
                and type(accepted) is bool,
                "invalid observation or acceptance",
            )
            action_rms = finite_number(record["max_action_rms"], "update action RMS")
            require(0 <= action_rms <= 0.250001, "update action RMS exceeds bound")
            if observed:
                before_loss = finite_number(record["loss_before"], "loss before")
                proposal = finite_number(record["loss_proposal"], "loss proposal")
                retained = finite_number(record["loss_retained"], "loss retained")
                require(
                    accepted == (proposal < before_loss),
                    "acceptance does not match strict loss descent",
                )
                require(
                    retained == (proposal if accepted else before_loss),
                    "retained loss mismatch",
                )
                require(
                    record["gradient_computed"] is True
                    and record["reason"]
                    == ("improved" if accepted else "not_improved"),
                    "invalid update reason",
                )
            else:
                require(
                    not accepted
                    and gradient_norm == 0
                    and not np.any(gradient)
                    and step_norm == 0,
                    "unobserved feedback changed state",
                )
                require(
                    all(
                        record[k] is None
                        for k in ("loss_before", "loss_proposal", "loss_retained")
                    ),
                    "unobserved feedback has loss",
                )
                require(
                    record["gradient_computed"] is False
                    and record["reason"] == "no_target",
                    "invalid no-target receipt",
                )
            if not accepted:
                require(
                    np.array_equal(before, after), "unaccepted update changed state"
                )
            previous = after
            tally = totals[arm]
            tally["events"] += 1
            tally["accepted"] += int(accepted)
            tally["nonzero_gradients"] += int(gradient_norm > 0)
            tally["observed_scalars"] += observed
    for group in zip(*(updates[a] for a in LEARNERS), strict=True):
        require(len({r["mask_sha256"] for r in group}) == 1, "feedback mask mismatch")
        require(len({r["observed"] for r in group}) == 1, "feedback observed mismatch")
    for arm in ARMS:
        stats = actions[arm]
        endpoints, capped = stats["endpoints"], stats["capped_endpoints"]
        rms = finite_number(stats["max_rms"], "action RMS")
        require(
            type(endpoints) is int and endpoints == (0 if warmup else length),
            "action endpoint count mismatch",
        )
        require(
            type(capped) is int and 0 <= capped <= endpoints,
            "invalid capped endpoint count",
        )
        require(
            0 <= rms <= 0.250001 and (endpoints > 0 or rms == 0),
            "action RMS exceeds bound",
        )
    return totals, actions


def read_curves(data, keys, *, exact=True):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        names = set(archive.files)
        require(
            set(keys) <= names and (not exact or names == set(keys)),
            "invalid curve archive keys",
        )
        arrays = {key: archive[key] for key in keys}
    times = arrays["times"]
    require(
        times.ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid timestamps",
    )
    for key in set(keys) - {"times"}:
        value = arrays[key]
        require(
            value.shape == times.shape
            and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1)),
            "invalid probability curve",
        )
    return arrays


def verify(output):
    """Read the completed ledger only; never perform inference or modify artifacts."""
    output = Path(output)
    completed = read_json(output / "completed.json")
    require(
        completed.get("status") == "COMPLETED",
        "completed run required before reading records",
    )
    p = read_json(output / "protocol.json")
    fixed = {
        "scope": "DEVELOPMENT_COMPLETE360_SHOT_RESET_FFN_GRADIENT",
        "arms": list(ARMS),
        "warmup20_cost_included": True,
        "outcome_authority": "HDF5",
        "temporary_parameters": 256,
        "persistent_neural_updates": 0,
        "first_update": 159,
        "first_possible_effect": 160,
        "update_interval": 128,
        "forecast_horizon": 25,
        "step_size": 0.1,
        "parameter_anchor": 0.01,
        "endpoint_hidden_rms_cap": 0.25,
        "cap_uses_only_current_window": True,
        "forecast_objective": "observed_temporal_centered_SmoothL1",
        "no_label_input_to_predict": True,
    }
    for key, expected in fixed.items():
        require(p.get(key) == expected, f"protocol mismatch: {key}")
    require(p["source_sha256"] == SOURCE_SHA256, "fixed source hash mismatch")
    require(
        p["reference_completed_sha256"] == REFERENCE_COMPLETED_SHA256,
        "fixed reference hash mismatch",
    )
    threshold = finite_number(p["threshold"], "threshold")
    require(0 < threshold < 1, "invalid threshold")
    order = p["order"]
    require(isinstance(order, list) and len(order) == 360, "360 ordered cases required")
    ids = [row["shot"] for row in order]
    require(
        all(type(shot) is int and shot > 0 for shot in ids) and len(set(ids)) == 360,
        "invalid shot order",
    )
    reference, calibration = Path(p["reference"]), Path(p["calibration_reference"])
    require(
        sha256_file(reference / "completed.json") == p["reference_completed_sha256"],
        "reference hash mismatch",
    )
    ref = read_json(reference / "protocol.json")
    require(
        ref["order"] == order and Path(ref["reference"]) == calibration,
        "reference order or calibration mismatch",
    )
    for key in ("source_sha256", "cold_state_sha256", "input_map_sha256", "threshold"):
        require(ref[key] == p[key], f"reference protocol mismatch: {key}")
    for path, expected in (
        (Path(p["source_checkpoint"]), p["source_sha256"]),
        (Path(p["cold_state"]), p["cold_state_sha256"]),
        (output / "source_basis.pt", p["basis_sha256"]),
        (calibration / "input_map.json", p["input_map_sha256"]),
    ):
        require(sha256_file(path) == expected, f"file hash mismatch: {path.name}")
    require(
        read_json(calibration / "input_map.json")["calibration_ids"] == ids[:20],
        "calibration identities mismatch",
    )
    snapshot_hash = check_snapshot(output, p)
    hashes_path = calibration / "input_hashes.json"
    raw_hashes = read_json(hashes_path)
    saved_rows = read_json(output / "per_shot.json")
    require(
        set(saved_rows) == set(ARMS) and all(len(saved_rows[a]) == 360 for a in ARMS),
        "invalid per-shot rows",
    )
    rows = {arm: [] for arm in ARMS}
    totals = {
        arm: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for arm in LEARNERS
    }
    actions = {arm: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for arm in ARMS}
    for index, shot in enumerate(ids, 1):
        prefix = f"{index:04d}_{shot}"
        curve_path = output / "shots" / f"{prefix}.npz"
        log_path = output / "shots" / f"{prefix}_updates.json"
        seal = read_json(output / "shots" / f"{prefix}_seal.json")
        require(
            seal["index"] == index
            and seal["shot"] == shot
            and seal["outcome_available"] is False
            and seal["warmup_curves_reused"] is (index <= 20),
            "invalid curve seal",
        )
        curve_bytes, log_bytes = read_bytes(curve_path), read_bytes(log_path)
        require(
            sha256_bytes(curve_bytes) == seal["sha256"]
            and sha256_bytes(log_bytes) == seal["updates_sha256"],
            "curve or log hash mismatch",
        )
        data = read_curves(curve_bytes, ("times", *ARMS))
        ref_data = read_curves(
            read_bytes(reference / "shots" / f"{prefix}.npz"),
            ("times", "cold"),
            exact=False,
        )
        require(
            np.array_equal(data["times"], ref_data["times"])
            and np.array_equal(data["cold"], ref_data["cold"]),
            "reference cold mismatch",
        )
        for arm in ARMS:
            require(
                np.array_equal(data[arm][:160], data["cold"][:160]),
                "prefix160 mismatch",
            )
            if index <= 20:
                require(
                    np.array_equal(data[arm], ref_data["cold"]), "warmup reuse mismatch"
                )
        log = read_json(log_path)
        case_totals, case_actions = check_log(
            log, len(data["times"]), warmup=index <= 20
        )
        raw = ROOT / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
        require(sha256_file(raw) == raw_hashes[str(shot)], "raw input hash mismatch")
        disruptive, down = read_outcome(raw)
        for arm in ARMS:
            row = dict(
                shot=shot,
                disruptive=disruptive,
                **evaluate_curve(data["times"], data[arm], disruptive, down, threshold),
            )
            require(row == saved_rows[arm][index - 1], "per-shot decision mismatch")
            rows[arm].append(row)
            for key in ("endpoints", "capped_endpoints"):
                actions[arm][key] += case_actions[arm][key]
            actions[arm]["max_rms"] = max(
                actions[arm]["max_rms"], case_actions[arm]["max_rms"]
            )
        for arm in LEARNERS:
            for key in totals[arm]:
                totals[arm][key] += case_totals[arm][key]
    metrics = {
        arm: {
            scope: aggregate(part)
            for scope, part in (
                ("full360", rows[arm]),
                ("warmup20", rows[arm][:20]),
                ("suffix340", rows[arm][20:]),
            )
        }
        for arm in ARMS
    }
    flips = {arm: compare_flips(rows["cold"], rows[arm]) for arm in LEARNERS}
    require(metrics == completed["metrics"], "completed metrics mismatch")
    require(flips == completed["flips"], "completed flips mismatch")
    require(
        totals == completed["update_statistics"], "completed update statistics mismatch"
    )
    require(
        actions == completed["action_statistics"],
        "completed action statistics mismatch",
    )
    require(
        completed["fresh_curves"] == 1360 and completed["reused_warmup_curves"] == 80,
        "curve counts mismatch",
    )
    require(
        completed["source_state_unchanged"] is True
        and completed["scientific_goal_achieved"] is False
        and completed["independent_future_confirmation"] is False,
        "completion claims mismatch",
    )
    real, cold = metrics["real"]["full360"], metrics["cold"]["full360"]
    require(
        real["fpr"] is not None and real["J"] is not None,
        "full cohort needs both outcomes",
    )
    operational = (
        real["fpr"] <= 0.2 and real["tp"] >= cold["tp"] and real["J"] > cold["J"]
    )
    separation = all(
        real["J"] > metrics[arm]["full360"]["J"] for arm in ("reverse", "hold")
    )
    require(
        completed["operational_gate"] == operational
        and completed["control_separation_gate"] == separation
        and completed["method_promoted"] == (operational and separation),
        "completion gate mismatch",
    )
    return dict(
        status="PASS",
        cases=360,
        curves=1440,
        metrics=metrics,
        flips=flips,
        update_statistics=totals,
        action_statistics=actions,
        protocol_sha256=sha256_file(output / "protocol.json"),
        completed_sha256=sha256_file(output / "completed.json"),
        code_snapshot_sha256=snapshot_hash,
        input_hashes_sha256=sha256_file(hashes_path),
        limitations=(
            "Recorded action RMS, masks and gradients were checked as logs, not independently recomputed from the neural model or signals. "
            "Record consistency does not establish physical contribution or independent future validation."
        ),
    )


def write_report(output, report):
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with (Path(output) / "verification_records.json").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(rendered)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify(args.output)
    write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "cases", "curves")}))


if __name__ == "__main__":
    main()
