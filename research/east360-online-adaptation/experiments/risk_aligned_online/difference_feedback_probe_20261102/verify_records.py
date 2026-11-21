"""CPU verification of the ended-20 local probe; JSON goes to stdout by default.

This verifies saved records, hashes and HDF5 metadata, not neural inference or
future performance. Pass ``--write-report`` to create the fixed report once.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "probe_outputs/difference_feedback_probe20_v1"
CALIBRATION = ROOT / "probe_outputs/stratified_cold20_context360_v1"
SOURCE = (
    ROOT / "experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
)
COLD = CALIBRATION / "models/cold.pt"
SOURCE_SHA256 = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
REFERENCE_SHA256 = "76ebb4035873ba3c6b85da9d89b995233d2ce36222237bf1733058d4c9482f4c"
PLAN = "DIFFERENCE_FEEDBACK_PROBE_PLAN_20261102.md"
MODULE = "experiments/risk_aligned_online/difference_feedback_probe_20261102"
ARMS = ("centered_real", "difference_real", "difference_reverse", "difference_hold")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name}: expected number",
    )
    value = float(value)
    require(math.isfinite(value), f"{name}: nonfinite")
    return value


def close(actual, expected, name, *, atol=1e-12, rtol=1e-9):
    actual = finite(actual, name)
    require(
        math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), f"{name}: mismatch"
    )


def vector(values, name):
    require(
        isinstance(values, list) and len(values) == 256, f"{name}: expected 256 values"
    )
    result = np.asarray([finite(value, name) for value in values], dtype=np.float64)
    require(result.shape == (256,), f"{name}: invalid shape")
    return result


def stable_bce(logit, label):
    """Softplus of the signed logit without exponent overflow."""
    logit = finite(logit, "logit")
    require(type(label) is int and label in (0, 1), "invalid label")
    signed = (1 - 2 * label) * logit
    return max(signed, 0.0) + math.log1p(math.exp(-abs(signed)))


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def recompute_summary(rows):
    """Independent label-then-shot and safe-shot statistics and plan gates."""
    arms = {}
    for arm in ARMS:
        groups = {group: defaultdict(list) for group in ("class0", "class1", "safe")}
        for row in rows:
            value = finite(row["results"][arm]["delta_bce"], f"{arm} delta_bce")
            groups[f"class{row['label']}"][row["shot"]].append(value)
            if not row["disruptive"] and row["label"] == 0:
                groups["safe"][row["shot"]].append(value)
        stats = {}
        for group, shots in groups.items():
            shot_means = [_mean(points) for points in shots.values()]
            stats[group] = {
                "n_points": sum(map(len, shots.values())),
                "n_shots": len(shots),
                "mean_delta_bce": _mean(shot_means),
            }
        zero = stats["class0"]["mean_delta_bce"]
        one = stats["class1"]["mean_delta_bce"]
        stats["balanced_delta_bce"] = (
            _mean([zero, one]) if zero is not None and one is not None else None
        )
        arms[arm] = stats
    positive = len({r["shot"] for r in rows if r["disruptive"] and r["label"] == 1})
    safe = len({r["shot"] for r in rows if not r["disruptive"] and r["label"] == 0})
    coverage = {
        "positive_disruptive_n_shots": positive,
        "negative_safe_n_shots": safe,
        "positive_disruptive": positive >= 6,
        "negative_safe": safe >= 8,
        "passed": positive >= 6 and safe >= 8,
    }
    real = arms["difference_real"]
    c0, c1, safe_mean = (
        real[g]["mean_delta_bce"] for g in ("class0", "class1", "safe")
    )
    balanced = real["balanced_delta_bce"]
    base = arms["centered_real"]["balanced_delta_bce"]
    reverse = arms["difference_reverse"]["balanced_delta_bce"]
    gates = {
        "coverage": coverage["passed"],
        "hold_zero": bool(rows)
        and all(
            abs(r["results"]["difference_hold"]["delta_bce"]) <= 1e-10 for r in rows
        ),
        "class1": c1 is not None and c1 < -1e-5,
        "class0": c0 is not None and c0 <= 1e-6,
        "safe": safe_mean is not None and safe_mean <= 1e-6,
        "balanced": balanced is not None and balanced < -1e-5,
        "vs_centered_real": balanced is not None
        and base is not None
        and base - balanced >= 1e-6,
        "vs_difference_reverse": balanced is not None
        and reverse is not None
        and reverse - balanced >= 1e-6,
    }
    gates["passed"] = all(gates.values())
    if not coverage["passed"]:
        decision = "coverage_insufficient"
    elif not gates["hold_zero"]:
        decision = "implementation_check_failed"
    elif gates["passed"]:
        decision = "pass"
    else:
        decision = "screen_failed"
    return {"arms": arms, "coverage": coverage, "gates": gates, "decision": decision}


def validate_membership(rows, sampling, ids):
    require(len(ids) == len(set(ids)), "duplicate allowed shot")
    allowed = set(ids)
    require(len(sampling) == len(ids), "sampling shot count mismatch")
    require([s["shot"] for s in sampling] == ids, "sampling shot order or IDs mismatch")
    chosen = {}
    for shot in sampling:
        require(shot["shot"] in allowed, "sampling shot outside allowed IDs")
        for selected in shot["selected"]:
            key = (shot["shot"], selected["query"])
            require(key not in chosen, "duplicate selected shot/query")
            chosen[key] = selected
    seen = set()
    for row in rows:
        require(row["shot"] in allowed, "sample shot outside allowed IDs")
        key = (row["shot"], row["query"])
        require(key not in seen, "duplicate sample shot/query")
        require(key in chosen, "sample not in selected queries")
        for field in ("requested_lead_ms", "actual_lead_ms"):
            require(
                row[field] == chosen[key][field],
                f"sample {field} differs from selection",
            )
        seen.add(key)
    require(seen == set(chosen), "selected queries and samples differ")


def expected_sampling(meta):
    length, start, rate = meta["length"], meta["start"], meta["rate"]
    if length <= 160:
        return [], [{"reason": "insufficient_prefix"}]
    if not meta["disruptive"]:
        chosen = np.unique(np.linspace(160, length - 1, 5, dtype=int))
        return [
            dict(query=int(q), requested_lead_ms=None, actual_lead_ms=None)
            for q in chosen
        ], []
    times = start + np.arange(length) / rate
    leads = np.round((meta["down"] - times) * 1000, 9)
    eligible = np.flatnonzero((np.arange(length) >= 160) & (leads > 30))
    selected, skipped, used = [], [], set()
    for requested in (250, 150, 100, 60, 35):
        if not len(eligible):
            skipped.append(
                dict(requested_lead_ms=requested, reason="requested_lead_not_covered")
            )
            continue
        q = int(eligible[np.argmin(abs(leads[eligible] - requested))])
        if abs(leads[q] - requested) > 2 or q in used:
            skipped.append(
                dict(requested_lead_ms=requested, reason="requested_lead_not_covered")
            )
            continue
        used.add(q)
        selected.append(
            dict(query=q, requested_lead_ms=requested, actual_lead_ms=float(leads[q]))
        )
    return selected, skipped


def compare_tree(actual, expected, name):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected),
            f"{name}: keys mismatch",
        )
        for key in expected:
            compare_tree(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"{name}: length mismatch",
        )
        for index, value in enumerate(expected):
            compare_tree(actual[index], value, f"{name}[{index}]")
    elif isinstance(expected, float):
        close(actual, expected, name)
    else:
        require(
            type(actual) is type(expected) and actual == expected, f"{name}: mismatch"
        )


def validate_sample(row, meta):
    q = row["query"]
    require(type(q) is int and 160 <= q < meta["length"], "query outside HDF5 bounds")
    require(
        type(row["disruptive"]) is bool and row["disruptive"] == meta["disruptive"],
        "disruptive flag mismatch",
    )
    time = meta["start"] + q / meta["rate"]
    close(row["query_time"], time, "query_time", atol=1e-12, rtol=0)
    if meta["disruptive"]:
        lead = (meta["down"] - time) * 1000
        rounded = float(np.round(lead, 9))
        require(rounded > 30, "query violates strict lead")
        close(row["actual_lead_ms"], rounded, "actual lead", atol=1e-8, rtol=0)
        require(
            row["requested_lead_ms"] in (250, 150, 100, 60, 35),
            "invalid requested lead",
        )
        require(
            abs(rounded - row["requested_lead_ms"]) <= 2, "requested lead not covered"
        )
        label = int(lead >= -1e-6 and lead <= 125 + 1e-6)
    else:
        require(
            row["actual_lead_ms"] is None and row["requested_lead_ms"] is None,
            "safe lead must be None",
        )
        label = 0
    require(
        type(row["label"]) is int and row["label"] == label,
        "label does not match HDF5 metadata",
    )
    for name, expected in (
        ("origin", q - 26),
        ("target_first", q - 25),
        ("target_last", q - 1),
        ("first_affected", q),
    ):
        require(
            type(row[name]) is int and row[name] == expected, f"{name}: clock mismatch"
        )
    observed = row["observed"]
    require(
        type(observed) is int and 0 <= observed <= 25 * 23, "observed count invalid"
    )
    require(
        np.all(vector(row["state256_before"], "state256_before") == 0),
        "initial state must be zero",
    )
    task_gradient = vector(row["task_gradient256"], "task_gradient256")
    hold_gradient = vector(row["base_hold_gradient256"], "base_hold_gradient256")
    if observed == 0:
        require(not np.any(hold_gradient), "no-target base hold gradient must be zero")
    require(set(row["results"]) == set(ARMS), "four arms required")
    verified = {key: row[key] for key in ("shot", "query", "disruptive", "label")}
    verified["results"] = {}
    gradients = {}
    for arm in ARMS:
        result = row["results"][arm]
        before = stable_bce(result["before_logit"], label)
        after = stable_bce(result["after_logit"], label)
        delta = after - before
        close(result["before_bce"], before, f"{arm} before_bce")
        close(result["after_bce"], after, f"{arm} after_bce")
        close(result["delta_bce"], delta, f"{arm} delta_bce")
        rms = finite(result["query_action_rms"], f"{arm} query_action_rms")
        require(0 <= rms <= 0.2500001, f"{arm} query_action_rms outside recorded range")
        rec = result["receipt"]
        grad = vector(rec["gradient256"], f"{arm} gradient256")
        state = vector(rec["state256_after"], f"{arm} state256_after")
        gradients[arm] = grad
        grad_norm, state_norm = (
            float(np.linalg.norm(grad)),
            float(np.linalg.norm(state)),
        )
        close(
            rec["gradient_norm"],
            grad_norm,
            f"{arm} gradient_norm",
            atol=1e-12,
            rtol=1e-6,
        )
        close(rec["step_norm"], state_norm, f"{arm} step_norm", atol=1e-12, rtol=1e-6)
        require(
            type(rec["observed"]) is int and rec["observed"] == observed,
            f"{arm} observed mismatch",
        )
        require(
            type(rec["accepted"]) is bool and type(rec["gradient_computed"]) is bool,
            f"{arm} receipt flags invalid",
        )
        if observed == 0:
            require(
                not rec["accepted"]
                and not rec["gradient_computed"]
                and not np.any(grad),
                f"{arm} no-target gradient invalid",
            )
            require(
                rec["reason"] == "no_target"
                and all(
                    rec[k] is None
                    for k in ("loss_before", "loss_proposal", "loss_retained")
                ),
                f"{arm} no-target loss invalid",
            )
            expected_rms = 0.0 if arm == "centered_real" else None
            require(
                rec["max_action_rms"] == expected_rms,
                f"{arm} no-target max_action_rms mismatch",
            )
        else:
            require(rec["gradient_computed"], f"{arm} missing gradient")
            loss_before = finite(rec["loss_before"], f"{arm} loss_before")
            loss_proposal = finite(rec["loss_proposal"], f"{arm} loss_proposal")
            require(
                rec["accepted"] == (loss_proposal < loss_before),
                f"{arm} strict loss descent mismatch",
            )
            close(
                rec["loss_retained"],
                loss_proposal if rec["accepted"] else loss_before,
                f"{arm} loss_retained",
                atol=0,
                rtol=0,
            )
            receipt_rms = finite(rec["max_action_rms"], f"{arm} max_action_rms")
            require(
                0 <= receipt_rms <= 0.2500001, f"{arm} max_action_rms outside range"
            )
        if rec["accepted"]:
            require(grad_norm > 0, f"{arm} accepted zero gradient")
            close(state_norm, 0.1, f"{arm} accepted step norm", atol=1e-7, rtol=1e-6)
            require(
                np.allclose(state, -0.1 * grad / grad_norm, atol=2e-8, rtol=2e-6),
                f"{arm} update direction mismatch",
            )
        else:
            require(not np.any(state), f"{arm} rejected state must remain zero")
        norms = grad_norm * float(np.linalg.norm(task_gradient))
        cosine = float(np.clip(grad @ task_gradient / norms, -1, 1)) if norms else None
        if cosine is None:
            require(
                result["risk_gradient_cosine"] is None, f"{arm} cosine must be None"
            )
        else:
            close(
                result["risk_gradient_cosine"],
                cosine,
                f"{arm} risk_gradient_cosine",
                atol=1e-9,
                rtol=1e-6,
            )
        if arm == "difference_hold":
            require(
                not np.any(grad) and not np.any(state) and not rec["accepted"],
                "difference_hold gradient/state must be zero",
            )
            require(
                delta == 0
                and result["delta_bce"] == 0
                and result["before_logit"] == result["after_logit"],
                "difference_hold delta must be zero",
            )
            if observed:
                require(
                    all(
                        rec[key] == 0
                        for key in ("loss_before", "loss_proposal", "loss_retained")
                    ),
                    "difference_hold objective must be zero",
                )
        verified["results"][arm] = {"delta_bce": delta}
    expected = gradients["centered_real"] - hold_gradient
    require(
        np.allclose(gradients["difference_real"], expected, atol=2e-7, rtol=2e-4),
        "gradient identity mismatch",
    )
    discrepancy = gradients["difference_real"] - expected
    close(
        row["difference_identity_max_absolute"],
        float(np.max(np.abs(discrepancy))),
        "gradient identity absolute diagnostic",
        atol=1e-14,
        rtol=1e-6,
    )
    relative = float(np.linalg.norm(discrepancy) / max(np.linalg.norm(expected), 1e-30))
    close(
        row["difference_identity_relative"],
        relative,
        "gradient identity relative diagnostic",
        atol=1e-12,
        rtol=1e-6,
    )
    return verified


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def check_snapshot(protocol):
    declared = protocol["code_sha256"]
    required = {
        PLAN,
        f"{MODULE}/__init__.py",
        f"{MODULE}/run.py",
        f"{MODULE}/core.py",
        f"{MODULE}/summary.py",
    }
    require(
        isinstance(declared, dict) and required <= set(declared),
        "missing declared source files",
    )
    require(declared[PLAN] == protocol["plan_sha256"], "plan hash mismatch")
    archive_path = OUTPUT / "code_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(
            len(names) == len(set(names)) and set(names) == set(declared),
            "source snapshot entries mismatch",
        )
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and not entry.is_dir(),
                "invalid source path",
            )
            require(entry.file_size <= 64 * 1024 * 1024, "oversized source entry")
            current = (ROOT / name).resolve()
            require(
                current.is_relative_to(ROOT.resolve()), "source path escapes workspace"
            )
            require(
                hashlib.sha256(archive.read(entry)).hexdigest() == declared[name],
                f"snapshot hash mismatch: {name}",
            )
            require(
                sha256_file(current) == declared[name],
                f"current source hash mismatch: {name}",
            )
    return sha256_file(archive_path)


def read_metadata(path):
    # Do not read any data/ip samples or any other raw signals.
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ValueError(
            "HDF5 metadata verification requires h5py; use rl_env"
        ) from exc
    with h5py.File(path, "r") as hdf:
        ip = hdf["data/ip"]
        require(len(ip.shape) == 1, "invalid ip shape")
        rate = float(np.asarray(ip.attrs["SampleRate"]).reshape(-1)[0])
        start = float(np.asarray(ip.attrs["StartTime"]).reshape(-1)[0])
        flag = bool(np.asarray(hdf["meta/IsDisrupt"][()]).reshape(-1)[0])
        raw_down = np.asarray(hdf["meta/DownTime"][()]).reshape(-1)
        down = float(raw_down[0]) if flag and len(raw_down) else None
        length = ip.shape[0]
    require(
        math.isfinite(rate) and abs(rate - 1000) <= 0.5, "invalid nominal sampling rate"
    )
    require(math.isfinite(start), "invalid sampling start")
    require(not flag or (down is not None and math.isfinite(down)), "invalid DownTime")
    return dict(length=length, rate=rate, start=start, disruptive=flag, down=down)


def verify_probe():
    completed_path = OUTPUT / "completed.json"
    done = read_json(completed_path)
    require(
        done.get("status") == "COMPLETED_ENDED20_LOCAL_PROBE", "probe is not completed"
    )
    protocol_path = OUTPUT / "protocol.json"
    protocol = read_json(protocol_path)
    fixed = {
        "scope": "CONDITIONAL_ENDED20_LOCAL_PROBE_NOT_ONLINE",
        "arms": list(ARMS),
        "source_sha256": SOURCE_SHA256,
        "reference_completed_sha256": REFERENCE_SHA256,
        "outcome_authority": "HDF5",
        "east_horizon_ms": 125,
        "strict_lead_ms": 30,
        "label_used_for_update": False,
        "query_selection_uses_ended_outcomes": True,
        "reset_each_local_sample": True,
        "step_size": 0.1,
        "anchor": 0.01,
        "radius": 0.25,
    }
    for key, expected in fixed.items():
        require(
            type(protocol.get(key)) is type(expected) and protocol[key] == expected,
            f"protocol mismatch: {key}",
        )
    ids = protocol["ids"]
    require(
        isinstance(ids, list)
        and len(ids) == len(set(ids)) == 20
        and all(type(x) is int and x > 0 for x in ids),
        "invalid protocol IDs",
    )
    require(protocol["raw_signal_ids_allowed"] == ids, "allowed IDs mismatch")
    require(
        done["raw_signal_ids_read"] == ids and done["raw_outcome_ids_read"] == ids,
        "recorded accessed IDs mismatch",
    )
    require(
        done["n_shots"] == 20 and done["suffix_raw_signals_read"] == 0,
        "unexpected shot count or suffix access",
    )
    require(
        done["source_tensors_unchanged"] is True
        and done["online_performance_measured"] is False
        and done["independent_future_confirmation"] is False,
        "completion scope mismatch",
    )
    require(
        Path(protocol["source_checkpoint"]).resolve() == SOURCE.resolve(),
        "unexpected source path",
    )
    require(
        Path(protocol["cold_state"]).resolve() == COLD.resolve(), "unexpected cold path"
    )
    require(
        Path(protocol["calibration_reference"]).resolve() == CALIBRATION.resolve(),
        "unexpected calibration path",
    )
    require(
        Path(protocol["reference"]).resolve()
        == (ROOT / "probe_outputs/lora_stream_v1").resolve(),
        "unexpected reference path",
    )
    for path, key in (
        (SOURCE, "source_sha256"),
        (COLD, "cold_state_sha256"),
        (CALIBRATION / "input_map.json", "input_map_sha256"),
        (OUTPUT / "source_basis.pt", "basis_sha256"),
    ):
        require(sha256_file(path) == protocol[key], f"hash mismatch: {key}")
    mapping = read_json(CALIBRATION / "input_map.json")
    require(
        mapping["calibration_ids"] == ids,
        "protocol IDs differ from fixed calibration IDs",
    )
    require(
        set(protocol["input_hashes"]) == {str(sid) for sid in ids},
        "input hash IDs mismatch",
    )
    snapshot_hash = check_snapshot(protocol)
    samples_path, sampling_path = OUTPUT / "samples.json", OUTPUT / "sampling.json"
    require(
        sha256_file(samples_path) == done["samples_sha256"], "samples hash mismatch"
    )
    require(
        sha256_file(sampling_path) == done["sampling_sha256"], "sampling hash mismatch"
    )
    rows, sampling = read_json(samples_path), read_json(sampling_path)
    require(len(rows) == done["n_samples"], "sample count mismatch")
    validate_membership(rows, sampling, ids)
    metadata = {}
    for sid in ids:
        path = ROOT / "data/EAST/unified_hdf5" / str(sid // 100 * 100) / f"{sid}.hdf5"
        require(
            sha256_file(path) == protocol["input_hashes"][str(sid)],
            "20-shot raw file hash mismatch",
        )
        metadata[sid] = read_metadata(path)
    for shot in sampling:
        meta = metadata[shot["shot"]]
        require(
            type(shot["disruptive"]) is bool
            and shot["disruptive"] == meta["disruptive"],
            "sampling disruptive mismatch",
        )
        selected, skipped = expected_sampling(meta)
        compare_tree(shot["selected"], selected, "selected")
        compare_tree(shot["skipped"], skipped, "skipped")
    validated = [validate_sample(row, metadata[row["shot"]]) for row in rows]
    summary = recompute_summary(validated)
    compare_tree(done["summary"], summary, "summary")
    return {
        "status": "PASS",
        "n_shots": len(ids),
        "n_samples": len(rows),
        "summary_recomputed": summary,
        "protocol_sha256": sha256_file(protocol_path),
        "completed_sha256": sha256_file(completed_path),
        "samples_sha256": done["samples_sha256"],
        "sampling_sha256": done["sampling_sha256"],
        "code_snapshot_sha256": snapshot_hash,
        "source_sha256": protocol["source_sha256"],
        "cold_state_sha256": protocol["cold_state_sha256"],
        "input_map_sha256": protocol["input_map_sha256"],
        "basis_sha256": protocol["basis_sha256"],
        "raw_files_checked": len(ids),
        "limitations": "Recorded RMS and gradients were checked without neural or signal replay. Protocol digest is reported, not externally authenticated. This is not future validation or online performance.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="create verification_records.json exclusively",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_probe()
        rendered = (
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        if args.write_report:
            with (OUTPUT / "verification_records.json").open(
                "x", encoding="utf-8"
            ) as report:
                report.write(rendered)
    except (ValueError, KeyError, OSError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
