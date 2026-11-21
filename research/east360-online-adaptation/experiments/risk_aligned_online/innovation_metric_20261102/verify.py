"""CPU-only consistency audit of sealed innovation-metric records.

This does not replay the neural model or read HDF5 signal datasets.  A PASS
describes record consistency, not a scientific gate or future validation.
The earlier v1 run was stopped and retained after a concurrent verifier edit
changed a file included in its runtime snapshot.  This verifier defaults to the
identical-scientific-configuration v2 infrastructure rerun.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "probe_outputs/innovation_metric360_v2"
REFERENCE = ROOT / "probe_outputs/critical_shot_meta360_v1"
TRAINING = ROOT / "probe_outputs/innovation_metric_heads20_v1"
OLD_TRAINING = ROOT / "probe_outputs/critical_shot_heads20_v1"
PLAN = "INNOVATION_METRIC_PLAN_20261102.md"
REFERENCE_SHA = "fb05bb2ea6e89f6924fb9e4b0b304b4f40da6734b5bd50be39ffaed123198ed1"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
COLD_SHA = "a06377a218a0f06def1eb3652cc091b5c514653681bffcf5dbfda7906f9db344"
THRESHOLD = 0.7258946895599365
HEADS = ("unit_operational", "metric_operational", "metric_forecast")
ARMS = (
    "cold",
    "contrast_unit",
    "innovation_unit",
    "innovation_metric",
    "metric_forecast",
    "metric_zero",
    "metric_reverse",
    "metric_hold",
)
LEARNERS = ARMS[1:]
REAL_ARMS = (
    "contrast_unit",
    "innovation_unit",
    "innovation_metric",
    "metric_forecast",
    "metric_zero",
)
CONTROLS = ("metric_reverse", "metric_hold", "metric_forecast", "metric_zero")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float)), name)
    require(math.isfinite(value), name)
    return float(value)


def bitwise_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def classify_curve(times, scores, disruptive, down, threshold):
    """Independently score the first preterminal crossing; 30 ms is not timely."""
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
    require(
        type(disruptive) is bool and 0 < finite(threshold, "threshold") < 1,
        "outcome/threshold",
    )
    valid = (
        (finite(down, "DownTime") - times) * 1000 >= -1e-6
        if disruptive
        else np.ones(len(times), dtype=bool)
    )
    crossings = np.flatnonzero(valid & (scores >= threshold))
    lead = (
        float(round((down - times[crossings[0]]) * 1000, 9))
        if disruptive and len(crossings)
        else None
    )
    return dict(
        alarm=bool(len(crossings)),
        lead_ms=lead,
        timely=bool(lead is not None and lead > 30),
    )


def recount(rows):
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    tp = sum(r["timely"] for r in rows)
    fp = sum(not r["disruptive"] and r["alarm"] for r in rows)
    tpr, fpr = (tp / nd if nd else None), (fp / ns if ns else None)
    leads = [r["lead_ms"] for r in rows if r["timely"]]
    return dict(
        n=len(rows),
        n_disrupt=nd,
        n_safe=ns,
        tp=tp,
        fp=fp,
        tpr=tpr,
        fpr=fpr,
        J=tpr - fpr if nd and ns else None,
        early_over3000=sum(lead > 3000 for lead in leads),
        median_timely_lead_ms=float(np.median(leads)) if leads else None,
    )


def paired_flips(reference, candidate):
    require(len(reference) == len(candidate), "flip length")
    result = {k: [] for k in ("rescued_D", "lost_D", "removed_FP", "added_FP")}
    for before, after in zip(reference, candidate, strict=True):
        require(
            (before["shot"], before["disruptive"])
            == (after["shot"], after["disruptive"]),
            "flip identity",
        )
        if before["disruptive"]:
            if after["timely"] and not before["timely"]:
                result["rescued_D"].append(before["shot"])
            if before["timely"] and not after["timely"]:
                result["lost_D"].append(before["shot"])
        else:
            if before["alarm"] and not after["alarm"]:
                result["removed_FP"].append(before["shot"])
            if after["alarm"] and not before["alarm"]:
                result["added_FP"].append(before["shot"])
    return result


def posthoc_caps(maxima, disruptive):
    """Development-cohort workpoints, not deployed thresholds."""
    values = np.asarray(maxima, dtype=np.float64)
    flags = np.asarray(disruptive, dtype=bool)
    require(
        values.shape == flags.shape and flags.any() and (~flags).any(), "posthoc cohort"
    )
    require(np.all(np.isfinite(values) | np.isneginf(values)), "posthoc maxima")
    thresholds = np.unique(
        np.r_[values[np.isfinite(values)], np.nextafter(1.0, np.inf)]
    )
    result = []
    for cap in (0.10, 0.15, 0.20):
        choices = []
        for threshold in thresholds:
            hits = values >= threshold
            tpr, fpr = float(hits[flags].mean()), float(hits[~flags].mean())
            if fpr <= cap:
                choices.append((tpr, -fpr, float(threshold)))
        tpr, neg_fpr, threshold = max(choices)
        result.append(
            dict(
                fpr_cap=cap,
                tpr=tpr,
                fpr=-neg_fpr,
                threshold=threshold,
                scope="POSTHOC_DEVELOPMENT_NOT_DEPLOYED",
            )
        )
    return result


def validate_curves(times, curves, ref_times, ref_cold, ref_operational, *, warmup):
    require(set(curves) == set(ARMS), "curve arms")
    times, ref_cold = np.asarray(times), np.asarray(ref_cold)
    require(bitwise_equal(times, ref_times), "reference times drift")
    require(
        times.ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    require(bitwise_equal(curves["cold"], ref_cold), "reference cold drift")
    require(
        bitwise_equal(curves["contrast_unit"], ref_operational),
        "reference contrast drift",
    )
    for arm in ARMS:
        value = np.asarray(curves[arm])
        require(
            value.shape == times.shape
            and value.dtype.str == ref_cold.dtype.str
            and np.isfinite(value).all()
            and np.all((0 <= value) & (value <= 1)),
            "invalid curve: " + arm,
        )
        require(bitwise_equal(value[:160], ref_cold[:160]), "first 160 drift: " + arm)
        if warmup:
            require(bitwise_equal(value, ref_cold), "warmup curve drift: " + arm)


def score_gates(scores):
    base, candidate = scores["cold"]["full360"], scores["innovation_metric"]["full360"]
    require(
        candidate["fpr"] is not None
        and candidate["J"] is not None
        and base["J"] is not None,
        "full cohort must contain both classes",
    )
    stage = (
        candidate["fpr"] <= 0.20
        and candidate["tp"] >= base["tp"]
        and candidate["fp"] <= base["fp"]
        and candidate["J"] > base["J"]
    )
    comparisons = {a: candidate["J"] > scores[a]["full360"]["J"] for a in CONTROLS}
    separation = all(comparisons.values())
    return dict(
        operational_gate=bool(stage),
        control_separation_gate=bool(separation),
        control_comparisons=comparisons,
        stage_success_pending_verification=bool(stage and separation),
    )


def read_outcome(shot):
    """Read only the authoritative outcome metadata, never signal datasets."""
    import h5py

    path = ROOT / "data/EAST/unified_hdf5" / str(shot // 100 * 100) / f"{shot}.hdf5"
    with h5py.File(path, "r") as handle:
        raw_flag = np.asarray(handle["meta/IsDisrupt"][()]).reshape(-1)
        raw_down = np.asarray(handle["meta/DownTime"][()]).reshape(-1)
    require(len(raw_flag) == 1 and raw_flag[0] in (0, 1), "invalid HDF outcome")
    flag = bool(raw_flag[0])
    down = float(raw_down[0]) if flag and len(raw_down) else None
    require(
        not flag or (down is not None and math.isfinite(down)), "invalid HDF DownTime"
    )
    return flag, down


def check_trace(info, arrays, length, warmup):
    require(
        info.get("reset_each_shot") is True and info.get("temporary_parameters") == 256,
        "trace protocol",
    )
    updates, actions = info["updates"], info["actions"]
    require(set(updates) == set(ARMS) and set(actions) == set(ARMS), "trace arms")
    require(
        set(arrays)
        == {a + suffix for a in ARMS for suffix in ("_before", "_after", "_gradient")},
        "trace arrays",
    )
    totals = {}
    for arm in ARMS:
        records = updates[arm]
        ends = [] if warmup or arm == "cold" else list(range(159, length, 128))
        require([r["maturity"] for r in records] == ends, "clock: " + arm)
        before, after, gradient = (
            np.asarray(arrays[arm + suffix])
            for suffix in ("_before", "_after", "_gradient")
        )
        require(
            before.shape == after.shape == (len(ends), 4, 64)
            and gradient.shape == (len(ends), 256),
            "state shape",
        )
        require(
            all(
                a.dtype == np.float32 and np.isfinite(a).all()
                for a in (before, after, gradient)
            ),
            "state finite/dtype",
        )
        if ends:
            require(
                bitwise_equal(before[0], np.zeros((4, 64), dtype=np.float32))
                and bitwise_equal(after[:-1], before[1:]),
                "state reset/continuity",
            )
        totals[arm] = dict(
            events=len(ends), accepted=0, nonzero_gradients=0, observed_scalars=0
        )
        for i, (record, end) in enumerate(zip(records, ends, strict=True)):
            require(
                record["origin"] == end - 25 and record["first_affected"] == end + 1,
                "causal clock",
            )
            for key in ("forecast_sha256", "mask_sha256", "target_sha256"):
                require(
                    isinstance(record.get(key), str) and HEX.fullmatch(record[key]),
                    "invalid " + key,
                )
            observed, accepted, computed = (
                record["observed"],
                record["accepted"],
                record["gradient_computed"],
            )
            require(
                type(observed) is int
                and 0 <= observed <= 550
                and type(accepted) is bool
                and type(computed) is bool,
                "receipt types",
            )
            grad_norm = finite(record["gradient_norm"], "gradient norm")
            step_norm = finite(record["step_norm"], "step norm")
            actual_grad = float(np.linalg.norm(gradient[i].astype(np.float64)))
            actual_step = float(
                np.linalg.norm((after[i] - before[i]).astype(np.float64))
            )
            require(
                grad_norm >= 0
                and math.isclose(grad_norm, actual_grad, rel_tol=1e-9, abs_tol=1e-14),
                "gradient norm",
            )
            require(
                step_norm >= 0
                and math.isclose(step_norm, actual_step, rel_tol=1e-9, abs_tol=1e-14),
                "step norm",
            )
            # Only the two unit-direction methods have a parameter norm bound.
            # The metric bound is in observed hidden space and cannot be checked
            # from these parameter receipts without the actual hidden features.
            if arm in ("contrast_unit", "innovation_unit"):
                require(step_norm <= 0.100001, "unit step norm")
            if observed:
                old = finite(record["loss_before"], "loss before")
                proposed = finite(record["loss_proposal"], "loss proposal")
                require(
                    computed
                    and accepted == (proposed < old)
                    and record["reason"]
                    == ("improved" if accepted else "not_improved"),
                    "acceptance",
                )
            else:
                require(
                    not computed
                    and not accepted
                    and record["reason"] == "no_target"
                    and grad_norm == 0
                    and step_norm == 0
                    and not np.any(gradient[i])
                    and "loss_before" not in record
                    and "loss_proposal" not in record,
                    "no-target sentinel",
                )
            if not accepted:
                require(bitwise_equal(before[i], after[i]), "rejected state changed")
            totals[arm]["accepted"] += int(accepted)
            totals[arm]["nonzero_gradients"] += int(grad_norm > 0)
            totals[arm]["observed_scalars"] += observed
    for group in zip(*(updates[a] for a in LEARNERS), strict=True):
        require(
            len({r["mask_sha256"] for r in group}) == 1
            and len({r["observed"] for r in group}) == 1,
            "learner mask",
        )
    for group in zip(*(updates[a] for a in REAL_ARMS), strict=True):
        require(len({r["target_sha256"] for r in group}) == 1, "real target hash")
    for stats in actions.values():
        endpoints, capped = stats["endpoints"], stats["capped_endpoints"]
        rms = finite(stats["max_rms"], "action RMS")
        require(
            type(endpoints) is int
            and endpoints == (0 if warmup else length)
            and type(capped) is int
            and 0 <= capped <= endpoints
            and 0 <= rms <= 0.250001
            and (endpoints > 0 or rms == 0),
            "action summary",
        )
    return totals, actions


def check_training(protocol, training_done, histories, cohort, warmup):
    require(
        [c["shot"] for c in cohort] == protocol["ids"] and len(cohort) == 20,
        "training cohort",
    )
    require(sum(c["disruptive"] for c in cohort) == 8, "training outcome balance")
    known = {c["shot"]: c for c in cohort}
    for shot, item in known.items():
        times, flag, down = warmup[shot]
        require(
            item["disruptive"] is flag
            and item["down"] == down
            and item["points"] == len(times),
            "training HDF metadata",
        )
    require(len(histories) == 128, "training step count")
    support = {a: 0 for a in HEADS}
    groups = {
        flag: [c["shot"] for c in cohort if c["disruptive"] is flag]
        for flag in (True, False)
    }
    rng = np.random.default_rng(131)
    for step, row in enumerate(histories, 1):
        ids = row["sample_shots"]
        expected = [
            groups[flag][int(rng.integers(len(groups[flag])))] for flag in (True, False)
        ]
        require(
            row["step"] == step
            and len(ids) == 2
            and all(s in known for s in ids)
            and ids == expected
            and known[ids[0]]["disruptive"] is True
            and known[ids[1]]["disruptive"] is False,
            "training sample",
        )
        require(
            set(row["episodes"]) == set(HEADS)
            and set(row["gradient_norms"]) == set(HEADS),
            "training heads",
        )
        for arm in HEADS:
            require(
                finite(row["gradient_norms"][arm], "head gradient") >= 0,
                "head gradient",
            )
            episodes = row["episodes"][arm]
            require(
                len(episodes) == 2 and [e["shot"] for e in episodes] == ids,
                "episode shots",
            )
            for event in episodes:
                times, flag, down = warmup[event["shot"]]
                query = event["query"]
                require(
                    type(query) is int and 0 <= query < len(times), "training query"
                )
                require(
                    not flag or round((down - times[query]) * 1000, 9) > 30,
                    "training timely query",
                )
                supported = query >= 160
                maturity = 159 + ((query - 160) // 128) * 128 if supported else None
                require(
                    event["supported"] is supported and event["maturity"] == maturity,
                    "training support clock",
                )
                require(
                    type(event["label"]) is int and event["label"] == int(flag),
                    "training task label",
                )
                for key in ("risk_before", "risk_after", "signal", "objective"):
                    finite(event[key], "training " + key)
                if not supported:
                    require(
                        event["risk_before"] == event["risk_after"],
                        "unsupported risk changed",
                    )
                support[arm] += int(supported)
    require(training_done["support_counts"] == support, "training support totals")
    return support


def check_training_contract(
    done, protocol, ids, source_hash, cold_hash, map_hash, plan_hash
):
    completion = dict(
        status="COMPLETED",
        steps=128,
        head_optimizer_steps=128,
        source_tensors_unchanged=True,
        ids=ids,
        outcome_reads=20,
        suffix_shots_read=0,
        plan_sha256=plan_hash,
    )
    fixed = dict(
        scope="ENDED20_INNOVATION_METRIC_EQUAL_BUDGET",
        ids=ids,
        source_sha256=source_hash,
        cold_state_sha256=cold_hash,
        input_map_sha256=map_hash,
        plan_sha256=plan_hash,
        heads=dict(
            unit_operational="unit",
            metric_operational="metric",
            metric_forecast="metric",
        ),
        forecast_parameters_per_head=575,
        steps=128,
        optimizer="Adam",
        learning_rate=0.001,
        seed=131,
        gradient_clip=1.0,
        temporal_gradient_truncation_updates=1,
        first_update=159,
        update_interval=128,
        parameter_unit_step=0.1,
        metric_learning_rate=1.0,
        relative_metric_damping=0.01,
        absolute_metric_damping=1e-8,
        metric_increment_hidden_rms_cap=0.05,
        total_hidden_rms_cap=0.25,
        warmup_cost_retained=True,
        independent_confirmation=False,
    )
    for key, expected in completion.items():
        require(done.get(key) == expected, "training completion: " + key)
    for key, expected in fixed.items():
        require(protocol.get(key) == expected, "training protocol: " + key)


def check_shot_seal(
    index, shot, seal, manifest, seal_hash, curve_hash, state_hash, updates_hash
):
    require(
        seal.get("shot") == shot
        and seal.get("index") == index
        and seal.get("warmup_curves_reused") is (index <= 20)
        and seal.get("warmup_outcome_already_known_for_head_fit") is (index <= 20)
        and seal.get("current_outcome_used_in_prediction") is False,
        "seal identity/flags",
    )
    require(
        manifest == dict(shot=shot, index=index, seal_sha256=seal_hash),
        "completion-anchored seal",
    )
    require(
        seal.get("sha256") == curve_hash
        and seal.get("states_sha256") == state_hash
        and seal.get("updates_sha256") == updates_hash,
        "sealed file hash",
    )


def check_snapshot(root, protocol, *, training=False):
    declared = protocol["code_sha256"]
    module = "experiments/risk_aligned_online/innovation_metric_20261102/"
    required = {PLAN, module + "__init__.py", module + "core.py", module + "stream.py"}
    required.add(module + ("prepare.py" if training else "run.py"))
    require(
        isinstance(declared, dict) and required <= set(declared), "snapshot entries"
    )
    require(declared[PLAN] == protocol["plan_sha256"], "snapshot plan hash")
    current = {}
    archive_path = root / "code_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(
            len(names) == len(set(names)) and set(names) == set(declared),
            "archive names",
        )
        require(sum(e.file_size for e in entries) <= 512 * 1024 * 1024, "archive size")
        for entry in entries:
            name, path = entry.filename, PurePosixPath(entry.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and not entry.is_dir()
                and entry.file_size <= 64 * 1024 * 1024,
                "archive path",
            )
            require(
                name == PLAN
                or (
                    path.parts[:2] == ("experiments", "risk_aligned_online")
                    and path.suffix == ".py"
                ),
                "unexpected non-code snapshot entry",
            )
            expected = declared[name]
            require(
                isinstance(expected, str) and HEX.fullmatch(expected),
                "archive declared hash",
            )
            require(
                hashlib.sha256(archive.read(entry)).hexdigest() == expected,
                "archive entry hash: " + name,
            )
            current_path = ROOT / name
            if current_path.is_file():
                resolved = current_path.resolve()
                require(
                    resolved.is_relative_to(ROOT.resolve())
                    and resolved.suffix == path.suffix,
                    "current code path is not a workspace code file",
                )
                current[name] = sha_file(resolved)
            else:
                current[name] = None
    drift = {
        name: {"snapshot": declared[name], "actual_current": actual}
        for name, actual in current.items()
        if actual != declared[name]
    }
    return dict(
        archive_bytes_sha256=sha_file(archive_path),
        archive_entries_verified=len(declared),
        actual_current_sha256=current,
        current_drift=drift,
    )


def check_run_contract(protocol, reference):
    fixed = dict(
        scope="DEVELOPMENT_COMPLETE360_INNOVATION_METRIC",
        arms=list(ARMS),
        threshold=THRESHOLD,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        input_window_points=128,
        diagnostic_forecast_points=25,
        parameter_step=0.1,
        parameter_unit_step=0.1,
        metric_learning_rate=1.0,
        metric_relative_damping=0.01,
        metric_absolute_damping=1e-8,
        metric_increment_hidden_rms_cap=0.05,
        hidden_rms_cap=0.25,
        temporary_parameters=256,
        reset_each_shot=True,
        head_fitted_ended20_only=True,
        head_training_shots=20,
        head_optimizer_steps=128,
        suffix_head_updates=0,
        suffix_inter_supervision=0,
        warmup_cost_included=True,
        suffix_uncached_real_gpu=True,
        outcome_authority="HDF5",
        outcome_fields=["IsDisrupt", "DownTime"],
        lead_strictly_greater_ms=30,
        run_selection_or_threshold_tuning=False,
        no_outcome_input_to_simulate=True,
        independent_future_confirmation=False,
        real_time_latency_verified=False,
    )
    for key, expected in fixed.items():
        require(protocol.get(key) == expected, "run protocol: " + key)
    assignment = {
        "cold": dict(head=None, mode="real", geometry="contrast"),
        "contrast_unit": dict(head="old_operational", mode="real", geometry="contrast"),
        "innovation_unit": dict(head="unit_operational", mode="real", geometry="unit"),
        "innovation_metric": dict(
            head="metric_operational", mode="real", geometry="metric"
        ),
        "metric_forecast": dict(head="metric_forecast", mode="real", geometry="metric"),
        "metric_zero": dict(head="zero", mode="real", geometry="metric"),
        "metric_reverse": dict(
            head="metric_operational", mode="reverse", geometry="metric"
        ),
        "metric_hold": dict(head="metric_operational", mode="hold", geometry="metric"),
    }
    require(protocol.get("head_assignment") == assignment, "run head assignment")
    order = protocol["order"]
    ids = [entry["shot"] for entry in order]
    require(
        len(ids) == len(set(ids)) == 360
        and all(type(s) is int and s > 0 for s in ids)
        and order == reference["order"],
        "360 distinct ordered shots",
    )
    for key in ("source_sha256", "cold_state_sha256", "input_map_sha256", "threshold"):
        require(protocol[key] == reference[key], "reference protocol: " + key)
    for key, expected in (
        ("reference", REFERENCE),
        ("training", TRAINING),
        ("old_training", OLD_TRAINING),
    ):
        require(
            Path(protocol[key]).resolve() == expected.resolve(),
            "unexpected path: " + key,
        )
    return ids


def check_model_files(protocol, reference):
    for key, hash_key, expected in (
        ("source_checkpoint", "source_sha256", SOURCE_SHA),
        ("cold_state", "cold_state_sha256", COLD_SHA),
    ):
        path = Path(protocol[key]).resolve(strict=True)
        require(path == Path(reference[key]).resolve(strict=True), "model path: " + key)
        require(
            protocol[hash_key] == reference[hash_key] == expected == sha_file(path),
            "model hash: " + key,
        )


def check_heads(path, names):
    heads = read_json(path)
    require(set(heads) == set(names), "training head names")
    for head in heads.values():
        table = np.asarray(head, dtype=np.float64)
        require(
            table.shape == (25, 23) and np.isfinite(table).all(), "head shape/finite"
        )


def verify(output):
    """Check a completed run only; this function does not write a report."""
    output = Path(output)
    done = read_json(output / "completed.json")
    require(
        done.get("status") == "COMPLETED",
        "completed 360 run required before reading records",
    )
    for filename, key in (
        ("protocol.json", "protocol_sha256"),
        ("code_snapshot.zip", "code_snapshot_sha256"),
        ("per_shot.json", "per_shot_sha256"),
        ("shot_seals.json", "shot_seals_sha256"),
    ):
        require(
            sha_file(output / filename) == done[key], "completion hash: " + filename
        )
    protocol = read_json(output / "protocol.json")
    require(
        sha_file(REFERENCE / "completed.json")
        == protocol["reference_completed_sha256"]
        == REFERENCE_SHA,
        "reference completion hash",
    )
    for filename, key in (
        ("protocol.json", "reference_protocol_sha256"),
        ("verification_records.json", "reference_records_sha256"),
        ("verification_neural.json", "reference_neural_sha256"),
    ):
        require(
            sha_file(REFERENCE / filename) == protocol[key],
            "reference hash: " + filename,
        )
    reference = read_json(REFERENCE / "protocol.json")
    records = read_json(REFERENCE / "verification_records.json")
    neural = read_json(REFERENCE / "verification_neural.json")
    refhash = records["hashes"]
    require(
        records["status"] == neural["status"] == "PASS"
        and refhash["completed_sha256"] == neural["completed_sha256"] == REFERENCE_SHA
        and refhash["protocol_sha256_at_verification"]
        == protocol["reference_protocol_sha256"],
        "reference verification binding",
    )
    ids = check_run_contract(protocol, reference)
    check_model_files(protocol, reference)
    require(
        refhash["source_sha256"] == SOURCE_SHA
        and refhash["cold_state_sha256"] == COLD_SHA,
        "reference model hashes",
    )
    require(len(refhash["shot_files"]) == 360, "reference per-shot verification count")
    calibration = Path(protocol["calibration_reference"])
    require(
        calibration.resolve() == Path(reference["calibration_reference"]).resolve(),
        "calibration path",
    )
    upstream = Path(reference["reference"])
    upstream_check = read_json(upstream / "verification_records.json")
    require(
        upstream_check["status"] == "PASS"
        and sha_file(upstream / "completed.json")
        == upstream_check["completed_sha256"]
        == reference["reference_completed_sha256"]
        and sha_file(upstream / "protocol.json") == upstream_check["protocol_sha256"],
        "input-hash authority verification",
    )
    map_hash = sha_file(calibration / "input_map.json")
    input_hashes = sha_file(calibration / "input_hashes.json")
    require(
        map_hash == protocol["input_map_sha256"] == refhash["input_map_sha256"]
        and input_hashes
        == protocol["input_hashes_sha256"]
        == upstream_check["input_hashes_sha256"],
        "input map/hash drift",
    )
    require(
        read_json(calibration / "input_map.json")["calibration_ids"] == ids[:20]
        and set(read_json(calibration / "input_hashes.json")) == {str(s) for s in ids},
        "calibration cohort",
    )
    plan_hash = sha_file(ROOT / PLAN)
    require(plan_hash == protocol["plan_sha256"], "plan hash")
    train_done = read_json(TRAINING / "completed.json")
    train_protocol = read_json(TRAINING / "protocol.json")
    check_training_contract(
        train_done, train_protocol, ids[:20], SOURCE_SHA, COLD_SHA, map_hash, plan_hash
    )
    for filename, key in (
        ("completed.json", "training_completed_sha256"),
        ("protocol.json", "training_protocol_sha256"),
        ("heads.json", "heads_sha256"),
        ("history.json", "history_sha256"),
        ("training_cohort.json", "training_cohort_sha256"),
    ):
        require(
            sha_file(TRAINING / filename) == protocol[key], "training hash: " + filename
        )
    require(
        train_done["heads_sha256"] == protocol["heads_sha256"]
        and train_done["history_sha256"] == protocol["history_sha256"],
        "training completion hashes",
    )
    old_done = read_json(OLD_TRAINING / "completed.json")
    old_protocol = read_json(OLD_TRAINING / "protocol.json")
    require(
        sha_file(OLD_TRAINING / "completed.json")
        == protocol["old_training_completed_sha256"]
        == reference["training_completed_sha256"]
        == refhash["training_completed_sha256"]
        and old_done["status"] == "COMPLETED"
        and old_done["head_optimizer_steps"] == 128
        and sha_file(OLD_TRAINING / "heads.json")
        == protocol["old_heads_sha256"]
        == old_done["heads_sha256"]
        == reference["heads_sha256"]
        == refhash["heads_sha256"]
        and sha_file(OLD_TRAINING / "history.json") == old_done["history_sha256"]
        and old_protocol["ids"] == ids[:20]
        and old_protocol["source_sha256"] == SOURCE_SHA
        and old_protocol["cold_state_sha256"] == COLD_SHA
        and old_protocol["input_map_sha256"] == map_hash,
        "old contrast training",
    )
    check_heads(TRAINING / "heads.json", HEADS)
    check_heads(OLD_TRAINING / "heads.json", ("forecast", "point", "operational"))
    snapshots = dict(
        run=check_snapshot(output, protocol),
        training=check_snapshot(TRAINING, train_protocol, training=True),
    )
    saved, manifests = (
        read_json(output / "per_shot.json"),
        read_json(output / "shot_seals.json"),
    )
    require(
        set(saved) == set(ARMS)
        and all(len(saved[a]) == 360 for a in ARMS)
        and len(manifests) == 360,
        "per-shot records",
    )
    rows, maxima, warmup, shot_hashes = (
        {a: [] for a in ARMS},
        {a: [] for a in ARMS},
        {},
        [],
    )
    totals = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, shot in enumerate(ids, 1):
        stem = f"{index:04d}_{shot}"
        base, refbase = output / "shots" / stem, REFERENCE / "shots" / stem
        curve, state, info_path, seal_path = (
            Path(str(base) + suffix)
            for suffix in (".npz", "_states.npz", "_updates.json", "_seal.json")
        )
        refcurve, refseal_path = (
            Path(str(refbase) + ".npz"),
            Path(str(refbase) + "_seal.json"),
        )
        seal, refseal = read_json(seal_path), read_json(refseal_path)
        seal_hash, curve_hash = sha_file(seal_path), sha_file(curve)
        state_hash, updates_hash = sha_file(state), sha_file(info_path)
        check_shot_seal(
            index,
            shot,
            seal,
            manifests[index - 1],
            seal_hash,
            curve_hash,
            state_hash,
            updates_hash,
        )
        refentry = refhash["shot_files"][index - 1]
        require(
            refentry["shot"] == shot
            and refentry["index"] == index
            and sha_file(refseal_path) == refentry["seal_sha256_at_verification"]
            and refseal["shot"] == shot
            and refseal["index"] == index
            and refseal["current_outcome_used_in_prediction"] is False
            and sha_file(refcurve) == refseal["sha256"] == refentry["curve_sha256"],
            "reference shot seal",
        )
        shot_hashes.append(
            dict(
                shot=shot,
                index=index,
                seal_sha256=seal_hash,
                curve_sha256=curve_hash,
                states_sha256=state_hash,
                updates_sha256=updates_hash,
                reference_curve_sha256=refseal["sha256"],
                reference_seal_sha256=refentry["seal_sha256_at_verification"],
            )
        )
        with np.load(curve, allow_pickle=False) as archive:
            require(set(archive.files) == {"times", *ARMS}, "curve keys")
            data = {name: archive[name] for name in archive.files}
        with np.load(refcurve, allow_pickle=False) as archive:
            ref_times, ref_cold, ref_operational = (
                archive["times"],
                archive["cold"],
                archive["operational"],
            )
        times = data["times"]
        validate_curves(
            times,
            {a: data[a] for a in ARMS},
            ref_times,
            ref_cold,
            ref_operational,
            warmup=index <= 20,
        )
        flag, down = read_outcome(shot)
        if index <= 20:
            warmup[shot] = (times, flag, down)
        with np.load(state, allow_pickle=False) as archive:
            state_arrays = {name: archive[name] for name in archive.files}
        shot_totals, shot_actions = check_trace(
            read_json(info_path), state_arrays, len(times), index <= 20
        )
        eligible = (
            np.round((down - times) * 1000, 9) > 30
            if flag
            else np.ones(len(times), bool)
        )
        for arm in ARMS:
            values = data[arm]
            row = dict(
                shot=shot,
                disruptive=flag,
                **classify_curve(times, values, flag, down, THRESHOLD),
            )
            require(
                row == saved[arm][index - 1], "per-shot scoring: " + stem + " " + arm
            )
            rows[arm].append(row)
            maxima[arm].append(
                float(values[eligible].max()) if eligible.any() else -np.inf
            )
            for key in totals[arm]:
                totals[arm][key] += shot_totals[arm][key]
            for key in ("endpoints", "capped_endpoints"):
                actions[arm][key] += shot_actions[arm][key]
            actions[arm]["max_rms"] = max(
                actions[arm]["max_rms"], shot_actions[arm]["max_rms"]
            )
    support = check_training(
        train_protocol,
        train_done,
        read_json(TRAINING / "history.json"),
        read_json(TRAINING / "training_cohort.json"),
        warmup,
    )
    metrics = {
        a: {
            name: recount(part)
            for name, part in (
                ("full360", rows[a]),
                ("warmup20", rows[a][:20]),
                ("suffix340", rows[a][20:]),
            )
        }
        for a in ARMS
    }
    flips = {a: paired_flips(rows["cold"], rows[a]) for a in LEARNERS}
    require(
        metrics == done["metrics"] and flips == done["flips"], "metrics or paired flips"
    )
    require(
        totals == done["update_statistics"] and actions == done["action_statistics"],
        "update/action totals",
    )
    gates = score_gates(metrics)
    require(all(done.get(key) == value for key, value in gates.items()), "gates")
    require(
        done["fresh_curves"] == 340 * len(ARMS)
        and done["reused_warmup_curves"] == 20 * len(ARMS)
        and done["source_tensors_unchanged"] is True
        and done["heads_unchanged"] is True
        and done["method_promoted"] is False
        and done["scientific_goal_achieved"] is False
        and done["independent_future_confirmation"] is False,
        "completion flags",
    )
    flags = [r["disruptive"] for r in rows["cold"]]
    return dict(
        status="PASS",
        scope="RECORDED_DATA_CONSISTENCY_ONLY_NOT_INDEPENDENT_NEURAL_REPLAY",
        cases=360,
        curves=360 * len(ARMS),
        metadata_outcomes_read=360,
        raw_signal_datasets_read=0,
        raw_hdf5_file_hashes_recomputed=False,
        threshold=THRESHOLD,
        metrics=metrics,
        flips=flips,
        update_statistics=totals,
        action_statistics=actions,
        training_supported_episodes=support,
        **gates,
        posthoc_workpoints={a: posthoc_caps(maxima[a], flags) for a in ARMS},
        hashes=dict(
            completed_sha256=sha_file(output / "completed.json"),
            protocol_sha256=done["protocol_sha256"],
            per_shot_sha256=done["per_shot_sha256"],
            shot_seals_sha256=done["shot_seals_sha256"],
            source_sha256=SOURCE_SHA,
            cold_state_sha256=COLD_SHA,
            reference_completed_sha256=REFERENCE_SHA,
            training_completed_sha256=protocol["training_completed_sha256"],
            heads_sha256=protocol["heads_sha256"],
            history_sha256=protocol["history_sha256"],
            input_map_sha256=map_hash,
            plan_sha256=plan_hash,
            verifier_sha256=sha_file(Path(__file__)),
            snapshots=snapshots,
            shot_files=shot_hashes,
        ),
        infrastructure_note=(
            "The operator stopped and retained v1 after concurrent verifier snapshot drift; "
            "v2 is the stated identical-scientific-configuration infrastructure rerun."
        ),
        limitations=[
            "No independent neural replay, signal reconstruction, current mask or action recomputation.",
            "Training critical maxima, neural derivatives and seal-before-outcome timing are recorded claims, not independently reproduced.",
            "Recorded action summaries and parameter receipts cannot independently prove actual total or incremental hidden RMS.",
            "Raw HDF5 hashes were not recomputed; metadata outcome reads are not raw-file authentication.",
            "Artifact hashes provide consistency, not externally authenticated provenance.",
            "Posthoc operating points use the development cohort and are NOT_DEPLOYED.",
            "No independent future validation, real-time latency or physical contribution is established.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report_path = args.output / "verification_records.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = verify(args.output)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(json.dumps({key: report[key] for key in ("status", "cases", "curves")}))


if __name__ == "__main__":
    main()
