"""CPU record consistency only; run with python -m ...full_path_meta_20261102.verify.

No neural model, runner classifier/gates, or raw HDF5 signals are imported/read.
Only explicit CLI execution writes a report, after a COMPLETED artifact exists.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from ..innovation_metric_20261102 import verify as prior

ROOT = prior.ROOT
OUTPUT = ROOT / "probe_outputs/full_path_forecast_controls360_v1"
REFERENCE = ROOT / "probe_outputs/innovation_metric360_v2"
TRAINING = ROOT / "probe_outputs/full_path_heads20_v1"
OLD_TRAINING = ROOT / "probe_outputs/innovation_metric_heads20_v1"
REFERENCE_SHA = "f558ee30e33192da29daa7863f58267d25913a9c8dc25a645188ab32cac11014"
FULL_HEAD_SHA = "26370f1eb13a19fbd4f0200114fbca77b025cc2d8bdd1b7ae8f24d8e9ecbbf34"
FULL_DONE_SHA = "5b4baf0581fad9cbeeb3626dc161e63fd4bf5c71c6e6225c2031871539466542"
PLAN = "FULL_PATH_META_PLAN_20261102.md"
EVALUATION = "FULL_PATH_CONTROL_EVALUATION_20261102.md"
ACCEPTANCE = "USER_ACCEPTANCE_ALIGNMENT_20261102.md"
MODULE = "experiments/risk_aligned_online/full_path_meta_20261102/"
INNOVATION = "experiments/risk_aligned_online/innovation_metric_20261102/"
RUN_REQUIRED = {
    PLAN,
    EVALUATION,
    MODULE + "__init__.py",
    MODULE + "run.py",
    INNOVATION + "__init__.py",
    INNOVATION + "core.py",
    INNOVATION + "stream.py",
}
REUSED = dict(cold="cold", truncated_metric="innovation_metric")
FRESH = (
    "forecast_real",
    "forecast_reverse",
    "forecast_hold",
    "full_real",
    "full_reverse",
    "full_hold",
)
ARMS = (*REUSED, *FRESH)
STATISTICS_SCOPE = "recomputed_this_run_only_not_historical_activity"
THRESHOLD = prior.THRESHOLD
require, finite = prior.require, prior.finite
read_json, sha_file = prior.read_json, prior.sha_file
classify_curve, recount, paired_flips = (
    prior.classify_curve,
    prior.recount,
    prior.paired_flips,
)
bitwise_equal, read_outcome = prior.bitwise_equal, prior.read_outcome


def score_gates(scores, zero):
    metrics = [zero, *(scores[a]["full360"] for a in ARMS)]
    require(
        all((m["n_disrupt"], m["n_safe"]) == (180, 180) for m in metrics),
        "fixed balanced180/180 required",
    )
    require(
        all(
            type(m[k]) is int and 0 <= m[k] <= 180
            for m in metrics
            for k in ("tp", "fp")
        ),
        "integer counts required",
    )

    def net(m):
        return m["tp"] - m["fp"]

    base, full = scores["cold"]["full360"], scores["full_real"]["full360"]
    stage = full["fp"] <= 36 and full["tp"] >= base["tp"] and net(full) > net(base)
    controls = {
        a: net(full) > net(scores[a]["full360"])
        for a in ("full_reverse", "full_hold", "forecast_real", "truncated_metric")
    }
    controls["reference_metric_zero"] = net(full) > net(zero)
    forecast = {
        a: net(scores["forecast_real"]["full360"]) > net(scores[a]["full360"])
        for a in ("forecast_reverse", "forecast_hold")
    }
    return dict(
        operational_gate=stage,
        control_comparisons=controls,
        control_separation_gate=all(controls.values()),
        forecast_control_comparisons=forecast,
        forecast_control_separation_gate=all(forecast.values()),
        stage_success_pending_verification=stage and all(controls.values()),
    )


def summarize(rows, zero):
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
    flips = {a: paired_flips(rows["cold"], rows[a]) for a in ARMS[1:]}
    base = metrics["cold"]["full360"]
    deltas = {}
    for a, f in flips.items():
        m = metrics[a]["full360"]
        deltas[a] = dict(
            delta_tp=m["tp"] - base["tp"],
            delta_fp=m["fp"] - base["fp"],
            delta_tpr=m["tpr"] - base["tpr"],
            delta_J=(m["tp"] - m["fp"] - base["tp"] + base["fp"]) / 180,
            net_favorable_flips=len(f["rescued_D"])
            + len(f["removed_FP"])
            - len(f["lost_D"])
            - len(f["added_FP"]),
        )
    return dict(
        metrics=metrics,
        flips=flips,
        deltas_vs_cold=deltas,
        forecast_matched_flips={
            a: paired_flips(rows[a], rows["forecast_real"])
            for a in ("forecast_reverse", "forecast_hold")
        },
        full360_integer_net={
            a: metrics[a]["full360"]["tp"] - metrics[a]["full360"]["fp"] for a in ARMS
        },
        **score_gates(metrics, zero),
    )


def validate_curves(times, curves, reference, *, warmup):
    require(
        set(curves) == set(ARMS) and bitwise_equal(times, reference["times"]),
        "curve arms/reference times",
    )
    times = np.asarray(times)
    require(
        times.ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    for a, original in {**REUSED, "forecast_real": "metric_forecast"}.items():
        require(bitwise_equal(curves[a], reference[original]), "reference curve: " + a)
    for a, values in curves.items():
        require(
            values.shape == times.shape
            and values.dtype.str == reference["cold"].dtype.str
            and np.isfinite(values).all()
            and np.all((0 <= values) & (values <= 1)),
            "invalid curve: " + a,
        )
        require(bitwise_equal(values[:160], reference["cold"][:160]), "first160: " + a)
        if warmup:
            require(bitwise_equal(values, reference["cold"]), "warmup: " + a)


def check_event(record, before, after, gradient, end):
    require(
        record["maturity"] == end
        and record["origin"] == end - 25
        and record["first_affected"] == end + 1,
        "causal receipt clock",
    )
    for key in ("forecast_sha256", "mask_sha256", "target_sha256"):
        require(
            isinstance(record.get(key), str) and prior.HEX.fullmatch(record[key]),
            "invalid receipt hash: " + key,
        )
    observed, accepted, computed = (
        record[k] for k in ("observed", "accepted", "gradient_computed")
    )
    require(
        type(observed) is int
        and 0 <= observed <= 550
        and type(accepted) is bool
        and type(computed) is bool,
        "receipt types",
    )
    g, step = (
        finite(record["gradient_norm"], "gradient norm"),
        finite(record["step_norm"], "step norm"),
    )
    require(
        g >= 0
        and np.isclose(
            g, np.linalg.norm(gradient.astype(np.float64)), rtol=1e-9, atol=1e-14
        ),
        "gradient norm",
    )
    require(
        step >= 0
        and np.isclose(
            step,
            np.linalg.norm((after - before).astype(np.float64)),
            rtol=1e-9,
            atol=1e-14,
        ),
        "step norm",
    )
    # Metric steps are constrained in hidden space, NOT by parameter norm <= .1.
    if observed:
        old, proposed = (
            finite(record["loss_before"], "loss before"),
            finite(record["loss_proposal"], "loss proposal"),
        )
        require(
            computed
            and accepted == (proposed < old)
            and record["reason"] == ("improved" if accepted else "not_improved"),
            "acceptance",
        )
    else:
        require(
            not computed
            and not accepted
            and g == step == 0
            and not np.any(gradient)
            and record["reason"] == "no_target"
            and "loss_before" not in record
            and "loss_proposal" not in record,
            "no target",
        )
    require(accepted or bitwise_equal(before, after), "rejected state changed")
    return dict(
        events=1,
        accepted=int(accepted),
        nonzero_gradients=int(g > 0),
        observed_scalars=observed,
    )


def check_trace(info, arrays, length, *, index, reference_path, reference_sha):
    require(
        info.get("reset_each_shot") is True
        and info.get("temporary_parameters") == 256
        and info.get("statistics_scope") == STATISTICS_SCOPE
        and info.get("reference_curve_sha256") == reference_sha,
        "trace contract",
    )
    require(
        all(set(info[k]) == set(ARMS) for k in ("updates", "actions", "execution")),
        "trace arms",
    )
    require(
        set(arrays)
        == {a + s for a in ARMS for s in ("_before", "_after", "_gradient")},
        "trace arrays",
    )
    totals = {}
    for a in ARMS:
        fresh = index > 20 and a in FRESH
        execution = info["execution"][a]
        expected = dict(
            kind="reused_warmup"
            if index <= 20
            else ("fresh_uncached_gpu" if fresh else "reused_reference"),
            reference=None if fresh else str(reference_path),
            reference_arm="cold" if index <= 20 else REUSED.get(a),
        )
        require(execution == expected, "execution/reference: " + a)
        ends = list(range(159, length, 128)) if fresh else []
        records = info["updates"][a]
        require([e["maturity"] for e in records] == ends, "update clock: " + a)
        before, after, gradient = (
            arrays[a + s] for s in ("_before", "_after", "_gradient")
        )
        require(
            before.shape == after.shape == (len(ends), 4, 64)
            and gradient.shape == (len(ends), 256),
            "state shapes",
        )
        require(
            all(
                x.dtype == np.float32 and np.isfinite(x).all()
                for x in (before, after, gradient)
            ),
            "state dtype/finite",
        )
        if ends:
            require(
                bitwise_equal(before[0], np.zeros((4, 64), np.float32))
                and bitwise_equal(after[:-1], before[1:]),
                "reset/stitch",
            )
        totals[a] = dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for i, end in enumerate(ends):
            for k, value in check_event(
                records[i], before[i], after[i], gradient[i], end
            ).items():
                totals[a][k] += value
        action = info["actions"][a]
        n, capped, rms = (
            action["endpoints"],
            action["capped_endpoints"],
            finite(action["max_rms"], "recorded RMS"),
        )
        require(
            type(n) is int
            and n == (length if fresh else 0)
            and type(capped) is int
            and 0 <= capped <= n
            and 0 <= rms <= 0.250001
            and (fresh or (capped == 0 and rms == 0)),
            "action execution summary",
        )
    for group in zip(*(info["updates"][a] for a in FRESH), strict=True):
        require(
            len({r["mask_sha256"] for r in group}) == 1
            and len({r["observed"] for r in group}) == 1,
            "shared mask",
        )
    for x, y in zip(
        info["updates"]["forecast_real"], info["updates"]["full_real"], strict=True
    ):
        require(x["target_sha256"] == y["target_sha256"], "real target equality")
    return totals, info["actions"]


def check_training(protocol, done, history, cohort, warmup):
    require(
        len(cohort) == 20
        and [c["shot"] for c in cohort] == protocol["ids"]
        and sum(c["disruptive"] for c in cohort) == 8,
        "training cohort",
    )
    for c in cohort:
        times, flag, down = warmup[c["shot"]]
        require(
            c["disruptive"] is flag and c["down"] == down and c["points"] == len(times),
            "training HDF metadata",
        )
    groups = {
        b: [c["shot"] for c in cohort if c["disruptive"] is b] for b in (True, False)
    }
    rng, supported = np.random.default_rng(131), 0
    require(len(history) == 128, "training steps")
    for step, row in enumerate(history, 1):
        expected = [groups[b][int(rng.integers(len(groups[b])))] for b in (True, False)]
        require(
            row["step"] == step and row["sample_shots"] == expected, "training sample"
        )
        require(
            set(row["episodes"]) == set(row["gradient_norms"]) == {"full_operational"},
            "training head",
        )
        require(
            finite(row["gradient_norms"]["full_operational"], "head gradient") >= 0,
            "head gradient",
        )
        es = row["episodes"]["full_operational"]
        require(len(es) == 2 and [e["shot"] for e in es] == expected, "episode sample")
        for e in es:
            times, flag, down = warmup[e["shot"]]
            q = e["query"]
            require(
                type(q) is int
                and 0 <= q < len(times)
                and (not flag or round((down - times[q]) * 1000, 9) > 30),
                "training query",
            )
            maturity = 159 + ((q - 160) // 128) * 128 if q >= 160 else None
            require(
                e["supported"] is (q >= 160) and e["maturity"] == maturity,
                "training support clock",
            )
            count = 1 + (maturity - 159) // 128 if maturity is not None else 0
            require(
                type(e["replay_updates"]) is int and e["replay_updates"] == count,
                "full replay clock",
            )
            require(
                type(e["accepted_path_updates"]) is int
                and 0 <= e["accepted_path_updates"] <= count,
                "accepted path count",
            )
            require(
                type(e["label"]) is int and e["label"] == int(flag),
                "shot outcome label",
            )
            for k in ("risk_before", "risk_after", "signal", "objective"):
                finite(e[k], "training " + k)
            if maturity is None:
                require(e["risk_before"] == e["risk_after"], "unsupported risk")
            supported += int(q >= 160)
    result = {"full_operational": supported}
    require(done["support_counts"] == result, "training support totals")
    return result


def check_bound_files(protocol, expected):
    declared = protocol["bound_files_sha256"]
    require(set(declared) == {str(p) for p in expected}, "bound file roster")
    for path, digest in declared.items():
        require(
            isinstance(digest, str)
            and prior.HEX.fullmatch(digest)
            and sha_file(Path(path)) == digest,
            "bound file hash: " + path,
        )
    return dict(declared)


def check_snapshot(root, protocol, *, training=False):
    required = (
        {
            PLAN,
            MODULE + "__init__.py",
            MODULE + "prepare.py",
            MODULE + "core.py",
            INNOVATION + "core.py",
            INNOVATION + "stream.py",
        }
        if training
        else RUN_REQUIRED
    )
    declared = protocol["code_sha256"]
    require(
        required <= set(declared) and declared[PLAN] == protocol["plan_sha256"],
        "snapshot entries/plan",
    )
    if not training:
        require(
            declared[EVALUATION] == protocol["evaluation_plan_sha256"],
            "evaluation plan hash",
        )
    plans, current = {PLAN} if training else {PLAN, EVALUATION}, {}
    archive_path = root / "code_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        require(
            len(names) == len(set(names)) and set(names) == set(declared),
            "archive names",
        )
        require(sum(e.file_size for e in entries) <= 512 * 1024 * 1024, "archive size")
        for e in entries:
            name, path = e.filename, PurePosixPath(e.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in name
                and not e.is_dir()
                and e.file_size <= 64 * 1024 * 1024,
                "archive path",
            )
            require(
                name in plans
                or (
                    path.parts[:2] == ("experiments", "risk_aligned_online")
                    and path.suffix == ".py"
                ),
                "non-code snapshot member",
            )
            require(
                isinstance(declared[name], str)
                and prior.HEX.fullmatch(declared[name])
                and hashlib.sha256(archive.read(e)).hexdigest() == declared[name],
                "archive hash",
            )
            actual = ROOT / name
            if actual.is_file():
                require(
                    actual.resolve().is_relative_to(ROOT.resolve()),
                    "current code outside worktree",
                )
                current[name] = sha_file(actual)
            else:
                current[name] = None
    return dict(
        archive_bytes_sha256=sha_file(archive_path),
        archive_entries_verified=len(entries),
        actual_current_sha256=current,
        current_drift={
            n: dict(snapshot=declared[n], actual_current=h)
            for n, h in current.items()
            if h != declared[n]
        },
    )


def check_protocol(p, reference):
    fixed = dict(
        scope="DEVELOPMENT_COMPLETE360_FULL_PATH_AND_POSTHOC_FORECAST",
        arms=list(ARMS),
        reused_arms=REUSED,
        fresh_arms=list(FRESH),
        threshold=THRESHOLD,
        reference_completed_sha256=REFERENCE_SHA,
        J_comparison_basis="integer_tp_minus_fp_fixed_180_180",
        geometry="metric",
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        metric_learning_rate=1.0,
        metric_increment_hidden_rms_cap=0.05,
        hidden_rms_cap=0.25,
        temporary_parameters=256,
        reset_each_shot=True,
        head_training_shots=20,
        head_optimizer_steps=128,
        suffix_head_updates=0,
        suffix_inter_supervision=0,
        warmup_cost_included=True,
        suffix_uncached_real_gpu=True,
        forecast_followup_posthoc_selected_from_prior_eight_arms=True,
        forecast_preregistered_winner=False,
        old_forecast_additional_training_steps=0,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        independent_future_confirmation=False,
        real_time_latency_verified=False,
        no_outcome_input_to_simulate=True,
        within_run_selection_or_threshold_tuning=False,
    )
    for key, value in fixed.items():
        require(p.get(key) == value, "run protocol: " + key)
    for key, path in (
        ("reference", REFERENCE),
        ("training", TRAINING),
        ("old_training", OLD_TRAINING),
    ):
        require(Path(p[key]).resolve() == path.resolve(), "run path: " + key)
    ids = [x["shot"] for x in p["order"]]
    require(
        p["order"] == reference["order"]
        and len(ids) == len(set(ids)) == 360
        and all(type(s) is int and s > 0 for s in ids),
        "ordered 360 roster",
    )
    for key in (
        "source_sha256",
        "cold_state_sha256",
        "input_map_sha256",
        "input_hashes_sha256",
        "threshold",
    ):
        require(p[key] == reference[key], "reference contract: " + key)
    require(
        Path(p["calibration_reference"]).resolve()
        == Path(reference["calibration_reference"]).resolve(),
        "calibration path",
    )
    return ids


def bound_paths(p):
    paths = [
        ROOT / PLAN,
        ROOT / EVALUATION,
        ROOT / ACCEPTANCE,
        Path(p["source_checkpoint"]),
        Path(p["cold_state"]),
        TRAINING / "training_cohort.json",
    ]
    paths += [
        REFERENCE / n
        for n in (
            "completed.json",
            "protocol.json",
            "verification_records.json",
            "verification_neural.json",
        )
    ]
    paths += [
        r / n
        for r in (TRAINING, OLD_TRAINING)
        for n in ("completed.json", "protocol.json", "heads.json", "history.json")
    ]
    return paths + [
        Path(p["calibration_reference"]) / n
        for n in ("input_map.json", "input_hashes.json")
    ]


def training_artifacts(p, reference, ids, output):
    full, fp, old, op = (
        read_json(r / n)
        for r, n in (
            (TRAINING, "completed.json"),
            (TRAINING, "protocol.json"),
            (OLD_TRAINING, "completed.json"),
            (OLD_TRAINING, "protocol.json"),
        )
    )
    require(
        sha_file(TRAINING / "completed.json") == FULL_DONE_SHA
        and sha_file(TRAINING / "heads.json") == full["heads_sha256"] == FULL_HEAD_SHA,
        "distinct full completion/head hashes",
    )
    require(
        sha_file(OLD_TRAINING / "completed.json")
        == reference["training_completed_sha256"]
        and sha_file(OLD_TRAINING / "protocol.json")
        == reference["training_protocol_sha256"]
        and old["heads_sha256"] == reference["heads_sha256"]
        and sha_file(OLD_TRAINING / "training_cohort.json")
        == reference["training_cohort_sha256"],
        "unchanged old forecast training",
    )
    prior.check_training_contract(
        old,
        op,
        ids[:20],
        prior.SOURCE_SHA,
        prior.COLD_SHA,
        p["input_map_sha256"],
        reference["plan_sha256"],
    )
    for root, done in ((TRAINING, full), (OLD_TRAINING, old)):
        require(
            sha_file(root / "heads.json") == done["heads_sha256"]
            and sha_file(root / "history.json") == done["history_sha256"],
            "training hashes",
        )
    completion = dict(
        status="COMPLETED",
        steps=128,
        head_optimizer_steps=128,
        number_heads=1,
        ids=ids[:20],
        outcome_reads=20,
        suffix_shots_read=0,
        source_tensors_unchanged=True,
        online_gain_measured=False,
        independent_future_confirmation=False,
    )
    for key, value in completion.items():
        require(full.get(key) == value, "full training completion: " + key)
    common = (
        "ids",
        "source_sha256",
        "cold_state_sha256",
        "input_map_sha256",
        "forecast_parameters_per_head",
        "steps",
        "optimizer",
        "learning_rate",
        "gradient_clip",
        "seed",
        "first_update",
        "update_interval",
        "metric_learning_rate",
        "relative_metric_damping",
        "absolute_metric_damping",
        "metric_increment_hidden_rms_cap",
        "total_hidden_rms_cap",
        "warmup_cost_retained",
        "independent_confirmation",
    )
    require(all(fp[k] == op[k] for k in common), "equal-budget full training")
    extra = dict(
        scope="ENDED20_FULL_PATH_RISK_META",
        heads={"full_operational": "metric"},
        initial_head="zero",
        temporal_gradient_truncation_updates=None,
        risk_meta_gradient_horizon="all_matured_updates_before_fixed_query",
        diagnostic_incoming_state_detached=True,
        first_possible_effect=160,
        reset_each_shot=True,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        suffix_shots_read=0,
    )
    for key, value in extra.items():
        require(fp.get(key) == value, "full training protocol: " + key)
    require(
        full["plan_sha256"] == fp["plan_sha256"] == p["plan_sha256"]
        and sha_file(TRAINING / "training_cohort.json")
        == full["training_cohort_sha256"],
        "full training plan/cohort",
    )
    prior.check_heads(TRAINING / "heads.json", ("full_operational",))
    prior.check_heads(OLD_TRAINING / "heads.json", prior.HEADS)
    snapshots = dict(
        run=check_snapshot(output, p),
        full_training=check_snapshot(TRAINING, fp, training=True),
        old_training=prior.check_snapshot(OLD_TRAINING, op, training=True),
    )
    return full, fp, old, op, snapshots


def verify(output):
    """Read completed saved artifacts only; no report is written here."""
    output = Path(output)
    done = read_json(output / "completed.json")
    require(
        done.get("status") == "COMPLETED", "COMPLETED run required before other reads"
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
    p = read_json(output / "protocol.json")
    reference, rd, records, neural = (
        read_json(REFERENCE / n)
        for n in (
            "protocol.json",
            "completed.json",
            "verification_records.json",
            "verification_neural.json",
        )
    )
    require(
        sha_file(REFERENCE / "completed.json") == REFERENCE_SHA
        and rd["status"] == "COMPLETED"
        and records["status"] == neural["status"] == "PASS"
        and records["hashes"]["completed_sha256"]
        == neural["completed_sha256"]
        == REFERENCE_SHA
        and sha_file(REFERENCE / "protocol.json")
        == rd["protocol_sha256"]
        == records["hashes"]["protocol_sha256"],
        "verified reference binding",
    )
    require(
        records["cases"] == 360
        and records["curves"] == 2880
        and neural["curves_replayed"] == 32
        and neural["actual_layer_hook_batches"] == 96
        and neural["fixed_preselected_shots"] == [74087, 73403, 63656, 62573]
        and rd["metrics"] == records["metrics"],
        "reference verification scope",
    )
    ids = check_protocol(p, reference)
    prior.check_model_files(p, reference)
    approved = (
        ROOT.parent
        / "Fuison/experiments/risk_aligned_online/runs/gradient_aligned512_20260929/source.pt"
    )
    require(
        Path(p["source_checkpoint"]).resolve() == approved.resolve(),
        "approved shared source",
    )
    bindings = check_bound_files(p, bound_paths(p))
    for name, key in (
        (PLAN, "plan_sha256"),
        (EVALUATION, "evaluation_plan_sha256"),
        (ACCEPTANCE, "acceptance_alignment_sha256"),
    ):
        require(bindings[str(ROOT / name)] == p[key], "bound document: " + name)
    calibration = Path(p["calibration_reference"])
    require(
        bindings[str(calibration / "input_map.json")] == p["input_map_sha256"]
        and bindings[str(calibration / "input_hashes.json")] == p["input_hashes_sha256"]
        and read_json(calibration / "input_map.json")["calibration_ids"] == ids[:20]
        and set(read_json(calibration / "input_hashes.json")) == {str(s) for s in ids},
        "input map/roster hashes",
    )
    # Pass the selected output explicitly, without importing or changing a runner.
    full, fp, old, op, snapshots = training_artifacts(p, reference, ids, output)
    saved, seals = (
        read_json(output / "per_shot.json"),
        read_json(output / "shot_seals.json"),
    )
    ref_records = records["hashes"]["shot_files"]
    require(
        set(saved) == set(ARMS)
        and all(len(saved[a]) == 360 for a in ARMS)
        and len(seals) == len(ref_records) == 360,
        "curve/seal roster",
    )
    rows, warmup, shot_hashes = {a: [] for a in ARMS}, {}, []
    totals = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, sid in enumerate(ids, 1):
        stem = f"{index:04d}_{sid}"
        base, refbase = output / "shots" / stem, REFERENCE / "shots" / stem
        curve, state, info_path, seal_path = (
            Path(str(base) + s)
            for s in (".npz", "_states.npz", "_updates.json", "_seal.json")
        )
        refcurve, refseal_path = (
            Path(str(refbase) + ".npz"),
            Path(str(refbase) + "_seal.json"),
        )
        seal, refseal, rr = (
            read_json(seal_path),
            read_json(refseal_path),
            ref_records[index - 1],
        )
        ch, sh, ih, se = (sha_file(x) for x in (curve, state, info_path, seal_path))
        prior.check_shot_seal(index, sid, seal, seals[index - 1], se, ch, sh, ih)
        require(
            rr["shot"] == refseal["shot"] == sid
            and rr["index"] == refseal["index"] == index
            and sha_file(refseal_path) == rr["seal_sha256"]
            and refseal["current_outcome_used_in_prediction"] is False
            and sha_file(refcurve) == refseal["sha256"] == rr["curve_sha256"],
            "reference shot seal",
        )
        with np.load(curve, allow_pickle=False) as z:
            require(set(z.files) == {"times", *ARMS}, "curve keys")
            data = {k: z[k] for k in z.files}
        with np.load(refcurve, allow_pickle=False) as z:
            ref = {
                k: z[k]
                for k in ("times", "cold", "innovation_metric", "metric_forecast")
            }
        times = data["times"]
        validate_curves(times, {a: data[a] for a in ARMS}, ref, warmup=index <= 20)
        with np.load(state, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        ts, ac = check_trace(
            read_json(info_path),
            arrays,
            len(times),
            index=index,
            reference_path=refcurve,
            reference_sha=rr["curve_sha256"],
        )
        flag, down = read_outcome(sid)
        if index <= 20:
            warmup[sid] = (times, flag, down)
        for a in ARMS:
            row = dict(
                shot=sid,
                disruptive=flag,
                **classify_curve(times, data[a], flag, down, THRESHOLD),
            )
            require(
                row == saved[a][index - 1], "independent scoring: " + stem + " " + a
            )
            rows[a].append(row)
            for k in totals[a]:
                totals[a][k] += ts[a][k]
            for k in ("endpoints", "capped_endpoints"):
                actions[a][k] += ac[a][k]
            actions[a]["max_rms"] = max(actions[a]["max_rms"], ac[a]["max_rms"])
        shot_hashes.append(
            dict(
                shot=sid,
                index=index,
                curve_sha256=ch,
                states_sha256=sh,
                updates_sha256=ih,
                seal_sha256=se,
                reference_curve_sha256=rr["curve_sha256"],
            )
        )
    support = check_training(
        fp,
        full,
        read_json(TRAINING / "history.json"),
        read_json(TRAINING / "training_cohort.json"),
        warmup,
    )
    old_support = prior.check_training(
        op,
        old,
        read_json(OLD_TRAINING / "history.json"),
        read_json(OLD_TRAINING / "training_cohort.json"),
        warmup,
    )
    summary = summarize(rows, rd["metrics"]["metric_zero"]["full360"])
    require(
        all(done.get(k) == val for k, val in summary.items()),
        "metrics/flips/deltas/integer gates",
    )
    require(
        done["reference_metric_zero"] == rd["metrics"]["metric_zero"]
        and done["recomputed_update_statistics"] == totals
        and done["recomputed_action_statistics"] == actions,
        "reference/totals",
    )
    fixed = dict(
        statistics_scope=STATISTICS_SCOPE,
        fresh_curves=2040,
        reused_suffix_curves=680,
        reused_warmup_curves=160,
        total_curves=2880,
        curve_execution_counts=dict(
            fresh_suffix=2040, copied_preparation=120, referenced_parent=720
        ),
        source_tensors_unchanged=True,
        heads_unchanged=True,
        forecast_followup_posthoc_selected_from_prior_eight_arms=True,
        forecast_preregistered_winner=False,
        method_promoted=False,
        scientific_goal_achieved=False,
        independent_future_confirmation=False,
    )
    require(
        all(done.get(k) == val for k, val in fixed.items()),
        "completion execution/flags",
    )
    return dict(
        status="PASS",
        scope="RECORDED_DATA_CONSISTENCY_ONLY_NOT_INDEPENDENT_NEURAL_REPLAY",
        cases=360,
        curves=2880,
        metadata_outcomes_read=360,
        raw_signal_datasets_read=0,
        raw_hdf5_file_hashes_recomputed=False,
        threshold=THRESHOLD,
        **summary,
        **fixed,
        recomputed_update_statistics=totals,
        recomputed_action_statistics=actions,
        reference_metric_zero=rd["metrics"]["metric_zero"],
        training_supported_episodes=support,
        old_training_supported_episodes=old_support,
        hashes=dict(
            completed_sha256=sha_file(output / "completed.json"),
            protocol_sha256=done["protocol_sha256"],
            per_shot_sha256=done["per_shot_sha256"],
            shot_seals_sha256=done["shot_seals_sha256"],
            reference_completed_sha256=REFERENCE_SHA,
            full_training_completed_sha256=FULL_DONE_SHA,
            full_heads_sha256=FULL_HEAD_SHA,
            verifier_sha256=sha_file(Path(__file__)),
            pure_helper_sha256=sha_file(Path(prior.__file__)),
            bound_files_sha256=bindings,
            snapshots=snapshots,
            shot_files=shot_hashes,
        ),
        limitations=[
            "Reused-arm empty new traces/actions describe this run only, not absence of historical adaptation.",
            "No independent neural or training replay, critical-maximum verification, or gradient recomputation.",
            "Mask/target hash agreement and recorded accepted-path counts are consistency checks, not recomputation from signals.",
            "Recorded actions and parameter receipts cannot independently prove actual hidden RMS.",
            "HDF5 reads use only IsDisrupt/DownTime; stored raw-file hashes are not independently authenticated.",
            "Current imported-code drift is reported separately from archived-code consistency.",
            "Forecast followup was selected posthoc from the prior eight arms; not independent future validation.",
            "No physical contribution, real-time latency, or scientific-goal success is established by PASS.",
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
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({k: report[k] for k in ("status", "cases", "curves")}))


if __name__ == "__main__":
    main()
