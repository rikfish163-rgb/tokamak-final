"""Fixed complete360 development replay of shot-local innovation geometry."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..critical_shot_meta_20261102.run import export_traces, metrics
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lifecycle_stream_20261025.audit import classify, flips
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform
from .stream import simulate

PLAN = io23.ROOT / "INNOVATION_METRIC_PLAN_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/critical_shot_meta360_v1"
REFERENCE_SHA = "fb05bb2ea6e89f6924fb9e4b0b304b4f40da6734b5bd50be39ffaed123198ed1"
TRAINING = io23.ROOT / "probe_outputs/innovation_metric_heads20_v1"
OLD_TRAINING = io23.ROOT / "probe_outputs/critical_shot_heads20_v1"
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
CONTROLS = ("metric_reverse", "metric_hold", "metric_forecast", "metric_zero")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_training(done, protocol, ids, source_hash, cold_hash, map_hash, plan_hash):
    """Reject a different training budget, cohort, parent, or input contract."""
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
    )
    for name, expected in completion.items():
        require(done.get(name) == expected, "training completion: " + name)
    for name, expected in fixed.items():
        require(protocol.get(name) == expected, "training protocol: " + name)


def build_heads(tables, old_operational):
    """Use one immutable metric head for all three matched target controls."""
    require(set(tables) == set(HEADS), "unexpected innovation head names")
    for table in [*tables.values(), old_operational]:
        require(
            isinstance(table, torch.Tensor)
            and table.shape == (25, 23)
            and table.dtype == torch.float32
            and table.device == old_operational.device
            and bool(torch.isfinite(table).all()),
            "expected finite 25x23 float32 heads on one device",
        )
    metric = tables["metric_operational"]
    return dict(
        cold=(None, "real", "contrast"),
        contrast_unit=(old_operational, "real", "contrast"),
        innovation_unit=(tables["unit_operational"], "real", "unit"),
        innovation_metric=(metric, "real", "metric"),
        metric_forecast=(tables["metric_forecast"], "real", "metric"),
        metric_zero=(torch.zeros_like(metric), "real", "metric"),
        metric_reverse=(metric, "reverse", "metric"),
        metric_hold=(metric, "hold", "metric"),
    )


def bitwise_equal(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def validate_curves(times, curves, ref_times, ref_cold, ref_operational, *, warmup):
    require(set(curves) == set(ARMS), "curve arms")
    times, ref_times = np.asarray(times), np.asarray(ref_times)
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
            and value.dtype.str == np.asarray(ref_cold).dtype.str
            and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1)),
            "invalid curve: " + arm,
        )
        require(
            bitwise_equal(value[:160], np.asarray(ref_cold)[:160]),
            "first 160 drift: " + arm,
        )
        if warmup:
            require(bitwise_equal(value, ref_cold), "warmup curve drift: " + arm)


def score_gates(scores):
    base = scores["cold"]["full360"]
    candidate = scores["innovation_metric"]["full360"]
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


def seal_and_score(root, index, sid, times, curves, arrays, info, threshold):
    """Write all prediction evidence and its seal before opening HDF outcomes."""
    stem = root / "shots" / f"{index:04d}_{sid}"
    curve_path = Path(str(stem) + ".npz")
    trace_path = Path(str(stem) + "_states.npz")
    info_path = Path(str(stem) + "_updates.json")
    seal_path = Path(str(stem) + "_seal.json")
    with curve_path.open("xb") as handle:
        np.savez_compressed(handle, times=times, **curves)
    with trace_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    write(info_path, info)
    write(
        seal_path,
        dict(
            shot=sid,
            index=index,
            sha256=sha(curve_path),
            states_sha256=sha(trace_path),
            updates_sha256=sha(info_path),
            warmup_curves_reused=index <= 20,
            warmup_outcome_already_known_for_head_fit=index <= 20,
            current_outcome_used_in_prediction=False,
        ),
    )
    sealed = dict(shot=sid, index=index, seal_sha256=sha(seal_path))
    flag, down = io23.read_outcome("EAST", sid)
    rows = {
        a: dict(
            shot=sid,
            disruptive=flag,
            **classify(times, curves[a], flag, down, threshold),
        )
        for a in ARMS
    }
    return rows, sealed


def run_shot(
    root, index, sid, reference_record, expected_hash, model, basis, heads, mapping
):
    """Run one full uncached suffix stream, or reuse its sealed warmup curve."""
    require(sha(io23.path_for("EAST", sid)) == expected_hash, "raw input hash drift")
    stem = f"{index:04d}_{sid}"
    ref_curve = REFERENCE / "shots" / (stem + ".npz")
    ref_seal_path = REFERENCE / "shots" / (stem + "_seal.json")
    require(
        reference_record["shot"] == sid and reference_record["index"] == index,
        "reference audit shot order",
    )
    require(
        sha(ref_seal_path) == reference_record["seal_sha256_at_verification"],
        "reference seal changed since verification",
    )
    ref_seal = read(ref_seal_path)
    require(
        ref_seal["shot"] == sid
        and ref_seal["index"] == index
        and ref_seal["current_outcome_used_in_prediction"] is False
        and sha(ref_curve) == ref_seal["sha256"] == reference_record["curve_sha256"],
        "reference curve seal",
    )
    with np.load(ref_curve, allow_pickle=False) as reference:
        ref_times, ref_cold, ref_operational = (
            reference["times"].copy(),
            reference["cold"].copy(),
            reference["operational"].copy(),
        )
    if index <= 20:
        times = ref_times.copy()
        curves = {a: ref_cold.copy() for a in ARMS}
        actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
        arrays, info = export_traces({a: [] for a in ARMS}, actions)
    else:
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        times = raw["times"]
        # No cache and no outcome or metadata are passed into the real stream.
        result = simulate(model, signal, basis, heads)
        curves = result["probability"]
        arrays, info = export_traces(result["tapes"], result["actions"])
    validate_curves(
        times, curves, ref_times, ref_cold, ref_operational, warmup=index <= 20
    )
    rows, seal = seal_and_score(
        root, index, sid, times, curves, arrays, info, THRESHOLD
    )
    return rows, info, seal


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    require(
        sha(REFERENCE / "completed.json") == REFERENCE_SHA, "reference completion hash"
    )
    reference = read(REFERENCE / "protocol.json")
    records = read(REFERENCE / "verification_records.json")
    neural = read(REFERENCE / "verification_neural.json")
    hashes = records["hashes"]
    require(
        records["status"] == neural["status"] == "PASS"
        and hashes["completed_sha256"] == neural["completed_sha256"] == REFERENCE_SHA
        and hashes["protocol_sha256_at_verification"]
        == sha(REFERENCE / "protocol.json"),
        "reference verification binding",
    )
    order = reference["order"]
    ids = [r["shot"] for r in order]
    require(
        len(ids) == len(set(ids)) == 360 and all(type(s) is int and s > 0 for s in ids),
        "reference must contain 360 distinct ordered shots",
    )
    require(len(hashes["shot_files"]) == 360, "reference per-shot verification count")
    upstream = Path(reference["reference"])
    upstream_check = read(upstream / "verification_records.json")
    require(
        upstream_check["status"] == "PASS"
        and sha(upstream / "completed.json")
        == upstream_check["completed_sha256"]
        == reference["reference_completed_sha256"]
        and sha(upstream / "protocol.json") == upstream_check["protocol_sha256"],
        "input-hash authority verification",
    )
    calibration = Path(reference["calibration_reference"])
    map_hash = sha(calibration / "input_map.json")
    input_hashes_hash = sha(calibration / "input_hashes.json")
    require(
        map_hash == reference["input_map_sha256"] == hashes["input_map_sha256"]
        and input_hashes_hash == upstream_check["input_hashes_sha256"],
        "input map/hash drift",
    )
    mapping = read(calibration / "input_map.json")
    expected_inputs = read(calibration / "input_hashes.json")
    require(
        mapping["calibration_ids"] == ids[:20]
        and set(expected_inputs) == {str(s) for s in ids},
        "calibration cohort",
    )
    source_hash, cold_hash, plan_hash = (
        sha(DEFAULT_PARENT),
        reference["cold_state_sha256"],
        sha(PLAN),
    )
    require(
        source_hash == reference["source_sha256"] == hashes["source_sha256"]
        and reference["threshold"] == THRESHOLD,
        "source or threshold drift",
    )
    done, train_protocol = (
        read(TRAINING / "completed.json"),
        read(TRAINING / "protocol.json"),
    )
    validate_training(
        done, train_protocol, ids[:20], source_hash, cold_hash, map_hash, plan_hash
    )
    require(
        sha(TRAINING / "heads.json") == done["heads_sha256"]
        and sha(TRAINING / "history.json") == done["history_sha256"],
        "new training hash drift",
    )
    history = read(TRAINING / "history.json")
    cohort_path = TRAINING / "training_cohort.json"
    cohort = read(cohort_path)
    require(
        [r["shot"] for r in cohort] == ids[:20]
        and all(type(r["disruptive"]) is bool for r in cohort)
        and sum(r["disruptive"] for r in cohort) == 8,
        "new training cohort",
    )
    cohort_flags = {r["shot"]: r["disruptive"] for r in cohort}
    require(
        len(history) == 128
        and all(
            r["step"] == step
            and len(r["sample_shots"]) == 2
            and set(r["sample_shots"]) <= set(ids[:20])
            and [cohort_flags[s] for s in r["sample_shots"]] == [True, False]
            for step, r in enumerate(history, 1)
        ),
        "new training history budget/cohort",
    )
    old_done, old_protocol = (
        read(OLD_TRAINING / "completed.json"),
        read(OLD_TRAINING / "protocol.json"),
    )
    require(
        sha(OLD_TRAINING / "completed.json")
        == reference["training_completed_sha256"]
        == hashes["training_completed_sha256"]
        and old_done["status"] == "COMPLETED"
        and old_done["head_optimizer_steps"] == 128
        and sha(OLD_TRAINING / "heads.json")
        == old_done["heads_sha256"]
        == reference["heads_sha256"]
        == hashes["heads_sha256"]
        and sha(OLD_TRAINING / "history.json") == old_done["history_sha256"]
        and old_protocol["ids"] == ids[:20]
        and old_protocol["source_sha256"] == source_hash
        and old_protocol["cold_state_sha256"] == cold_hash
        and old_protocol["input_map_sha256"] == map_hash,
        "old contrast training drift",
    )
    new_data, old_data = (
        read(TRAINING / "heads.json"),
        read(OLD_TRAINING / "heads.json"),
    )
    require(
        set(new_data) == set(HEADS)
        and set(old_data) == {"forecast", "point", "operational"},
        "head names",
    )

    configure()
    model, configuration, cold = load_model()
    require(model.base.center.device.type == "cuda", "real GPU model required")
    require(
        sha(cold) == cold_hash == hashes["cold_state_sha256"]
        and float(configuration["threshold"]) == THRESHOLD,
        "cold parent/threshold drift",
    )
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    device = model.base.center.device
    tables = {
        k: torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in new_data.items()
    }
    old = torch.tensor(old_data["operational"], dtype=torch.float32, device=device)
    heads = build_heads(tables, old)
    head_copies = {
        a: head.detach().clone()
        for a, (head, _, _) in heads.items()
        if head is not None
    }

    root.mkdir(parents=True)
    (root / "shots").mkdir()
    # Bind this entry point and its actually imported experiment dependencies.
    # Unused verification scripts are independent sidecars and may still be
    # under development; globbing the entire package falsely bound those edits
    # to inference (the incomplete v1 retained the resulting drift evidence).
    files = {Path(__file__), PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    code_hashes = {str(f.relative_to(io23.ROOT)): sha(f) for f in files}
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_INNOVATION_METRIC",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        reference_protocol_sha256=sha(REFERENCE / "protocol.json"),
        reference_records_sha256=sha(REFERENCE / "verification_records.json"),
        reference_neural_sha256=sha(REFERENCE / "verification_neural.json"),
        training=str(TRAINING),
        training_completed_sha256=sha(TRAINING / "completed.json"),
        training_protocol_sha256=sha(TRAINING / "protocol.json"),
        heads_sha256=done["heads_sha256"],
        history_sha256=done["history_sha256"],
        training_cohort_sha256=sha(cohort_path),
        old_training=str(OLD_TRAINING),
        old_training_completed_sha256=sha(OLD_TRAINING / "completed.json"),
        old_heads_sha256=old_done["heads_sha256"],
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=source_hash,
        cold_state=str(cold),
        cold_state_sha256=cold_hash,
        calibration_reference=str(calibration),
        input_map_sha256=map_hash,
        input_hashes_sha256=input_hashes_hash,
        threshold=THRESHOLD,
        head_assignment={
            a: dict(
                head=(
                    "zero"
                    if a == "metric_zero"
                    else {
                        "cold": None,
                        "contrast_unit": "old_operational",
                        "innovation_unit": "unit_operational",
                        "innovation_metric": "metric_operational",
                        "metric_forecast": "metric_forecast",
                        "metric_reverse": "metric_operational",
                        "metric_hold": "metric_operational",
                    }.get(a)
                ),
                mode=mode,
                geometry=geometry,
            )
            for a, (_, mode, geometry) in heads.items()
        },
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
        plan_sha256=plan_hash,
        code_sha256=code_hashes,
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    # Guard against a concurrent code edit while constructing the archive.
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        require(
            all(
                hashlib.sha256(archive.read(name)).hexdigest() == digest
                for name, digest in code_hashes.items()
            ),
            "code changed during snapshot",
        )

    rows = {a: [] for a in ARMS}
    totals = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    seals = []
    for index, sid in enumerate(ids, 1):
        shot_rows, info, seal = run_shot(
            root,
            index,
            sid,
            hashes["shot_files"][index - 1],
            expected_inputs[str(sid)],
            model,
            basis,
            heads,
            mapping,
        )
        seals.append(seal)
        for arm in ARMS:
            rows[arm].append(shot_rows[arm])
            stat = info["actions"][arm]
            actions[arm]["endpoints"] += stat["endpoints"]
            actions[arm]["capped_endpoints"] += stat["capped_endpoints"]
            actions[arm]["max_rms"] = max(actions[arm]["max_rms"], stat["max_rms"])
            events = info["updates"][arm]
            totals[arm]["events"] += len(events)
            totals[arm]["accepted"] += sum(e["accepted"] for e in events)
            totals[arm]["nonzero_gradients"] += sum(
                e["gradient_norm"] > 0 for e in events
            )
            totals[arm]["observed_scalars"] += sum(e["observed"] for e in events)
        progress = dict(completed=index, total=360, seconds=time.monotonic() - started)
        _durable_json(root / "progress.json", progress)
        if index % 10 == 0:
            print(json.dumps(progress), flush=True)
    require(
        all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()),
        "source tensors changed",
    )
    require(
        all(
            torch.equal(head_copies[a], head)
            for a, (head, _, _) in heads.items()
            if head is not None
        ),
        "head tensors changed",
    )
    require(
        sha(DEFAULT_PARENT) == source_hash
        and sha(cold) == cold_hash
        and sha(TRAINING / "heads.json") == done["heads_sha256"]
        and sha(OLD_TRAINING / "heads.json") == old_done["heads_sha256"]
        and sha(calibration / "input_map.json") == map_hash
        and sha(calibration / "input_hashes.json") == input_hashes_hash
        and all(
            sha(io23.ROOT / name) == digest for name, digest in code_hashes.items()
        ),
        "source, head, input contract or code changed during replay",
    )
    scores = {
        a: {
            part: metrics(data)
            for part, data in (
                ("full360", rows[a]),
                ("warmup20", rows[a][:20]),
                ("suffix340", rows[a][20:]),
            )
        }
        for a in ARMS
    }
    gates = score_gates(scores)
    write(root / "per_shot.json", rows)
    write(root / "shot_seals.json", seals)
    result = dict(
        status="COMPLETED",
        metrics=scores,
        flips={a: flips(rows["cold"], rows[a]) for a in ARMS[1:]},
        update_statistics=totals,
        action_statistics=actions,
        **gates,
        fresh_curves=340 * len(ARMS),
        reused_warmup_curves=20 * len(ARMS),
        source_tensors_unchanged=True,
        heads_unchanged=True,
        method_promoted=False,
        scientific_goal_achieved=False,
        independent_future_confirmation=False,
        per_shot_sha256=sha(root / "per_shot.json"),
        shot_seals_sha256=sha(root / "shot_seals.json"),
        protocol_sha256=sha(root / "protocol.json"),
        code_snapshot_sha256=sha(root / "code_snapshot.zip"),
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "metrics",
                    "operational_gate",
                    "control_separation_gate",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
