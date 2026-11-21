"""Complete360 full-path head comparison and posthoc matched forecast controls."""

import argparse
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
from ..innovation_metric_20261102.run import THRESHOLD, bitwise_equal, require
from ..innovation_metric_20261102.stream import simulate
from ..lifecycle_stream_20261025.audit import classify, flips
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform

PLAN = io23.ROOT / "FULL_PATH_META_PLAN_20261102.md"
EVALUATION = io23.ROOT / "FULL_PATH_CONTROL_EVALUATION_20261102.md"
ACCEPTANCE = io23.ROOT / "USER_ACCEPTANCE_ALIGNMENT_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/innovation_metric360_v2"
REFERENCE_SHA = "f558ee30e33192da29daa7863f58267d25913a9c8dc25a645188ab32cac11014"
TRAINING = io23.ROOT / "probe_outputs/full_path_heads20_v1"
OLD_TRAINING = io23.ROOT / "probe_outputs/innovation_metric_heads20_v1"
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


def build_heads(full, forecast):
    for head in (full, forecast):
        require(
            isinstance(head, torch.Tensor)
            and head.shape == (25, 23)
            and head.dtype == torch.float32
            and head.device == full.device
            and bool(torch.isfinite(head).all()),
            "finite 25x23 float32 heads required",
        )
    return {
        prefix + "_" + mode: (head, mode, "metric")
        for prefix, head in (("forecast", forecast), ("full", full))
        for mode in ("real", "reverse", "hold")
    }


def net_correct(metric):
    return metric["tp"] - metric["fp"]


def score_gates(scores, zero):
    base, full = scores["cold"]["full360"], scores["full_real"]["full360"]
    require(
        all(
            (m["n_disrupt"], m["n_safe"]) == (180, 180)
            for m in [zero, *(scores[a]["full360"] for a in ARMS)]
        ),
        "fixed balanced180/180 required",
    )
    stage = (
        full["fp"] <= 36
        and full["tp"] >= base["tp"]
        and net_correct(full) > net_correct(base)
    )
    controls = {
        a: net_correct(full) > net_correct(scores[a]["full360"])
        for a in ("full_reverse", "full_hold", "forecast_real", "truncated_metric")
    }
    controls["reference_metric_zero"] = net_correct(full) > net_correct(zero)
    forecast = {
        a: net_correct(scores["forecast_real"]["full360"])
        > net_correct(scores[a]["full360"])
        for a in ("forecast_reverse", "forecast_hold")
    }
    return dict(
        operational_gate=bool(stage),
        control_comparisons=controls,
        control_separation_gate=all(controls.values()),
        forecast_control_comparisons=forecast,
        forecast_control_separation_gate=all(forecast.values()),
        stage_success_pending_verification=bool(stage and all(controls.values())),
    )


def load_training(root, reference, *, full):
    done, protocol, tables = (
        read(root / name) for name in ("completed.json", "protocol.json", "heads.json")
    )
    ids = [r["shot"] for r in reference["order"][:20]]
    require(
        done["status"] == "COMPLETED"
        and done["steps"] == done["head_optimizer_steps"] == protocol["steps"] == 128
        and done["ids"] == protocol["ids"] == ids
        and done["source_tensors_unchanged"] is True
        and done["outcome_reads"] == 20
        and done["suffix_shots_read"] == 0,
        "training budget/cohort",
    )
    require(
        all(
            protocol[k] == reference[k]
            for k in ("source_sha256", "cold_state_sha256", "input_map_sha256")
        )
        and sha(root / "heads.json") == done["heads_sha256"]
        and sha(root / "history.json") == done["history_sha256"],
        "training parent/hash drift",
    )
    require(
        all(
            np.asarray(t).shape == (25, 23)
            and np.isfinite(np.asarray(t, np.float64)).all()
            for t in tables.values()
        ),
        "invalid training head shape/values",
    )
    if full:
        require(
            protocol["scope"] == "ENDED20_FULL_PATH_RISK_META"
            and done["number_heads"] == 1
            and protocol["heads"] == {"full_operational": "metric"}
            and set(tables) == {"full_operational"}
            and protocol["temporal_gradient_truncation_updates"] is None
            and protocol["diagnostic_incoming_state_detached"] is True
            and done["plan_sha256"] == protocol["plan_sha256"] == sha(PLAN),
            "full-path training contract",
        )
        require(
            sha(root / "training_cohort.json") == done["training_cohort_sha256"],
            "training cohort hash",
        )
        history = read(root / "history.json")
        require(
            len(history) == 128
            and all(
                r["step"] == step
                and len(r["sample_shots"]) == 2
                and set(r["sample_shots"]) <= set(ids)
                for step, r in enumerate(history, 1)
            ),
            "training history budget/IDs",
        )
    else:
        require(
            sha(root / "completed.json") == reference["training_completed_sha256"]
            and sha(root / "protocol.json") == reference["training_protocol_sha256"]
            and done["heads_sha256"] == reference["heads_sha256"]
            and set(tables)
            == {"unit_operational", "metric_operational", "metric_forecast"},
            "frozen forecast head drift",
        )
    return tables


def validate_curves(times, curves, reference, *, warmup):
    require(
        set(curves) == set(ARMS) and bitwise_equal(times, reference["times"]),
        "curve arms/times drift",
    )
    require(
        np.asarray(times).ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    for arm, reference_arm in {**REUSED, "forecast_real": "metric_forecast"}.items():
        require(
            bitwise_equal(curves[arm], reference[reference_arm]),
            "reference curve drift: " + arm,
        )
    for arm, curve in curves.items():
        require(
            curve.shape == times.shape
            and np.isfinite(curve).all()
            and np.all((curve >= 0) & (curve <= 1))
            and curve.dtype == reference["cold"].dtype,
            "invalid curve: " + arm,
        )
        require(
            bitwise_equal(curve[:160], reference["cold"][:160]),
            "first160 drift: " + arm,
        )
        if warmup:
            require(
                bitwise_equal(curve, reference["cold"]), "warmup curve drift: " + arm
            )


def seal_and_score(root, index, sid, times, curves, arrays, info):
    stem = root / "shots" / f"{index:04d}_{sid}"
    curve, state, meta, seal = [
        Path(str(stem) + suffix)
        for suffix in (".npz", "_states.npz", "_updates.json", "_seal.json")
    ]
    for path, values in ((curve, dict(times=times, **curves)), (state, arrays)):
        with path.open("xb") as handle:
            np.savez_compressed(handle, **values)
    write(meta, info)
    write(
        seal,
        dict(
            shot=sid,
            index=index,
            sha256=sha(curve),
            states_sha256=sha(state),
            updates_sha256=sha(meta),
            warmup_curves_reused=index <= 20,
            warmup_outcome_already_known_for_head_fit=index <= 20,
            current_outcome_used_in_prediction=False,
        ),
    )
    sealed = dict(shot=sid, index=index, seal_sha256=sha(seal))
    flag, down = io23.read_outcome("EAST", sid)
    rows = {
        a: dict(
            shot=sid,
            disruptive=flag,
            **classify(times, curves[a], flag, down, THRESHOLD),
        )
        for a in ARMS
    }
    return rows, sealed


def run_shot(root, index, sid, record, expected_hash, model, basis, heads, mapping):
    require(sha(io23.path_for("EAST", sid)) == expected_hash, "raw input hash drift")
    path = REFERENCE / "shots" / f"{index:04d}_{sid}.npz"
    seal_path = path.with_name(path.stem + "_seal.json")
    require(
        (record["shot"], record["index"]) == (sid, index)
        and sha(seal_path) == record["seal_sha256"],
        "reference seal/order drift",
    )
    seal = read(seal_path)
    require(
        (seal["shot"], seal["index"]) == (sid, index)
        and seal["current_outcome_used_in_prediction"] is False
        and sha(path) == seal["sha256"] == record["curve_sha256"],
        "reference curve hash drift",
    )
    with np.load(path, allow_pickle=False) as archive:
        reference = {
            k: archive[k].copy()
            for k in ("times", "cold", "innovation_metric", "metric_forecast")
        }
    curves = {
        a: reference["cold" if index <= 20 else original].copy()
        for a, original in REUSED.items()
    }
    tapes = {a: [] for a in ARMS}
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    times = reference["times"]
    if index <= 20:
        curves.update({a: reference["cold"].copy() for a in FRESH})
    else:
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        result = simulate(
            model, signal, basis, heads
        )  # Six arms, no cached encoding or duplicate cold prediction.
        require(
            set(result["probability"])
            == set(result["tapes"])
            == set(result["actions"])
            == set(FRESH),
            "fresh arms",
        )
        curves.update(result["probability"])
        tapes.update(result["tapes"])
        actions.update(result["actions"])
        times = raw["times"]
    validate_curves(times, curves, reference, warmup=index <= 20)
    arrays, info = export_traces(tapes, actions)
    info["statistics_scope"] = "recomputed_this_run_only_not_historical_activity"
    info["execution"] = {
        a: dict(
            kind="reused_warmup"
            if index <= 20
            else ("reused_reference" if a in REUSED else "fresh_uncached_gpu"),
            reference=str(path) if index <= 20 or a in REUSED else None,
            reference_arm="cold" if index <= 20 else REUSED.get(a),
        )
        for a in ARMS
    }
    info["reference_curve_sha256"] = record["curve_sha256"]
    rows, sealed = seal_and_score(root, index, sid, times, curves, arrays, info)
    return rows, info, sealed


def snapshot(root, protocol):
    files = {Path(__file__), PLAN, EVALUATION}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    protocol["code_sha256"] = {str(f.relative_to(io23.ROOT)): sha(f) for f in files}
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    reference, done, records, neural = [
        read(REFERENCE / name)
        for name in (
            "protocol.json",
            "completed.json",
            "verification_records.json",
            "verification_neural.json",
        )
    ]
    require(
        sha(REFERENCE / "completed.json") == REFERENCE_SHA
        and done["status"] == "COMPLETED"
        and records["status"] == neural["status"] == "PASS"
        and records["hashes"]["completed_sha256"]
        == neural["completed_sha256"]
        == REFERENCE_SHA
        and sha(REFERENCE / "protocol.json")
        == done["protocol_sha256"]
        == records["hashes"]["protocol_sha256"],
        "verified reference binding",
    )
    order = reference["order"]
    ids = [r["shot"] for r in order]
    require(
        len(ids) == len(set(ids)) == len(records["hashes"]["shot_files"]) == 360
        and all(type(s) is int and s > 0 for s in ids)
        and reference["threshold"] == THRESHOLD,
        "roster/threshold",
    )
    calibration = Path(reference["calibration_reference"])
    mapping, expected_inputs = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    require(
        mapping["calibration_ids"] == ids[:20]
        and set(expected_inputs) == {str(s) for s in ids}
        and sha(calibration / "input_map.json") == reference["input_map_sha256"]
        and sha(calibration / "input_hashes.json") == reference["input_hashes_sha256"]
        and sha(DEFAULT_PARENT) == reference["source_sha256"],
        "input/source contract",
    )
    full, old = (
        load_training(TRAINING, reference, full=True),
        load_training(OLD_TRAINING, reference, full=False),
    )
    paths = [
        PLAN,
        EVALUATION,
        ACCEPTANCE,
        DEFAULT_PARENT,
        Path(reference["cold_state"]),
        TRAINING / "training_cohort.json",
    ]
    paths += [
        REFERENCE / name
        for name in (
            "completed.json",
            "protocol.json",
            "verification_records.json",
            "verification_neural.json",
        )
    ]
    paths += [
        r / name
        for r in (TRAINING, OLD_TRAINING)
        for name in ("completed.json", "protocol.json", "heads.json", "history.json")
    ]
    paths += [calibration / name for name in ("input_map.json", "input_hashes.json")]
    bound = {str(path): sha(path) for path in paths}
    configure()
    model, configuration, cold = load_model()
    require(
        model.base.center.device.type == "cuda"
        and float(configuration["threshold"]) == THRESHOLD
        and sha(cold) == reference["cold_state_sha256"]
        and all(not p.requires_grad for p in model.parameters()),
        "frozen CUDA cold parent required",
    )
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    tables = [
        torch.tensor(data, dtype=torch.float32, device=model.base.center.device)
        for data in (full["full_operational"], old["metric_forecast"])
    ]
    copies = [t.clone() for t in tables]
    heads = build_heads(*tables)
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_FULL_PATH_AND_POSTHOC_FORECAST",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        training=str(TRAINING),
        old_training=str(OLD_TRAINING),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=reference["source_sha256"],
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        calibration_reference=str(calibration),
        threshold=THRESHOLD,
        input_map_sha256=reference["input_map_sha256"],
        input_hashes_sha256=reference["input_hashes_sha256"],
        bound_files_sha256=bound,
        plan_sha256=sha(PLAN),
        evaluation_plan_sha256=sha(EVALUATION),
        acceptance_alignment_sha256=sha(ACCEPTANCE),
        J_comparison_basis="integer_tp_minus_fp_fixed_180_180",
        reused_arms=REUSED,
        fresh_arms=FRESH,
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
    snapshot(root, protocol)
    rows, seals = {a: [] for a in ARMS}, []
    updates = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, sid in enumerate(ids, 1):
        shot, info, sealed = run_shot(
            root,
            index,
            sid,
            records["hashes"]["shot_files"][index - 1],
            expected_inputs[str(sid)],
            model,
            basis,
            heads,
            mapping,
        )
        seals.append(sealed)
        for arm in ARMS:
            rows[arm].append(shot[arm])
            events, action = info["updates"][arm], info["actions"][arm]
            for key, value in dict(
                events=len(events),
                accepted=sum(e["accepted"] for e in events),
                nonzero_gradients=sum(e["gradient_norm"] > 0 for e in events),
                observed_scalars=sum(e["observed"] for e in events),
            ).items():
                updates[arm][key] += value
            for key in ("endpoints", "capped_endpoints"):
                actions[arm][key] += action[key]
            actions[arm]["max_rms"] = max(actions[arm]["max_rms"], action["max_rms"])
        progress = dict(completed=index, total=360, seconds=time.monotonic() - started)
        _durable_json(root / "progress.json", progress)
        if index % 10 == 0:
            print(json.dumps(progress), flush=True)
    require(
        all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
        and all(torch.equal(a, b) for a, b in zip(tables, copies, strict=True)),
        "source/head tensors changed",
    )
    require(
        all(sha(Path(path)) == digest for path, digest in bound.items())
        and all(
            sha(io23.ROOT / path) == digest
            for path, digest in protocol["code_sha256"].items()
        ),
        "bound file/code drift",
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
    paired = {a: flips(rows["cold"], rows[a]) for a in ARMS[1:]}
    base = scores["cold"]["full360"]
    deltas = {
        a: dict(
            delta_tp=scores[a]["full360"]["tp"] - base["tp"],
            delta_fp=scores[a]["full360"]["fp"] - base["fp"],
            delta_tpr=scores[a]["full360"]["tpr"] - base["tpr"],
            delta_J=(net_correct(scores[a]["full360"]) - net_correct(base)) / 180,
            net_favorable_flips=len(paired[a]["rescued_D"])
            + len(paired[a]["removed_FP"])
            - len(paired[a]["lost_D"])
            - len(paired[a]["added_FP"]),
        )
        for a in ARMS[1:]
    }
    write(root / "per_shot.json", rows)
    write(root / "shot_seals.json", seals)
    result = dict(
        status="COMPLETED",
        metrics=scores,
        flips=paired,
        deltas_vs_cold=deltas,
        forecast_matched_flips={
            a: flips(rows[a], rows["forecast_real"])
            for a in ("forecast_reverse", "forecast_hold")
        },
        full360_integer_net={a: net_correct(scores[a]["full360"]) for a in ARMS},
        **score_gates(scores, done["metrics"]["metric_zero"]["full360"]),
        reference_metric_zero=done["metrics"]["metric_zero"],
        recomputed_update_statistics=updates,
        recomputed_action_statistics=actions,
        statistics_scope="recomputed_this_run_only_not_historical_activity",
        fresh_curves=340 * 6,
        reused_suffix_curves=340 * 2,
        reused_warmup_curves=20 * 8,
        total_curves=360 * 8,
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
