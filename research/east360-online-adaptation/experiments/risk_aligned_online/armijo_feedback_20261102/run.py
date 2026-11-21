"""Complete360: matched bounded-step neural feedback and frozen references."""

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
from ..lifecycle_stream_20261025.audit import classify, flips
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform
from .prepare import PLAN
from .stream import simulate

EVALUATION = io23.ROOT / "ARMIJO_CONTROL_EVALUATION_20261102.md"
REFERENCE = io23.ROOT / "probe_outputs/full_path_forecast_controls360_v1"
REFERENCE_SHA = "e63f2a10d4f917279f0e47d38d3851b5cb93bba5b1184e5529a2cf41207b4587"
TRAINING = io23.ROOT / "probe_outputs/armijo_heads20_v1"
REUSED = dict(cold="cold", fixed_full="full_real")
FRESH = tuple(
    p + "_" + m for p in ("forecast", "meta") for m in ("real", "reverse", "hold")
)
ARMS = (*REUSED, *FRESH)


def build_heads(tables):
    require(set(tables) == {"meta", "forecast"}, "matched meta/forecast required")
    for table in tables.values():
        require(
            isinstance(table, torch.Tensor)
            and table.shape == (25, 23)
            and table.dtype == torch.float32
            and bool(torch.isfinite(table).all()),
            "finite float32 25x23 head required",
        )
    require(tables["meta"].device == tables["forecast"].device, "head devices")
    return {
        p + "_" + m: (tables[p], m)
        for p in ("forecast", "meta")
        for m in ("real", "reverse", "hold")
    }


def net_correct(score):
    return score["tp"] - score["fp"]


def score_gates(scores):
    require(set(scores) == set(ARMS), "score arms")
    require(
        all(
            (scores[a]["full360"]["n_disrupt"], scores[a]["full360"]["n_safe"])
            == (180, 180)
            for a in ARMS
        ),
        "fixed180/180 required",
    )
    base, real = (scores[a]["full360"] for a in ("cold", "meta_real"))
    operating = (
        real["fp"] <= 36
        and real["tp"] >= base["tp"]
        and net_correct(real) > net_correct(base)
    )
    comparisons = {
        a: net_correct(real) > net_correct(scores[a]["full360"])
        for a in ("meta_reverse", "meta_hold", "forecast_real", "fixed_full")
    }
    forecast = {
        a: net_correct(scores["forecast_real"]["full360"])
        > net_correct(scores[a]["full360"])
        for a in ("forecast_reverse", "forecast_hold")
    }
    return dict(
        operational_gate=bool(operating),
        control_comparisons=comparisons,
        control_separation_gate=all(comparisons.values()),
        forecast_control_comparisons=forecast,
        forecast_control_separation_gate=all(forecast.values()),
        stage_success_pending_verification=bool(
            operating and all(comparisons.values())
        ),
    )


def validate_curves(times, curves, reference, *, warmup):
    require(
        set(curves) == set(ARMS) and bitwise_equal(times, reference["times"]),
        "curve roster/time drift",
    )
    require(
        np.asarray(times).ndim == 1
        and len(times) > 0
        and np.isfinite(times).all()
        and np.all(np.diff(times) > 0),
        "invalid times",
    )
    for a, curve in curves.items():
        require(
            curve.shape == times.shape
            and curve.dtype == reference["cold"].dtype
            and np.isfinite(curve).all()
            and np.all((curve >= 0) & (curve <= 1)),
            "invalid probability: " + a,
        )
        require(
            bitwise_equal(curve[:160], reference["cold"][:160]), "first160 drift: " + a
        )
        if warmup:
            require(bitwise_equal(curve, reference["cold"]), "preparation drift: " + a)
        elif a in REUSED:
            require(bitwise_equal(curve, reference[REUSED[a]]), "reference drift: " + a)


def load_training(reference):
    done, protocol, tables, history = [
        read(TRAINING / name)
        for name in ("completed.json", "protocol.json", "heads.json", "history.json")
    ]
    ids = [r["shot"] for r in reference["order"][:20]]
    require(
        done["status"] == "COMPLETED"
        and done["head_optimizer_steps"] == protocol["steps"] == 128
        and done["number_heads"] == 2
        and done["ids"] == protocol["ids"] == ids
        and done["source_tensors_unchanged"]
        and done["outcome_reads"] == 20
        and done["suffix_shots_read"] == 0,
        "training budget/cohort",
    )
    require(
        protocol["scope"] == "ENDED20_ARMIJO_FULL_PATH_MATCHED_TRAINING"
        and protocol["heads"] == ["meta", "forecast"]
        and protocol["risk_meta_gradient_horizon"]
        == "all_matured_updates_before_fixed_query"
        and protocol["diagnostic_incoming_state_detached"] is True
        and protocol["max_proposal_evaluations"] == 11
        and protocol["armijo_c"] == 1e-4,
        "training objective/solver",
    )
    require(
        all(
            protocol[k] == reference[k]
            for k in ("source_sha256", "cold_state_sha256", "input_map_sha256")
        )
        and done["plan_sha256"] == protocol["plan_sha256"] == sha(PLAN)
        and done["heads_sha256"] == sha(TRAINING / "heads.json")
        and done["history_sha256"] == sha(TRAINING / "history.json"),
        "training hash/parent drift",
    )
    require(
        set(tables) == {"meta", "forecast"}
        and len(history) == 128
        and all(
            r["step"] == step
            and len(r["sample_shots"]) == 2
            and set(r["sample_shots"]) <= set(ids)
            and set(r["episodes"]) == {"meta", "forecast"}
            for step, r in enumerate(history, 1)
        ),
        "training history",
    )
    return tables


def run_shot(root, index, sid, record, expected_hash, model, basis, heads, mapping):
    require(sha(io23.path_for("EAST", sid)) == expected_hash, "raw input drift")
    path = REFERENCE / "shots" / f"{index:04d}_{sid}.npz"
    oldseal = path.with_name(path.stem + "_seal.json")
    require(
        (record["shot"], record["index"]) == (sid, index)
        and sha(oldseal) == record["seal_sha256"],
        "reference order/seal",
    )
    old = read(oldseal)
    require(
        (old["shot"], old["index"]) == (sid, index)
        and old["current_outcome_used_in_prediction"] is False
        and sha(path) == old["sha256"] == record["curve_sha256"],
        "reference file drift",
    )
    with np.load(path, allow_pickle=False) as f:
        reference = {k: f[k].copy() for k in ("times", "cold", "full_real")}
    times = reference["times"]
    curves = {
        a: reference["cold" if index <= 20 else original].copy()
        for a, original in REUSED.items()
    }
    tapes = {a: [] for a in ARMS}
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    if index <= 20:
        curves.update({a: reference["cold"].copy() for a in FRESH})
    else:
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        result = simulate(model, signal, basis, heads)
        require(
            all(
                set(result[k]) == set(FRESH)
                for k in ("probability", "tapes", "actions")
            ),
            "fresh arm roster",
        )
        curves.update(result["probability"])
        tapes.update(result["tapes"])
        actions.update(result["actions"])
        times = raw["times"]
    validate_curves(times, curves, reference, warmup=index <= 20)
    arrays, info = export_traces(tapes, actions)
    info["statistics_scope"] = "fresh_recomputed_only_not_historical_activity"
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
    stem = root / "shots" / f"{index:04d}_{sid}"
    curve, state, meta, seal = [
        Path(str(stem) + s)
        for s in (".npz", "_states.npz", "_updates.json", "_seal.json")
    ]
    for filename, data in ((curve, dict(times=times, **curves)), (state, arrays)):
        with filename.open("xb") as handle:
            np.savez_compressed(handle, **data)
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
    # Predictions, states and forecast seals above precede this outcome read.
    flag, down = io23.read_outcome("EAST", sid)
    rows = {
        a: dict(
            shot=sid,
            disruptive=flag,
            **classify(times, curves[a], flag, down, THRESHOLD),
        )
        for a in ARMS
    }
    return rows, info, dict(shot=sid, index=index, seal_sha256=sha(seal))


def main(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    reference, done, records, neural = [
        read(REFERENCE / n)
        for n in (
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
        and reference["threshold"] == THRESHOLD,
        "roster/threshold",
    )
    calibration = Path(reference["calibration_reference"])
    mapping, expected = [
        read(calibration / n) for n in ("input_map.json", "input_hashes.json")
    ]
    require(
        mapping["calibration_ids"] == ids[:20]
        and set(expected) == {str(s) for s in ids}
        and sha(calibration / "input_map.json") == reference["input_map_sha256"]
        and sha(calibration / "input_hashes.json") == reference["input_hashes_sha256"]
        and sha(DEFAULT_PARENT) == reference["source_sha256"],
        "input/source contract",
    )
    trained = load_training(reference)
    paths = [PLAN, EVALUATION, DEFAULT_PARENT, Path(reference["cold_state"])]
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
        TRAINING / n
        for n in (
            "completed.json",
            "protocol.json",
            "heads.json",
            "history.json",
            "training_cohort.json",
        )
    ]
    paths += [calibration / n for n in ("input_map.json", "input_hashes.json")]
    bound = {str(p): sha(p) for p in paths}
    configure()
    model, configuration, cold = load_model()
    require(
        model.base.center.device.type == "cuda"
        and float(configuration["threshold"]) == THRESHOLD
        and sha(cold) == reference["cold_state_sha256"]
        and all(not p.requires_grad for p in model.parameters()),
        "frozen CUDA parent",
    )
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    tables = {
        a: torch.tensor(v, dtype=torch.float32, device=model.base.center.device)
        for a, v in trained.items()
    }
    copies = {a: t.clone() for a, t in tables.items()}
    heads = build_heads(tables)
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_BOUNDED_ARMIJO_MATCHED_FEEDBACK",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        training=str(TRAINING),
        training_completed_sha256=sha(TRAINING / "completed.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=reference["source_sha256"],
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        calibration_reference=str(calibration),
        input_map_sha256=reference["input_map_sha256"],
        input_hashes_sha256=reference["input_hashes_sha256"],
        threshold=THRESHOLD,
        bound_files_sha256=bound,
        plan_sha256=sha(PLAN),
        evaluation_plan_sha256=sha(EVALUATION),
        reused_arms=REUSED,
        fresh_arms=FRESH,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        temporary_parameters=256,
        reset_each_shot=True,
        max_proposal_evaluations=11,
        armijo_c=1e-4,
        metric_increment_hidden_rms_cap=0.05,
        hidden_rms_cap=0.25,
        head_training_shots=20,
        parameters_per_head=575,
        head_optimizer_steps=128,
        suffix_head_updates=0,
        suffix_inter_supervision=0,
        warmup_cost_included=True,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        J_comparison_basis="integer_tp_minus_fp_fixed_180_180",
        no_outcome_input_to_simulate=True,
        independent_future_confirmation=False,
        real_time_latency_verified=False,
        equal_floating_point_compute_budget_claimed=False,
    )
    files = {Path(__file__), PLAN, EVALUATION}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    protocol["code_sha256"] = {str(p.relative_to(io23.ROOT)): sha(p) for p in files}
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(files):
            archive.write(p, str(p.relative_to(io23.ROOT)))
    rows, seals = {a: [] for a in ARMS}, []
    updates = {
        a: dict(
            events=0, accepted=0, evaluations=0, nonzero_gradients=0, observed_scalars=0
        )
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, sid in enumerate(ids, 1):
        shot, info, seal = run_shot(
            root,
            index,
            sid,
            records["hashes"]["shot_files"][index - 1],
            expected[str(sid)],
            model,
            basis,
            heads,
            mapping,
        )
        seals.append(seal)
        for a in ARMS:
            rows[a].append(shot[a])
            events, action = info["updates"][a], info["actions"][a]
            for k, v in dict(
                events=len(events),
                accepted=sum(e["accepted"] for e in events),
                evaluations=sum(e["evaluations"] for e in events),
                nonzero_gradients=sum(e["gradient_norm"] > 0 for e in events),
                observed_scalars=sum(e["observed"] for e in events),
            ).items():
                updates[a][k] += v
            for k in ("endpoints", "capped_endpoints"):
                actions[a][k] += action[k]
            actions[a]["max_rms"] = max(actions[a]["max_rms"], action["max_rms"])
        progress = dict(completed=index, total=360, seconds=time.monotonic() - started)
        _durable_json(root / "progress.json", progress)
        if index % 10 == 0:
            print(json.dumps(progress), flush=True)
    require(
        all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
        and all(torch.equal(tables[a], copies[a]) for a in tables),
        "source/head drift",
    )
    require(
        all(sha(Path(p)) == h for p, h in bound.items())
        and all(sha(io23.ROOT / p) == h for p, h in protocol["code_sha256"].items()),
        "bound/code drift",
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
    write(root / "per_shot.json", rows)
    write(root / "shot_seals.json", seals)
    result = dict(
        status="COMPLETED",
        metrics=scores,
        flips=paired,
        full360_integer_net={a: net_correct(scores[a]["full360"]) for a in ARMS},
        **score_gates(scores),
        recomputed_update_statistics=updates,
        recomputed_action_statistics=actions,
        statistics_scope="fresh_recomputed_only_not_historical_activity",
        source_tensors_unchanged=True,
        heads_unchanged=True,
        curve_execution_counts=dict(
            fresh_suffix=2040, copied_preparation=120, referenced_parent=720
        ),
        total_curves=2880,
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
