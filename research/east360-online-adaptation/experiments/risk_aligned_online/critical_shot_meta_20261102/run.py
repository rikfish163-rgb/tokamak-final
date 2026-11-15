"""Complete360 equal-parent, equal-budget shot-local neural feedback trial."""

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
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..warmstart_alignment_20261022.alignment import transform
from .prepare import PLAN
from .stream import simulate

REFERENCE = io23.ROOT / "probe_outputs/lora_stream_v1"
REFERENCE_SHA = "76ebb4035873ba3c6b85da9d89b995233d2ce36222237bf1733058d4c9482f4c"
TRAINING = io23.ROOT / "probe_outputs/critical_shot_heads20_v1"
ARMS = (
    "cold",
    "original_difference",
    "forecast",
    "point",
    "operational",
    "reverse",
    "hold",
)


def export_traces(tapes, actions):
    arrays = {}
    metadata = dict(
        reset_each_shot=True, temporary_parameters=256, actions=actions, updates={}
    )
    for arm, events in tapes.items():
        arrays[arm + "_before"] = (
            np.stack([e["before"].detach().cpu().numpy() for e in events])
            if events
            else np.empty((0, 4, 64), np.float32)
        )
        arrays[arm + "_after"] = (
            np.stack([e["after"].detach().cpu().numpy() for e in events])
            if events
            else np.empty((0, 4, 64), np.float32)
        )
        gradients = []
        rows = []
        for e in events:
            receipt = e["receipt"]
            computed = "gradient256" in receipt
            if computed:
                gradients.append(np.asarray(receipt["gradient256"], np.float32))
            else:
                # Explicit storage sentinel, not a measured/fabricated gradient.
                # The upstream no-target branch computes no gradient or update.
                assert e["observed"] == 0 and receipt["reason"] == "no_target"
                gradients.append(np.zeros(256, np.float32))
            row = {
                k: v for k, v in e.items() if k not in ("before", "after", "receipt")
            }
            row.update(
                {
                    k: v
                    for k, v in receipt.items()
                    if k not in ("gradient256", "state256_after")
                }
            )
            row["gradient_computed"] = computed
            rows.append(row)
        arrays[arm + "_gradient"] = (
            np.stack(gradients) if events else np.empty((0, 256), np.float32)
        )
        metadata["updates"][arm] = rows
    return arrays, metadata


def metrics(rows):
    result = counts(rows)
    leads = [r["lead_ms"] for r in rows if r["timely"]]
    result["median_timely_lead_ms"] = float(np.median(leads)) if leads else None
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    reference_protocol = read(REFERENCE / "protocol.json")
    assert read(REFERENCE / "verification_records.json")["status"] == "PASS"
    preparation = read(TRAINING / "completed.json")
    assert (
        preparation["status"] == "COMPLETED"
        and preparation["head_optimizer_steps"] == 128
    )
    assert sha(TRAINING / "heads.json") == preparation["heads_sha256"]
    assert sha(TRAINING / "history.json") == preparation["history_sha256"]
    assert preparation["plan_sha256"] == sha(PLAN)
    training_protocol = read(TRAINING / "protocol.json")
    order = reference_protocol["order"]
    assert len(order) == 360 and len({r["shot"] for r in order}) == 360
    assert training_protocol["ids"] == [r["shot"] for r in order[:20]]
    model, configuration, cold = load_model()
    assert (
        sha(cold)
        == reference_protocol["cold_state_sha256"]
        == training_protocol["cold_state_sha256"]
    )
    threshold = float(configuration["threshold"])
    assert threshold == reference_protocol["threshold"]
    calibration = Path(reference_protocol["calibration_reference"])
    assert (
        sha(calibration / "input_map.json")
        == reference_protocol["input_map_sha256"]
        == training_protocol["input_map_sha256"]
    )
    mapping, expected_inputs = (
        read(calibration / "input_map.json"),
        read(calibration / "input_hashes.json"),
    )
    assert mapping["calibration_ids"] == training_protocol["ids"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    tables = {
        k: torch.tensor(v, dtype=torch.float32, device="cuda")
        for k, v in read(TRAINING / "heads.json").items()
    }
    assert set(tables) == {"forecast", "point", "operational"}
    assert all(b.shape == (25, 23) and torch.isfinite(b).all() for b in tables.values())
    table_copy = {k: v.clone() for k, v in tables.items()}
    heads = dict(
        cold=(None, "real"),
        original_difference=(torch.zeros_like(tables["operational"]), "real"),
        forecast=(tables["forecast"], "real"),
        point=(tables["point"], "real"),
        operational=(tables["operational"], "real"),
        reverse=(tables["operational"], "reverse"),
        hold=(tables["operational"], "hold"),
    )
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_CRITICAL_ENDPOINT_NEURAL_META",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        training=str(TRAINING),
        training_completed_sha256=sha(TRAINING / "completed.json"),
        heads_sha256=sha(TRAINING / "heads.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        calibration_reference=str(calibration),
        input_map_sha256=sha(calibration / "input_map.json"),
        threshold=threshold,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        parameter_step=0.1,
        hidden_rms_cap=0.25,
        temporary_parameters=256,
        reset_each_shot=True,
        head_fitted_ended20_only=True,
        suffix_head_updates=0,
        suffix_inter_supervision=0,
        warmup_cost_included=True,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        no_outcome_input_to_simulate=True,
        independent_future_confirmation=False,
        real_time_latency_verified=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows = {arm: [] for arm in ARMS}
    totals = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in ARMS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, entry in enumerate(order, 1):
        sid = entry["shot"]
        stem = f"{index:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == expected_inputs[str(sid)]
        assert (
            sha(REFERENCE / "shots" / (stem + ".npz"))
            == read(REFERENCE / "shots" / (stem + "_seal.json"))["sha256"]
        )
        with np.load(REFERENCE / "shots" / (stem + ".npz"), allow_pickle=False) as ref:
            if index <= 20:
                times = ref["times"].copy()
                curves = {a: ref["cold"].copy() for a in ARMS}
                arrays, info = export_traces(
                    {a: [] for a in ARMS},
                    {
                        a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0)
                        for a in ARMS
                    },
                )
            else:
                raw = apply_input_contract(
                    simple_data.read_shot("EAST", sid, rate_contract="nominal")
                )
                signal = transform(
                    {k: raw[k] for k in ("values", "available")}, mapping
                )
                times = raw["times"]
                result = simulate(model, signal, basis, heads)
                curves = result["probability"]
                assert np.array_equal(times, ref["times"])
                assert np.array_equal(curves["cold"], ref["cold"]), (sid, "cold drift")
                assert all(
                    np.array_equal(curves[a][:160], curves["cold"][:160]) for a in ARMS
                )
                arrays, info = export_traces(result["tapes"], result["actions"])
        curve_path = root / "shots" / (stem + ".npz")
        trace_path = root / "shots" / (stem + "_states.npz")
        info_path = root / "shots" / (stem + "_updates.json")
        with curve_path.open("xb") as handle:
            np.savez_compressed(handle, times=times, **curves)
        with trace_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        write(info_path, info)
        write(
            root / "shots" / (stem + "_seal.json"),
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
        flag, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            rows[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(times, curves[arm], flag, down, threshold),
                )
            )
            stats = info["actions"][arm]
            actions[arm]["endpoints"] += stats["endpoints"]
            actions[arm]["capped_endpoints"] += stats["capped_endpoints"]
            actions[arm]["max_rms"] = max(actions[arm]["max_rms"], stats["max_rms"])
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
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    assert all(torch.equal(v, tables[k]) for k, v in table_copy.items())
    scores = {
        a: {
            part: metrics(data)
            for part, data in [
                ("full360", rows[a]),
                ("warmup20", rows[a][:20]),
                ("suffix340", rows[a][20:]),
            ]
        }
        for a in ARMS
    }
    base, candidate = scores["cold"]["full360"], scores["operational"]["full360"]
    stage = (
        candidate["fpr"] <= 0.20
        and candidate["tp"] >= base["tp"]
        and candidate["fp"] <= base["fp"]
        and candidate["J"] > base["J"]
    )
    separation = all(
        candidate["J"] > scores[a]["full360"]["J"] for a in ("reverse", "hold")
    )
    write(root / "per_shot.json", rows)
    result = dict(
        status="COMPLETED",
        metrics=scores,
        flips={a: flips(rows["cold"], rows[a]) for a in ARMS[1:]},
        update_statistics=totals,
        action_statistics=actions,
        operational_gate=stage,
        control_separation_gate=separation,
        task_comparisons={
            a: candidate["J"] > scores[a]["full360"]["J"]
            for a in ("original_difference", "forecast", "point")
        },
        fresh_curves=340 * len(ARMS),
        reused_warmup_curves=20 * len(ARMS),
        source_tensors_unchanged=True,
        heads_unchanged=True,
        stage_success_pending_verification=bool(stage and separation),
        method_promoted=False,
        scientific_goal_achieved=False,
        independent_future_confirmation=False,
        per_shot_sha256=sha(root / "per_shot.json"),
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
                    "task_comparisons",
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
