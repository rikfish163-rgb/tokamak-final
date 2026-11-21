"""Complete360, label-free shot-reset first-FFN diagnostic feedback."""

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
from ..dual_history_ttt_20261101.run import digest_array, load_model
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .core import direct, encode, forward, step

REFERENCE = io23.ROOT / "probe_outputs/centered_gate_stream_v1"
REFERENCE_SHA = "2b28ac02039b9106627c5aca865d23bcdd340ca2eda5d55d328aea46850f0d42"
PLAN = io23.ROOT / "LORA_STREAM_PLAN_20261102.md"
ARMS = ("cold", "real", "reverse", "hold")
LEARNERS = ARMS[1:]


def empty_info():
    return dict(
        reset_each_shot=True,
        temporary_parameters=256,
        updates={a: [] for a in LEARNERS},
        action_statistics={
            a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS
        },
    )


def predict(model, signal, basis):
    """No outcome metadata accepted; targets are used only at maturity."""
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only")
    device = model.base.center.device
    state = {a: torch.zeros(4, 64, device=device) for a in ARMS}
    curves = {a: [] for a in ARMS}
    info = empty_info()
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        end = int(ends[-1])
        origin = feedback_origin(end)
        pack = encode(model, signal, ends, basis)
        seals = {}
        with torch.no_grad():
            for arm in ARMS:
                result = forward(
                    model,
                    pack,
                    state[arm],
                    forecast=origin is not None and arm in LEARNERS,
                )
                curves[arm].append(result["logit"].sigmoid().cpu().numpy())
                stats = info["action_statistics"][arm]
                stats["endpoints"] += len(ends)
                stats["capped_endpoints"] += int(result["capped"].sum())
                stats["max_rms"] = max(stats["max_rms"], float(result["rms"].max()))
                if origin is not None and arm in LEARNERS:
                    seals[arm] = digest_array(
                        result["forecast"][origin - lo : origin - lo + 1].cpu().numpy()
                    )
        if origin is None:
            continue
        assert lo <= origin <= end and origin + 25 == end
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        truth, available = targets(signal, origin, end)
        mask = torch.as_tensor(available, device=device)
        for arm in LEARNERS:
            before = state[arm].flatten().cpu().tolist()
            state[arm], receipt = step(
                model,
                point,
                state[arm],
                torch.as_tensor(truth[arm], device=device),
                mask,
            )
            info["updates"][arm].append(
                dict(
                    origin=origin,
                    target_first=origin + 1,
                    target_last=end,
                    risk_sealed_through=end,
                    first_affected=end + 1,
                    forecast_sha256=seals[arm],
                    mask_sha256=digest_array(available),
                    target_sha256=digest_array(truth[arm]),
                    state256_before=before,
                    **receipt,
                )
            )
    curves = {a: np.concatenate(parts) for a, parts in curves.items()}
    assert all(np.array_equal(curves[a][:160], curves["cold"][:160]) for a in LEARNERS)
    assert max(s["max_rms"] for s in info["action_statistics"].values()) <= 0.250001
    return curves, info


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    assert read(REFERENCE / "verification.json")["status"] == "PASS"
    reference_protocol = read(REFERENCE / "protocol.json")
    calibration = Path(reference_protocol["reference"])
    order = reference_protocol["order"]
    mapping = read(calibration / "input_map.json")
    assert mapping["calibration_ids"] == [r["shot"] for r in order[:20]]
    hashes = read(calibration / "input_hashes.json")
    model, parent, cold = load_model()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert sha(cold) == reference_protocol["cold_state_sha256"]
    basis = source_basis(model)
    if preflight:
        sid = order[0]["shot"]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k][:416] for k in ("values", "available")}, mapping)
        curves, info = predict(model, signal, basis)
        direct_cold = []
        with torch.no_grad():
            for lo in range(0, len(signal["values"]), 32):
                ends = np.arange(lo, min(lo + 32, len(signal["values"])))
                pack = encode(model, signal, ends, basis)
                direct_cold.append(
                    model.risk(
                        pack["h"], pack["valid"], pack["h"].new_zeros(len(ends), 92)
                    )
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
            assert np.array_equal(curves["cold"], np.concatenate(direct_cold))
            state = torch.as_tensor(
                info["updates"]["real"][0]["state256_after"], device="cuda"
            ).reshape(4, 64)
            ends = np.arange(256, 288)
            pack = encode(model, signal, ends, basis)
            cached = forward(model, pack, state, forecast=True)
            actual = direct(model, signal, ends, basis, state, forecast=True)
            assert torch.equal(cached["logit"], actual["logit"])
            assert torch.equal(cached["forecast"], actual["forecast"])
        assert all(
            torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
        )
        root.mkdir(parents=True)
        _durable_json(
            root / "preflight.json",
            dict(
                status="PASS",
                shot=sid,
                points=len(signal["values"]),
                real_updates=len(info["updates"]["real"]),
                cold_exact=True,
                direct_layer_path_exact=True,
                source_state_unchanged=True,
                outcomes_read=0,
                target_suffix_shots_read=0,
                receipt=info,
            ),
        )
        print("CUDA preflight PASS", flush=True)
        return
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    torch.save({"basis": basis.cpu()}, root / "source_basis.pt")
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_SHOT_RESET_FFN_GRADIENT",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        calibration_reference=str(calibration),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(calibration / "input_map.json"),
        basis_sha256=sha(root / "source_basis.pt"),
        threshold=float(parent["threshold"]),
        warmup20_cost_included=True,
        outcome_authority="HDF5",
        temporary_parameters=256,
        persistent_neural_updates=0,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        forecast_horizon=25,
        step_size=0.1,
        parameter_anchor=0.01,
        endpoint_hidden_rms_cap=0.25,
        cap_uses_only_current_window=True,
        forecast_objective="observed_temporal_centered_SmoothL1",
        no_label_input_to_predict=True,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows = {a: [] for a in ARMS}
    totals = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in LEARNERS
    }
    actions = {a: dict(endpoints=0, capped_endpoints=0, max_rms=0.0) for a in ARMS}
    for index, record in enumerate(order, 1):
        sid = record["shot"]
        prefix = f"{index:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        with np.load(REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False) as ref:
            if index <= 20:
                times = ref["times"].copy()
                curves = {a: ref["cold"].copy() for a in ARMS}
                info = empty_info()
            else:
                raw = apply_input_contract(
                    simple_data.read_shot("EAST", sid, rate_contract="nominal")
                )
                signal = transform(
                    {k: raw[k] for k in ("values", "available")}, mapping
                )
                times = raw["times"]
                curves, info = predict(model, signal, basis)
                assert np.array_equal(times, ref["times"])
                assert np.array_equal(curves["cold"], ref["cold"]), (sid, "cold drift")
        curve_path, log_path = (
            root / "shots" / f"{prefix}.npz",
            root / "shots" / f"{prefix}_updates.json",
        )
        with curve_path.open("xb") as handle:
            np.savez_compressed(handle, times=times, **curves)
        _durable_json(log_path, info)
        _durable_json(
            root / "shots" / f"{prefix}_seal.json",
            dict(
                shot=sid,
                index=index,
                outcome_available=False,
                warmup_curves_reused=index <= 20,
                sha256=sha(curve_path),
                updates_sha256=sha(log_path),
            ),
        )
        flag, down = io23.read_outcome("EAST", sid)
        for arm in ARMS:
            rows[arm].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(times, curves[arm], flag, down, protocol["threshold"]),
                )
            )
            old, new = actions[arm], info["action_statistics"][arm]
            old["endpoints"] += new["endpoints"]
            old["capped_endpoints"] += new["capped_endpoints"]
            old["max_rms"] = max(old["max_rms"], new["max_rms"])
        for arm in LEARNERS:
            history = info["updates"][arm]
            totals[arm]["events"] += len(history)
            totals[arm]["accepted"] += sum(r["accepted"] for r in history)
            totals[arm]["nonzero_gradients"] += sum(
                r["gradient_norm"] > 0 for r in history
            )
            totals[arm]["observed_scalars"] += sum(r["observed"] for r in history)
        progress = dict(
            completed=index, total=len(order), seconds=time.monotonic() - start
        )
        _durable_json(root / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    metrics = {
        a: {
            scope: counts(part)
            for scope, part in (
                ("full360", rows[a]),
                ("warmup20", rows[a][:20]),
                ("suffix340", rows[a][20:]),
            )
        }
        for a in ARMS
    }
    m, baseline = metrics["real"]["full360"], metrics["cold"]["full360"]
    operational = (
        m["fpr"] <= 0.2 and m["tp"] >= baseline["tp"] and m["J"] > baseline["J"]
    )
    separation = all(m["J"] > metrics[a]["full360"]["J"] for a in ("reverse", "hold"))
    _durable_json(root / "per_shot.json", rows)
    completed = dict(
        status="COMPLETED",
        metrics=metrics,
        flips={a: flips(rows["cold"], rows[a]) for a in LEARNERS},
        update_statistics=totals,
        action_statistics=actions,
        fresh_curves=340 * len(ARMS),
        reused_warmup_curves=20 * len(ARMS),
        source_state_unchanged=True,
        operational_gate=operational,
        control_separation_gate=separation,
        method_promoted=operational and separation,
        scientific_goal_achieved=False,
        independent_future_confirmation=False,
        seconds=time.monotonic() - start,
    )
    _durable_json(root / "completed.json", completed)
    print(
        json.dumps(
            {
                k: completed[k]
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
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
