"""Only objective replacement; four fresh arms and six exact archived controls."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.model import encode_views, latent_forward
from ..dual_history_ttt_20261101.run import digest_array, load_model
from ..dual_history_ttt_20261101.stream import feedback_origin, targets
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from .model import step

OLD = io23.ROOT / "probe_outputs/dual_history_ttt_v1"
OLD_SHA = "dc1708ebacb64294443f3c24dbb69be53494671e209329905b0032215d9ff626"
PLAN = io23.ROOT / "CENTERED_GATE_STREAM_PLAN_20261101.md"
FRESH = ("cold", "centered_real", "centered_reverse", "centered_hold")
COPY = dict(
    absolute_real="real",
    absolute_reverse="reverse",
    absolute_hold="hold",
    long_data="long_data",
    snapshot_data="snapshot_data",
    equal_data="equal_data",
)
ARMS = (*FRESH, *COPY)


def predict(model, signal):
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only")
    device = model.base.center.device
    betas = {a: torch.zeros(16, 2, device=device) for a in FRESH}
    curves = {a: [] for a in FRESH}
    updates = {a: [] for a in FRESH[1:]}
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        pack = encode_views(model, signal, ends)
        with torch.no_grad():
            for a in FRESH:
                z = latent_forward(model, pack, betas[a])["logit"]
                curves[a].append(z.sigmoid().cpu().numpy())
        end = int(ends[-1])
        origin = feedback_origin(end)
        if origin is None:
            continue
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        with torch.no_grad():
            forecast_seals = {
                a: digest_array(
                    latent_forward(model, point, betas[a], forecast=True)["forecast"]
                    .cpu()
                    .numpy()
                )
                for a in updates
            }
        y, available = targets(signal, origin, end)
        mask = torch.as_tensor(available, device=device)
        for a in updates:
            control = a.removeprefix("centered_")
            before = betas[a].flatten().cpu().tolist()
            betas[a], info = step(
                model, point, betas[a], torch.as_tensor(y[control], device=device), mask
            )
            updates[a].append(
                dict(
                    origin=origin,
                    target_first=origin + 1,
                    target_last=end,
                    risk_sealed_through=end,
                    first_affected=end + 1,
                    forecast_sha256=forecast_seals[a],
                    mask_sha256=digest_array(available),
                    target_sha256=digest_array(y[control]),
                    beta32_before=before,
                    **info,
                )
            )
    curves = {a: np.concatenate(v) for a, v in curves.items()}
    assert all(np.array_equal(curves[a][:160], curves["cold"][:160]) for a in updates)
    return curves, dict(reset_each_shot=True, temporary_parameters=32, updates=updates)


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(OLD / "completed.json") == OLD_SHA
    old_protocol = read(OLD / "protocol.json")
    old_verification = read(OLD / "verification.json")
    assert (
        old_verification["status"] == "PASS"
        and old_verification["completed_sha256"] == OLD_SHA
    )
    model, parent, cold = load_model()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mapping = read(REFERENCE / "input_map.json")
    order = old_protocol["order"]
    threshold = float(parent["threshold"])
    assert (
        threshold == old_protocol["threshold"]
        and sha(cold) == old_protocol["cold_state_sha256"]
    )
    assert sha(REFERENCE / "input_map.json") == old_protocol["input_map_sha256"]
    if preflight:
        sid = order[0]["shot"]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k][:416] for k in ("values", "available")}, mapping)
        curves, receipt = predict(model, signal)
        other = {k: v.copy() for k, v in signal.items()}
        other["values"][320:] += 1e5
        other["available"][320:] = ~other["available"][320:]
        changed, changed_receipt = predict(model, other)
        assert all(np.array_equal(curves[a][:320], changed[a][:320]) for a in FRESH)
        for a in FRESH[1:]:
            assert [r for r in receipt["updates"][a] if r["target_last"] < 320] == [
                r for r in changed_receipt["updates"][a] if r["target_last"] < 320
            ]
        root.mkdir(parents=True)
        _durable_json(
            root / "preflight.json",
            dict(
                status="PASS",
                shot=sid,
                points=len(signal["values"]),
                curves_prefix_exact=4,
                cutoff=320,
                receipt=receipt,
                outcomes_read=0,
                suffix_shots_read=0,
            ),
        )
        print("centered CUDA preflight PASS", flush=True)
        return
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_SINGLE_OBJECTIVE_REPLACEMENT",
        order=order,
        arms=ARMS,
        fresh_arms=FRESH,
        copied_arms=COPY,
        old=str(OLD),
        old_completed_sha256=OLD_SHA,
        old_protocol_sha256=sha(OLD / "protocol.json"),
        old_verification_sha256=sha(OLD / "verification.json"),
        reference=str(REFERENCE),
        source_checkpoint=old_protocol["source_checkpoint"],
        source_sha256=old_protocol["source_sha256"],
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        threshold=threshold,
        outcome_authority="HDF5",
        warmup20_cost_included=True,
        first_update=159,
        update_interval=128,
        forecast_horizon=25,
        step_size=0.1,
        anchor=0.01,
        temporary_parameters=32,
        source_and_head_frozen=True,
        persistent_neural_updates=0,
        independent_confirmation=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    hashes = read(REFERENCE / "input_hashes.json")
    rows = {a: [] for a in ARMS}
    aggregate = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in FRESH[1:]
    }
    for i, record in enumerate(order):
        sid, prefix = record["shot"], f"{i + 1:04d}_{record['shot']}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        old_path = OLD / "shots" / f"{prefix}.npz"
        old_seal = read(OLD / "shots" / f"{prefix}_seal.json")
        assert sha(old_path) == old_seal["sha256"]
        with np.load(old_path, allow_pickle=False) as old:
            assert np.array_equal(s["times"], old["times"])
            if i < 20:
                curves = {a: old["cold"].copy() for a in FRESH}
                info = dict(
                    reset_each_shot=True,
                    temporary_parameters=0,
                    updates={a: [] for a in FRESH[1:]},
                )
            else:
                curves, info = predict(
                    model,
                    transform({k: s[k] for k in ("values", "available")}, mapping),
                )
            assert np.array_equal(curves["cold"], old["cold"]), (sid, "cold drift")
            curves.update({new: old[old_name].copy() for new, old_name in COPY.items()})
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=s["times"], **curves)
            handle.flush()
            os.fsync(handle.fileno())
        info_path = root / "shots" / f"{prefix}_updates.json"
        _durable_json(info_path, info)
        _durable_json(
            root / "shots" / f"{prefix}_seal.json",
            dict(
                index=i + 1,
                shot=sid,
                sha256=sha(path),
                updates_sha256=sha(info_path),
                copied_from_sha256=sha(old_path),
                outcome_available=False,
            ),
        )
        flag, down = io23.read_outcome("EAST", sid)
        for a, curve in curves.items():
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(s["times"], curve, flag, down, threshold),
                )
            )
        for a in aggregate:
            for u in info["updates"][a]:
                aggregate[a]["events"] += 1
                aggregate[a]["accepted"] += u["accepted"]
                aggregate[a]["nonzero_gradients"] += u["gradient_norm"] > 0
                aggregate[a]["observed_scalars"] += u["observed"]
        if (i + 1) % 20 == 0:
            progress = dict(
                shots=i + 1,
                total=360,
                seconds=time.monotonic() - start,
                gradient_events=aggregate["centered_real"]["events"],
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items())
    metrics = {
        a: {
            scope: counts(subset)
            for scope, subset in (
                ("full360", r),
                ("warmup20", r[:20]),
                ("suffix340", r[20:]),
            )
        }
        for a, r in rows.items()
    }
    m, c = metrics["centered_real"]["full360"], metrics["cold"]["full360"]
    operation = m["fpr"] <= 0.20 and m["tp"] >= c["tp"] and m["J"] > c["J"]
    controls = (
        "centered_reverse",
        "centered_hold",
        "long_data",
        "snapshot_data",
        "equal_data",
    )
    separation = all(m["J"] > metrics[a]["full360"]["J"] for a in controls)
    result = dict(
        status="COMPLETED",
        metrics=metrics,
        update_statistics=aggregate,
        flips={a: flips(rows["cold"], r) for a, r in rows.items()},
        operational_gate=operation,
        control_separation_gate=separation,
        source_state_unchanged=True,
        reference_cold_exact=360,
        fresh_arm_curve_count=1440,
        fresh_neural_curve_count=1360,
        reused_prefix_curve_count=80,
        copied_curve_count=2160,
        seconds=time.monotonic() - start,
        method_promoted=False,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "per_shot.json", rows)
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
