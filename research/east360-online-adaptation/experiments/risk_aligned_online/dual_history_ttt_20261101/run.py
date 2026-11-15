"""Complete360 fixed-head, shot-local temporal-view gradient experiment."""

import argparse
import hashlib
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
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import transform
from .model import encode_views, latent_forward, step
from .stream import feedback_origin, targets

PLAN = io23.ROOT / "DUAL_HISTORY_TTT_PLAN_20261101.md"
ARMS = ("cold", "long_data", "snapshot_data", "equal_data", "real", "reverse", "hold")
LEARNERS = ("real", "reverse", "hold")


def digest_array(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def predict(model, signal):
    """Batch causal windows; no result metadata is accepted by this function."""
    if set(signal) != {"values", "available"}:
        raise ValueError("signals only")
    device = model.base.center.device
    beta = {a: torch.zeros(16, 2, device=device) for a in ARMS}
    beta["long_data"][:, 0] = 1
    beta["snapshot_data"][:, 1] = 1
    beta["equal_data"][:] = 1 / 3
    curves, updates = {a: [] for a in ARMS}, {a: [] for a in LEARNERS}
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        pack = encode_views(model, signal, ends)
        with torch.no_grad():
            for arm in ARMS:
                z = latent_forward(model, pack, beta[arm])["logit"]
                curves[arm].append(z.sigmoid().cpu().numpy())
        end = int(ends[-1])
        origin = feedback_origin(end)
        if origin is None:
            continue
        assert lo <= origin <= end and origin + 25 == end
        point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
        # Seal the original forecast before target construction. No intermediate
        # update occurs between origin and maturity at the fixed128 cadence.
        with torch.no_grad():
            forecast_seals = {
                a: digest_array(
                    latent_forward(model, point, beta[a], forecast=True)["forecast"]
                    .cpu()
                    .numpy()
                )
                for a in LEARNERS
            }
        truth, available = targets(signal, origin, end)
        mask = torch.as_tensor(available, device=device)
        for arm in LEARNERS:
            before = beta[arm].flatten().cpu().tolist()
            beta[arm], receipt = step(
                model,
                point,
                beta[arm],
                torch.as_tensor(truth[arm], device=device),
                mask,
            )
            updates[arm].append(
                dict(
                    origin=origin,
                    target_first=origin + 1,
                    target_last=end,
                    risk_sealed_through=end,
                    first_affected=end + 1,
                    forecast_sha256=forecast_seals[arm],
                    mask_sha256=digest_array(available),
                    target_sha256=digest_array(truth[arm]),
                    beta32_before=before,
                    **receipt,
                )
            )
    curves = {a: np.concatenate(v) for a, v in curves.items()}
    for a in LEARNERS:
        assert np.array_equal(curves[a][:160], curves["cold"][:160])
    info = dict(reset_each_shot=True, temporary_parameters=32, updates=updates)
    return curves, info


def load_model():
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    cold = REFERENCE / "models/cold.pt"
    model.restore(torch.load(cold, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False), parent, cold


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    model, parent, cold = load_model()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent["threshold"])
    order = read(REFERENCE / "order.json")
    hashes = read(REFERENCE / "input_hashes.json")
    mapping = read(REFERENCE / "input_map.json")
    assert mapping["calibration_ids"] == [r["shot"] for r in order[:20]]
    if preflight:
        # Ended first20 only; all360 identities are fixed, no suffix inference.
        sid = order[0]["shot"]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: s[k][:416] for k in ("values", "available")}, mapping)
        curves, receipt = predict(model, signal)
        with torch.no_grad():
            direct = []
            for lo in range(0, len(signal["values"]), 32):
                ends = np.arange(lo, min(lo + 32, len(signal["values"])))
                pack = encode_views(model, signal, ends)
                z = model.risk(
                    pack["h"], pack["valid"], pack["h"].new_zeros(len(ends), 92)
                )
                direct.append(z.sigmoid().cpu().numpy())
        assert np.array_equal(curves["cold"], np.concatenate(direct))
        assert all(
            torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items()
        )
        root.mkdir(parents=True)
        _durable_json(
            root / "preflight.json",
            dict(
                status="PASS",
                shot=sid,
                points=len(signal["values"]),
                real_updates=len(receipt["updates"]["real"]),
                receipt=receipt,
                cold_exact=True,
                source_state_unchanged=True,
                target_suffix_signals_read=0,
                outcomes_read=0,
            ),
        )
        print("CUDA preflight PASS", flush=True)
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
        scope="DEVELOPMENT_COMPLETE360_TEMPORARY_INTERNAL_VIEW_GRADIENT",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        threshold=threshold,
        warmup20_cost_included=True,
        outcome_authority="HDF5",
        temporary_parameters=32,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        forecast_horizon=25,
        step_size=0.1,
        anchor=0.01,
        classifier_alarm_source_frozen=True,
        persistent_neural_updates=0,
        target_masks="origin & future & reversed_future; exclude discrete mode_number_n",
        plan_sha256=sha(PLAN),
        independent_confirmation=False,
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
    rows = {a: [] for a in ARMS}
    aggregate = {
        a: dict(events=0, accepted=0, nonzero_gradients=0, observed_scalars=0)
        for a in LEARNERS
    }
    for index, record in enumerate(order):
        sid, prefix = record["shot"], f"{index + 1:04d}_{record['shot']}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        with np.load(REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False) as ref:
            assert np.array_equal(s["times"], ref["times"])
            if index < 20:
                curves = {a: ref["source"].copy() for a in ARMS}
                info = dict(
                    reset_each_shot=True,
                    temporary_parameters=0,
                    updates={a: [] for a in LEARNERS},
                )
            else:
                curves, info = predict(
                    model,
                    transform({k: s[k] for k in ("values", "available")}, mapping),
                )
            assert np.array_equal(curves["cold"], ref["cold"]), (sid, "cold drift")
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
                index=index + 1,
                shot=sid,
                sha256=sha(path),
                updates_sha256=sha(info_path),
                outcome_available=False,
            ),
        )
        # No learning occurs below this point; result fields never enter predict.
        flag, down = io23.read_outcome("EAST", sid)
        for a, curve in curves.items():
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(s["times"], curve, flag, down, threshold),
                )
            )
        for a in LEARNERS:
            for u in info["updates"][a]:
                aggregate[a]["events"] += 1
                aggregate[a]["accepted"] += u["accepted"]
                aggregate[a]["nonzero_gradients"] += u["gradient_norm"] > 0
                aggregate[a]["observed_scalars"] += u["observed"]
        if (index + 1) % 20 == 0:
            progress = dict(
                shots=index + 1,
                total=360,
                seconds=time.monotonic() - start,
                gradient_events=aggregate["real"]["events"],
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
    m, c = metrics["real"]["full360"], metrics["cold"]["full360"]
    operational = m["fpr"] <= 0.20 and m["tp"] >= c["tp"] and m["J"] > c["J"]
    controls = [a for a in ARMS if a not in ("cold", "real")]
    mechanism = all(m["J"] > metrics[a]["full360"]["J"] for a in controls)
    result = dict(
        status="COMPLETED",
        metrics=metrics,
        update_statistics=aggregate,
        flips={a: flips(rows["cold"], r) for a, r in rows.items()},
        source_state_unchanged=True,
        reference_cold_exact=360,
        operational_gate=operational,
        control_separation_gate=mechanism,
        scientific_goal_achieved=False,
        method_promoted=False,
        seconds=time.monotonic() - start,
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
