"""CPU/NumPy recount of sealed ordered-motion forecasts, without outcomes.

This is not a replay of the fitted model.  The sealed normalization, local,
joint, reverse, and stale forecasts are inputs; the latter four are never
recomputed from the fitted network.  Only the zero and past-motion persistence
controls, support, targets, and losses are independently reconstructed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np


H, W, C, LAG = 25, 128, 23, 5
PAIR_OFFSET, START_ORIGIN, FIRST, STRIDE = 25, 177, 135, 32
ARMS = ("zero", "persistence", "local", "joint", "reverse", "stale")
PREDICTION_KEYS = ("shot", "origin", "predictions", "origin_valid", "past_valid")
ARTIFACTS = frozenset(
    (
        "protocol.json",
        "inventory.json",
        "head.npz",
        "normalization.npz",
        "val_predictions.npz",
        "cold_predictions.npz",
    )
)
ATOL, RTOL = 3e-6, 2e-6
SOURCE_REFERENCE = (
    Path(__file__).resolve().parent
    / "probe_outputs/layer_interaction_masks360_20261105_v1"
)
SOURCE_REFERENCE_COMPLETED_SHA256 = (
    "c887f88cf6a7183074d781d1655d86a15f8fdcf29d14273fcb4137b41e62cec8"
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_origins(n):
    stop = n - H - 1
    if stop < START_ORIGIN:
        return np.empty(0, dtype=np.int64)
    return np.linspace(
        START_ORIGIN, stop, min(16, stop - START_ORIGIN + 1), dtype=np.int64
    )


def cold_origins(n):
    origins = np.arange(FIRST, n, STRIDE, dtype=np.int64) - H
    return origins[(origins >= START_ORIGIN) & (origins <= n - H - 1)]


def check_norm(norm):
    try:
        center, scale = (np.asarray(x, dtype=np.float64) for x in norm)
    except (TypeError, ValueError):
        raise ValueError("finite flat23 normalization required") from None
    need(
        center.shape == scale.shape == (C,)
        and np.isfinite(center).all()
        and np.isfinite(scale).all()
        and (scale > 0).all(),
        "finite flat23 center and positive scale required",
    )
    return center, scale


def ordered_block(z, available, origin_valid):
    """Return the signed antisymmetric five-step motion and four-endpoint mask."""
    z = np.asarray(z)
    available = np.asarray(available)
    origin_valid = np.asarray(origin_valid)
    need(
        z.ndim == 3
        and z.shape[1:] == (H, C)
        and z.dtype.kind in "fiu"
        and available.shape == z.shape
        and available.dtype == bool
        and origin_valid.shape == (len(z), C)
        and origin_valid.dtype == bool,
        "ordered block shape or mask invalid",
    )
    need(np.isfinite(z[available]).all(), "observed ordered block must be finite")
    z = np.where(available, z, 0).astype(np.float64, copy=False)
    j = np.arange(10)
    support = (
        available[:, j]
        & available[:, j + LAG]
        & available[:, 19 - j]
        & available[:, 24 - j]
        & origin_valid[:, None, :]
    )
    support[:, :, [2, 10]] = False
    difference = 0.5 * ((z[:, j + LAG] - z[:, j]) + (z[:, 24 - j] - z[:, 19 - j]))
    need(np.isfinite(difference[support]).all(), "nonfinite ordered motion")
    return np.where(support, difference, 0.0), support


def reconstruct(values, available, origins, role, norm):
    """Rebuild targets and causal persistence from one shot's signal only."""
    center, scale = check_norm(norm)
    values, available, origins = map(np.asarray, (values, available, origins))
    need(
        role in ("val", "cold")
        and values.ndim == 2
        and values.shape[1] == C
        and values.dtype.kind in "fiu"
        and available.shape == values.shape
        and available.dtype == bool
        and origins.ndim == 1
        and origins.dtype.kind in "iu"
        and (origins >= START_ORIGIN).all()
        and (origins + H < len(values)).all(),
        "invalid shot signals or origins",
    )
    need(np.isfinite(values[available]).all(), "observed signals must be finite")
    past_ix = origins[:, None] + np.arange(1 - H, 1)
    future_ix = origins[:, None] + np.arange(1, H + 1)
    past_mask = available[past_ix].copy()
    if role == "val":
        past_mask[:, :, 2] = False
        past_mask[1::2, :, 1] = False
    origin_valid = past_mask[:, -1].copy()
    future_mask = available[future_ix].copy()
    past_z = np.arcsinh((np.where(past_mask, values[past_ix], center) - center) / scale)
    future_z = np.arcsinh(
        (np.where(future_mask, values[future_ix], center) - center) / scale
    )
    persistence, past_valid = ordered_block(past_z, past_mask, origin_valid)
    target, future_valid = ordered_block(future_z, future_mask, origin_valid)
    return target, past_valid & future_valid, persistence, past_valid, origin_valid


def window_losses(predictions, target, mask):
    predictions, target, mask = map(np.asarray, (predictions, target, mask))
    n = len(predictions)
    need(
        predictions.shape == (n, len(ARMS), 10, C)
        and predictions.dtype.kind == "f"
        and target.shape == (n, 10, C)
        and target.dtype.kind in "fiu"
        and mask.shape == target.shape
        and mask.dtype == bool
        and np.isfinite(predictions).all()
        and np.isfinite(target[mask]).all(),
        "finite paired ordered-motion predictions and boolean support required",
    )
    residual = np.where(
        mask[:, None],
        predictions.astype(np.float64) - target[:, None].astype(np.float64),
        0.0,
    )
    absolute = np.abs(residual)
    small = np.minimum(absolute, 1.0)
    loss = 0.5 * small * small + np.maximum(absolute - 1.0, 0.0)
    need(np.isfinite(loss).all(), "nonfinite SmoothL1 loss")
    counts = mask.sum(axis=(1, 2))
    return loss.sum(axis=(2, 3)) / np.maximum(counts, 1)[:, None], counts > 0, counts


def check_packed(packed):
    keys = set(packed.files if hasattr(packed, "files") else packed)
    need(keys == set(PREDICTION_KEYS), "unexpected sealed prediction keys")
    shot, origin, predictions, origin_valid, past_valid = (
        np.asarray(packed[k]) for k in PREDICTION_KEYS
    )
    n = len(shot)
    need(
        shot.ndim == origin.ndim == 1
        and shot.shape == origin.shape
        and shot.dtype.kind in "iu"
        and origin.dtype.kind in "iu"
        and predictions.shape == (n, len(ARMS), 10, C)
        and predictions.dtype.kind == "f"
        and origin_valid.shape == (n, C)
        and origin_valid.dtype == bool
        and past_valid.shape == (n, 10, C)
        and past_valid.dtype == bool
        and np.isfinite(predictions).all()
        and len(set(zip(shot.tolist(), origin.tolist()))) == n,
        "invalid or duplicate sealed forecast rows",
    )
    need(np.all(predictions[:, 0] == 0), "sealed zero arm is not zero")
    midpoint = 0.5 * predictions[:, 3].astype(np.float64) + 0.5 * predictions[:, 4]
    need(
        np.allclose(midpoint, predictions[:, 2], atol=ATOL, rtol=RTOL),
        "joint/reverse midpoint differs from local",
    )
    return shot, origin, predictions, origin_valid, past_valid


def summarize(observed, n_shots, n_forecasts, valid_cells):
    need(bool(observed), "no supported held shots")
    x = np.stack([r["mean"] for r in observed])
    need(
        x.shape == (len(observed), len(ARMS)) and np.isfinite(x).all(),
        "invalid per-shot losses",
    )
    means = dict(zip(ARMS, map(float, x.mean(axis=0)), strict=True))
    wins = {
        arm: int((x[:, 3] < x[:, i]).sum())
        for i, arm in enumerate(ARMS)
        if arm != "joint"
    }
    strict_mean = all(means["joint"] < means[arm] for arm in ARMS if arm != "joint")
    majority = all(
        2 * wins[arm] > len(observed)
        for arm in ("zero", "persistence", "local", "stale")
    )
    return dict(
        n_shots=n_shots,
        supported_shots=len(observed),
        n_forecasts=n_forecasts,
        supported_forecasts=sum(r["n_supported"] for r in observed),
        valid_cells=int(valid_cells),
        mean=means,
        joint_lower_loss_shots=wins,
        pass_gate=bool(strict_mean and majority),
        per_shot=[
            dict(shot=r["shot"], n_supported=r["n_supported"], mean=r["mean"].tolist())
            for r in observed
        ],
    )


def score_group(
    packed, inventory, role, mapping, norm, hashes, path_for, read_shot, prepare_actual
):
    need(role in ("val", "cold"), "only validation and cold signals may be scored")
    shot, origins, predictions, sealed_origin, sealed_past = check_packed(packed)
    observed, total, valid_cells = [], 0, 0
    for row in inventory:
        sid = row["shot"]
        device = "EAST" if role == "cold" else "J-TEXT"
        need(
            hashes(path_for(device, sid)) == row["raw_sha256"],
            "held signal bytes changed",
        )
        raw = read_shot(
            device, sid, **({"rate_contract": "nominal"} if role == "cold" else {})
        )
        need(
            raw["shot"] == sid and raw["device"] == device,
            "held signal identity changed",
        )
        data = prepare_actual(raw, 21, mapping) if role == "cold" else raw
        values, available = data["values"], data["available"]
        want = (
            cold_origins(len(values)) if role == "cold" else source_origins(len(values))
        )
        select = shot == sid
        these = origins[select]
        need(
            len(values) == row["n_points"]
            and len(these) == row["n_forecasts"]
            and np.array_equal(these, want),
            "held inventory or sampling schedule differs",
        )
        total += len(these)
        if not len(these):
            continue
        target, common, persistence, past_valid, origin_valid = reconstruct(
            values, available, these, role, norm
        )
        need(
            np.array_equal(sealed_origin[select], origin_valid),
            "origin validity differs",
        )
        need(np.array_equal(sealed_past[select], past_valid), "past support differs")
        sealed_persistence = predictions[select, 1]
        need(
            np.all(sealed_persistence[~past_valid] == 0)
            and np.allclose(sealed_persistence, persistence, atol=ATOL, rtol=RTOL),
            "sealed persistence differs from causal past motion",
        )
        losses, supported, counts = window_losses(predictions[select], target, common)
        valid_cells += int(counts.sum())
        if supported.any():
            observed.append(
                dict(
                    shot=int(sid),
                    n_supported=int(supported.sum()),
                    mean=losses[supported].mean(axis=0),
                )
            )
    need(total == len(shot), "sealed forecast shot assignment differs from inventory")
    return summarize(observed, len(inventory), total, valid_cells)


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def check_metadata(run, done, cfg, inventory, roles, mapping, hashes):
    need(
        done.get("status") == "COMPLETE_ORDERED_MOTION_SOURCE_AND_HELD_FORECASTS",
        "fit is not complete",
    )
    artifacts = done.get("artifact_sha256")
    need(
        isinstance(artifacts, dict)
        and ARTIFACTS <= set(artifacts)
        and all(
            isinstance(k, str)
            and Path(k).name == k
            and k not in (".", "..")
            and is_sha256(v)
            for k, v in artifacts.items()
        ),
        "invalid or incomplete artifact seal",
    )
    need(
        all(hashes(run / name) == digest for name, digest in artifacts.items()),
        "sealed artifact changed",
    )
    inputs = cfg.get("inputs")
    need(
        isinstance(inputs, dict)
        and bool(inputs)
        and all(isinstance(k, str) and is_sha256(v) for k, v in inputs.items())
        and isinstance(cfg.get("mapping"), str)
        and any(Path(p).resolve() == Path(cfg["mapping"]).resolve() for p in inputs)
        and all(hashes(p) == digest for p, digest in inputs.items()),
        "invalid or changed input binding",
    )
    need(
        is_sha256(cfg.get("source_state_sha256"))
        and tuple(cfg.get("arms", ())) == ARMS
        and all(
            type(cfg.get(k)) is int and cfg[k] == v
            for k, v in {
                "forecast_horizon": H,
                "window": W,
                "shape_lag": LAG,
                "pair_offset": PAIR_OFFSET,
                "start_origin": START_ORIGIN,
                "first": FIRST,
                "stride": STRIDE,
                "outcomes_read": 0,
                "source_test_signals_read": 0,
                "source_risk_updates": 0,
            }.items()
        ),
        "invalid ordered-motion protocol",
    )
    train, val, test = (roles[k] for k in ("train", "val", "test"))
    cold = cfg.get("cold_signal_only")
    need(
        isinstance(cold, list)
        and len(cold) == 20
        and len(set(cold)) == 20
        and cold == mapping.get("calibration_ids")
        and cfg.get("training") == train
        and cfg.get("validation") == val
        and not (
            set(train) & set(val) or set(train) & set(test) or set(val) & set(test)
        )
        and all(len(ids) == len(set(ids)) for ids in (train, val, test)),
        "locked source roles or cold20 mapping differ",
    )
    for key, expected in (("train", train), ("val", val), ("cold", cold)):
        rows = inventory.get(key)
        need(
            isinstance(rows, list)
            and all(isinstance(r, dict) for r in rows)
            and [r.get("shot") for r in rows] == expected
            and all(
                type(r.get("shot")) is int
                and is_sha256(r.get("raw_sha256"))
                and type(r.get("n_points")) is int
                and r["n_points"] >= 0
                and type(r.get("n_forecasts")) is int
                and r["n_forecasts"] >= 0
                for r in rows
            ),
            "invalid or reordered inventory",
        )
    need(
        all(
            r["n_forecasts"] == len(source_origins(r["n_points"]))
            for r in inventory["train"]
        ),
        "TRAIN inventory sampling schedule differs",
    )


def signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


class Hashes:
    """Hash inputs once; fail if any file changes during the recount."""

    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        path = Path(path).resolve(strict=True)
        before = signature(path)
        if path not in self.cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            need(signature(path) == before, f"file changed while hashing: {path}")
            self.cache[path] = (before, digest.hexdigest())
        need(self.cache[path][0] == before, f"file changed during recount: {path}")
        return self.cache[path][1]

    def check(self):
        for path, (before, _) in self.cache.items():
            need(signature(path) == before, f"file changed during recount: {path}")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def check_source_reference(state_sha256, mapping_path, hashes):
    """Bind the source state and cold20 map to the locked layer-0 reference."""
    completed = SOURCE_REFERENCE / "completed.json"
    protocol = SOURCE_REFERENCE / "protocol.json"
    need(
        hashes(completed) == SOURCE_REFERENCE_COMPLETED_SHA256,
        "locked source reference changed",
    )
    done = read_json(completed)
    expected = done.get("artifact_sha256", {}).get("protocol.json")
    need(
        is_sha256(expected) and hashes(protocol) == expected,
        "locked source reference protocol changed",
    )
    reference = read_json(protocol)
    need(
        reference.get("source_state_sha256") == state_sha256,
        "source state differs from locked qualified reference",
    )
    inputs = reference.get("input_sha256")
    need(isinstance(inputs, dict), "locked reference input hashes missing")
    expected_map = [
        digest
        for path, digest in inputs.items()
        if Path(path).resolve() == Path(mapping_path).resolve()
    ]
    need(
        len(expected_map) == 1
        and is_sha256(expected_map[0])
        and hashes(mapping_path) == expected_map[0],
        "cold20 map differs from locked reference",
    )


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def signal_dependencies():
    # The existing signal reader and fixed EAST input mapping are shared data
    # contracts, not the ordered-motion fitter or scoring implementation.
    from experiments.risk_aligned_online import io23, simple_data
    from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
        prepare_actual,
    )

    return io23, simple_data, prepare_actual


def verify(run, out):
    """Recount an already sealed fit; this function reads held signals, not labels."""
    io23, simple_data, prepare_actual = signal_dependencies()
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(out)
    start, hashes = time.monotonic(), Hashes()
    completed_sha = hashes(run / "completed.json")
    done = read_json(run / "completed.json")
    hashes(run / "completed.json")
    hashes(run / "protocol.json")
    hashes(run / "inventory.json")
    cfg, inventory = (
        read_json(run / name) for name in ("protocol.json", "inventory.json")
    )
    inputs = cfg.get("inputs")
    need(
        isinstance(inputs, dict)
        and all(isinstance(k, str) and is_sha256(v) for k, v in inputs.items())
        and any(Path(p).resolve() == io23.JTEXT_INDEX.resolve() for p in inputs),
        "canonical source split index must be bound by the fit",
    )
    need(
        isinstance(cfg.get("mapping"), str)
        and any(Path(p).resolve() == Path(cfg["mapping"]).resolve() for p in inputs),
        "cold20 map must be bound by the fit",
    )
    need(
        all(hashes(path) == expected for path, expected in inputs.items()),
        "fit input changed before reading",
    )
    roles = simple_data.source_roles()
    hashes(cfg["mapping"])
    mapping = read_json(cfg["mapping"])
    check_metadata(run, done, cfg, inventory, roles, mapping, hashes)
    check_source_reference(cfg["source_state_sha256"], cfg["mapping"], hashes)
    need(
        all(
            hashes(io23.path_for("J-TEXT", r["shot"])) == r["raw_sha256"]
            for r in inventory["train"]
        ),
        "TRAIN signal bytes changed",
    )
    with np.load(run / "normalization.npz", allow_pickle=False) as saved:
        need(set(saved.files) == {"center", "scale"}, "invalid normalization archive")
        norm = check_norm((saved["center"], saved["scale"]))
    groups = {}
    with patch.object(
        io23, "read_outcome", side_effect=RuntimeError("outcomes forbidden")
    ):
        for role in ("val", "cold"):
            with np.load(run / f"{role}_predictions.npz", allow_pickle=False) as packed:
                groups[role] = score_group(
                    packed,
                    inventory[role],
                    role,
                    mapping,
                    norm,
                    hashes,
                    io23.path_for,
                    simple_data.read_shot,
                    prepare_actual,
                )
            print(
                role,
                groups[role]["mean"],
                "gate",
                groups[role]["pass_gate"],
                flush=True,
            )
    hashes.check()
    write_json(
        out,
        dict(
            status="PASS_ORDERED_MOTION_FIDELITY_RECOUNT",
            enabled=all(group["pass_gate"] for group in groups.values()),
            fit_completed_sha256=completed_sha,
            fit_head_sha256=hashes(run / "head.npz"),
            source_state_sha256=cfg["source_state_sha256"],
            source_reference_completed_sha256=SOURCE_REFERENCE_COMPLETED_SHA256,
            groups=groups,
            normalization_provenance="sealed fit artifact; source-checkpoint equality not independently checked",
            scope="Signal-only forecast fidelity; not an EAST alarm or outcome evaluation.",
            seconds=time.monotonic() - start,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    verify(args.run, args.output)


if __name__ == "__main__":
    main()
