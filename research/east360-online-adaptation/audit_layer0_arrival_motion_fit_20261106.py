"""Re-score sealed forecasts on raw observed signals; do not read any outcomes."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from verify_confirmation_raw_replay_20261105 import Hashes, need, read
from audit_layer0_action_utility_20261106 import dump


H, C, LAG = 25, 23, 5
ARMS = ("hold", "linear", "native", "local", "joint")
METRICS = ("level", "shape", "combined")


def errors(pred, target, mask):
    if (
        pred.ndim != 4
        or target.shape != (len(pred), H, C)
        or mask.shape != target.shape
        or pred.shape[2:] != (H, C)
        or mask.dtype != bool
        or not np.isfinite(pred).all()
        or not np.isfinite(target[mask]).all()
    ):
        raise ValueError("paired H25 observations and scores required")
    residue = np.where(mask[:, None], pred.astype(np.float64) - target[:, None], 0.0)

    def smooth(x):
        z = np.abs(x)
        return np.where(z < 1, 0.5 * z**2, z - 0.5)

    pair = mask[:, LAG:] & mask[:, :-LAG]
    nl, ns = mask.sum((1, 2)), pair.sum((1, 2))
    level = smooth(residue).sum((2, 3)) / np.maximum(nl, 1)[:, None]
    shape = (
        np.where(
            pair[:, None], smooth(residue[:, :, LAG:] - residue[:, :, :-LAG]), 0.0
        ).sum((2, 3))
        / np.maximum(ns, 1)[:, None]
    )
    supported = (nl > 0) & (ns > 0)
    return np.stack((level, shape, 0.5 * (level + shape)), axis=-1), supported


def promotion(s):
    return all(
        s["joint"][key] < s[k][key]
        for key in ("shape", "combined")
        for k in ("hold", "linear", "local")
    )


def score_group(packed, inventory, role, mapping, norm, hashes):
    center, scale = norm
    shot, origins, predictions, valid_origin = (
        packed[k] for k in ("shot", "origin", "predictions", "origin_valid")
    )
    need(
        predictions.shape == (len(shot), len(ARMS), H, C)
        and valid_origin.shape == (len(shot), C)
        and valid_origin.dtype == bool
        and shot.shape == origins.shape
        and shot.dtype.kind in "iu"
        and origins.dtype.kind in "iu"
        and len(set(zip(shot, origins))) == len(shot),
        "held forecast schema",
    )
    observed, total = [], 0
    for row in inventory:
        sid = row["shot"]
        dev = "EAST" if role == "cold" else "J-TEXT"
        path = io23.path_for(dev, sid)
        need(hashes(path) == row["raw_sha256"], "held raw changed")
        raw = simple_data.read_shot(
            dev, sid, **({"rate_contract": "nominal"} if role == "cold" else {})
        )
        data = prepare_actual(raw, 21, mapping) if role == "cold" else raw
        v, a = data["values"], data["available"]
        want = (
            np.arange(135, len(v), 32) - H
            if role == "cold"
            else (
                np.linspace(127, len(v) - H - 1, min(16, len(v) - H - 127), dtype=int)
                if len(v) >= 128 + H
                else np.empty(0, dtype=int)
            )
        )
        k = shot == sid
        o = origins[k]
        need(
            len(v) == row["n_points"]
            and len(o) == row["n_forecasts"]
            and np.array_equal(o, want),
            "held sampling schedule differs",
        )
        if not len(o):
            continue
        total += len(o)
        available = a[o].copy()
        past = o[:, None] + np.arange(-24, 0)
        past_mask = a[np.maximum(past, 0)] & (past >= 0)[:, :, None]
        if role == "val":
            available[:, 2] = False
            past_mask[:, :, 2] = False
            available[1::2, 1] = False
            past_mask[1::2, :, 1] = False
        need(
            np.array_equal(available, valid_origin[k]),
            "held origin availability changed",
        )
        level = np.arcsinh((np.where(available, v[o], center) - center) / scale)
        past_level = np.arcsinh(
            (np.where(past_mask, v[np.maximum(past, 0)], center) - center) / scale
        )
        delta = np.where(
            past_mask & available[:, None], past_level - level[:, None], 0.0
        )
        clock = np.arange(-24, 0)[None, :, None]
        denom = np.where(past_mask & available[:, None], clock**2, 0).sum(1)
        slope = (delta * clock).sum(1) / np.maximum(denom, 1)
        linear = level[:, None] + np.arange(1, H + 1)[None, :, None] * slope[:, None]
        need(
            np.allclose(predictions[k, 0], level[:, None], atol=3e-6, rtol=2e-6)
            and np.allclose(predictions[k, 1], linear, atol=3e-6, rtol=2e-6),
            "baseline not causal hold/anchored linear",
        )
        j = o[:, None] + np.arange(1, H + 1)
        m = a[j] & available[:, None]
        m[..., [2, 10]] = False
        y = np.arcsinh((np.where(m, v[j], center) - center) / scale)
        values, supported = errors(predictions[k], y, m)
        if supported.any():
            observed.append(
                dict(
                    shot=sid,
                    n_supported=int(supported.sum()),
                    mean=values[supported].mean(0),
                )
            )
    need(total == len(shot) and observed, "held ID assignment or support mismatch")
    x = np.stack([v["mean"] for v in observed])
    avg = {
        k: dict(zip(METRICS, map(float, val), strict=True))
        for k, val in zip(ARMS, x.mean(0), strict=True)
    }
    pairs = {
        k: {m: int((x[:, -1, j] < x[:, i, j]).sum()) for j, m in enumerate(METRICS)}
        for i, k in enumerate(ARMS[:-1])
    }
    return dict(
        n_shots=len(inventory),
        supported_shots=len(observed),
        n_forecasts=total,
        supported_forecasts=sum(r["n_supported"] for r in observed),
        mean=avg,
        joint_lower_loss_shots=pairs,
        pass_gate=promotion(avg),
        per_shot=[dict(r, mean=r["mean"].tolist()) for r in observed],
    )


def verify(run, out):
    start = time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    h = Hashes()
    done, cfg, inv = [
        read(run / n) for n in ("completed.json", "protocol.json", "inventory.json")
    ]
    need(
        done["status"] == "COMPLETE_ARRIVAL_MOTION_SOURCE_AND_HELD_FORECASTS"
        and all(h(run / p) == sha for p, sha in done["artifact_sha256"].items())
        and all(h(p) == sha for p, sha in cfg["inputs"].items())
        and tuple(cfg["arms"]) == ARMS
        and (cfg["forecast_horizon"], cfg["shape_lag"], cfg["window"]) == (H, LAG, 128)
        and cfg["outcomes_read"]
        == cfg["source_test_signals_read"]
        == cfg["source_risk_updates"]
        == 0,
        "incomplete or changed forecast seal",
    )
    roles = simple_data.source_roles()
    mapping = read(cfg["mapping"])
    need(
        cfg["training"] == roles["train"]
        and cfg["validation"] == roles["val"]
        and cfg["cold_signal_only"] == mapping["calibration_ids"]
        and len(cfg["cold_signal_only"]) == 20
        and [x["shot"] for x in inv["train"]] == roles["train"]
        and [x["shot"] for x in inv["val"]] == roles["val"]
        and [x["shot"] for x in inv["cold"]] == cfg["cold_signal_only"],
        "invalid fitted/held roles",
    )
    need(
        all(
            h(io23.path_for("J-TEXT", x["shot"])) == x["raw_sha256"]
            for x in inv["train"]
        ),
        "TRAIN HDF5 bytes changed",
    )
    base, _ = load_parent(DEFAULT_PARENT)
    with np.load(run / "normalization.npz", allow_pickle=False) as a:
        norm = (a["center"], a["scale"])
        need(
            all(
                np.array_equal(v, u.numpy())
                for v, u in zip(norm, (base.center, base.scale), strict=True)
            ),
            "wrong source normalization",
        )
    groups = {}
    with patch.object(
        io23, "read_outcome", side_effect=RuntimeError("forecast audit has no labels")
    ):
        for key in ("val", "cold"):
            with np.load(run / f"{key}_predictions.npz", allow_pickle=False) as p:
                groups[key] = score_group(p, inv[key], key, mapping, norm, h)
            print(
                key,
                {k: round(v["shape"], 8) for k, v in groups[key]["mean"].items()},
                "gate",
                groups[key]["pass_gate"],
                flush=True,
            )
    h.check()
    dump(
        out,
        dict(
            status="PASS_ARRIVAL_MOTION_FIDELITY_RECOUNT",
            enabled=all(v["pass_gate"] for v in groups.values()),
            fit_completed_sha256=h(run / "completed.json"),
            fit_head_sha256=h(run / "head.npz"),
            groups=groups,
            seconds=time.monotonic() - start,
            scope="Observed signals only; not EAST alarm evaluation.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    verify(a.run, a.output)
