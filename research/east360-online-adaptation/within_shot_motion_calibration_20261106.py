"""Causal per-shot recursive calibration of direct, fixed-window motion forecasts."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from audit_layer0_action_utility_20261106 import dump
from audit_layer0_arrival_motion_fit_20261106 import errors, H, C, LAG, METRICS
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from verify_confirmation_raw_replay_20261105 import Hashes, need, read


ROOT = io23.ROOT
PLAN = ROOT / "LAYER0_WITHIN_SHOT_MOTION_CALIBRATION_PLAN_20261106.md"
FIT = ROOT / "probe_outputs/layer0_arrival_motion_fit_20261106_v1"
CHECK = ROOT / "probe_outputs/layer0_arrival_motion_fit_audit_20261106_v1.json"
DECAY, RIDGE = 0.98, 0.01
ARMS = (
    "hold",
    "source_local",
    "source_joint",
    "local_recursive",
    "latent_recursive",
    "lag_recursive",
)


def prequential(origins, hold, basis, target, valid):
    origins, hold, basis, target, valid = (
        np.asarray(x) for x in (origins, hold, basis, target, valid)
    )
    n = len(origins)
    if (
        origins.ndim != 1
        or origins.dtype.kind not in "iu"
        or not n
        or (np.diff(origins) <= 0).any()
        or hold.shape != (n, H, C)
        or target.shape != hold.shape
        or valid.shape != hold.shape
        or valid.dtype != bool
        or basis.ndim != 4
        or basis.shape[:3] != hold.shape
        or not 0 < basis.shape[3] <= 3
        or not np.isfinite(hold).all()
        or not np.isfinite(basis).all()
        or not np.isfinite(target[valid]).all()
    ):
        raise ValueError(
            "finite timed causal basis and matching matured observations required"
        )
    k = basis.shape[3]
    cov = np.zeros((C, k, k), dtype=float)
    rhs = np.zeros((C, k), dtype=float)
    ridge = RIDGE * np.eye(k)
    values, counts = np.empty_like(hold, dtype=float), np.zeros(n, dtype=np.int64)

    def add(x, y, valid):
        w = valid / np.maximum(valid.sum(0), 1)[None]
        cov[:] += 0.5 * np.einsum("hc,hci,hcj->cij", w, x, x)
        rhs[:] += 0.5 * np.einsum("hc,hci,hc->ci", w, x, np.where(valid, y, 0.0))

    older = 0
    for i, now in enumerate(origins):
        while older < i and origins[older] + H <= now:
            x, m = basis[older], valid[older]
            y = np.where(m, target[older] - hold[older], 0.0)
            cov *= DECAY
            rhs *= DECAY
            add(x, y, m)
            add(x[LAG:] - x[:-LAG], y[LAG:] - y[:-LAG], m[LAG:] & m[:-LAG])
            older += 1
        counts[i] = older
        a = np.linalg.solve(cov + ridge[None], rhs[..., None])[..., 0]
        values[i] = hold[i] + np.einsum("hck,ck->hc", basis[i], a)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite recursive predictor")
    return values, counts


def matched_context(raw):
    raw = np.asarray(raw, dtype=float)
    if (
        raw.ndim != 3
        or raw.shape[1:] != (H, C)
        or not len(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("one finite contextual forecast per issue time required")
    diff = np.diff(raw, axis=0, prepend=raw[:1])
    lag = np.concatenate((np.zeros_like(diff[:1]), diff[:-1]), axis=0)
    n, m = np.linalg.norm(diff, axis=1), np.linalg.norm(lag, axis=1)
    common = (n > 0) & (m > 0)
    scale = np.divide(n, m, out=np.zeros_like(n), where=common)
    return np.where(common[:, None], diff, 0.0), lag * scale[:, None]


def report_group(rows):
    need(rows, "no observed held diagnostic targets")
    x = np.array([v["mean"] for v in rows])
    mean = {
        k: dict(zip(METRICS, v.tolist(), strict=True))
        for k, v in zip(ARMS, x.mean(0), strict=True)
    }
    return dict(
        n_shots=len(rows),
        n_forecasts=sum(v["n"] for v in rows),
        mean=mean,
        win_shape_vs_hold={
            k: int((x[:, i, 1] < x[:, 0, 1]).sum()) for i, k in enumerate(ARMS[1:], 1)
        },
        context_gate=all(
            mean["latent_recursive"][m] < mean[base][m]
            for m in ("shape", "combined")
            for base in ("hold", "local_recursive", "lag_recursive")
        ),
        per_shot=rows,
    )


def run(out):
    start = time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    h = Hashes()
    done, cfg, inventory, prior = (
        read(p)
        for p in (
            FIT / "completed.json",
            FIT / "protocol.json",
            FIT / "inventory.json",
            CHECK,
        )
    )
    need(
        prior["status"] == "PASS_ARRIVAL_MOTION_FIDELITY_RECOUNT"
        and prior["fit_completed_sha256"] == h(FIT / "completed.json")
        and all(h(FIT / k) == sha for k, sha in done["artifact_sha256"].items())
        and cfg["arms"] == ["hold", "linear", "native", "local", "joint"],
        "frozen signal forecast record required",
    )
    mapping = read(cfg["mapping"])
    need(
        h(cfg["mapping"]) == cfg["inputs"][cfg["mapping"]]
        and cfg["cold_signal_only"] == mapping["calibration_ids"]
        and [v["shot"] for v in inventory["cold"]] == cfg["cold_signal_only"],
        "cold raw roles or map changed",
    )
    parent, _ = load_parent(DEFAULT_PARENT)
    center, scale = parent.center.numpy(), parent.scale.numpy()
    support, prefix = {"all": [], "matured10": []}, []
    with (
        np.load(FIT / "cold_predictions.npz", allow_pickle=False) as a,
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("recursive motion only observes signals"),
        ),
    ):
        roster, o_all, p_all, obs_all = (
            a[k] for k in ("shot", "origin", "predictions", "origin_valid")
        )
        for i, row in enumerate(inventory["cold"], 1):
            sid = row["shot"]
            path = io23.path_for("EAST", sid)
            need(h(path) == row["raw_sha256"], "cold diagnostic file changed")
            signal = prepare_actual(
                simple_data.read_shot("EAST", sid, rate_contract="nominal"), 21, mapping
            )
            k = roster == sid
            origins, pred, obs = o_all[k], p_all[k].astype(float), obs_all[k]
            hold, line, local, joint = (pred[:, j] for j in (0, 1, 3, 4))
            need(
                len(origins) == row["n_forecasts"]
                and np.array_equal(
                    origins, np.arange(135, len(signal["values"]), 32) - H
                )
                and np.array_equal(obs, signal["available"][origins]),
                "cold source forecast schedule changed",
            )
            j = origins[:, None] + np.arange(1, H + 1)
            valid = signal["available"][j] & obs[:, None]
            valid[..., [2, 10]] = False
            y = np.arcsinh(
                (np.where(valid, signal["values"][j], center) - center) / scale
            )
            real, lag = matched_context(joint - local)
            base = np.stack((line - hold, local - hold), axis=-1)
            terms = (
                base,
                np.concatenate((base, real[..., None]), axis=-1),
                np.concatenate((base, lag[..., None]), axis=-1),
            )
            pairs = [prequential(origins, hold, q, y, valid) for q in terms]
            matured = np.searchsorted(origins + H, origins, side="right")
            need(
                all(np.array_equal(c, matured) for _, c in pairs),
                "forecast read an unobserved maturity",
            )
            if i in (1, 10, 20):
                c = len(origins) // 2
                z = y.copy()
                z[c:] += 1000
                for q, (p, _) in zip(terms, pairs, strict=True):
                    altered = q.copy()
                    altered[c + 1 :] *= -1000
                    f, _ = prequential(origins, hold, altered, z, valid)
                    need(
                        np.array_equal(p[: c + 1], f[: c + 1]),
                        "unmatured outcome changed a prediction",
                    )
                prefix.append(
                    dict(shot=sid, through_origin=int(origins[c]), n_checked=c + 1)
                )
            loss, ok = errors(
                np.stack((hold, local, joint, *(p for p, _ in pairs)), axis=1), y, valid
            )
            for name, mask in (("all", ok), ("matured10", ok & (matured >= 10))):
                if mask.any():
                    support[name].append(
                        dict(
                            shot=sid,
                            n=int(mask.sum()),
                            mean=loss[mask].mean(0).tolist(),
                        )
                    )
    groups = {k: report_group(v) for k, v in support.items()}
    h.check()
    dump(
        out,
        dict(
            status="PASS_CAUSAL_RECURSIVE_MOTION",
            arms=ARMS,
            groups=groups,
            elapsed=time.monotonic() - start,
            prefix_witness=prefix,
            decay=DECAY,
            ridge=RIDGE,
            plan_sha256=h(PLAN),
            code_sha256=h(Path(__file__)),
            fit_sha256=h(FIT / "completed.json"),
            context_gate=all(v["context_gate"] for v in groups.values()),
            labels_read=0,
            alarm_scoring=False,
        ),
    )
    for k, v in groups.items():
        print(
            k,
            {a: round(x["shape"], 8) for a, x in v["mean"].items()},
            "context gate",
            v["context_gate"],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
