"""Causal shot-local shrinkage and a separately labeled Huber oracle bound."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import audit_layer0_ordered_motion_20261106 as audit


H, DECAY, MIN_MATURED = 25, 0.98, 3
ARMS = ("local", "joint", "reverse", "stale")
PLAN = Path(__file__).with_name("LAYER0_ORDERED_SHRINK_PLAN_20261106.md")


def causal(origins, prediction, truth, mask):
    """Update sufficient statistics only when a previous complete target arrived."""
    t, p, y, m = map(np.asarray, (origins, prediction, truth, mask))
    if t.ndim != 1 or t.dtype.kind not in "iu" or (t < 0).any():
        raise ValueError("integer nonnegative forecast clocks required")
    t = t.astype(np.int64, copy=False)
    if (np.diff(t) <= 0).any() or (t > np.iinfo(np.int64).max - H).any():
        raise ValueError("strictly increasing valid forecast clocks required")
    if p.ndim != 4 or p.shape[0] != len(t) or y.shape != p.shape[:1] + p.shape[2:]:
        raise ValueError("paired forecast/target windows required")
    if m.shape != y.shape or m.dtype != bool:
        raise ValueError("common observation support required")
    count = np.zeros(len(t), dtype=np.int64)
    alpha = np.zeros(p.shape[:2], dtype=np.float64)
    u = np.zeros(p.shape[1], dtype=np.float64)
    d = u.copy()
    j = n = 0
    for k, now in enumerate(t):
        while j < k and t[j] + H <= now:
            keep = m[j]
            if keep.any():
                v, q = p[j][:, keep].astype(np.float64), y[j][keep].astype(np.float64)
                if not np.isfinite(v).all() or not np.isfinite(q).all():
                    raise ValueError(
                        "observed completed forecast/target must be finite"
                    )
                u = DECAY * u + (v * q).mean(axis=1)
                d = DECAY * d + (v * v).mean(axis=1)
                n += 1
            j += 1
        count[k] = n
        if n >= MIN_MATURED:
            # An all-zero historical prediction has no estimable scale and remains zero.
            alpha[k] = np.clip(
                np.divide(u, d, out=np.zeros_like(d), where=d > 0), 0.0, 1.0
            )
    return alpha, count


def oracle_scale(start, delta, truth, mask):
    """Hindsight only: minimize shot-equal-window Huber along a bounded segment."""
    b, v, y, m = map(np.asarray, (start, delta, truth, mask))
    if b.ndim != 3 or not (b.shape == v.shape == y.shape == m.shape) or m.dtype != bool:
        raise ValueError("paired Huber oracle windows required")
    count = m.sum((1, 2))
    if not count.any() or not all(np.isfinite(x[m]).all() for x in (b, v, y)):
        raise ValueError("finite supported Huber oracle windows required")
    resid = np.where(m, b, 0.0).astype(np.float64) - np.where(m, y, 0.0)
    v = np.where(m, v, 0.0).astype(np.float64)
    weighted_v = v / np.maximum(count, 1)[:, None, None]

    def derivative(a):
        return float((np.clip(resid + a * v, -1.0, 1.0) * weighted_v).sum())

    if derivative(0) >= 0:
        return 0.0
    if derivative(1) <= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        a = (lo + hi) * 0.5
        if derivative(a) < 0:
            lo = a
        else:
            hi = a
    return (lo + hi) * 0.5


def loss_rows(p, y, mask):
    abs_error = np.abs(np.where(mask, p.astype(np.float64) - y, 0.0))
    loss = 0.5 * np.minimum(abs_error, 1.0) ** 2 + np.maximum(abs_error - 1.0, 0.0)
    return loss.sum((1, 2)) / np.maximum(mask.sum((1, 2)), 1)


def inspect_shot(origins, pred, y, mask):
    support = mask.any((1, 2))
    alpha, counts = causal(origins, pred[:, 2:], y, mask)
    warm = support & (counts >= MIN_MATURED)
    if not warm.any():
        raise ValueError(
            "cold diagnostic shot has no supported window after maturation"
        )
    avg = dict(
        zip(
            audit.ARMS,
            (
                float(loss_rows(p, y, mask)[support].mean())
                for p in pred.transpose(1, 0, 2, 3)
            ),
            strict=True,
        )
    )
    causal_avg = {
        k: float(
            loss_rows(alpha[:, j, None, None] * pred[:, 2 + j], y, mask)[support].mean()
        )
        for j, k in enumerate(ARMS)
    }
    causal_warm = {
        k: float(
            loss_rows(alpha[:, j, None, None] * pred[:, 2 + j], y, mask)[warm].mean()
        )
        for j, k in enumerate(ARMS)
    }
    causal_avg["zero"] = avg["zero"]
    causal_warm["zero"] = float(loss_rows(pred[:, 0], y, mask)[warm].mean())
    oracle = {}
    for j, k in enumerate(ARMS):
        for origin, b in (("zero", pred[:, 0]), ("local", pred[:, 2])):
            if origin == k:
                continue
            d = pred[:, 2 + j].astype(np.float64) - b
            a = oracle_scale(b, d, y, mask)
            error = float(loss_rows(b + a * d, y, mask)[support].mean())
            oracle[f"{origin}_to_{k}"] = dict(alpha=a, error=error)
    return dict(
        n=int(support.sum()),
        warm=int(warm.sum()),
        original=avg,
        causal=causal_avg,
        causal_warm=causal_warm,
        oracle=oracle,
        median_causal_scale=dict(
            zip(ARMS, np.median(alpha[warm], axis=0).tolist(), strict=True)
        ),
    )


def summarize_group(rows, key):
    arms = tuple(rows[0][key])
    groups = {k: np.array([s[key][k] for s in rows]) for k in arms}
    avg = {k: float(v.mean()) for k, v in groups.items()}
    wins = {
        k: int((groups["joint"] < v).sum()) for k, v in groups.items() if k != "joint"
    }
    return dict(
        mean=avg,
        joint_wins=wins,
        pass_gate=all(avg["joint"] < avg[k] and 2 * wins[k] > len(rows) for k in wins),
    )


def verify_prefix(origins, prediction, truth, mask, boundary):
    full, n = causal(origins, prediction, truth, mask)
    last = origins[boundary]
    # Mutate all labels not yet fully arrived at this decision boundary.
    unknown = origins + H > last
    y, a = truth.copy(), mask.copy()
    y[unknown] = 1e6
    a[unknown] = False
    short, k = causal(
        origins[: boundary + 1],
        prediction[: boundary + 1],
        y[: boundary + 1],
        a[: boundary + 1],
    )
    if not np.array_equal(full[: boundary + 1], short) or not np.array_equal(
        n[: boundary + 1], k
    ):
        raise ValueError("unarrived truth changed previous shot-local scaling")
    return dict(
        origins=int(boundary + 1), last=int(last), modified_targets=int(unknown.sum())
    )


def run(root, recount, output):
    from experiments.risk_aligned_online import io23, simple_data
    from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
        prepare_actual,
    )

    if output.exists():
        raise FileExistsError(output)
    start, h = time.monotonic(), audit.Hashes()
    done, cfg, inv, check = [
        audit.read_json(p)
        for p in (
            root / "completed.json",
            root / "protocol.json",
            root / "inventory.json",
            recount,
        )
    ]
    if (
        check["status"] != "PASS_ORDERED_MOTION_FIDELITY_RECOUNT"
        or check["enabled"]
        or h(root / "completed.json") != check["fit_completed_sha256"]
    ):
        raise ValueError(
            "an independently rechecked, failed ordered forecast gate is required"
        )
    mapping = audit.read_json(cfg["mapping"])
    audit.check_metadata(root, done, cfg, inv, simple_data.source_roles(), mapping, h)
    with np.load(root / "normalization.npz", allow_pickle=False) as x:
        norm = audit.check_norm((x["center"], x["scale"]))
    with np.load(root / "cold_predictions.npz", allow_pickle=False) as x:
        ids, origins, pred, _, _ = audit.check_packed(x)
    rows, witness = [], []
    with patch.object(
        io23, "read_outcome", side_effect=RuntimeError("no outcomes in shrink test")
    ):
        for i, row in enumerate(inv["cold"]):
            sid = row["shot"]
            if h(io23.path_for("EAST", sid)) != row["raw_sha256"]:
                raise ValueError("cold diagnostic file changed")
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            d = prepare_actual(raw, 21, mapping)
            k = ids == sid
            o = origins[k]
            y, m, _, _, _ = audit.reconstruct(
                d["values"], d["available"], o, "cold", norm
            )
            p = pred[k]
            record = inspect_shot(o, p, y, m)
            ref = check["groups"]["cold"]["per_shot"][i]
            if (
                ref["shot"] != sid
                or ref["n_supported"] != record["n"]
                or not np.allclose(
                    list(record["original"].values()),
                    ref["mean"],
                    atol=1e-12,
                    rtol=1e-12,
                )
            ):
                raise ValueError(
                    "original audited scores changed during shrink recount"
                )
            rows.append(dict(shot=sid, **record))
            if i in (0, 9, 19):
                witness.append(
                    dict(shot=sid, **verify_prefix(o, p[:, 2:], y, m, len(o) // 2))
                )
    h.check()
    decomposition = {}
    for name, group in check["groups"].items():
        a = np.array([s["mean"] for s in group["per_shot"]], dtype=np.float64)
        odd = (a[:, 3] - a[:, 4]) * 0.5
        even = (a[:, 3] + a[:, 4]) * 0.5 - a[:, 2]
        if even.min() < -2e-7:
            raise ValueError("unexpected negative midpoint convexity gap")
        decomposition[name] = dict(
            odd=float(odd.mean()),
            even=float(even.mean()),
            net=float((a[:, 3] - a[:, 2]).mean()),
        )
    oracle = {
        k: dict(
            mean=float(np.mean([r["oracle"][k]["error"] for r in rows])),
            median_scale=float(np.median([r["oracle"][k]["alpha"] for r in rows])),
        )
        for k in rows[0]["oracle"]
    }
    result = dict(
        status="COMPLETE_ORDERED_SHRINK_DIAGNOSTIC",
        scope="Mapped cold20 signal-only forecast adaptation; not alarm or neural update.",
        fit_completed_sha256=h(root / "completed.json"),
        recount_sha256=h(recount),
        plan_sha256=h(PLAN),
        code_sha256=h(Path(__file__)),
        horizon=H,
        decay=DECAY,
        min_matured=MIN_MATURED,
        outcomes_read=0,
        alarm_scoring=False,
        seconds=time.monotonic() - start,
        decomposition=decomposition,
        witnesses=witness,
        all=summarize_group(rows, "causal"),
        matured=summarize_group(rows, "causal_warm"),
        hindsight_only_oracle=oracle,
        shots=rows,
    )
    audit.write_json(output, result)
    print(result["all"], result["matured"], result["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "recount", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.fit, args.recount, args.output)
