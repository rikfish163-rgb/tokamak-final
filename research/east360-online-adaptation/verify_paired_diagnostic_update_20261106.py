"""Read original HDF5 outcomes only after the signed-update predictions are sealed."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit

import alarm_order_conditional_20261106 as extra
import moving_peak_replay_20261105 as r
from run_paired_diagnostic_update_20261106 import (
    ARMS,
    SEEDS,
    checked_fit,
    checked_protocol,
    dataset_context,
    train_counts,
)
from trajectory_replay_20261105 import baseline_for
from verify_confirmation_raw_replay_20261105 import Hashes, score


def verify_prediction(pred, times, mask, *, enabled):
    n = len(times)
    logits, mat, visible, common, active, norms, cosine, offsets, aux = (
        pred[k]
        for k in (
            "logits",
            "maturity",
            "visible",
            "common",
            "active",
            "norms",
            "cosine",
            "offsets",
            "auxiliary",
        )
    )
    maturity = np.arange(63, n, 32)
    m, s = len(maturity), len(SEEDS)
    if (
        not np.array_equal(times, pred["time"])
        or not np.array_equal(mat, maturity)
        or logits.shape != (n, s, len(ARMS))
        or logits.dtype != np.float32
        or any(v.dtype != bool for v in (visible, common, active))
        or visible.shape != (m,)
        or common.shape != (m,)
        or active.shape != (n,)
        or norms.shape != (m, s, 4)
        or cosine.shape != (m, s, 3)
        or offsets.shape != (m, s)
        or aux.shape != (m, s, 2)
        or not all(np.isfinite(q).all() for q in (logits, norms, cosine, offsets, aux))
    ):
        raise ValueError("invalid sealed shapes, clock or numeric payload")
    used = [3, 4, 5, 14, 15, 16, 17, 19, 20, 21, 22]
    k = (n // 32) * 32
    count = mask[:k, used].reshape(k // 32, 32, 11).sum(1)
    pair = (count[:-1] >= 16) & (count[1:] >= 16)
    groups = np.stack((pair[:, :3].any(1), pair[:, 3:7].any(1), pair[:, 7:].any(1)), 1)
    if (
        not np.array_equal(visible, groups.sum(1) >= 2)
        or (common & ~(visible & enabled)).any()
    ):
        raise ValueError("independently recounted block support differs")
    slots = np.arange(n) // 32 - 2
    use = slots >= 0
    expected = np.zeros(n, dtype=bool)
    expected[use] = common[slots[use]]
    if not np.array_equal(expected, active):
        raise ValueError("a write crossed the end-of-block causal boundary")
    native = logits[:, 0, 0]
    if (
        not np.array_equal(logits[:, :, 0], np.broadcast_to(native[:, None], (n, s)))
        or not np.array_equal(
            logits[~active],
            np.broadcast_to(native[~active, None, None], logits[~active].shape),
        )
        or (norms < 0).any()
        or (norms > 0.5 + 2e-6).any()
        or not np.allclose(norms, norms[..., :1], rtol=0, atol=2e-6)
        or (norms[~common] != 0).any()
        or (norms[common] <= 0).any()
        or (np.abs(cosine) > 1 + 2e-6).any()
    ):
        raise ValueError("parent identity or equal bounded actions violated")
    if not np.allclose(
        logits[active, :, 5],
        native[active, None] + offsets[slots[active]],
        atol=8e-7,
        rtol=0,
    ):
        raise ValueError("equal-anchor score control is not the committed block offset")
    return native


def auxiliaries(a):
    active, common = a["active"], a["common"]
    visible = a["visible"]
    out = dict(
        maturities=len(visible),
        visible=int(visible.sum()),
        common=int(common.sum()),
        active=int(active.sum()),
    )
    if common.any():
        out.update(
            norms=a["norms"][common, :, 0].mean(0, dtype=np.float64).tolist(),
            cosine=a["cosine"][common].mean(0, dtype=np.float64).tolist(),
            aux=a["auxiliary"][common].mean(0, dtype=np.float64).tolist(),
        )
    if active.any():
        selected = a["logits"][active].astype(np.float64)
        out.update(
            abs_risk_change=np.abs(selected[:, :, 1] - selected[:, :, 0]).mean(0).tolist(),
            abs_beyond_equal_score=np.abs(selected[:, :, 1] - selected[:, :, 5]).mean(0).tolist(),
        )
    return out


def summarize_aux(rows):
    return dict(
        **{
            k: sum(q[k] for q in rows)
            for k in ("maturities", "visible", "common", "active")
        },
        **{
            k: np.mean([q[k] for q in rows if k in q], axis=0).tolist()
            for k in (
                "norms",
                "cosine",
                "aux",
                "abs_risk_change",
                "abs_beyond_equal_score",
            )
            if any(k in q for q in rows)
        },
    )


def audit(root, role):
    import paired_diagnostic_metrics_20261106 as metrics

    out = root / f"{role}_audit.json"
    if out.exists():
        raise FileExistsError(out)
    if metrics.ARMS != ARMS or metrics.SEEDS != SEEDS:
        raise ValueError("independent scorer must use the locked paired arm/seed order")
    h, start = Hashes(), time.monotonic()
    cfg, fit = checked_protocol(root, h), checked_fit(root, h)
    context = dataset_context(h)
    if cfg["fixed"] != context["order"] or cfg["conditional"] != context["conditional"]:
        raise ValueError("original EAST roster changed")
    receipt = root / "train_coverage.json"
    inventory = r.read(receipt)
    if (
        fit["train_coverage_sha256"] != h(receipt)
        or [q["shot"] for q in inventory["inventory"]] != cfg["roles"]["train"]
        or not train_counts(inventory["inventory"])["passes"]
        or not all(
            h(r.io23.path_for("J-TEXT", q["shot"])) == q["sha256"]
            for q in inventory["inventory"]
        )
    ):
        raise ValueError("bound, independent source TRAIN evidence required")
    pp, ff = h(root / "protocol.json"), h(root / "fit_completed.json")
    stage = "source" if role == "val" else "east"
    done = r.read(root / f"{stage}_done.json")
    place = root / role
    seal_path = place / "completed.json"
    seal = r.read(seal_path)
    ids = cfg["roles"]["val"] if role == "val" else cfg[role]
    if (
        done["status"] != "COMPLETE_PAIRED_DIAGNOSTIC_STAGE"
        or done["stage"] != stage
        or (done["protocol_sha256"], done["fit_sha256"]) != (pp, ff)
        or seal["status"] != "SEALED_PAIRED_DIAGNOSTIC_UPDATE"
        or seal["role"] != role
        or (seal["protocol_sha256"], seal["fit_sha256"]) != (pp, ff)
        or seal["held_outcome_reads"] != 0
        or [q["shot"] for q in seal["rows"]] != ids
        or not any(q["active"] > 0 and q["error"] <= 5e-5 for q in seal["prefix"])
    ):
        raise ValueError(
            "complete blind held archive and active prefix witness required"
        )
    dev = "J-TEXT" if role == "val" else "EAST"
    for rank, q in enumerate(seal["rows"], 1):
        sid = q["shot"]
        if (q["rank"], q["file"]) != (rank, f"{rank:04d}_{sid}.npz") or (
            h(place / q["file"]) != q["sha256"]
            or h(r.io23.path_for(dev, sid)) != q["raw_sha256"]
        ):
            raise ValueError("prediction or raw provenance changed before label access")

    seal_sha = h(seal_path)
    adv = r.simple_data.advanced_times()
    summary, numbers, totals = [], [], dict(tp=0, fp=0)
    cutoff = 10 if role == "val" else 30
    for q in seal["rows"]:
        sid, rank = q["shot"], q["rank"]
        if role == "val":
            raw = r.simple_data.read_shot(dev, sid)
            p, mask = None, raw["available"]
            horizon, _, _ = r.io23.effective_horizon_ms(sid, adv)
        else:
            fixed = role == "fixed"
            raw = r.simple_data.read_shot(dev, sid, rate_contract="nominal")
            t, p = (
                baseline_for(rank, sid, context["baseline"])
                if fixed
                else extra.baseline(
                    h, rank, sid, context["extra_baseline"], q["raw_sha256"]
                )
            )
            if not np.array_equal(t, raw["times"]):
                raise ValueError("frozen EAST baseline clock changed")
            mask = r.prepare_actual(raw, rank if fixed else 21, context["mapping"])[
                "available"
            ]
            horizon = 125.0
        flag, down = r.io23.read_outcome(dev, sid)
        with np.load(place / q["file"], allow_pickle=False) as a:
            z = verify_prediction(
                a, raw["times"], mask, enabled=role != "fixed" or rank > 20
            )
            if p is not None:
                if not np.allclose(expit(z), p, rtol=0, atol=2e-6):
                    raise ValueError("frozen EAST probability no longer matches")
                s = score(
                    raw["times"],
                    p,
                    dict(shot=sid, disruptive=flag, down=down),
                    cfg["threshold"],
                )
                for k in totals:
                    totals[k] += s["status"] == k
            row = metrics.shot_metrics(
                raw["times"],
                a["logits"],
                a["active"],
                flag,
                down,
                horizon,
                cutoff,
                cfg["threshold"],
            )
            row["shot"] = sid
            summary.append(row)
            numbers.append(auxiliaries(a))
    expected = {"fixed": (116, 29), "conditional": (257, 68)}
    if role in expected and tuple(totals.values()) != expected[role]:
        raise ValueError("EAST first-alarm reference counts changed")
    result = metrics.aggregate(summary, cfg["threshold"], cutoff)
    gate = metrics.source_gate(result) if role == "val" else None
    h.check()
    r.write_new(
        out,
        dict(
            status="RECOUNT_PAIRED_DIAGNOSTIC_UPDATE",
            role=role,
            seal_sha256=seal_sha,
            auditor_sha256=h(Path(__file__)),
            metrics_sha256=h(Path(metrics.__file__)),
            threshold=cfg["threshold"],
            logit_threshold=float(logit(cfg["threshold"])),
            aggregate=result,
            gate=gate,
            auxiliary=summarize_aux(numbers),
            original_east_reference=totals if role != "val" else None,
            prefix=seal["prefix"],
            seconds=time.monotonic() - start,
        ),
    )
    print(role, "gate", gate, "auxiliary", summarize_aux(numbers), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("val", "fixed", "conditional"))
    p.add_argument("root", type=Path)
    a = p.parse_args()
    audit(a.root, a.role)
