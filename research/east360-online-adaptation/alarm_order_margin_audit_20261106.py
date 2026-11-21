"""Post-seal compare dense native margins with the four measured order perturbations."""

import argparse
from pathlib import Path
import time

import numpy as np

import alarm_order_conditional_20261106 as conditional
import moving_peak_replay_20261105 as r
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes


def measure(t, p, down, threshold, packed):
    lead = np.round((down - t) * 1000, 9)
    dense = np.flatnonzero((lead > 30) & (lead <= 3000))
    t_ = packed["times"]
    grid_lead = np.round((down - t_) * 1000, 9)
    use = (grid_lead > 30) & (grid_lead <= 3000) & ~packed["first"]
    grid = packed["index"][use]
    if not len(dense) or not np.array_equal(t[grid], t_[use]):
        raise ValueError("reasonable dense/grid clock mismatch")
    idx = int(dense[np.argmax(p[dense])])
    d = float(p[idx])
    if not 0 < d < threshold:
        raise ValueError(
            "originally untimely shot must have finite sub-threshold dense peak"
        )
    g = float(np.max(p[grid])) if len(grid) else None
    q = packed["logits"][use].astype(np.float64)
    support = packed["supported"][use]
    diff = (q[:, 1:] - q[:, :1])[support]
    up = max(0.0, float(diff.max())) if len(diff) else None
    gap = float(np.log(threshold / (1 - threshold)) - np.log(d / (1 - d)))
    return dict(
        dense_peak=d,
        dense_best_index=idx,
        dense_gap_logit=gap,
        grid_peak=g,
        dense_minus_grid=d - g if g is not None else None,
        dense_near=d >= threshold - 0.15,
        grid_near=g is not None and g >= threshold - 0.15,
        largest_sampled_uplift=up,
        measured_uplift_below_dense_gap=up is not None and up < gap,
    )


def aggregate(rows):
    out = dict(n=len(rows))
    for name in ("dense_near", "grid_near", "measured_uplift_below_dense_gap"):
        out[name] = sum(q[name] for q in rows)
    for name in (
        "dense_peak",
        "dense_gap_logit",
        "dense_minus_grid",
        "largest_sampled_uplift",
    ):
        values = [q[name] for q in rows if q[name] is not None]
        out[name] = dict(
            n=len(values), median=float(np.median(values)), max=max(values)
        )
    return out


def case(root, score_path, *, extra, h):
    done, prot = [r.read(root / k) for k in ("completed.json", "protocol.json")]
    receipt = r.read(score_path)
    if (
        receipt["seal_sha256"] != h(root / "completed.json")
        or done["protocol_sha256"] != h(root / "protocol.json")
        or not all(h(p) == v for p, v in prot["input_sha256"].items())
        or prot["labels_read"]
    ):
        raise ValueError("paired order study and outcome receipt required")
    if extra:
        order, cfg, prior, _, _ = conditional.references(h)
    else:
        order, cfg, prior = reference()
    if order != prot["order"] or len(order) != len(receipt["per_shot"]):
        raise ValueError("frozen cohort/order mismatch")
    rows = []
    for rank, (sid, src, y) in enumerate(
        zip(order, done["shots"], receipt["per_shot"], strict=True), 1
    ):
        if sid != y["shot"] or sid != src["shot"]:
            raise ValueError("shot identity differs")
        if y["status"] not in ("miss", "tardy"):
            continue
        t, p = (
            conditional.baseline(h, rank, sid, prior, cfg["raw_sha256"][str(sid)])
            if extra
            else baseline_for(rank, sid, prior)
        )
        archive = root / "shots" / src["path" if extra else "name"]
        if h(archive) != src["sha256"]:
            raise ValueError("changed sealed reordered views")
        with np.load(archive, allow_pickle=False) as a:
            info = measure(t, p, y["down"], cfg["threshold"], a)
        rows.append(dict(shot=sid, status=y["status"], **info))
    return dict(
        rows=rows,
        all=aggregate(rows),
        miss=aggregate([x for x in rows if x["status"] == "miss"]),
    )


def run(fixed, fixed_score, extra, extra_score, out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    a = case(fixed, fixed_score, extra=False, h=h)
    b = case(extra, extra_score, extra=True, h=h)
    h.check()
    r.write_new(
        out,
        dict(
            status="RECOUNT_DENSE_MARGIN_AND_MEASURED_UPLIFT",
            code_sha256=h(Path(__file__)),
            fixed_score_sha256=h(fixed_score),
            conditional_score_sha256=h(extra_score),
            fixed=a,
            conditional=b,
            seconds=time.monotonic() - start,
            scope="Posthoc diagnostic at original misses; not a bound for unmeasured times or neural updates.",
        ),
    )
    print(a["all"], b["all"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixed", "fixed_score", "extra", "extra_score", "out"):
        parser.add_argument(field, type=Path)
    x = parser.parse_args()
    run(x.fixed, x.fixed_score, x.extra, x.extra_score, x.out)
