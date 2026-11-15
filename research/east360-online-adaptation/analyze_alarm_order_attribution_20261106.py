"""Independently recount held views and join the original outcomes after seal."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import raw_for
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes, score


ARMS = ("recent_reverse", "old_reverse", "recent_inner", "old_inner")
REVERSE_PAIRS = (((11, 14), (12, 13)), ((7, 10), (8, 9)), ((12, 13),), ((8, 9),))
W, P, C, STRIDE, ADVANCE = 128, 8, 23, 32, 24
NEAR = 0.15
DIAG = ("all", "strict", "reasonable", "inside125", "reasonable_near")


def independent_inputs(v, a, i, center, scale):
    raw = i[:, None] - 127 + np.arange(128)
    safe = raw.clip(min=0)
    m = (a[safe] & (raw[:, :, None] >= 0)).reshape(len(i), 16, 8, C)
    x = v[safe].copy().reshape(m.shape)
    x = np.where(raw.reshape(len(i), 16, 8, 1) >= 0, x, 0.0)
    z = np.arcsinh((np.where(m, x, center) - center) / scale).astype(np.float64)
    supported = np.zeros((len(i), len(ARMS)), dtype=bool)
    motion = np.full(supported.shape, np.nan)
    for j, pairs in enumerate(REVERSE_PAIRS):
        use = (i >= 127).copy()
        count = np.zeros(len(i), dtype=np.int64)
        sq = np.zeros(len(i))
        for left, right in pairs:
            use &= (m[:, left] == m[:, right]).all((1, 2))
            sq += (np.square(z[:, left] - z[:, right]) * m[:, left]).sum((1, 2))
            count += m[:, left].sum((1, 2))
        supported[:, j] = use
        motion[use, j] = np.sqrt(sq[use] / np.maximum(count[use], 1))
    return dict(
        values=x, mask=m, coverage=m.mean((1, 2, 3)), supported=supported, motion=motion
    )


def point_report(a, disrupted, down, tau):
    i = ~a["first"]
    if disrupted:
        dt = np.round((down - a["times"]) * 1000, decimals=9)
        strict = i & (dt > 30)
        reasonable = strict & (dt <= 3000)
        t = (i, strict, reasonable, strict & (dt <= 125), reasonable & a["near"])
    else:
        t = (i,)
    out = {}
    for name, use in zip(DIAG[: len(t)], t, strict=True):
        q = a["supported"] & use[:, None]
        up = q & (a["logits"][:, 1:] >= tau)
        out[name] = dict(
            n=int(use.sum()),
            available=dict(zip(ARMS, q.any(0).tolist(), strict=True)),
            up=dict(zip(ARMS, up.any(0).tolist(), strict=True)),
            up_points=dict(zip(ARMS, up.sum(0).tolist(), strict=True)),
        )
    return out


def first_report(a, diag, rank, shot, tau):
    i = np.flatnonzero(a["first"])
    if not len(i):
        return None
    j = int(i[0])
    tag = diag["status"]
    if tag == "tp":
        tag = "reasonable_tp" if diag["lead_ms"] <= 3000 else "early_tp"
    z = a["logits"][j].astype(np.float64)
    response = [float(z[0] - x) if np.isfinite(x) else None for x in z[1:]]
    m = [float(v) if np.isfinite(v) else None for v in a["motion"][j]]
    f = a["future"][j].astype(np.float64)
    return dict(
        shot=shot,
        rank=rank,
        index=int(a["index"][j]),
        class_tag=tag,
        logit=float(z[0]),
        coverage=float(a["coverage"][j]),
        lead_ms=diag["lead_ms"],
        motion_recent=m[0],
        motion_old=m[1],
        delta_recent=response[0],
        delta_old=response[1],
        delta_recent_inner=response[2],
        delta_old_inner=response[3],
        future_residual=float(f[2] - f[1]) if np.isfinite(f).all() else None,
        down=dict(
            zip(
                ARMS,
                [bool(x < tau) if np.isfinite(x) else None for x in z[1:]],
                strict=True,
            )
        ),
    )


def reference_anchors(p, threshold):
    crossing = np.nonzero(p >= threshold)[0]
    first = int(crossing[0]) if len(crossing) else None
    grid = np.arange(127, len(p), 32, dtype=np.int64)
    kept = grid[(p[grid] < threshold) & ((grid < first) if first is not None else True)]
    ix = kept if first is None else np.r_[kept, first]
    f = (ix == first) if first is not None else np.zeros(len(ix), dtype=bool)
    return ix, f, ~f & (p[ix] >= threshold - NEAR)


def check_archive(a, v, m, t, p, threshold, center, scale):
    ix, f, near = reference_anchors(p, threshold)
    n = len(ix)
    if not (
        np.array_equal(a["index"], ix)
        and np.array_equal(a["times"], t[ix])
        and np.array_equal(a["first"], f)
        and np.array_equal(a["near"], near)
        and a["logits"].shape == (n, 5)
        and a["logits"].dtype == np.float32
        and a["future"].shape == (n, 3)
        and a["future"].dtype == np.float32
        and a["placebo"].shape == (n,)
    ):
        raise ValueError("invalid candidate seal or fixed clocks")
    q = independent_inputs(v, m, ix, center, scale)
    if not (
        np.array_equal(q["supported"], a["supported"])
        and np.array_equal(q["coverage"], a["coverage"])
        and np.allclose(
            q["motion"], a["motion"], rtol=5e-13, atol=5e-13, equal_nan=True
        )
        and np.array_equal(np.isfinite(a["logits"][:, 1:]), q["supported"])
        and np.isfinite(a["logits"][:, 0]).all()
    ):
        raise ValueError(
            "view support or perturbation magnitude differs from original HDF5"
        )
    use = q["supported"][:, 0] & (ix + ADVANCE < len(t))
    if not (
        np.array_equal(np.where(use, ix + ADVANCE, -1), a["future_index"])
        and np.array_equal(np.isfinite(a["future"]).all(1), use)
    ):
        raise ValueError("sealed transported clocks differ")
    x, y = a["logits"][use], a["future"][use]
    if not np.array_equal(y[:, 2], y[:, 0] + (x[:, 1] - x[:, 0])):
        raise ValueError("scalar control does not match the anchor logit shift")

    def sigmoid(x):
        return torch.from_numpy(x).sigmoid().numpy()

    if not (
        np.allclose(sigmoid(a["logits"][:, 0]), p[ix], atol=2e-6, rtol=0.0)
        and np.allclose(sigmoid(y[:, 0]), p[ix[use] + ADVANCE], atol=2e-6, rtol=0.0)
    ):
        raise ValueError("native score parity failed in recount")
    seen = q["supported"][:, 0] & (f | (np.arange(n) % 16 == 0))
    if not np.array_equal(np.isfinite(a["placebo"]), seen):
        raise ValueError("joint token-position negative control schedule differs")
    return q


def cpu_witness(base, packed, v, m, q, center, scale):
    # Rebuild raw variants directly, without importing the producer's view helper.
    if not len(packed["index"]):
        return None
    rows = np.unique(np.r_[len(packed["index"]) // 2, np.flatnonzero(packed["first"])])
    out, want = [], []
    for j in rows:
        for k, pairs in enumerate(((), *REVERSE_PAIRS)):
            if k and not packed["supported"][j, k - 1]:
                continue
            z = q["values"][j].copy()
            for i, h in pairs:
                z[[i, h]] = z[[h, i]]
            out.append((z.reshape(W, C), q["mask"][j].reshape(W, C)))
            want.append(packed["logits"][j, k])
        if packed["future_index"][j] >= 0:
            ix = np.array([packed["future_index"][j]])
            f = independent_inputs(v, m, ix, center, scale)
            z = f["values"][0].copy()
            out.append((z.reshape(W, C), f["mask"][0].reshape(W, C)))
            for i, h in REVERSE_PAIRS[0]:
                u, w = i - 3, h - 3
                z[[u, w]] = z[[w, u]]
            # Append copies: the same z variable has just been rearranged.
            out[-1] = (f["values"][0].reshape(W, C), out[-1][1])
            out.append((z.reshape(W, C), out[-1][1]))
            want.extend(packed["future"][j, :2].tolist())
    with torch.no_grad():
        x, a = map(np.stack, zip(*out, strict=True))
        pred = base(torch.from_numpy(x), torch.from_numpy(a))["logit"].numpy()
    err = float(np.max(np.abs(pred - np.array(want))))
    if err > 2e-5:
        raise ValueError(
            "independent CPU replay did not reproduce saved raw view logits"
        )
    return dict(n=len(want), max_abs_logit=err)


def summarize_first(rows):
    keys = (
        "delta_recent",
        "delta_old",
        "delta_recent_inner",
        "delta_old_inner",
        "future_residual",
    )
    out = {}
    for tag in ("fp", "reasonable_tp", "early_tp", "tardy"):
        q = [x for x in rows if x["class_tag"] == tag]
        d = {}
        for key in keys:
            v = np.array([x[key] for x in q if x[key] is not None])
            d[key] = dict(
                n=len(v),
                mean=float(v.mean()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,
                median_abs=float(np.median(np.abs(v))) if len(v) else None,
            )
        d["down"] = {k: sum(x["down"][k] is True for x in q) for k in ARMS}
        out[tag] = dict(
            n=len(q),
            before320=sum(x["index"] < 320 for x in q),
            unsupported_before127=sum(x["index"] < 127 for x in q),
            views=d,
        )
    return out


def summarize_grid(rows):
    out = {}
    for label, types in (
        ("miss_or_tardy", ("miss", "tardy")),
        ("miss", ("miss",)),
        ("tn", ("tn",)),
    ):
        q = [x for x in rows if x["status"] in types]
        out[label] = dict(n=len(q), areas={})
        for name in DIAG if label != "tn" else ("all",):
            v = [a["grid"][name] for a in q]
            out[label]["areas"][name] = dict(
                with_candidate=sum(x["n"] > 0 for x in v),
                available={k: sum(x["available"][k] for x in v) for k in ARMS},
                up={k: sum(x["up"][k] for x in v) for k in ARMS},
                up_points={k: sum(x["up_points"][k] for x in v) for k in ARMS},
            )
    return out


def analyze(root, output):
    from alarm_order_match_20261106 import match_summary

    if output.exists():
        raise FileExistsError(output)
    start, h = time.monotonic(), Hashes()
    done, p = (r.read(root / name) for name in ("completed.json", "protocol.json"))
    order, cfg, baseline = reference()
    if (
        done["status"] != "COMPLETE_ALARM_ORDER_BLIND_SEAL"
        or h(root / "protocol.json") != done["protocol_sha256"]
        or not all(h(f) == v for f, v in p["input_sha256"].items())
        or [x["shot"] for x in done["shots"]] != order
        or p["order"] != order
        or p["labels_read"]
        or p["new_alarm_curves"]
        or p["learned_parameters"]
        or tuple(p["arms"]) != ("native", *ARMS)
        or (p["window"], p["stride"], p["advance"], p["near"])
        != (W, STRIDE, ADVANCE, NEAR)
        or p["threshold"] != cfg["threshold"]
    ):
        raise ValueError("invalid sealed experiment or not the fixed reference")
    mpath = r.COLD / "input_map.json"
    mapping = r.read(mpath)
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    if r.state_digest(base) != p["source_state_sha256"]:
        raise ValueError("qualified source state differs from blind phase")
    center, scale = base.center.numpy(), base.scale.numpy()
    tau = float(np.log(p["threshold"] / (1 - p["threshold"])))
    records, first, cpu = [], [], []
    total = dict(
        anchors=0,
        supported=dict.fromkeys(ARMS, 0),
        future=0,
        placebo=0,
        placebo_max=0.0,
    )
    for rank, item in enumerate(done["shots"], start=1):
        sid = item["shot"]
        path = root / "shots" / item["name"]
        if (
            item["rank"] != rank
            or h(path) != item["sha256"]
            or h(r.io23.path_for("EAST", sid)) != item["raw_sha256"]
        ):
            raise ValueError("raw or prediction archive changed")
        t, prior = baseline_for(rank, sid, baseline)
        raw = raw_for(sid, cfg["raw_sha256"])
        if not np.array_equal(t, raw["times"]):
            raise ValueError("raw time grid changed")
        data = r.prepare_actual(raw, rank, mapping)
        v, m = data["values"], data["available"]
        with np.load(path, allow_pickle=False) as a:
            if len(a["index"]) != item["anchors"]:
                raise ValueError("saved anchor inventory differs")
            q = check_archive(a, v, m, t, prior, p["threshold"], center, scale)
            if rank in (21, 69, 165, 225, 281, 320):
                cpu.append(
                    dict(shot=sid, check=cpu_witness(base, a, v, m, q, center, scale))
                )
            # This is the only stage that accesses outcomes, after verifying all blind input bindings.
            flag, down = r.io23.read_outcome("EAST", sid)
            y = score(
                t, prior, dict(shot=sid, disruptive=flag, down=down), p["threshold"]
            )
            if y["first_index"] != (
                int(a["index"][a["first"]][0]) if a["first"].any() else None
            ):
                raise ValueError(
                    "HDF5 first crossing does not reproduce blind reference anchor"
                )
            anchor = first_report(a, y, rank, sid, tau)
            if anchor is not None:
                first.append(anchor)
            y["grid"] = point_report(a, flag, down, tau)
            y["rank"] = rank
            records.append(y)
            total["anchors"] += len(a["index"])
            for k, count in zip(ARMS, a["supported"].sum(0), strict=True):
                total["supported"][k] += int(count)
            total["future"] += int((a["future_index"] >= 0).sum())
            finite = a["placebo"][np.isfinite(a["placebo"])]
            total["placebo"] += len(finite)
            if len(finite):
                total["placebo_max"] = max(
                    total["placebo_max"], float(np.abs(finite).max())
                )
    if total["placebo_max"] > 2e-5:
        raise ValueError(
            "permuting already position-labeled tokens affected CLS unexpectedly"
        )
    first_groups = summarize_first(first)
    if tuple(first_groups[k]["n"] for k in ("reasonable_tp", "early_tp", "fp")) != (
        100,
        16,
        29,
    ):
        raise ValueError("fixed first alarm counts did not reproduce")
    matched = match_summary(first)
    result = dict(
        status="RECOUNT_REAL_ALARM_ORDER_ATTRIBUTION",
        scope="frozen model sensitivity only, no new complete alarm curves",
        seal_sha256=h(root / "completed.json"),
        scorer_sha256=h(Path(__file__)),
        matching_sha256=h(Path(match_summary.__code__.co_filename)),
        totals=total,
        first_groups=first_groups,
        first_anchors=first,
        first_after20=summarize_first([x for x in first if x["rank"] > 20]),
        matched=matched,
        grid=summarize_grid(records),
        cpu_witness=cpu,
        per_shot=records,
        seconds=time.monotonic() - start,
    )
    h.check()
    r.write_new(output, result)
    print(total, flush=True)
    print("paired", matched.get("gate", "see saved receipt"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    analyze(a.run, a.output)
