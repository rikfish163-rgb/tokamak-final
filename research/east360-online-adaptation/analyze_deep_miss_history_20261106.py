"""Outcome-after-seal verification and shot-paired risk information diagnostics."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

import alarm_order_conditional_20261106 as conditional
import moving_peak_replay_20261105 as r
from run_deep_miss_history_20261106 import ARMS, MASS, SEEDS
from trajectory_replay_20261105 import baseline_for, reference
from verify_confirmation_raw_replay_20261105 import Hashes, score


COLS = [i for i in range(23) if i not in (1, 2, 10)]
RISK_KEYS = (
    "positive",
    "safe_high",
    "deep_positive",
    "positive_partial",
    "safe_high_partial",
)


def independent_support(mask, anchors):
    if (
        mask.ndim != 2
        or mask.shape[1] != 23
        or mask.dtype != bool
        or anchors.ndim != 1
        or anchors.dtype.kind not in "iu"
        or (anchors < 127).any()
        or (anchors >= len(mask)).any()
    ):
        raise ValueError("exact causal source-view endpoints required")
    # Direct cumulative counts, independently of the feature builder.
    older_end = anchors[:, None] - 128 - 256 * np.arange(4)[None, :]
    older_start = older_end - 255
    past = np.r_[0, np.cumsum(mask[:, COLS].any(1), dtype=np.int64)]
    q = (past[np.maximum(older_end + 1, 0)] - past[np.maximum(older_start, 0)]) > 0
    q &= older_start >= 0
    visible = np.r_[0, np.cumsum(mask.sum(1), dtype=np.int64)]
    coverage = (visible[anchors + 1] - visible[anchors - 127]) / (128 * 23)
    return q[:, :3], q[:, 1:], coverage


def point_roles(times, p, flag, down, window, cutoff, threshold):
    if flag:
        lead = np.round((down - times) * 1000, 9)
        return dict(
            positive=(lead > cutoff) & (lead <= window + 1e-6),
            safe_high=np.zeros(len(times), dtype=bool),
            reasonable=(lead > cutoff) & (lead <= 3000),
        )
    return dict(
        positive=np.zeros(len(times), dtype=bool),
        safe_high=p >= threshold - 0.15,
        reasonable=np.zeros(len(times), dtype=bool),
    )


def describe(rows):
    if not rows:
        return dict(shots=0, anchors=0, mean=None, native=None)
    v = np.stack([x[1] for x in rows])
    if v.shape != (len(rows), len(SEEDS), len(ARMS)) or not np.isfinite(v).all():
        raise ValueError("every held comparison must preserve seeds and pairing")
    shot_means = v.mean(1)
    delta = shot_means[:, 1] - shot_means[:, 0]
    boot = (
        np.random.default_rng(131)
        .choice(delta, size=(10000, len(delta)), replace=True)
        .mean(1)
    )
    return dict(
        shots=len(rows),
        anchors=sum(x[0] for x in rows),
        native=float(np.mean([x[2] for x in rows])),
        mean=dict(zip(ARMS, shot_means.mean(0).tolist(), strict=True)),
        seeds={
            str(s): dict(zip(ARMS, value.tolist(), strict=True))
            for s, value in zip(SEEDS, v.mean(0), strict=True)
        },
        real_minus_short_descriptive95=np.quantile(boot, [0.025, 0.975]).tolist(),
        real_lower_shots=int((delta < 0).sum()),
    )


def window_loss(score, native, q, positive):
    if q.any():
        z, b = score[q].astype(np.float64), native[q].astype(np.float64)
        s = -1 if positive else 1
        return (
            int(q.sum()),
            np.logaddexp(0.0, s * z).mean(0),
            float(np.logaddexp(0.0, s * b).mean()),
        )
    return None


def reach_summary(flags):
    if not flags:
        return dict(shots=0, per_seed=None)
    v = np.stack(flags)
    n = v.sum(0)
    return dict(
        shots=len(flags),
        mean=dict(zip(ARMS, n.mean(0).tolist(), strict=True)),
        per_seed={
            str(s): dict(zip(ARMS, x.tolist(), strict=True))
            for s, x in zip(SEEDS, n, strict=True)
        },
        real_only=(v[:, :, 1] & ~v[:, :, 0]).sum(0).tolist(),
        short_only=(v[:, :, 0] & ~v[:, :, 1]).sum(0).tolist(),
    )


def matched_safety(positives, negatives):
    """Match distinct shots on native logit, elapsed index and current availability."""
    if not positives or not negatives:
        return dict(n_pairs=0)
    reject = len(positives) + len(negatives) + 1
    c = np.full((len(positives), len(negatives) + len(positives)), float(reject))
    c[:, : len(negatives)] = reject * (len(positives) + 1)
    selected = {}
    for i, (_sid, pos, _diff) in enumerate(positives):
        for j, (_s, v, _d) in enumerate(negatives):
            rel = np.abs(v - pos) / np.array([0.30, np.log(2.0), 0.05])
            keep = np.flatnonzero((rel <= 1.0).all(1))
            if len(keep):
                z = rel[keep].mean(1)
                h = int(np.argmin(z))
                c[i, j] = z[h]
                selected[(i, j)] = int(keep[h])
    rows, cols = linear_sum_assignment(c)
    pairs, delta = [], []
    for i, j in zip(rows, cols, strict=True):
        if (i, j) not in selected:
            continue
        sid, _, value = positives[i]
        safe, _, v = negatives[j]
        q = value - v[selected[(i, j)]]
        pairs.append(
            dict(
                disruptive=sid,
                safe=safe,
                distance=float(c[i, j]),
                correction_delta=q.tolist(),
            )
        )
        delta.append(q)
    return dict(
        n_pairs=len(pairs),
        eligible_positive_shots=len(positives),
        pairs=pairs,
        mean_delta=dict(zip(ARMS[1:], np.stack(delta).mean(0).tolist(), strict=True))
        if pairs
        else None,
        real_higher=sum(bool(q[0] > 0) for q in delta),
    )


def gates(groups):
    out = {}
    for name, x in groups.items():
        p = x["risk"]["positive" if name == "val" else "deep_positive"]
        s = x["risk"]["safe_high"]
        checks = dict(
            enough_positive=p["shots"] >= 15,
            real_lower_than_all=p["mean"] is not None
            and all(
                p["mean"]["real"] < p["mean"][k]
                for k in ("short", "stale", "scrambled")
            ),
            safe_within_tolerance=s["mean"] is not None
            and s["mean"]["real"] <= s["mean"]["short"] + 0.02,
        )
        if name != "val":
            e, f = x["reach_deep_common"], x["new_safe_common"]
            checks["more_reasonable_grid_crossings"] = (
                e["shots"] > 0 and e["mean"]["real"] > e["mean"]["short"]
            )
            checks["no_more_safe_grid_crossings"] = (
                f["shots"] > 0 and f["mean"]["real"] <= f["mean"]["short"]
            )
        out[name] = dict(checks=checks, passed=all(checks.values()))
    return dict(
        groups=out, passed=all(x["passed"] for x in out.values()), promotes_method=False
    )


def read_bound(root, h, roles):
    done, p, fit = [
        r.read(root / name)
        for name in ("completed.json", "protocol.json", "fit_completed.json")
    ]
    if (
        done["status"] != "COMPLETE_SOURCE_ONLY_HISTORY_RISK_BLIND_SEAL"
        or done["protocol_sha256"] != h(root / "protocol.json")
        or done["fit_sha256"] != h(root / "fit_completed.json")
        or not all(h(k) == v for k, v in p["input_sha256"].items())
        or p["held_outcome_reads"]
        or p["source_test_reads"]
        or p["learned_base_parameters"]
        or tuple(p["seeds"]) != SEEDS
        or tuple(p["arms"]) != ARMS
        or tuple(p["training_mass"]) != MASS
        or p["train"] != roles["train"]
        or p["val"] != roles["val"]
        or [q["shot"] for q in fit["train"]] != roles["train"]
        or fit["labels"] != "J-TEXT TRAIN only"
        or not all(
            h(root / "fit" / f"{s}.npz") == fit["fits"][str(s)]["sha256"] for s in SEEDS
        )
    ):
        raise ValueError("verified source-only held seal required")
    for q in fit["train"]:
        if h(r.io23.path_for("J-TEXT", q["shot"])) != q["sha256"]:
            raise ValueError("source training raw bytes changed")
    for name, rows in done["rows"].items():
        if (
            name not in ("val", "fixed", "conditional")
            or [q["shot"] for q in rows] != p[name]
        ):
            raise ValueError("held shot roster changed")
        for rank, q in enumerate(rows, 1):
            if q["rank"] != rank or h(root / name / q["file"]) != q["sha256"]:
                raise ValueError("held prediction bytes changed")
            dev = "J-TEXT" if name == "val" else "EAST"
            if h(r.io23.path_for(dev, q["shot"])) != q["raw_sha256"]:
                raise ValueError("held raw bytes changed")
    return p, done, fit


def check_shot(a, q, t, mask, threshold, p, stride):
    ix = np.arange(127, len(t), stride, dtype=np.int64)
    first = None
    if p is not None:
        hits = np.flatnonzero(p >= threshold)
        if len(hits):
            first = int(hits[0])
            if first >= 127:
                ix = np.unique(np.r_[ix, first])
    if (
        q["samples"] != len(t)
        or q["anchors"] != len(ix)
        or q["first"] != first
        or not np.array_equal(ix, a["index"])
        or not np.array_equal(t[ix], a["times"])
        or a["risk"].shape != (len(ix), len(SEEDS), len(ARMS))
        or a["native"].shape != (len(ix),)
        or not np.isfinite(a["risk"]).all()
        or not np.isfinite(a["native"]).all()
        or (np.abs(a["risk"] - a["native"][:, None, None]) > 4.00001).any()
    ):
        raise ValueError("invalid held clock, native shape or risk residual bound")
    sr, ss, coverage = independent_support(mask, ix)
    if (
        not np.array_equal(sr, a["support_real"])
        or not np.array_equal(ss, a["support_stale"])
        or not np.array_equal(coverage, a["coverage"])
        or (
            p is not None
            and not np.allclose(expit(a["native"]), p[ix], atol=2e-6, rtol=0.0)
        )
    ):
        raise ValueError(
            "original masks, elapsed clocks or native probabilities differ"
        )
    return sr.all(1) & ss.all(1)


def one_group(name, rows, h, root, fixed_ctx, adv, threshold):
    from deep_miss_history_features_20261106 import FEATURE_NAMES, FEAT_DIM

    fixed_prior, fixed_cfg, extra_prior, extra_cfg, mapping = fixed_ctx
    risk = {k: [] for k in RISK_KEYS}
    reaches, safes, pairs_pos, pairs_neg, headroom = [], [], [], [], []
    coverage = dict(
        anchors=0,
        complete=0,
        any_real=0,
        shots_with_complete=0,
        first=0,
        first_complete=0,
    )
    clip = np.zeros(4 * FEAT_DIM + 72, dtype=np.int64)
    native = dict(tp=0, fp=0, miss=0, tardy=0, tn=0)
    tau = float(np.log(threshold / (1.0 - threshold)))
    for q in rows:
        sid, rank = q["shot"], q["rank"]
        p = None
        if name == "val":
            raw = r.simple_data.read_shot("J-TEXT", sid)
            mask = raw["available"]
            flag, down = r.io23.read_outcome("J-TEXT", sid)
            window, _, _ = r.io23.effective_horizon_ms(sid, adv)
            cutoff = 10.0
            status = None
        else:
            cfg = fixed_cfg if name == "fixed" else extra_cfg
            raw = r.simple_data.read_shot("EAST", sid, rate_contract="nominal")
            mask = r.prepare_actual(raw, rank if name == "fixed" else 21, mapping)[
                "available"
            ]
            t, p = (
                baseline_for(rank, sid, fixed_prior)
                if name == "fixed"
                else conditional.baseline(
                    h, rank, sid, extra_prior, cfg["raw_sha256"][str(sid)]
                )
            )
            if not np.array_equal(t, raw["times"]):
                raise ValueError("fixed baseline clock changed")
            flag, down = r.io23.read_outcome("EAST", sid)
            status = score(t, p, dict(shot=sid, disruptive=flag, down=down), threshold)[
                "status"
            ]
            native[status] += 1
            window, cutoff = 125.0, 30.0
        with np.load(root / name / q["file"], allow_pickle=False) as a:
            common = check_shot(
                a, q, raw["times"], mask, threshold, p, 8 if name == "val" else 32
            )
            ix, native_logit, scores = a["index"], a["native"], a["risk"]
            p_sample = expit(native_logit) if p is None else p[ix]
            role = point_roles(
                a["times"], p_sample, flag, down, window, cutoff, threshold
            )
            deep = status in ("miss", "tardy")
            coverage["anchors"] += len(ix)
            coverage["complete"] += int(common.sum())
            coverage["any_real"] += int(a["support_real"].any(1).sum())
            coverage["shots_with_complete"] += bool(common.any())
            c = a["real_clipped_counts"]
            if (
                c.shape != clip.shape
                or c.dtype.kind not in "iu"
                or (c < 0).any()
                or (c > common.sum()).any()
            ):
                raise ValueError("invalid feature clipping ledger")
            clip += c
            if q["first"] is not None:
                coverage["first"] += 1
                coverage["first_complete"] += bool((common & (ix == q["first"])).any())
            for key, select, pos in (
                ("positive", role["positive"] & common, True),
                ("safe_high", role["safe_high"] & common, False),
                ("deep_positive", role["positive"] & common & deep, True),
                (
                    "positive_partial",
                    role["positive"] & ~common & a["support_real"].any(1),
                    True,
                ),
                (
                    "safe_high_partial",
                    role["safe_high"] & ~common & a["support_real"].any(1),
                    False,
                ),
            ):
                m = window_loss(scores, native_logit, select, pos)
                if m is not None:
                    risk[key].append(m)
            if name != "val":
                coords = np.column_stack(
                    (native_logit, np.log(ix + 128), a["coverage"])
                )
                v = scores.mean(1).astype(np.float64)
                diff = v[:, 1:] - v[:, :1]
                if deep:
                    k = np.flatnonzero(role["positive"] & common)
                    if len(k):
                        j = k[np.argmax(native_logit[k])]
                        pairs_pos.append((sid, coords[j], diff[j]))
                    timely = role["reasonable"] & common
                    if timely.any():
                        reaches.append((scores[timely] >= tau).any(0))
                        ttd = np.round((down - raw["times"]) * 1000, 9)
                        peak = float(np.max(p[(ttd > 30) & (ttd <= 3000)]))
                        gap = tau - np.log(peak / (1.0 - peak))
                        headroom.append(
                            [
                                float(gap),
                                float((v[timely, 0] - native_logit[timely]).max()),
                                float((v[timely, 1] - native_logit[timely]).max()),
                                float(diff[timely, 0].max()),
                            ]
                        )
                if not flag and common.any():
                    pairs_neg.append((sid, coords[common], diff[common]))
                    if status == "tn":
                        safes.append((scores[common] >= tau).any(0))
    split = FEAT_DIM + 72
    d = max(1, coverage["complete"])
    top = np.argsort(-clip[split:], kind="stable")[:8]
    shift = dict(
        common_anchors=coverage["complete"],
        shared_fraction=float(clip[:split].sum() / (d * split)),
        history_fraction=float(clip[split:].sum() / (d * (len(clip) - split))),
        largest_history=[
            dict(
                slot=int(i // FEAT_DIM),
                feature=FEATURE_NAMES[i % FEAT_DIM],
                fraction=float(clip[split + i] / d),
            )
            for i in top
            if clip[split + i]
        ],
    )
    return dict(
        risk={k: describe(v) for k, v in risk.items()},
        coverage=coverage,
        native=native,
        reach_deep_common=reach_summary(reaches),
        new_safe_common=reach_summary(safes),
        paired_native_clock_availability=matched_safety(pairs_pos, pairs_neg),
        real_feature_clip=shift,
        headroom=dict(
            shots=len(headroom),
            medians=dict(
                zip(
                    (
                        "dense_gap",
                        "max_short_uplift",
                        "max_real_uplift",
                        "max_real_minus_short",
                    ),
                    np.median(headroom, axis=0).tolist(),
                    strict=True,
                )
            )
            if headroom
            else None,
        ),
    )


def analyze(root, out):
    if out.exists():
        raise FileExistsError(out)
    start, h = time.monotonic(), Hashes()
    p, done, fit = read_bound(root, h, r.simple_data.source_roles())
    fixed, cfg, a = reference()
    extra, ec, b, _, mapping = conditional.references(h)
    if (
        fixed != p["fixed"]
        or extra != p["conditional"]
        or p["threshold"] != cfg["threshold"]
    ):
        raise ValueError("locked held orders or threshold mismatch")
    context = a, cfg, b, ec, mapping
    adv = r.simple_data.advanced_times()
    groups = {}
    for name in ("val", "fixed", "conditional"):
        groups[name] = one_group(
            name, done["rows"][name], h, root, context, adv, p["threshold"]
        )
        print(name, groups[name]["coverage"], groups[name]["native"], flush=True)
    if (groups["fixed"]["native"]["tp"], groups["fixed"]["native"]["fp"]) != (
        116,
        29,
    ) or (
        groups["conditional"]["native"]["tp"],
        groups["conditional"]["native"]["fp"],
    ) != (257, 68):
        raise ValueError("original static scores no longer reproduce")
    gate = gates(groups)
    ans = dict(
        status="RECOUNT_DEEP_MISS_HISTORY_RISK",
        seal_sha256=h(root / "completed.json"),
        code_sha256=h(Path(__file__)),
        groups=groups,
        gate=gate,
        train_windows=np.array([a["counts"] for a in fit["train"]]).sum(0).tolist(),
        train_common_windows=np.array([a["common"] for a in fit["train"]])
        .sum(0)
        .tolist(),
        train_shots_by_stratum=(np.array([a["counts"] for a in fit["train"]]) > 0)
        .sum(0)
        .tolist(),
        train_common_shots_by_stratum=(
            np.array([a["common"] for a in fit["train"]]) > 0
        )
        .sum(0)
        .tolist(),
        new_alarm_curves=0,
        seconds=time.monotonic() - start,
    )
    h.check()
    r.write_new(out, ans)
    print(gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    analyze(a.root, a.out)
