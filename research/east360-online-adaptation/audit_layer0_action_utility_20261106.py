"""Audit revealed-history credit and post-hoc action capacity, never fit on scored flows."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from layer0_action_utility_20261106 import alarm_envelope, block_credit
from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    counts,
    need,
    raw_truth,
    read,
    score,
)
from verify_layer0_signed_projection_20261106 import (
    FIRST,
    FIT,
    MAP,
    ROOT,
    SOURCE,
    STRIDE,
    TAU,
    shot_path,
)
from verify_moving_peak_replay_20261105 import check_curve, parity


PREFIX = ROOT / "probe_outputs"
SEALED_HISTORY = PREFIX / "layer0_disruptive_feedback360_20261105_v1"
SIGNED = {
    k: (
        PREFIX / f"layer0_signed_{k}_20261106_v1",
        PREFIX / f"layer0_signed_{k}_verification_20261106_v1.json",
    )
    for k in ("fixed360", "conditional835")
}
CONTEXTS = ("telescoping", "isolated", "leaveout")
MENU = dict(
    binary=("static", "intra"), extended=("static", "intra", "signed", "blind_signed")
)


def dump(path, payload):
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False, indent=2)


def first_is_timely(times, down, i, *, capped):
    if i == len(times):
        return False
    lead = round(float((down - times[i]) * 1000), 9)
    return lead > 30 and (not capped or lead <= 3000)


def best_first(times, probs, truth, *, capped, first=FIRST, stride=STRIDE):
    possible = alarm_envelope(
        times,
        probs,
        disruptive=truth["disruptive"],
        down=truth["down"],
        threshold=TAU,
        first=first,
        stride=stride,
    )
    if truth["disruptive"]:
        return next(
            (
                int(i)
                for i in possible
                if first_is_timely(times, truth["down"], i, capped=capped)
            ),
            int(possible[0]),
        )
    return len(times) if possible[-1] == len(times) else int(possible[0])


def miss_reason(times, probs, truth, chosen, *, capped):
    lead = np.round((truth["down"] - times) * 1000, 9)
    if first_is_timely(times, truth["down"], chosen, capped=capped):
        return "selectable_rescue"
    valid = (lead > 30) & ((lead <= 3000) if capped else True)
    if not np.any(probs[:, valid] >= TAU):
        return "no_eligible_candidate_crossing"
    return "eligible_crossing_cannot_be_first"


def coverage(rows, key):
    result = {}
    for sign, label in ((1, "benefit"), (-1, "harm")):
        count = [(r, int((r[key] == sign).sum())) for r in rows]
        result[label] = dict(
            anchors=sum(n for _, n in count),
            shots=[r["shot"] for r, n in count if n],
            disruptive=[r["shot"] for r, n in count if n and r["disruptive"]],
            safe=[r["shot"] for r, n in count if n and not r["disruptive"]],
        )
    return result


def outcome_value(row, capped):
    if row["disruptive"]:
        return int(row["status"] == "tp" and (not capped or row["lead_ms"] <= 3000))
    return -int(row["status"] == "fp")


def score_first(times, first, truth):
    # Single-point representative scoring preserves the existing HDF5 first-crossing rules.
    p = np.zeros(len(times), dtype=np.float32)
    if first < len(times):
        p[first] = 1.0
    return score(times, p, truth, TAU)


def brute_credit(times, native, moved, truth, fast):
    """Independently verify selected actual splices against the original scorer."""
    idx = fast["anchor_indices"]
    if not len(idx):
        return 0
    controls = {0, len(idx) // 2, len(idx) - 1}
    for mode in CONTEXTS:
        hit = np.flatnonzero(fast[f"capped_{mode}"])
        if len(hit):
            controls.update([int(hit[0]), int(hit[-1])])
    for k in sorted(controls):
        a, b = idx[k], idx[k] + STRIDE
        after = native.copy()
        after[a:b] = moved[a:b]
        restored = moved.copy()
        restored[a:b] = native[a:b]
        pairs = dict(
            isolated=(after, native),
            leaveout=(moved, restored),
            telescoping=(np.r_[moved[:b], native[b:]], np.r_[moved[:a], native[a:]]),
        )
        for mode, (x, y) in pairs.items():
            sx, sy = (score(times, p, truth, TAU) for p in (x, y))
            for label, cap in (("strict", False), ("capped", True)):
                need(
                    fast[f"{label}_{mode}"][k]
                    == outcome_value(sx, cap) - outcome_value(sy, cap),
                    f"credit fails exact curve splice at {truth['shot']}/{a}/{mode}/{label}",
                )
    return len(controls)


def rows_for_probe(rows):
    xs, signs, ids = [], [], []
    for r in rows:
        u = r["capped_isolated"]
        sel = u != 0
        if np.any(sel):
            xs.append(r["features"][sel])
            signs.append(u[sel])
            ids.append(np.full(sel.sum(), r["shot"]))
    x, y, shot = (np.concatenate(v) for v in (xs, signs, ids))
    weights = np.zeros(len(x), dtype=float)
    for sign in (-1, 1):
        s = y == sign
        roles = np.unique(shot[s])
        for sid in roles:
            m = s & (shot == sid)
            weights[m] = 0.5 / len(roles) / m.sum()
    return x.astype(float), y.astype(float), weights, shot


def run_probe(history, queried):
    train = coverage(history, "capped_isolated")
    test = coverage(queried, "capped_isolated")
    enough = dict(
        cold=all(len(v["shots"]) >= 5 for v in train.values()),
        queried=all(len(v["shots"]) >= 3 for v in test.values()),
    )
    if not all(enough.values()):
        # Preregistered stopping rule: correlated windows do not supply new independent labels.
        return dict(
            status="INSUFFICIENT_DISTINCT_SHOT_COVERAGE", gates=enough, models_fitted=0
        )
    x, y, w, _ = rows_for_probe(history)
    xx, yy, ww, ids = rows_for_probe(queried)
    result = {}
    for name, width in (
        ("risk_clock", 2),
        ("anchor_effect", 4),
        ("latent_history", 132),
    ):
        m = (x[:, :width] * w[:, None]).sum(0)
        sd = np.sqrt(((x[:, :width] - m) ** 2 * w[:, None]).sum(0)).clip(min=1e-3)
        a = np.c_[np.ones(len(x)), (x[:, :width] - m) / sd]
        b = np.c_[np.ones(len(xx)), (xx[:, :width] - m) / sd]

        def loss(p):
            z = y * (a @ p)
            penalty = np.r_[0, p[1:]]
            return (
                float(w @ np.logaddexp(0, -z) + 0.5 * np.dot(penalty, penalty)),
                a.T @ (-y * w * expit(-z)) + penalty,
            )

        fit = minimize(loss, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B")
        need(
            fit.success and np.isfinite(fit.x).all(),
            "cold-only ridge-logistic probe did not converge",
        )
        z = b @ fit.x
        result[name] = dict(
            balanced_accuracy=float(ww @ ((z >= 0) == (yy > 0))),
            balanced_log_loss=float(ww @ np.logaddexp(0, -yy * z)),
            queried_predictions=[
                dict(shot=int(i), effect=int(u), logit=float(p))
                for i, u, p in zip(ids, yy, z, strict=True)
            ],
        )
    return dict(
        status="COLD_TRAINED_QUERY_DIAGNOSTIC",
        gates=enough,
        models_fitted=3,
        models=result,
    )


def audit_history(run, output):
    started = time.monotonic()
    h = Hashes()
    cfg, done = [read(run / p) for p in ("protocol.json", "completed.json")]
    old, old_cfg = [
        read(SEALED_HISTORY / p) for p in ("completed.json", "protocol.json")
    ]
    need(
        done["status"] == "COMPLETE_ENDED_ACTION_HISTORY"
        and done["protocol_sha256"] == h(run / "protocol.json")
        and done["n_shots"] == len(cfg["rows"]) == len(done["sealed"]) == 36
        and done["n_curves"] == 72
        and (cfg["first"], cfg["stride"], cfg["threshold"]) == (FIRST, STRIDE, TAU)
        and (
            cfg["source_state_sha256"],
            cfg["source_fit_sha256"],
            cfg["source_map_sha256"],
        )
        == (SOURCE, FIT, MAP)
        and cfg["new_labels"] == 0
        and cfg["fold_fits_use_other_19_only"] is True
        and cfg["fold_predictions_not_original_stream"] is True
        and cfg["scored_flow_predictions_untouched"] is True
        and all(h(p) == sha for p, sha in cfg["input_sha256"].items()),
        "incomplete or changed history archive",
    )
    parent_sha = h(run / "completed.json")
    original = []
    for block, note in enumerate(old["blocks"], 1):
        commit, reveal = [
            SEALED_HISTORY / "blocks" / f"{block:02d}_{tag}.json"
            for tag in ("commit", "reveals")
        ]
        c, r = read(commit), read(reveal)
        need(
            (h(commit), h(reveal), r["commit_sha256"])
            == (note["commit_sha256"], note["reveals_sha256"], h(commit))
            and c["rank_end"] == block * 20
            and c["selected"] == [x["shot"] for x in r["outcomes"]],
            "original release chain",
        )
        original.extend(
            (
                x["shot"],
                x["disruptive"],
                x["down"],
                c["rank_end"],
                x["read_end_ns"],
                h(commit),
            )
            for x in r["outcomes"]
        )
    roles = old_cfg["order"][:20]
    need(
        original
        == [
            tuple(
                r[k]
                for k in (
                    "shot",
                    "disruptive",
                    "down",
                    "available_after_rank",
                    "original_read_end_ns",
                    "original_commit_sha256",
                )
            )
            for r in cfg["rows"]
        ]
        and [r["shot"] for r in cfg["rows"][:20]] == roles,
        "original outcome roles changed",
    )
    groups = {"cold_loo": [], "queried": []}
    sample_splices = 0
    for n, (row, seal) in enumerate(zip(cfg["rows"], done["sealed"], strict=True)):
        sid, rank = row["shot"], row["rank"]
        cold = n < 20
        g = "cold_loo" if cold else "queried"
        path = run / seal["file"]
        deps = seal["dependencies"]
        need(
            (sid, rank, g) == (seal["shot"], seal["rank"], seal["group"])
            and row["group"] == g
            and old_cfg["order"][rank - 1] == sid
            and h(path) == seal["curve_sha256"]
            and all(h(run / p) == digest for p, digest in deps.items()),
            "history per-shot seal",
        )
        truth, clock = raw_truth(shot_path(sid), sid)
        need(
            h(shot_path(sid))
            == cfg["raw_sha256"][str(sid)]
            == old_cfg["raw_sha256"][str(sid)]
            and all(row[k] == truth[k] for k in ("disruptive", "down")),
            "original HDF5 label mismatch",
        )
        if cold:
            expected = [x for x in roles if x != sid]
            stem = path.stem
            map_path, log_path, rule_path = [
                f"folds/{stem}_{k}" for k in ("map.json", "fit.json", "rule.npz")
            ]
            mapping, fit = (read(run / p) for p in (map_path, log_path))
            need(
                set(deps) == {map_path, log_path, rule_path}
                and mapping["calibration_ids"] == expected
                and fit["training_ids"] == expected
                and fit["held"] == sid
                and fit["initial"]["optimizer_steps"] == 36
                and len(fit["rounds"]) == 3
                and all(
                    set(rd["selection"]) == set(map(str, expected))
                    and rd["steps"] == 12
                    for rd in fit["rounds"]
                ),
                "cold fold used wrong training roles, map or steps",
            )
        with np.load(path, allow_pickle=False) as a:
            native, moved, features = (a[k] for k in ("static", "adapted", "features"))
            need(
                np.array_equal(a["times"], clock)
                and set(a.files) == {"times", "static", "adapted", "features"},
                "history clock or fields changed",
            )
            if not cold:
                ref = SEALED_HISTORY / "shots" / path.name
                need(
                    deps == {str(ref): old["sealed"][rank - 1]["curve_sha256"]},
                    "query prediction not from original archive",
                )
                with np.load(ref, allow_pickle=False) as b:
                    parity(native, b["static"], TAU, f"{sid}/static")
                    parity(moved, b["intra"], TAU, f"{sid}/adapted")
            detail = block_credit(
                clock,
                native,
                moved,
                disruptive=truth["disruptive"],
                down=truth["down"],
                threshold=TAU,
            )
            idx = detail["anchor_indices"]
            need(
                features.shape == (len(idx), 132)
                and np.isfinite(features).all()
                and np.allclose(
                    features[:, 1],
                    (clock[idx] - clock[0]).astype(np.float32),
                    rtol=0,
                    atol=0,
                )
                and np.array_equal(features[:, 2], moved[idx] - native[idx]),
                "noncausal or inconsistent anchor columns",
            )
            scores = {
                k: score(clock, v, truth, TAU)
                for k, v in (("static", native), ("adapted", moved))
            }
            for label, capped in (("strict", False), ("capped", True)):
                need(
                    detail[f"{label}_telescoping"].sum()
                    == outcome_value(scores["adapted"], capped)
                    - outcome_value(scores["static"], capped),
                    "credit must telescope to actual first alarm",
                )
            sample_splices += brute_credit(clock, native, moved, truth, detail)
        groups[g].append(
            dict(
                shot=sid,
                disruptive=truth["disruptive"],
                scores=scores,
                features=features,
                **detail,
            )
        )
    h.check()
    summaries = {
        g: dict(
            n_shots=len(rows),
            actual={
                k: counts([r["scores"][k] for r in rows]) for k in ("static", "adapted")
            },
            coverage={
                f"{t}_{c}": coverage(rows, f"{t}_{c}")
                for t in ("strict", "capped")
                for c in CONTEXTS
            },
        )
        for g, rows in groups.items()
    }
    probe = run_probe(groups["cold_loo"], groups["queried"])
    dump(
        output,
        dict(
            status="PASS_HISTORY_ACTION_CREDIT_HDF5",
            parent_sha256=parent_sha,
            groups=summaries,
            n_independent_splice_sites=sample_splices,
            probe=probe,
            seconds=time.monotonic() - started,
            per_shot={
                g: [
                    {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in r.items()
                        if k != "features"
                    }
                    for r in rows
                ]
                for g, rows in groups.items()
            },
        ),
    )
    for g, info in summaries.items():
        print(
            g,
            {
                t: {a: len(v["shots"]) for a, v in d.items()}
                for t, d in info["coverage"].items()
            },
            flush=True,
        )
    print(probe["status"], flush=True)


def audit_oracle(output):
    started = time.monotonic()
    h = Hashes()
    results = {}
    for mode, (run, checked) in SIGNED.items():
        cfg, done, audit = [
            read(p) for p in (run / "protocol.json", run / "completed.json", checked)
        ]
        order = cfg["order"]
        parent_sha = h(run / "completed.json")
        need(
            audit["status"] == "PASS_SIGNED_PROJECTION_HDF5_RECOUNT"
            and audit["producer_sha256"] == parent_sha
            and done["protocol_sha256"] == h(run / "protocol.json")
            and len(order) == len(done["sealed"])
            and (
                cfg["first"],
                cfg["stride"],
                cfg["threshold"],
                cfg["source_state_sha256"],
            )
            == (FIRST, STRIDE, TAU, SOURCE),
            "verified signed replay required",
        )
        summary = {k: {c: [] for c in ("strict", "capped")} for k in MENU}
        reasons = {k: {c: Counter() for c in ("strict", "capped")} for k in MENU}
        records = []
        for rank, sid in enumerate(order, 1):
            seal = done["sealed"][rank - 1]
            curve = run / "shots" / f"{rank:04d}_{sid}.npz"
            raw = shot_path(sid)
            need(
                (seal["rank"], seal["shot"]) == (rank, sid)
                and seal["curve_sha256"] == h(curve)
                and cfg["raw_sha256"][str(sid)] == seal["raw_sha256"] == h(raw),
                "oracle input hashes",
            )
            truth, clock = raw_truth(raw, sid)
            report = dict(
                shot=sid,
                rank=rank,
                **{k: truth[k] for k in ("disruptive", "down")},
                menus={},
            )
            with np.load(curve, allow_pickle=False) as a:
                need(
                    np.array_equal(a["times"], clock), "oracle clock differs from HDF5"
                )
                original = score(clock, a["static"], truth, TAU)
                report["static"] = original
                for menu, arms in MENU.items():
                    ps = np.stack([a[k] for k in arms])
                    for k, p in zip(arms, ps, strict=True):
                        check_curve(p, clock, k)
                    res = {}
                    for kind, cap in (("strict", False), ("capped", True)):
                        i = best_first(clock, ps, truth, capped=cap)
                        row = score_first(clock, i, truth)
                        summary[menu][kind].append(row)
                        res[kind] = row
                        if truth["disruptive"] and not outcome_value(original, cap):
                            why = miss_reason(clock, ps, truth, i, capped=cap)
                            need(
                                (why == "selectable_rescue")
                                == bool(outcome_value(row, cap)),
                                "oracle outcome mismatch",
                            )
                            reasons[menu][kind][why] += 1
                            res[kind + "_relative_to_static_miss"] = why
                    report["menus"][menu] = res
            records.append(report)
        metrics = {
            k: {c: counts(v) for c, v in sub.items()} for k, sub in summary.items()
        }
        for cap in ("strict", "capped"):
            key = "tp" if cap == "strict" else "tp_30_to_3000"
            native, basic, rich = (
                audit["full"]["static"],
                metrics["binary"][cap],
                metrics["extended"][cap],
            )
            need(
                native[key] <= basic[key] <= rich[key]
                and native["fp"] >= basic["fp"] >= rich["fp"],
                "expanded oracle cannot lose options",
            )
        results[mode] = dict(
            parent_sha256=parent_sha,
            n_shots=len(order),
            metrics=metrics,
            reasons_relative_to_static_miss=reasons,
            static=audit["full"]["static"],
            intra=audit["full"]["intra"],
            per_shot=records,
        )
        print(
            mode,
            {
                k: {
                    c: (d["tp"] if c == "strict" else d["tp_30_to_3000"], d["fp"])
                    for c, d in v.items()
                }
                for k, v in metrics.items()
            },
            flush=True,
        )
    h.check()
    dump(
        output,
        dict(
            status="PASS_POSTHOC_ACTION_MENU_HDF5",
            seconds=time.monotonic() - started,
            cohorts=results,
            scope="Oracle knows the ended shot; no fitting, no new online prediction, not a strategy or independent validation.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="mode", required=True)
    hist, oracle = cmd.add_parser("history"), cmd.add_parser("oracle")
    hist.add_argument("run", type=Path)
    for q in (hist, oracle):
        q.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode == "history":
        audit_history(a.run, a.output)
    else:
        audit_oracle(a.output)
