"""TRAIN-only oracle ranking of identical sealed one-write neural actions."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

import fit_output_trust_20261106 as fit
from inspect_peak_contrast_first_alarm_20261106 import first
from verify_confirmation_raw_replay_20261105 import Hashes


PLAN = fit.r.ROOT / "FIRST_PASSAGE_TARGET_PROTOCOL_20261106.md"
TABLE = fit.r.ROOT / "probe_outputs/output_trust_20261106_v1"
TEMPERATURE, GUARD_POINTS = 0.05, 9
METHODS = ("block_peak", "first_passage")


def first_utility(z, tau, role, ix):
    z, ix = np.asarray(z), np.asarray(ix)
    if role not in ("safe", "recover", "protect") or z.ndim != 2 or not len(ix):
        raise ValueError(
            "nonempty trajectory range and a recognized shot role required"
        )
    if (
        (np.diff(ix) != 1).any()
        or ix[0] < 0
        or ix[-1] >= z.shape[1]
        or not np.isfinite(z).all()
    ):
        raise ValueError(
            "valid contiguous clock range and finite trajectories required"
        )
    cdf = expit((np.maximum.accumulate(z, axis=1) - tau) / TEMPERATURE)
    if role == "safe":
        return 1.0 - cdf[:, ix[-1]]
    s = cdf[:, ix].mean(1)
    return s - cdf[:, ix[0] - 1] if role == "recover" and ix[0] > 0 else s


def best(gain):
    if not len(gain):
        return None
    k = int(np.argmax(gain))
    return k if gain[k] > 1e-12 else None


def chosen(data, slots, raw, mask, mode, seed, native, times, rec, role, ix):
    en, an = np.nonzero(data["available"][slots, mode, seed] & mask.any(1)[:, None])
    if not len(en):
        return dict(n_candidates=0, block_peak=None, first_passage=None)
    rows = slots[en]
    block = data["future"][rows, mode, seed, an].astype(np.float64)
    local = np.where(mask[en], block, -np.inf).max(1)
    diff = local - raw[en]
    if role == "safe":
        diff = -diff
    whole = np.broadcast_to(native, (len(rows), len(native))).copy()
    j, k = np.nonzero(data["present"][rows])
    t = data["maturity"][rows[j]] + 1 + k
    whole[j, t] = block[j, k]
    new = first_utility(whole, fit.TAU, role, ix)
    ref = first_utility(native[None], fit.TAU, role, ix)[0]
    passed = new - ref
    results = dict(n_candidates=len(rows))
    for method, i in zip(METHODS, (best(diff), best(passed)), strict=True):
        results[method] = (
            dict(
                maturity=int(data["maturity"][rows[i]]),
                arm=int(an[i]),
                peak_gain=float(diff[i]),
                first_utility_gain=float(passed[i]),
                **first(times, whole[i], rec["disruptive"], rec["down"], fit.TAU),
            )
            if i is not None
            else None
        )
    return results


def select(data, slots, rec, times, native):
    b = first(times, native, rec["disruptive"], rec["down"], fit.TAU)
    if not rec["disruptive"]:
        role, field, ix = "safe", "present", np.arange(len(times))
    elif b["timely"]:
        end = b["first"]
        role, field, ix = (
            "protect",
            "protect",
            np.arange(max(0, end + 1 - GUARD_POINTS), end + 1),
        )
    else:
        lead = ((rec["down"] - times) * 1000).round(9)
        role, field, ix = (
            "recover",
            "positive",
            np.flatnonzero((lead > 10) & (lead <= rec["horizon_ms"] + 1e-6)),
        )
    mask = data[field][slots]
    if mask.any() and not len(ix):
        raise ValueError("archive role is inconsistent with the first-alarm clock")
    raw = np.where(mask, data["frozen"][slots], -np.inf).max(1)
    arms = [
        [
            chosen(data, slots, raw, mask, m, j, native, times, rec, role, ix)
            for j in range(len(fit.SEEDS))
        ]
        for m in range(len(fit.bank.MODES))
    ]
    return dict(shot=rec["shot"], role=role, baseline=b, modes=arms)


def summary(rows):
    result = {}
    for m, mode in enumerate(fit.bank.MODES):
        result[mode] = {}
        for j, seed in enumerate(fit.SEEDS):
            per_method = {}
            for arm in METHODS:
                pairs = [
                    (q["baseline"], q["modes"][m][j][arm], q["shot"]) for q in rows
                ]
                final = [
                    (base, x if x is not None else base, sid) for base, x, sid in pairs
                ]
                per_method[arm] = dict(
                    writes=sum(x is not None for _, x, _ in pairs),
                    candidates=sum(q["modes"][m][j]["n_candidates"] for q in rows),
                    timely=sum(x["timely"] for _, x, _ in final),
                    fp=sum(x["false_alarm"] for _, x, _ in final),
                    recovered=[
                        sid for b, x, sid in final if not b["timely"] and x["timely"]
                    ],
                    lost=[
                        sid for b, x, sid in final if b["timely"] and not x["timely"]
                    ],
                    delayed=[
                        sid
                        for b, x, sid in final
                        if b["timely"]
                        and x["first"] is not None
                        and x["first"] > b["first"]
                    ],
                    advanced=[
                        sid
                        for b, x, sid in final
                        if b["timely"]
                        and x["first"] is not None
                        and x["first"] < b["first"]
                    ],
                )
            per_method["different_choices"] = sum(
                p["modes"][m][j][METHODS[0]] != p["modes"][m][j][METHODS[1]]
                for p in rows
            )
            result[mode][str(seed)] = per_method
    return dict(
        static_timely=sum(q["baseline"]["timely"] for q in rows),
        static_fp=sum(q["baseline"]["false_alarm"] for q in rows),
        modes=result,
    )


def run(out):
    if out.exists():
        raise FileExistsError(out)
    h, start = Hashes(), time.monotonic()
    for path in (PLAN, Path(__file__), Path(fit.__file__)):
        h(path)
    cfg, _, shards = fit.checked_table(TABLE, h)
    data, recs = fit.load_table(shards)
    idx = defaultdict(list)
    for i, sid in enumerate(data["row_shot"]):
        idx[int(sid)].append(i)
    if [q["shot"] for q, _, _ in recs] != cfg["roles"]["train"]:
        raise ValueError("same TRAIN IDs required")
    rows = []
    for i, (rec, t, z) in enumerate(recs, 1):
        rows.append(
            select(data, np.asarray(idx[rec["shot"]], dtype=np.int64), rec, t, z)
        )
        if i % 250 == 0 or i == len(recs):
            print(f"First-passage TRAIN oracle {i}/{len(recs)} shots", flush=True)
    h.check()
    s = summary(rows)
    fit.r.write_new(
        out,
        dict(
            status="OFFLINE_TARGET_AUDIT_NOT_POLICY",
            temperature=TEMPERATURE,
            guard_points=GUARD_POINTS,
            seeds=fit.SEEDS,
            tau=fit.TAU,
            protocol_sha256=h(PLAN),
            action_table_sha256=h(TABLE / "completed.json"),
            code_sha256=h(Path(__file__)),
            n_shots=len(rows),
            n_windows=len(data["row_shot"]),
            rows=rows,
            summary=s,
            seconds=time.monotonic() - start,
            val_test_east_reads=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)
