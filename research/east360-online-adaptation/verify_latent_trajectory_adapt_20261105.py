"""Independent HDF/curve replay audit and safe-window coverage diagnosis."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    RAW_ROOT,
    SOURCE,
    SOURCE_DONE_SHA,
    Hashes,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)
from joint_lora_hindsight_capacity_20261104 import write_new

ARMS = [
    "static",
    "state20",
    "history20",
    "stale20",
    "state100",
    "history100",
    "stale100",
]


def expected_queries(times, p, flag, down, first):
    if flag:
        lead = np.round(1000 * (down - times), 9)
        eligible = np.flatnonzero(
            (lead > 30) & (lead <= 125) & (np.arange(len(lead)) >= first)
        )
        n = min(8, len(eligible))
        ix = np.array(
            [i * (len(eligible) - 1) // max(n - 1, 1) for i in range(n)], dtype=int
        )
        return eligible[ix], lead[eligible[ix]]
    width, extra = divmod(max(len(times) - first, 0), 8)
    q, start = [], first
    for b in range(8):
        stop = start + width + int(b < extra)
        if stop > start:
            q.append(start + int(np.argmax(p[start:stop])))
        start = stop
    return np.array(q, dtype=int), np.full(len(q), np.nan)


def audit(run, output):
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    completion = run / "completed.json"
    done = read(completion)
    need(
        done["status"] == "COMPLETE_LATENT_TRAJECTORY_ADAPT_DEVELOPMENT",
        "terminal run required",
    )
    producer_sha = h(completion)
    for name, value in done["artifact_sha256"].items():
        need(h(run / name) == value, f"run artifact changed: {name}")
    cfg, saved, fits, summary = [
        read(run / name)
        for name in (
            "protocol.json",
            "per_shot.json",
            "fit_receipts.json",
            "summary.json",
        )
    ]
    need(all(h(p) == v for p, v in cfg["input_sha256"].items()), "bound inputs changed")
    need(
        h(SOURCE / "completed.json") == SOURCE_DONE_SHA, "locked parent receipt changed"
    )
    original, binding = read(SOURCE / "protocol.json"), read(SOURCE / "completed.json")
    need(
        h(SOURCE / "protocol.json") == binding["artifact_sha256"]["protocol.json"],
        "locked parent protocol changed",
    )
    ids, tau, first, stride = (
        cfg["order"],
        cfg["threshold"],
        cfg["first_anchor"],
        cfg["stride"],
    )
    need(
        ids == [r["shot"] for r in original["order"]]
        and len(ids) == len(set(ids)) == 360,
        "fixed identity",
    )
    need(
        cfg["arms"] == ARMS
        and tau == original["threshold"]
        and cfg["raw_sha256"] == original["raw_sha256"],
        "fixed scorer and raw cohort",
    )
    need(
        cfg["label_budgets"] == [20, 100]
        and cfg["independent_future_validation"] is False,
        "history and claims scope",
    )
    need(
        done["n_shots"] == len(ids) and done["n_curves"] == len(ids) * len(ARMS),
        "terminal count",
    )
    for n in (20, 100):
        f = fits[str(n)]
        need(
            set(f["ids"]) == set(ids[:n])
            and f["n_shots"] == n
            and f["after_all_historical_reveals"],
            "fit did not use only ended prefix",
        )
        need(f["params_sha256"] == h(run / f"fit{n}.npz"), "fit artifact changed")
    with np.load(run / "latent_bank.npz", allow_pickle=False) as file:
        bank = {k: file[k] for k in file.files}
    need(set(bank["rank"]) == set(range(1, 361)), "bank shot coverage")
    need(
        all(np.isfinite(bank[k]).all() for k in ("state", "history", "native_logit")),
        "bank finiteness",
    )
    for n in (20, 100):
        need(
            int((bank["rank"] <= n).sum()) == fits[str(n)]["n_examples"],
            "fit window count",
        )
    observed = {k: [] for k in ARMS}
    updates, norm_mismatch = Counter(), Counter()
    safe_coverage = {k: Counter() for k in ARMS if k != "static"}
    groups = {k: {"deep_positive": [], "high_safe": []} for k in ARMS if k != "static"}
    for i, (sid, receipt) in enumerate(zip(ids, done["sealed"], strict=True), 1):
        path = run / "shots" / f"{i:04d}_{sid}.npz"
        seal, rev = [
            path.with_name(path.stem + x) for x in ("_seal.json", "_revealed.json")
        ]
        note, label = read(seal), read(rev)
        need(
            receipt
            == dict(
                shot=sid,
                curve_sha256=h(path),
                seal_sha256=h(seal),
                revealed_sha256=h(rev),
            ),
            "seal binding",
        )
        need(
            (note["shot"], note["rank"], label["shot"]) == (sid, i, sid)
            and note["curve_sha256"] == receipt["curve_sha256"],
            "seal identity",
        )
        need(
            note["current_outcome_used_in_prediction"] is False
            and label["after_seal_sha256"] == receipt["seal_sha256"],
            "outcome boundary receipt",
        )
        need(
            path.stat().st_mtime_ns
            <= seal.stat().st_mtime_ns
            <= rev.stat().st_mtime_ns,
            "outcome file order",
        )
        active = {str(b): fits[str(b)]["params_sha256"] for b in (20, 100) if b < i}
        need(note["active_fit_sha256"] == active, "wrong historical fit used")
        if i in (20, 100):
            next_path = path.with_name(f"{i + 1:04d}_{ids[i]}.npz")
            need(
                rev.stat().st_mtime_ns
                <= (run / f"fit{i}.npz").stat().st_mtime_ns
                <= next_path.stat().st_mtime_ns,
                "fit must follow its last reveal",
            )
        raw = RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"
        need(h(raw) == note["raw_sha256"] == cfg["raw_sha256"][str(sid)], "raw hash")
        truth, times = raw_truth(raw, sid)
        h(raw)
        flag, down = truth["disruptive"], truth["down"]
        orig = SOURCE / "shots" / path.name
        need(
            h(orig) == binding["seals"][i - 1]["curve_sha256"],
            "archived baseline identity",
        )
        with (
            np.load(path, allow_pickle=False) as f,
            np.load(orig, allow_pickle=False) as p,
        ):
            need(
                np.array_equal(times, f["times"]) and np.array_equal(times, p["times"]),
                "HDF clock",
            )
            static = f["static"]
            need(np.array_equal(static, p["map_after20"]), "raw native parity")
            teacher = p["retrospective_map_all"] if i <= 20 else static
            q, lead = expected_queries(times, teacher, flag, down, first)
            mask = bank["rank"] == i
            need(
                np.array_equal(bank["query"][mask], q)
                and np.array_equal(bank["lead_ms"][mask], lead, equal_nan=True)
                and (bank["positive"][mask] == flag).all()
                and (bank["shot"][mask] == sid).all(),
                "teaching selection/truth",
            )
            native_q = np.exp(
                -np.logaddexp(0.0, -bank["native_logit"][mask].astype(float))
            )
            need(
                np.max(abs(native_q - teacher[q]), initial=0) <= 4e-6,
                "bank native logit",
            )
            arm_is_active = {
                k: k != "static"
                and (k.endswith("20") and i > 20 or k.endswith("100") and i > 100)
                for k in ARMS
            }
            need(
                set(note["updates"]) == {k for k, v in arm_is_active.items() if v},
                "update arm presence",
            )
            for k in ARMS:
                v = f[k]
                need(
                    v.shape == times.shape
                    and np.isfinite(v).all()
                    and ((v >= 0) & (v <= 1)).all(),
                    "valid probability curve",
                )
                prefix = first if arm_is_active[k] else len(v)
                need(
                    np.array_equal(v[:prefix], static[:prefix]),
                    "updates before eligible sample",
                )
                row = score(times, v, truth, tau)
                need(row == label["rows"][k] == saved[k][i - 1], "score recount")
                observed[k].append(row)
                if not arm_is_active[k]:
                    continue
                stats = note["updates"][k]
                need(
                    0 <= stats["nonzero"] <= len(range(first, len(times), stride))
                    and 0 <= stats["maximum"] <= cfg["radius"] + 1e-6,
                    "bounded recorded update",
                )
                updates[k] += stats["nonzero"]
                norm_mismatch[k] += stats["real_norm_mismatch"]
                if (
                    not flag
                    and row["status"] == "fp"
                    and observed["static"][-1]["status"] == "tn"
                ):
                    c = safe_coverage[k]
                    c["new_fp"] += 1
                    c["new_fp_first_outside_teacher_peaks"] += (
                        row["first_index"] not in q
                    )
                    c["new_fp_not_visible_at_any_teacher_peak"] += not (
                        v[q] >= tau
                    ).any()
                if i > 100:
                    hard = native_q <= tau - 0.25 if flag else native_q >= tau
                    if hard.any():
                        groups[k]["deep_positive" if flag else "high_safe"].append(
                            float((v[q][hard] - static[q][hard]).mean())
                        )
    for group, section in (
        ("full360", slice(None)),
        ("after20", slice(20, None)),
        ("after100", slice(100, None)),
    ):
        partial = {k: v[section] for k, v in observed.items()}
        expected = dict(
            metrics={k: counts(v) for k, v in partial.items()},
            pairs={
                k: pair(v, partial["static"])
                for k, v in partial.items()
                if k != "static"
            },
            pairs_capped={
                k: pair(v, partial["static"], capped=True)
                for k, v in partial.items()
                if k != "static"
            },
        )
        need(expected == summary[group], f"summary recount {group}")
    h.check()
    paired = {
        str(b): {
            c: pair(observed[f"history{b}"][100:], observed[f"{c}{b}"][100:])
            for c in ("state", "stale")
        }
        for b in (20, 100)
    }
    group_means = {
        k: {
            g: dict(shots=len(v), mean_delta_p=float(np.mean(v)) if v else None)
            for g, v in x.items()
        }
        for k, x in groups.items()
    }
    report = dict(
        status="PASS_LATENT_ADAPTATION_INDEPENDENT_RECOUNT",
        n_shots=360,
        n_curves=2520,
        bank_rows=len(bank["shot"]),
        producer_completion_sha256=producer_sha,
        verifier_sha256=h(Path(__file__)),
        fitted_after_prefix=True,
        native_probability_exact=True,
        nonzero_update_receipts=dict(updates),
        norm_mismatch_receipts=dict(norm_mismatch),
        summary=summary,
        history_pairwise_suffix=paired,
        new_safe_false_alarm_coverage=safe_coverage,
        suffix_hard_window_mean_change=group_means,
        independent_future_validation=False,
    )
    write_new(output, report)
    print(report["status"], safe_coverage, group_means, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.run, a.output)
