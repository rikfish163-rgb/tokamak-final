"""Independently audit sealed patch-selection results against first alarms.

Only the producer's sealed-bank loader supplies HDF5 truth and bank arrays.  The
scoring, finite-menu reachability, folds, model evaluation and choices below do
not call the producer scorer, target replay, or fitter.  This is a TRAIN-only
record audit, not a raw neural-forward or prospective-performance witness.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VIEWS = ("risk", "true", "reverse")
ORDER = tuple((view, mode) for view in VIEWS for mode in ("open", "guarded"))
MARGINS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, np.inf)
WIDTH = 32


def need(condition, message):
    if not condition:
        raise ValueError(message)


def first_alarm(times, curve, disruptive, down, tau):
    """Compare scores in float64; HDF5 leads use ms and 9-decimal rounding."""
    times = np.asarray(times, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    need(times.ndim == 1 and curve.shape == times.shape, "bad shot arrays")
    eligible = (down - times) * 1000 >= -1e-6 if disruptive else np.ones(len(times), bool)
    hits = np.flatnonzero(eligible & (curve >= np.float64(tau)))
    first = int(hits[0]) if len(hits) else None
    lead = round(float((down - times[first]) * 1000), 9) if disruptive and first is not None else None
    return dict(first=first, lead_ms=lead, timely=bool(lead is not None and lead > 10),
                false_alarm=bool(not disruptive and first is not None))


def replay(native, clock, curves, allowed, choices, *, effect=None):
    """Splice the selected real block, or its same-anchor scalar control."""
    native, clock = np.asarray(native), np.asarray(clock)
    curves, allowed, choices = np.asarray(curves), np.asarray(allowed), np.asarray(choices)
    need(choices.shape == (len(clock),) and choices.dtype.kind in "iu", "bad choice vector")
    need(np.all((choices >= 0) & (choices < curves.shape[1])), "choice outside menu")
    need(allowed[np.arange(len(clock)), choices].all(), "choice unavailable")
    output = np.array(native, dtype=np.result_type(native.dtype, curves.dtype), copy=True)
    for b, c in enumerate(clock):
        start, end = int(c) + 1, min(int(c) + 1 + WIDTH, len(native))
        block = curves[b, choices[b], :end - start]
        if effect is not None:
            block = curves[b, 0, :end - start] + effect[b, choices[b], 2]
        output[start:end] = block
    return output


def menu_analysis(shot, clock, curves, allowed, tau):
    """Derive every single-edit label and all reachable first-hit indices.

    Each local action first-hit is combined with unaffected native hits for
    single edits.  For the exact oracle, a later block is reachable only when
    every earlier editable block has an available no-hit choice and no
    unaffected native sample has already crossed.
    """
    native = np.asarray(shot.native, dtype=np.float64)
    times = np.asarray(shot.times, dtype=np.float64)
    clock, curves, allowed = np.asarray(clock), np.asarray(curves), np.asarray(allowed)
    n = len(native)
    need(n > 0 and times.shape == (n,) and np.all(np.diff(times) > 0), "invalid time grid")
    need(np.isfinite(native).all() and np.isfinite(times).all() and np.isfinite(curves).all(),
         "non-finite menu")
    need(clock.ndim == 1 and clock.dtype.kind in "iu" and np.all((clock >= 0) & (clock < n)),
         "invalid release clock")
    need(curves.ndim == 3 and curves.shape[0] == len(clock) and curves.shape[2] == WIDTH,
         "invalid curve shape")
    need(allowed.dtype.kind == "b" and allowed.shape == curves.shape[:2]
         and allowed.shape[1] > 0 and allowed[:, 0].all(), "invalid menu availability")
    starts = clock.astype(np.int64) + 1
    ends = np.minimum(starts + WIDTH, n)
    order = np.argsort(starts, kind="stable")
    need(all(starts[order[i]] >= ends[order[i - 1]] for i in range(1, len(order))),
         "overlapping edit blocks")
    for b, (start, end) in enumerate(zip(starts, ends)):
        need(np.array_equal(curves[b, 0, :end - start], native[start:end]),
             f"identity curve mismatch in block {b}")
    leads = (shot.down - times) * 1000 if shot.disruptive else None
    eligible = leads >= -1e-6 if shot.disruptive else np.ones(n, bool)
    hits = np.flatnonzero((native >= np.float64(tau)) & eligible)
    base = first_alarm(times, native, shot.disruptive, shot.down, tau)
    timely = np.zeros(n + 1, bool)
    if shot.disruptive:
        timely[:n] = leads > 10
        for i in np.flatnonzero(np.abs(leads - 10) <= 2e-9):
            timely[i] = round(float(leads[i]), 9) > 10
    hard = np.zeros(allowed.shape, dtype=np.int8)
    local = []
    for b, (start, end) in enumerate(zip(starts, ends)):
        width = int(end - start)
        crossing = (curves[b, :, :width].astype(np.float64) >= np.float64(tau))
        crossing &= eligible[start:end]
        if width:
            firsts = np.where(crossing.any(axis=1), start + crossing.argmax(axis=1), n)
        else:
            firsts = np.full(allowed.shape[1], n, dtype=np.int64)
        local.append(firsts)
        prefix = int(hits[0]) if len(hits) and hits[0] < start else n
        later = np.searchsorted(hits, end)
        suffix = int(hits[later]) if later < len(hits) else n
        overall = np.minimum(firsts, min(prefix, suffix)).astype(np.intp)
        change = (timely[overall].astype(np.int8) - int(base["timely"]) if shot.disruptive
                  else int(base["false_alarm"]) - (overall < n).astype(np.int8))
        hard[b, allowed[b]] = change[allowed[b]]
    need(not hard[:, 0].any(), "identity action changed hard label")
    reachable, cursor, alive = set(), 0, True
    for b in order:
        start, end = int(starts[b]), int(ends[b])
        at = np.searchsorted(hits, cursor)
        if at < len(hits) and hits[at] < start:
            reachable.add(int(hits[at]))
            alive = False
            break
        options = local[b][allowed[b]]
        reachable.update(int(v) for v in options if v < n)
        if not np.any(options == n):
            alive = False
            break
        cursor = end
    if alive:
        at = np.searchsorted(hits, cursor)
        reachable.add(int(hits[at]) if at < len(hits) else n)
    reachable = sorted(reachable)
    return dict(baseline=base, hard=hard, reachable_first=reachable,
                oracle_timely=bool(shot.disruptive and any(timely[i] for i in reachable)),
                oracle_no_false_alarm=bool(not shot.disruptive and n in reachable))


def deterministic_folds(ids):
    ids = [int(s) for s in ids]
    need(len(ids) == len(set(ids)), "duplicate shot IDs")
    order = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(str(ids[i]).encode("ascii")).digest(), ids[i]))
    result = np.empty(len(ids), dtype=np.int64)
    for rank, index in enumerate(order):
        result[index] = rank % 5
    return result


def features(data, rows, view, tau):
    """Rebuild the recorded feature recipe without the producer function."""
    history, effect = data["risk"][rows], data["effect"][rows]
    n, actions, _ = effect.shape
    action = np.broadcast_to(np.eye(actions, dtype=np.float32)[None, :, 1:], (n, actions, actions - 1))
    risk = np.broadcast_to(history[:, None], (n, actions, history.shape[1]))
    anchor = effect[..., 2:3]
    interactions = np.concatenate((action * (history[:, 7] - tau)[:, None, None],
                                   anchor * risk[..., [0, 7, 8, 9]]), axis=2)
    if view == "risk":
        motion = np.zeros((n, actions, 45), dtype=np.float32)
    else:
        need(view in ("true", "reverse"), "unknown view")
        which = 0 if view == "true" else 1
        window, g = data["ordered"][rows, which], data["g"][rows, which]
        norm = np.linalg.norm(g, axis=-1, keepdims=True)
        g = np.divide(g, norm, out=np.zeros_like(g), where=norm > 1e-8)
        agreement = np.einsum("baf,btf->bat", data["states"][rows], g, optimize=True)
        motion = np.concatenate((np.broadcast_to(window[:, None], (n, actions, 27)),
                                 agreement, anchor * window[:, None, 9:18]), axis=2)
    return np.concatenate((risk, action, effect, interactions, motion), axis=2).astype(np.float32)


def model_predict(x, allowed, model):
    x = np.asarray(x, dtype=np.float64)
    mean, scale, coefficient = (np.asarray(model[k], dtype=np.float64)
                                for k in ("mean", "scale", "coefficient"))
    need(x.ndim == 3 and x.shape[:2] == allowed.shape and mean.shape == (x.shape[2],)
         and scale.shape == coefficient.shape == mean.shape and np.all(scale > 0)
         and all(np.isfinite(v).all() for v in (x, mean, scale, coefficient))
         and np.isfinite(model["intercept"]), "invalid recorded model")
    pred = np.full(allowed.shape, -np.inf, dtype=np.float64)
    pred[:, 0] = 0.0
    positive = allowed.copy()
    positive[:, 0] = False
    pred[positive] = ((x[positive] - mean) / scale) @ coefficient + model["intercept"]
    need(np.isfinite(pred[positive]).all(), "non-finite model prediction")
    return pred


def choose(prediction, allowed, margin):
    scores = np.where(allowed & np.isfinite(prediction), prediction, -np.inf).copy()
    scores[:, 0] = -np.inf
    best = np.argmax(scores, axis=1)
    value = scores[np.arange(len(scores)), best]
    return np.where(np.isfinite(value) & (value > margin), best, 0).astype(np.int64)


def summarize(records):
    return dict(timely=sum(c["timely"] for _, _, c in records),
                fp=sum(c["false_alarm"] for _, _, c in records),
                recovered=[s for s, b, c in records if not b["timely"] and c["timely"]],
                lost=[s for s, b, c in records if b["timely"] and not c["timely"]],
                fp_added=[s for s, b, c in records if not b["false_alarm"] and c["false_alarm"]],
                fp_removed=[s for s, b, c in records if b["false_alarm"] and not c["false_alarm"]],
                first_changed=sum(b["first"] != c["first"] for _, b, c in records))


def verify_fits(shots, data, choices, utility, owner, recorded_folds, evidence, tau):
    ids = [q.sid for q in shots]
    folds = deterministic_folds(ids)
    allowed = data["allowed"]
    need(np.array_equal(folds, recorded_folds), "deterministic fold assignment mismatch")
    need(choices.shape == (len(owner), len(ORDER)) and utility.shape == (len(VIEWS), *allowed.shape),
         "stored choice or utility dimensions mismatch")
    need(set(evidence) == set(VIEWS), "fold views mismatch")
    max_error, checked = 0.0, 0

    def predict_rows(rows, view, model):
        result = np.empty((len(rows), allowed.shape[1]), dtype=np.float64)
        for start in range(0, len(rows), 2048):
            ix = rows[start:start + 2048]
            result[start:start + len(ix)] = model_predict(features(data, ix, view, tau), allowed[ix], model)
        return result

    for v, view in enumerate(VIEWS):
        records = evidence[view]
        need(len(records) == 5, f"{view}: incomplete fold receipts")
        for k, receipt in enumerate(records):
            cal_fold = (k + 1) % 5
            expected = dict(held_shots=[ids[i] for i in range(len(ids)) if folds[i] == k],
                            calibrate_shots=[ids[i] for i in range(len(ids)) if folds[i] == cal_fold],
                            fit_shots=[ids[i] for i in range(len(ids)) if folds[i] not in (k, cal_fold)])
            need(receipt["fit_fold"] == k and receipt["calibrate_fold"] == cal_fold
                 and all(receipt[key] == value for key, value in expected.items()),
                 f"{view} fold {k}: incomplete, overlapping or non-deterministic fold lists")
            held_rows = np.flatnonzero(folds[owner] == k)
            held = predict_rows(held_rows, view, receipt["model"])
            stored = utility[v, held_rows]
            need(np.array_equal(np.isneginf(stored), np.isneginf(held)), f"{view} fold {k}: utility mask")
            finite = allowed[held_rows]
            need(np.isfinite(stored[finite]).all(), f"{view} fold {k}: non-finite utility")
            diff = np.abs(stored[finite].astype(np.float64) - held[finite])
            max_error = max(max_error, float(diff.max(initial=0.0)))
            need(np.allclose(stored[finite], held[finite], rtol=1e-6, atol=2e-7),
                 f"{view} fold {k}: stored utility differs")
            margin = np.inf if receipt["margin"] is None else float(receipt["margin"])
            need(any(margin == m for m in MARGINS), f"{view} fold {k}: unknown margin")
            for j, m in enumerate((0.0, margin)):
                actual = choose(held, allowed[held_rows], m)
                need(np.array_equal(actual, choices[held_rows, 2 * v + j]),
                     f"{view} fold {k}: held choice mismatch")
            checked += 2 * len(held_rows)

            cal_rows = np.flatnonzero(folds[owner] == cal_fold)
            cal_pred = predict_rows(cal_rows, view, receipt["model"])
            candidates = []
            for m in MARGINS:
                selected = choose(cal_pred, allowed[cal_rows], m)
                per_shot = []
                for i in np.flatnonzero(folds == cal_fold):
                    q = shots[i]
                    lo, hi = np.searchsorted(cal_rows, [q.blocks.start, q.blocks.stop])
                    sl = q.blocks
                    z = replay(q.native, data["clock"][sl], data["curves"][sl], allowed[sl], selected[lo:hi])
                    base = first_alarm(q.times, q.native, q.disruptive, q.down, tau)
                    per_shot.append((q.sid, base, first_alarm(q.times, z, q.disruptive, q.down, tau)))
                summary = summarize(per_shot)
                eligible = not summary["fp_added"] and not summary["lost"]
                candidates.append(dict(margin=None if np.isinf(m) else float(m), eligible=eligible, **summary))
            need(candidates == receipt["calibration"], f"{view} fold {k}: calibration replay mismatch")
            feasible = [(len(c["recovered"]), len(c["fp_removed"]), m)
                        for m, c in zip(MARGINS, candidates) if c["eligible"]]
            need(feasible and max(feasible)[2] == margin, f"{view} fold {k}: selected calibration margin mismatch")
    return dict(held_choices_checked=checked, calibration_rules_checked=len(VIEWS) * 5 * len(MARGINS),
                max_recorded_float32_utility_error=max_error,
                utility_comparison="rtol=1e-6, atol=2e-7 against float64 reconstruction")


def verify_replays(shots, data, stored_hard, choices, folds, result, tau):
    allowed = data["allowed"]
    need(choices.shape == (len(allowed), len(ORDER)) and choices.dtype.kind in "iu"
         and stored_hard.shape == allowed.shape, "hard/choice array dimensions mismatch")
    labels = [f"{v}_{m}" for v, m in ORDER] + ["true_open_scalar", "true_guarded_scalar"]
    need(set(result["per_shot"]) == set(labels) and set(result["scores"]) == set(labels),
         "result arm list mismatch")
    capacity = dict(available_actions=int(allowed[:, 1:].sum()), positive_actions=0, negative_actions=0,
                    single_edit_timely=[], single_edit_fp_removed=[], oracle_timely=[], oracle_fp_removed=[])
    support_keys = ("positive_shots", "negative_shots", "single_edit_timely", "single_edit_fp_removed",
                    "multiwrite_only_timely", "multiwrite_only_fp_removed", "distinct_true_benefit")
    by_fold = {str(k): {key: [] for key in support_keys} for k in range(5)}
    arm_records = {label: [] for label in labels}
    arm_positive = {label: [] for label in labels}
    arm_negative = {label: [] for label in labels}
    bases = []
    for i, q in enumerate(shots):
        sl = q.blocks
        clock, curves, available = (data[key][sl] for key in ("clock", "curves", "allowed"))
        analysis = menu_analysis(q, clock, curves, available, tau)
        hard, base = analysis.pop("hard"), analysis["baseline"]
        need(np.array_equal(hard, stored_hard[sl]), f"shot {q.sid}: single-edit hard labels differ")
        for key in ("baseline", "reachable_first", "oracle_timely", "oracle_no_false_alarm"):
            need(analysis[key] == q.teacher[key], f"shot {q.sid}: independent menu {key} differs")
        positive, negative = bool((hard > 0).any()), bool((hard < 0).any())
        capacity["positive_actions"] += int((hard > 0).sum())
        capacity["negative_actions"] += int((hard < 0).sum())
        fold = by_fold[str(int(folds[i]))]
        if positive:
            fold["positive_shots"].append(q.sid)
        if negative:
            fold["negative_shots"].append(q.sid)
        if q.disruptive and not base["timely"]:
            if positive:
                capacity["single_edit_timely"].append(q.sid)
                fold["single_edit_timely"].append(q.sid)
            if analysis["oracle_timely"]:
                capacity["oracle_timely"].append(q.sid)
                if not positive:
                    fold["multiwrite_only_timely"].append(q.sid)
        if not q.disruptive and base["false_alarm"]:
            if positive:
                capacity["single_edit_fp_removed"].append(q.sid)
                fold["single_edit_fp_removed"].append(q.sid)
            if analysis["oracle_no_false_alarm"]:
                capacity["oracle_fp_removed"].append(q.sid)
                if not positive:
                    fold["multiwrite_only_fp_removed"].append(q.sid)
        bases.append([q.sid, base, base])
        for label in labels:
            scalar = label.endswith("_scalar")
            view, mode = label.split("_")[:2]
            col = ORDER.index((view, mode))
            selected = choices[sl, col]
            effect = data["effect"][sl] if scalar else None
            z = replay(q.native, clock, curves, available, selected, effect=effect)
            scored = first_alarm(q.times, z, q.disruptive, q.down, tau)
            record = [q.sid, base, scored]
            stored = result["per_shot"][label]
            need(len(stored) == len(shots) and record == stored[i],
                 f"{label} shot {q.sid}: per-shot tuple differs")
            arm_records[label].append(record)
            if not scalar:
                chosen_hard = hard[np.arange(len(selected)), selected]
                if (chosen_hard > 0).any():
                    arm_positive[label].append(q.sid)
                if (chosen_hard < 0).any():
                    arm_negative[label].append(q.sid)
    need(capacity == result["capacity"], "capacity lists or action counts differ")
    baseline = summarize(bases)
    need(baseline == result["baseline"], "baseline summary differs")
    need(result["n_shots"] == len(shots) and result["disruptive"] == sum(q.disruptive for q in shots)
         and result["blocks"] == len(allowed), "cohort totals differ")
    scores, evidence = {}, {}
    for label, records in arm_records.items():
        view, mode = label.split("_")[:2]
        col = ORDER.index((view, mode))
        summary = summarize(records) | {"writes": int(np.count_nonzero(choices[:, col]))}
        need(summary == result["scores"][label], f"{label}: summary differs")
        scores[label] = summary
        evidence[label] = summary | {"per_shot_tuples_checked": len(records)}
        if not label.endswith("_scalar"):
            evidence[label]["selected_positive_hard_shots"] = arm_positive[label]
            evidence[label]["selected_negative_hard_shots"] = arm_negative[label]
    def benefit(x):
        return set(x["recovered"] + x["fp_removed"])
    main = scores["true_guarded"]
    distinct = sorted(benefit(main) - set().union(*(benefit(scores[k]) for k in
        ("risk_guarded", "reverse_guarded", "true_guarded_scalar"))))
    promoted = bool(distinct and not main["lost"] and not main["fp_added"]
                    and (main["recovered"] or len(main["fp_removed"]) >= 3))
    need(distinct == result["distinct_true_benefit"] and promoted == result["promoted"],
         "distinct benefit or promotion flag differs")
    shot_to_fold = {q.sid: int(folds[i]) for i, q in enumerate(shots)}
    for sid in distinct:
        by_fold[str(shot_to_fold[sid])]["distinct_true_benefit"].append(sid)
    for k, group in by_fold.items():
        group["shot_count"] = sum(int(f) == int(k) for f in folds)
        group["disruptive_count"] = sum(q.disruptive and int(folds[i]) == int(k)
                                        for i, q in enumerate(shots))
        group["counts"] = {key: len(group[key]) for key in support_keys}
    return dict(baseline=baseline, arms=evidence, capacity=capacity,
                hard_support_by_fold=by_fold, distinct_true_benefit=distinct, promoted=promoted,
                all_single_edit_labels_checked=int(stored_hard.size),
                reachable_first_lists_checked=len(shots))


def verify(bank_root, result_dir, output):
    # Lazy import: synthetic tests never invoke the producer loader or touch data.
    import run_patch_utility_selection_20261106 as producer
    from verify_confirmation_raw_replay_20261105 import Hashes

    bank_root, result_dir, output = map(Path, (bank_root, result_dir, output))
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    result_path = result_dir / "result.json"
    choice_path = result_dir / "selected_actions.npz"
    fit_path = result_dir / "fit_and_calibration.json"
    h(result_path)
    with result_path.open(encoding="utf-8") as f:
        result = json.load(f)
    need(result["status"] == "SCORED_TRAIN_SHOT_DISJOINT_DIAGNOSTIC", "unexpected result status")
    manifest = {Path(path).resolve(strict=True): digest for path, digest in result["input_sha256"].items()}
    need(len(manifest) == len(result["input_sha256"]), "duplicate canonical input hashes")
    for path, digest in manifest.items():
        need(h(path) == digest, f"recorded input hash differs: {path}")
    for path, key in ((bank_root / "seal.json", "bank_seal_sha256"),
                      (choice_path, "choices_sha256"), (fit_path, "fit_sha256")):
        canonical = path.resolve(strict=True)
        need(canonical in manifest and h(path) == result[key] == manifest[canonical],
             f"missing or mismatched {key}")
    with fit_path.open(encoding="utf-8") as f:
        receipts = json.load(f)
    need([list(v) for v in ORDER] == receipts["order"]
         and receipts["bank_seal_sha256"] == result["bank_seal_sha256"], "fit receipt seal/order mismatch")
    with np.load(choice_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in ("choices", "utility", "hard", "soft", "folds", "owner")}
    print("Validated recorded hashes; loading sealed TRAIN bank and HDF5 truth", flush=True)
    shots, data, loader_hard, loader_soft, owner = producer.load_bank(bank_root, h)
    # The producer also hashed these inputs before calling its loader.
    for path in (Path(producer.__file__), Path(producer.fit.__file__), Path(producer.utility.__file__),
                 producer.bank.PLAN, producer.BASELINE):
        need(path.resolve(strict=True) in manifest, f"producer input absent from manifest: {path}")
        h(path)
    extra = set(h.cache) - set(manifest) - {result_path.resolve(strict=True)}
    need(not extra, f"loader read inputs absent from result hash manifest: {sorted(map(str, extra))}")
    need(np.array_equal(loader_hard, arrays["hard"]) and np.array_equal(loader_soft, arrays["soft"]),
         "saved targets differ from producer loader")
    need(np.array_equal(owner, arrays["owner"]) and owner.dtype.kind in "iu", "stored block owners differ")
    cursor = 0
    for i, q in enumerate(shots):
        need(q.blocks.start == cursor and q.blocks.stop >= cursor
             and np.all(owner[q.blocks] == i), f"shot {q.sid}: non-contiguous block ownership")
        cursor = q.blocks.stop
    need(cursor == len(owner), "unassigned blocks")
    tau = producer.bank.TAU
    replay_evidence = verify_replays(shots, data, arrays["hard"], arrays["choices"],
                                     arrays["folds"], result, tau)
    with producer.BASELINE.open(encoding="utf-8") as f:
        reference = json.load(f)["summary"]
    need(replay_evidence["baseline"]["timely"] == reference["static_timely"]
         and replay_evidence["baseline"]["fp"] == reference["static_fp"], "frozen baseline mismatch")
    print("Verified all shot/action replays and exact menu capacity; checking fold receipts", flush=True)
    fit_evidence = verify_fits(shots, data, arrays["choices"], arrays["utility"], owner,
                               arrays["folds"], receipts["evidence"], tau)
    h.check()
    verifier_sha = h(Path(__file__))
    audit = dict(status="INDEPENDENT_TRAIN_PATCH_SELECTION_RECOUNT", result_sha256=h(result_path),
                 bank_seal_sha256=result["bank_seal_sha256"], choices_sha256=result["choices_sha256"],
                 fit_sha256=result["fit_sha256"], verifier_sha256=verifier_sha,
                 recorded_input_hashes_checked=len(manifest), truth="producer loader: HDF5 IsDisrupt/DownTime",
                 threshold_float64=float(tau), replay=replay_evidence, fits=fit_evidence,
                 limitations=["No raw neural-forward witness or coefficient refit",
                              "Soft targets checked against loader, not independently recomputed",
                              "TRAIN diagnostic, not held-out or online performance"])
    with output.open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Verified TRAIN patch-selection records: {output}", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    verify(args.bank, args.result_dir, args.output_json)
