"""Pure NumPy post-reveal audit of aligned danger/safe peak effects.

Inputs are already aligned: each pair has a danger and safe block, three
seeds, and the same ordered arms. Danger padding outside ``positive`` and
``protect`` is ignored, while the entire safe block must be observed. The
same positive offsets are used on both blocks. A peak's shape delta is
``max(future - anchor_delta) - max(native)`` on its stated mask; this is not
the maximum of pointwise differences. Threshold crossings use ``>=``.

The report retains every pair and action, including empty/no-solution cases.
It reports raw and anchor-demeaned danger crossings, whole-safe-block
suppression/induction, and protected and positive-window losses. A new danger
crossing requires both a native miss/tardy flag and a native positive-mask
peak below threshold; an already-timely shot is never counted as new. An
empty protection mask imposes no protection constraint and is not evidence of
preservation or loss. A nonempty mask must witness a native crossing.

The returned ``pairs`` retain input order, IDs, masks and every action in
``seed_actions[seed_index][arm_index]``. Per-action feasibility conditions
and protection evidence are explicit. ``arms`` contains per-seed counts,
means and rates with their denominators. ``capacity`` counts the union of
feasible arms within each pair/seed, so its two-seed count may include a
different arm in each seed; the separate per-arm two-seed counts do not.
These are retrospective diagnostics, not a policy, action recommendation,
or deployable selection.

The descriptive 95% percentile bootstrap resamples independent pairs,
preserving all three seeds within each pair, and describes the mean paired
shape contrast across those seeds. The caller is responsible for pair
independence, outcome provenance and predict-before-reveal boundaries; this
module reads no data, files, or outcomes on its own.
"""

import math
from numbers import Real

import numpy as np


BOOTSTRAP_SEED = 61106
BOOTSTRAP_RESAMPLES = 2000
N_SEEDS = 3
BLOCK_SIZE = 32
DANGER_MIN_SHAPE_DELTA = 0.05
SAFE_MAX_SHAPE_DELTA = 1e-6


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers")
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc


def _bool_array(value, name, shape):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a boolean array of shape {shape}") from exc
    if result.shape != shape or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return result


def _inputs(
    native,
    future,
    anchor_delta,
    norms,
    positive,
    protect,
    native_timely,
    threshold_logit,
    arm_names,
    pair_ids,
):
    native = _real_array(native, "native")
    future = _real_array(future, "future")
    anchor_delta = _real_array(anchor_delta, "anchor_delta")
    norms = _real_array(norms, "norms")
    if native.ndim != 3 or native.shape[1:] != (2, BLOCK_SIZE):
        raise ValueError("native must have shape [P, 2, 32]")
    p = native.shape[0]
    if (
        future.ndim != 5
        or future.shape[:3] != (p, 2, N_SEEDS)
        or future.shape[-1] != BLOCK_SIZE
    ):
        raise ValueError("future must have shape [P, 2, 3, A, 32]")
    a = future.shape[3]
    if anchor_delta.shape != (p, 2, N_SEEDS, a):
        raise ValueError("anchor_delta must have shape [P, 2, 3, A]")
    if norms.shape != (p, 2, N_SEEDS, a):
        raise ValueError("norms must have shape [P, 2, 3, A]")
    positive = _bool_array(positive, "positive", (p, BLOCK_SIZE))
    protect = _bool_array(protect, "protect", (p, BLOCK_SIZE))
    native_timely = _bool_array(native_timely, "native_timely", (p,))
    if p and not positive.any(axis=1).all():
        raise ValueError("every positive mask must be nonempty")
    if (
        isinstance(threshold_logit, (bool, np.bool_))
        or not isinstance(threshold_logit, Real)
        or not math.isfinite(threshold_logit)
    ):
        raise ValueError("threshold_logit must be a finite real number")
    threshold_logit = float(threshold_logit)
    if isinstance(arm_names, (str, bytes)):
        raise ValueError("arm_names must contain A unique strings")
    try:
        arm_names = list(arm_names)
    except TypeError as exc:
        raise ValueError("arm_names must contain A unique strings") from exc
    if (
        len(arm_names) != a
        or any(not isinstance(name, str) for name in arm_names)
        or len(set(arm_names)) != a
    ):
        raise ValueError("arm_names must contain A unique strings")
    try:
        pair_ids = np.asarray(pair_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair_ids must be an integer array of shape [P, 2]") from exc
    if pair_ids.shape != (p, 2) or pair_ids.dtype.kind not in "iu":
        raise ValueError("pair_ids must be an integer array of shape [P, 2]")
    if any(np.unique(pair_ids[:, col]).size != p for col in range(2)):
        raise ValueError("pair_ids cannot reuse an ID within either stratum")

    danger_mask = positive | protect
    future_danger_mask = np.broadcast_to(
        danger_mask[:, None, None, :], future[:, 0].shape
    )
    if not all(
        np.isfinite(values).all()
        for values in (
            native[:, 0][danger_mask],
            native[:, 1],
            future[:, 0][future_danger_mask],
            future[:, 1],
            anchor_delta,
            norms,
        )
    ):
        raise ValueError("all observed values and action metadata must be finite")
    has_protect = protect.any(axis=1)
    if p:
        protected_native = np.max(np.where(protect, native[:, 0], -np.inf), axis=-1)
        if np.any(has_protect & (protected_native < threshold_logit)):
            raise ValueError(
                "each nonempty protect mask must witness a native crossing"
            )
    return (
        native,
        future,
        anchor_delta,
        norms,
        positive,
        protect,
        native_timely,
        threshold_logit,
        arm_names,
        pair_ids,
    )


def _peaks(native, future, anchor_delta, mask):
    base = np.max(np.where(mask, native, -np.inf), axis=-1)
    action_mask = mask[:, None, None, :]
    actual = np.max(np.where(action_mask, future, -np.inf), axis=-1)
    shifted = np.zeros_like(future)
    np.subtract(future, anchor_delta[..., None], out=shifted, where=action_mask)
    demeaned = np.max(np.where(action_mask, shifted, -np.inf), axis=-1)
    return {
        "native_peak": base,
        "actual_peak": actual,
        "score_only_peak": base[:, None, None] + anchor_delta,
        "shape_demeaned_peak": demeaned,
        "actual_delta": actual - base[:, None, None],
        "shape_delta": demeaned - base[:, None, None],
    }


def _peak_row(peaks, pair, seed, arm):
    return {
        name: float(values[pair] if name == "native_peak" else values[pair, seed, arm])
        for name, values in peaks.items()
    }


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    scale = np.max(np.abs(values))
    return float(np.mean(values / scale) * scale) if scale else 0.0


def _rate(count, denominator):
    return count / denominator if denominator else None


def _contrast_summary(contrast):
    """Percentile CI over pair resamples; seeds remain nested within pairs."""
    n = contrast.shape[0]
    per_seed_mean = [_mean(contrast[:, seed]) for seed in range(N_SEEDS)]
    per_seed_positive = [
        float(np.mean(contrast[:, seed] > 0)) if n else None for seed in range(N_SEEDS)
    ]
    result = {
        "n_pairs": n,
        "three_seed_mean": None,
        "three_seed_positive_proportion": None,
        "per_seed_mean": per_seed_mean,
        "per_seed_positive_proportion": per_seed_positive,
        "bootstrap_ci_95": None,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if not n:
        return result
    scale = np.max(np.abs(contrast))
    pair_mean = np.mean(contrast / scale, axis=1) * scale if scale else np.zeros(n)
    result["three_seed_mean"] = _mean(pair_mean)
    result["three_seed_positive_proportion"] = float(np.mean(pair_mean > 0))
    # Reinitialize per arm, giving all arms identical resampled pair indices.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_scale = np.max(np.abs(pair_mean))
    normalized = pair_mean / bootstrap_scale if bootstrap_scale else pair_mean
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 128):
        stop = min(start + 128, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples[start:stop] = np.mean(normalized[indices], axis=1)
    result["bootstrap_ci_95"] = (
        np.percentile(samples, [2.5, 97.5]) * bootstrap_scale
    ).tolist()
    return result


def _seed_summary(pairs, seed, arm):
    rows = [pair["seed_actions"][seed][arm] for pair in pairs]
    n = len(rows)
    counts = {
        "native_miss_or_tardy": sum(not pair["native_timely"] for pair in pairs),
        "new_danger_crossing_eligible": sum(
            pair["new_danger_crossing_eligible"] for pair in pairs
        ),
        "protected": sum(bool(pair["protect_offsets"]) for pair in pairs),
        "native_timely_protected": sum(
            pair["native_timely"] and bool(pair["protect_offsets"]) for pair in pairs
        ),
        "native_timely_without_protect": sum(
            pair["native_timely"] and not pair["protect_offsets"] for pair in pairs
        ),
        "native_danger_positive_crossing": sum(
            pair["native_danger_positive_crossing"] for pair in pairs
        ),
        "native_safe_full_crossing": sum(
            pair["native_safe_full_crossing"] for pair in pairs
        ),
    }
    event_names = (
        "new_danger_crossing_raw",
        "new_danger_crossing_shape",
        "lost_danger_positive_crossing_raw",
        "lost_danger_positive_crossing_shape",
        "safe_induced_raw",
        "safe_induced_shape",
        "safe_suppressed_raw",
        "safe_suppressed_shape",
        "lost_protected_crossing",
        "lost_native_timely_protected_crossing",
        "danger_shape_pass",
        "safe_shape_pass",
        "safe_no_new_crossing",
        "protection_satisfied",
        "feasible",
    )
    counts.update({name: sum(bool(row[name]) for row in rows) for name in event_names})
    denominators = {
        "new_danger_crossing_raw": counts["new_danger_crossing_eligible"],
        "new_danger_crossing_shape": counts["new_danger_crossing_eligible"],
        "lost_danger_positive_crossing_raw": counts["native_danger_positive_crossing"],
        "lost_danger_positive_crossing_shape": counts[
            "native_danger_positive_crossing"
        ],
        "safe_induced_raw": n - counts["native_safe_full_crossing"],
        "safe_induced_shape": n - counts["native_safe_full_crossing"],
        "safe_suppressed_raw": counts["native_safe_full_crossing"],
        "safe_suppressed_shape": counts["native_safe_full_crossing"],
        "lost_protected_crossing": counts["protected"],
        "lost_native_timely_protected_crossing": counts["native_timely_protected"],
        "danger_shape_pass": n,
        "safe_shape_pass": n,
        "safe_no_new_crossing": n,
        "protection_satisfied": n,
        "feasible": n,
    }
    rates = {name: _rate(counts[name], denominators[name]) for name in event_names}
    for suffix in ("raw", "shape"):
        key = f"new_danger_crossing_{suffix}_per_miss_or_tardy"
        denominators[key] = counts["native_miss_or_tardy"]
        rates[key] = _rate(
            counts[f"new_danger_crossing_{suffix}"], counts["native_miss_or_tardy"]
        )
    means = {}
    for scope in ("danger_positive", "safe_aligned", "safe_full"):
        for metric in (
            "native_peak",
            "actual_peak",
            "score_only_peak",
            "shape_demeaned_peak",
            "actual_delta",
            "shape_delta",
        ):
            means[f"{scope}_{metric}"] = _mean([row[scope][metric] for row in rows])
    for metric in (
        "shape_contrast",
        "danger_anchor_delta",
        "safe_anchor_delta",
        "danger_norm",
        "safe_norm",
    ):
        means[metric] = _mean([row[metric] for row in rows])
    return {
        "seed_index": seed,
        "n_pairs": n,
        "counts": counts,
        "rates": rates,
        "rate_denominators": denominators,
        "means": means,
        "feasible_pair_ids": [
            pair["pair_id"] for pair, row in zip(pairs, rows) if row["feasible"]
        ],
    }


def score_pairs(
    native,
    future,
    anchor_delta,
    norms,
    positive,
    protect,
    native_timely,
    threshold_logit,
    arm_names,
    pair_ids,
):
    """Return JSON-safe, per-action and pair-preserving diagnostic metrics.

    Shapes are ``[P,2,32]``, ``[P,2,3,A,32]``, ``[P,2,3,A]`` for both
    action metadata arrays, ``[P,32]`` for both masks, ``[P]`` for native
    timeliness, and integer ``[P,2]`` for pair IDs. Axis 1 is danger, safe.
    Seed indices are 0, 1, 2; no seed labels are inferred. ``P`` and ``A``
    may be zero. No input is modified. Empty samples have null statistics.
    """
    (
        native,
        future,
        anchor_delta,
        norms,
        positive,
        protect,
        native_timely,
        threshold_logit,
        arm_names,
        pair_ids,
    ) = _inputs(
        native,
        future,
        anchor_delta,
        norms,
        positive,
        protect,
        native_timely,
        threshold_logit,
        arm_names,
        pair_ids,
    )
    p, a = len(native), len(arm_names)
    has_protect = protect.any(axis=1)
    # The fallback is used only internally for empty protect masks; their
    # returned protection evidence remains null, never a substituted peak.
    protected_mask = np.where(has_protect[:, None], protect, positive)
    try:
        with np.errstate(over="raise", invalid="raise"):
            danger = _peaks(native[:, 0], future[:, 0], anchor_delta[:, 0], positive)
            safe = _peaks(native[:, 1], future[:, 1], anchor_delta[:, 1], positive)
            safe_full = _peaks(
                native[:, 1],
                future[:, 1],
                anchor_delta[:, 1],
                np.ones((p, BLOCK_SIZE), dtype=bool),
            )
            protected = _peaks(
                native[:, 0], future[:, 0], anchor_delta[:, 0], protected_mask
            )
            contrast = danger["shape_delta"] - safe["shape_delta"]
    except FloatingPointError as exc:
        raise ValueError("derived scores must be finite") from exc
    if (
        not all(
            np.isfinite(values).all()
            for peaks in (danger, safe, safe_full, protected)
            for values in peaks.values()
        )
        or not np.isfinite(contrast).all()
    ):
        raise ValueError("derived scores must be finite")

    pairs = []
    for i in range(p):
        d_native_cross = bool(danger["native_peak"][i] >= threshold_logit)
        s_native_cross = bool(safe_full["native_peak"][i] >= threshold_logit)
        eligible = bool(not native_timely[i] and not d_native_cross)
        pair = {
            "pair_id": [int(value) for value in pair_ids[i]],
            "native_timely": bool(native_timely[i]),
            "positive_offsets": np.flatnonzero(positive[i]).tolist(),
            "protect_offsets": np.flatnonzero(protect[i]).tolist(),
            "native_danger_positive_peak": float(danger["native_peak"][i]),
            "native_safe_aligned_peak": float(safe["native_peak"][i]),
            "native_safe_full_peak": float(safe_full["native_peak"][i]),
            "native_danger_protect_peak": (
                float(protected["native_peak"][i]) if has_protect[i] else None
            ),
            "native_danger_positive_crossing": d_native_cross,
            "native_safe_full_crossing": s_native_cross,
            "new_danger_crossing_eligible": eligible,
            "seed_actions": [],
            "feasible_arms_by_seed": [],
        }
        for seed in range(N_SEEDS):
            seed_actions, feasible_arms = [], []
            for arm in range(a):
                d = _peak_row(danger, i, seed, arm)
                s = _peak_row(safe, i, seed, arm)
                whole = _peak_row(safe_full, i, seed, arm)
                prot = _peak_row(protected, i, seed, arm) if has_protect[i] else None
                preserves = prot is None or prot["actual_peak"] >= threshold_logit
                safe_induced = (
                    not s_native_cross and whole["actual_peak"] >= threshold_logit
                )
                danger_pass = d["shape_delta"] >= DANGER_MIN_SHAPE_DELTA
                safe_pass = s["shape_delta"] <= SAFE_MAX_SHAPE_DELTA
                feasible = bool(
                    danger_pass and safe_pass and not safe_induced and preserves
                )
                row = {
                    "arm": arm_names[arm],
                    "danger_anchor_delta": float(anchor_delta[i, 0, seed, arm]),
                    "safe_anchor_delta": float(anchor_delta[i, 1, seed, arm]),
                    "danger_norm": float(norms[i, 0, seed, arm]),
                    "safe_norm": float(norms[i, 1, seed, arm]),
                    "danger_positive": d,
                    "safe_aligned": s,
                    "safe_full": whole,
                    "danger_protect": prot,
                    "shape_contrast": float(contrast[i, seed, arm]),
                    "new_danger_crossing_raw": bool(
                        eligible and d["actual_peak"] >= threshold_logit
                    ),
                    "new_danger_crossing_shape": bool(
                        eligible and d["shape_demeaned_peak"] >= threshold_logit
                    ),
                    "lost_danger_positive_crossing_raw": bool(
                        d_native_cross and d["actual_peak"] < threshold_logit
                    ),
                    "lost_danger_positive_crossing_shape": bool(
                        d_native_cross and d["shape_demeaned_peak"] < threshold_logit
                    ),
                    "safe_induced_raw": bool(safe_induced),
                    "safe_induced_shape": bool(
                        not s_native_cross
                        and whole["shape_demeaned_peak"] >= threshold_logit
                    ),
                    "safe_suppressed_raw": bool(
                        s_native_cross and whole["actual_peak"] < threshold_logit
                    ),
                    "safe_suppressed_shape": bool(
                        s_native_cross
                        and whole["shape_demeaned_peak"] < threshold_logit
                    ),
                    "protection_preserved": bool(preserves)
                    if prot is not None
                    else None,
                    "protection_satisfied": bool(preserves),
                    "danger_shape_pass": bool(danger_pass),
                    "safe_shape_pass": bool(safe_pass),
                    "safe_no_new_crossing": bool(not safe_induced),
                    "lost_protected_crossing": bool(not preserves)
                    if prot is not None
                    else None,
                    "lost_native_timely_protected_crossing": (
                        bool(not preserves)
                        if prot is not None and native_timely[i]
                        else None
                    ),
                    "feasible": feasible,
                }
                seed_actions.append(row)
                if feasible:
                    feasible_arms.append(arm_names[arm])
            pair["seed_actions"].append(seed_actions)
            pair["feasible_arms_by_seed"].append(feasible_arms)
        pair["feasible_seed_count"] = sum(
            bool(arms) for arms in pair["feasible_arms_by_seed"]
        )
        pairs.append(pair)

    arms = []
    for arm, name in enumerate(arm_names):
        arms.append(
            {
                "arm": name,
                "shape_contrast": _contrast_summary(contrast[:, :, arm]),
                "seeds": [_seed_summary(pairs, seed, arm) for seed in range(N_SEEDS)],
                "candidate_pairs_at_least_two_seeds": sum(
                    sum(
                        pair["seed_actions"][seed][arm]["feasible"]
                        for seed in range(N_SEEDS)
                    )
                    >= 2
                    for pair in pairs
                ),
            }
        )
    capacity = {
        "candidate_pairs_per_seed": [
            sum(bool(pair["feasible_arms_by_seed"][seed]) for pair in pairs)
            for seed in range(N_SEEDS)
        ],
        "candidate_pairs_any_seed": sum(
            pair["feasible_seed_count"] > 0 for pair in pairs
        ),
        "candidate_pairs_at_least_two_seeds": sum(
            pair["feasible_seed_count"] >= 2 for pair in pairs
        ),
        "no_solution_pair_ids": [
            pair["pair_id"] for pair in pairs if not pair["feasible_seed_count"]
        ],
    }
    return {
        "n_pairs": p,
        "n_arms": a,
        "threshold_logit": threshold_logit,
        "danger_min_shape_delta": DANGER_MIN_SHAPE_DELTA,
        "safe_max_shape_delta": SAFE_MAX_SHAPE_DELTA,
        "capacity_is_retrospective_not_policy": True,
        "arms": arms,
        "capacity": capacity,
        "pairs": pairs,
    }
