"""Pure CPU, post-reveal, whole-shot and common-support diagnostics.

The caller supplies verified outcomes after sealing predictions. ``times`` and
``down`` are seconds; all lead and horizon arguments are milliseconds. The
only required dependency is NumPy. No files, labels, or devices are accessed.

``shot_metrics`` returns one JSON-safe row with ``seeds[str(seed)][arm]``;
the caller must add a unique integer or string ``shot`` before aggregation.
Whole-shot first alarms use *all* eligible points, not only active ones. The
first pre-outcome crossing decides both strict and reasonable TP status.

The BCE strata use active points and a binary canonical timely-window target.
After rounding lead to 9 decimal places, ``positive`` has lead_ms < lead <=
horizon_ms + 1e-6 and ``early`` has lead > horizon_ms + 1e-6. ``late`` has
pre-outcome lead <= lead_ms, and ``outside`` combines early and late. Their
zero BCE targets are diagnostic window targets, never HDF outcome relabels.
``safe_high`` has target zero and uses each seed's static sigmoid score >=
threshold - 0.15 to select a common mask for all arms. The post-outcome 1e-6
ms tolerance is applied before lead rounding, matching the native scorer.

``aggregate`` reports per-seed counts and strata, followed by pooled BCEs:
points are averaged within each shot/seed, available seeds within each shot,
then shots equally. Pooled n_points counts seed-point exposures, not unique
time points. Descriptive percentile bootstraps resample paired *shots* and
average available seeds within each shot; they make no independence claim.
The gate uses all fixed seeds and raw strict-TP counts, never a best-seed
selection or a 3000 ms upper cap.
"""

import math
from numbers import Integral

import numpy as np


ARMS = ("static", "real", "reverse", "unsigned", "zero_target", "score")
SEEDS = (131, 317, 563)
STRATA = ("positive", "early", "late", "outside", "safe_high")
BOOTSTRAP_SEED = 61106
BOOTSTRAP_RESAMPLES = 2000
POST_OUTCOME_TOLERANCE_MS = 1e-6
HORIZON_TOLERANCE_MS = 1e-6


def _finite_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _config(threshold, lead_ms):
    threshold = _finite_float(threshold, "threshold")
    lead_ms = _finite_float(lead_ms, "lead_ms")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one")
    if lead_ms < 0:
        raise ValueError("lead_ms must be nonnegative")
    return threshold, lead_ms


def _mean(values):
    """Finite mean without overflowing on a sum of large finite values."""
    values = [float(value) for value in values]
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    return float((math.fsum(value / scale for value in values) / len(values)) * scale)


def _bce(logits, positive):
    losses = np.logaddexp(0.0, -logits if positive else logits)
    return _mean(losses)


def shot_metrics(
    times, logits, active, disruptive, down, horizon_ms, lead_ms, threshold
):
    """Score a single shot's (time, seed, arm) logits and supported points.

    Input clocks must be nonempty, finite, and strictly increasing; logits
    must be finite with shape (N, 3, 6), and ``active`` must be boolean (N,).
    A safe shot's down value is unused and returned as null. Empty strata have
    zero points and null BCE; no output contains NumPy values or NaN/Infinity.
    """
    threshold, lead_ms = _config(threshold, lead_ms)
    horizon_ms = _finite_float(horizon_ms, "horizon_ms")
    if horizon_ms <= lead_ms:
        raise ValueError("horizon_ms must exceed lead_ms")
    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError("disruptive must be a verified boolean")
    disruptive = bool(disruptive)
    down = _finite_float(down, "down") if disruptive else None
    try:
        times = np.asarray(times, dtype=np.float64)
        logits = np.asarray(logits, dtype=np.float64)
        active = np.asarray(active)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times, logits and active must be aligned arrays") from exc
    if times.ndim != 1 or not len(times):
        raise ValueError("times must be a nonempty vector")
    n = len(times)
    if (
        logits.shape != (n, len(SEEDS), len(ARMS))
        or active.shape != (n,)
        or active.dtype != np.bool_
        or not np.isfinite(times).all()
        or not np.isfinite(logits).all()
        or np.any(times[1:] <= times[:-1])
    ):
        raise ValueError(
            "finite ordered times, aligned logits, and boolean active required"
        )

    if disruptive:
        with np.errstate(over="ignore", invalid="ignore"):
            raw_leads = (down - times) * 1000
            leads = np.round(raw_leads, decimals=9)
        if not np.isfinite(raw_leads).all() or not np.isfinite(leads).all():
            raise ValueError("lead times must be finite")
        before = raw_leads >= -POST_OUTCOME_TOLERANCE_MS
        upper = horizon_ms + HORIZON_TOLERANCE_MS
        positive = active & (leads > lead_ms) & (leads <= upper)
        early = active & (leads > upper)
        late = active & before & (leads <= lead_ms)
    else:
        leads = None
        before = np.ones(n, dtype=bool)
        positive = early = late = np.zeros(n, dtype=bool)

    logit_threshold = math.log(threshold) - math.log1p(-threshold)
    result = {
        "disruptive": disruptive,
        "down": down,
        "horizon_ms": horizon_ms,
        "lead_ms": lead_ms,
        "threshold": threshold,
        "n_points": n,
        "n_active": int(active.sum()),
        "seeds": {},
    }
    for seed_index, seed in enumerate(SEEDS):
        static = logits[:, seed_index, 0]
        static_probability = np.empty_like(static)
        nonnegative = static >= 0
        static_probability[nonnegative] = 1 / (1 + np.exp(-static[nonnegative]))
        exp_negative = np.exp(static[~nonnegative])
        static_probability[~nonnegative] = exp_negative / (1 + exp_negative)
        safe_high = active & (static_probability >= threshold - 0.15) & (not disruptive)
        masks = {
            "positive": positive,
            "early": early,
            "late": late,
            "outside": early | late,
            "safe_high": safe_high,
        }
        seed_result = {}
        for arm_index, arm in enumerate(ARMS):
            scores = logits[:, seed_index, arm_index]
            crossings = np.flatnonzero(before & (scores >= logit_threshold))
            first = int(crossings[0]) if len(crossings) else None
            first_lead = (
                float(leads[first]) if disruptive and first is not None else None
            )
            raw_tp = disruptive and first_lead is not None and first_lead > lead_ms
            reasonable_tp = raw_tp and first_lead <= 3000.0
            fp = not disruptive and first is not None
            strata = {
                name: {
                    "n_points": int(mask.sum()),
                    "bce": _bce(scores[mask], name == "positive"),
                }
                for name, mask in masks.items()
            }
            seed_result[arm] = {
                "first_index": first,
                "first_lead_ms": first_lead,
                "raw_tp": raw_tp,
                "reasonable_tp": reasonable_tp,
                "fp": fp,
                "strata": strata,
            }
        result["seeds"][str(seed)] = seed_result
    return result


def _shot_id(value):
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        return value
    raise ValueError("shot must be a unique integer or string")


def _shot_sort(row):
    shot = row["shot"]
    return (0, shot) if isinstance(shot, int) else (1, shot)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _stratum_summary(observations):
    """Summarize one (stratum, arm), with equal shot weighting."""
    shot_values = []
    seed_shots = 0
    points = 0
    for seed_entries in observations:
        valid = [entry for entry in seed_entries if entry["bce"] is not None]
        if valid:
            shot_values.append(_mean(entry["bce"] for entry in valid))
            seed_shots += len(valid)
            points += sum(entry["n_points"] for entry in valid)
    return {
        "n_shots": len(shot_values),
        "n_seed_shot_observations": seed_shots,
        "n_points": points,
        "bce": _mean(shot_values),
    }


def _gains(rows, seed_key, arm, metric):
    gained, lost = [], []
    for row in rows:
        seed = row["seeds"][seed_key]
        base = seed["static"][metric]
        candidate = seed[arm][metric]
        if candidate and not base:
            gained.append(row["shot"])
        elif base and not candidate:
            lost.append(row["shot"])
    return {
        "gained_count": len(gained),
        "gained_shots": gained,
        "lost_count": len(lost),
        "lost_shots": lost,
    }


def _bootstrap(differences):
    """Fixed-seed percentile interval from shot-level resampling."""
    n = len(differences)
    result = {
        "n_shots": n,
        "mean_difference": _mean(differences),
        "ci_95": None,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }
    if n:
        scale = max(abs(value) for value in differences)
        if scale == 0:
            result["ci_95"] = [0.0, 0.0]
        else:
            normalized = np.asarray(differences, dtype=np.float64) / scale
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
            for start in range(0, BOOTSTRAP_RESAMPLES, 128):
                end = min(start + 128, BOOTSTRAP_RESAMPLES)
                indices = rng.integers(0, n, size=(end - start, n))
                means[start:end] = normalized[indices].mean(axis=1)
            result["ci_95"] = [
                float(value * scale) for value in np.quantile(means, [0.025, 0.975])
            ]
    return result


def _paired_differences(rows, control, metric):
    differences = []
    for row in rows:
        if metric in ("raw_tp", "reasonable_tp") and not row["disruptive"]:
            continue
        if metric == "fp" and row["disruptive"]:
            continue
        seed_differences = []
        for seed in SEEDS:
            arms = row["seeds"][str(seed)]
            if metric in STRATA:
                real = arms["real"]["strata"][metric]["bce"]
                other = arms[control]["strata"][metric]["bce"]
                if real is None or other is None:
                    continue
            else:
                real = int(arms["real"][metric])
                other = int(arms[control][metric])
            seed_differences.append(real - other)
        if seed_differences:
            differences.append(_mean(seed_differences))
    return differences


def aggregate(rows, threshold, lead_ms):
    """Aggregate unique post-reveal shot rows, without selecting any seed.

    Rows must be outputs of ``shot_metrics`` with a caller-assigned ``shot``.
    Per-shot horizons may differ. The returned seed/arm counts and paired shot
    IDs cover full trajectories; pooled BCE and bootstrap use supported masks.
    """
    threshold, lead_ms = _config(threshold, lead_ms)
    checked = []
    seen = set()
    for row in rows:
        if "shot" not in row:
            raise ValueError("each row requires a shot")
        shot = _shot_id(row["shot"])
        identity = (type(shot), shot)
        if identity in seen:
            raise ValueError("duplicate shot")
        seen.add(identity)
        if row["threshold"] != threshold:
            raise ValueError("row threshold does not match aggregate threshold")
        if row["lead_ms"] != lead_ms:
            raise ValueError("row lead_ms does not match aggregate lead_ms")
        checked.append({**row, "shot": shot})
    checked.sort(key=_shot_sort)
    n_disruptive = sum(row["disruptive"] for row in checked)
    n_safe = len(checked) - n_disruptive
    result = {
        "schema_version": 1,
        "threshold": threshold,
        "lead_ms": lead_ms,
        "horizons_ms": sorted({row["horizon_ms"] for row in checked}),
        "n_shots": len(checked),
        "n_disruptive": n_disruptive,
        "n_safe": n_safe,
        "per_seed": {},
        "pooled_strata": {},
        "mean_counts": {},
        "paired_bootstrap": {},
    }
    for seed in SEEDS:
        seed_key = str(seed)
        seed_result = {}
        for arm in ARMS:
            entries = [row["seeds"][seed_key][arm] for row in checked]
            raw = sum(entry["raw_tp"] for entry in entries)
            reasonable = sum(entry["reasonable_tp"] for entry in entries)
            fp = sum(entry["fp"] for entry in entries)
            misses = sum(
                row["disruptive"] and entry["first_index"] is None
                for row, entry in zip(checked, entries, strict=True)
            )
            seed_result[arm] = {
                "n_disruptive": n_disruptive,
                "n_safe": n_safe,
                "raw_tp": raw,
                "reasonable_tp": reasonable,
                "early_tp": raw - reasonable,
                "miss": misses,
                "tardy": n_disruptive - raw - misses,
                "fp": fp,
                "tn": n_safe - fp,
                "raw_tpr": _rate(raw, n_disruptive),
                "reasonable_tpr": _rate(reasonable, n_disruptive),
                "fpr": _rate(fp, n_safe),
                "vs_static": {
                    metric: _gains(checked, seed_key, arm, metric)
                    for metric in ("raw_tp", "reasonable_tp", "fp")
                },
                "strata": {
                    name: _stratum_summary(
                        [[entry["strata"][name]] for entry in entries]
                    )
                    for name in STRATA
                },
            }
        result["per_seed"][seed_key] = seed_result

    for name in STRATA:
        result["pooled_strata"][name] = {
            arm: _stratum_summary(
                [
                    [row["seeds"][str(seed)][arm]["strata"][name] for seed in SEEDS]
                    for row in checked
                ]
            )
            for arm in ARMS
        }
    for arm in ARMS:
        result["mean_counts"][arm] = {
            metric: _mean(result["per_seed"][str(seed)][arm][metric] for seed in SEEDS)
            for metric in ("raw_tp", "reasonable_tp", "fp")
        }
    for control in ("static", "reverse"):
        result["paired_bootstrap"][f"real_minus_{control}"] = {
            metric: _bootstrap(_paired_differences(checked, control, metric))
            for metric in (*STRATA, "raw_tp", "reasonable_tp", "fp")
        }
    return result


def source_gate(summary):
    """Return explicit, conservative source criteria and their conjunction."""
    positive = summary["pooled_strata"]["positive"]
    safe = summary["pooled_strata"]["safe_high"]
    real_positive = positive["real"]["bce"]
    real_safe = safe["real"]["bce"]
    static_safe = safe["static"]["bce"]
    criteria = {
        "positive_at_least_20_distinct_shots": positive["real"]["n_shots"] >= 20,
        "safe_high_at_least_20_distinct_shots": safe["real"]["n_shots"] >= 20,
    }
    for arm in ARMS:
        if arm == "real":
            continue
        other = positive[arm]["bce"]
        criteria[f"positive_real_better_than_{arm}_by_over_1e-6"] = (
            real_positive is not None
            and other is not None
            and real_positive + 1e-6 < other
        )
    criteria["safe_high_real_at_most_static_plus_0_02"] = (
        real_safe is not None
        and static_safe is not None
        and real_safe <= static_safe + 0.02
    )

    # All arms have the same fixed three seeds, so compare integer totals.
    # This is exactly equivalent to comparing means, without FP-boundary
    # rounding making equal rational values appear unequal.
    def total(arm, metric):
        return sum(summary["per_seed"][str(seed)][arm][metric] for seed in SEEDS)

    criteria["mean_raw_tp_at_least_static"] = summary["n_disruptive"] > 0 and total(
        "real", "raw_tp"
    ) >= total("static", "raw_tp")
    criteria["mean_fp_at_most_static_plus_1"] = summary["n_safe"] > 0 and total(
        "real", "fp"
    ) <= total("static", "fp") + len(SEEDS)
    return {"passed": all(criteria.values()), "criteria": criteria}
