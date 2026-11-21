"""Independent, CPU-only metrics for predictions already sealed by the caller.

This module never opens a prediction, receipt, raw-data, or outcome file, and
does not import the training objective.  The caller independently validates the
seal, reveals the authoritative HDF5 outcome, calls ``per_shot`` with the
supplied arrays, and adds a unique ``shot`` to each returned row.  Pass those
rows to ``summarize``.  The optional ``prediction_diagnostics`` accepts the
remaining prediction-only arrays and can be called before truth is revealed.

``times`` and ``down`` are seconds, while ``lead_ms`` and ``horizon_ms`` are
milliseconds.  First crossings use all eligible points, even inactive points.
The first alarm alone determines classification.  Raw leads no later than
1e-6 ms after disruption are eligible, then leads are rounded to nine decimals.
Raw TP requires a first-alarm lead strictly greater than ``lead_ms``; it has no
maximum-lead cap. ``reasonable_tp`` additionally requires lead > 30 ms and
<= 3000 ms.  For direct comparison with the existing paired scorer,
``native_reasonable_tp`` instead means raw TP with lead <= 3000 ms.

Crossings use float64 logits >= logit(threshold), matching the paired scorer.
The native scorer instead thresholds its supplied probability array; finite
precision, especially float32 sigmoid rounding at the threshold, can yield a
different first crossing.  Since those original native probabilities are not
in this interface, bit-for-bit native agreement at that boundary cannot be
asserted from the sealed logits alone; the caller should compare the original
probabilities if exact native reproduction is required.

The positive BCE mask is active & (lead > lead_ms) &
(lead <= horizon_ms + 1e-6); other diagnostic strata follow the prior paired
scorer, including its static-selected threshold-minus-0.15 safe slice.  Safe
peak probabilities use *all* time points and all safe shots.  A stratum mean
averages points per seed/shot, available seeds per shot, then shots equally.

``summarize`` returns JSON-safe counts by seed and arm; all safe peak values
and distributions; pooled BCE; and ``paired[block_minus_<arm>]`` for each
control.  Its paired entries retain seed-specific timely wins, losses, new and
removed FPs, and descriptive bootstrap intervals.  A bootstrap observation is
one shot's mean paired difference across the three seeds; shots, not seeds or
seed/shot rows, are resampled.  The fixed-seed percentile intervals are
descriptive, not evidence that repeated seeds are independent.

The source gate is reported in ``summary['source_gate']``.  Its integer net
objective is exactly sum_over_seeds(raw_tp - fp), compared strictly with point,
reverse and score; no best seed, reasonable-lead cap, or BCE substitutes for
the prespecified gate.  The gate also requires both outcome groups, no per-seed
timely regression or FP increase vs static, an aggregate timely gain, and at
least two seeds with strictly improved timely counts.
"""

import math
from numbers import Integral

import numpy as np


SEEDS = (131, 317, 563)
ARMS = ("static", "old", "point", "block", "reverse", "score")
STRATA = ("positive", "early", "late", "outside", "safe_high")
CONTROLS = tuple(arm for arm in ARMS if arm != "block")
POST_OUTCOME_TOLERANCE_MS = 1e-6
HORIZON_TOLERANCE_MS = 1e-6
BOOTSTRAP_SEED = 61106
BOOTSTRAP_RESAMPLES = 2000


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


def _mean(values):
    """Average finite values without overflowing an intermediate sum."""
    values = [float(value) for value in values]
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    return float(math.fsum(value / scale for value in values) / len(values) * scale)


def _probability(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1 / (1 + np.exp(-values[nonnegative]))
    exp_value = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_value / (1 + exp_value)
    return result


def _stratum(scores, mask, positive):
    selected = scores[mask]
    losses = np.logaddexp(0, -selected if positive else selected)
    return {"n_points": int(mask.sum()), "bce": _mean(losses)}


def per_shot(times, logits, active, disruptive, down, horizon_ms, threshold, lead_ms):
    """Return one post-reveal row for the fixed (time, seed, arm) array order.

    ``logits`` is (N, 3, 6); ``active`` is one shared boolean mask of length N.
    A safe shot ignores ``down``; a disruptive shot requires finite ``down``.
    The caller must assign ``row['shot']`` before aggregation.
    """
    threshold = _finite_float(threshold, "threshold")
    lead_ms = _finite_float(lead_ms, "lead_ms")
    horizon_ms = _finite_float(horizon_ms, "horizon_ms")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one")
    if lead_ms < 0 or horizon_ms <= lead_ms:
        raise ValueError("lead_ms must be nonnegative and horizon_ms must exceed lead_ms")
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
    if (logits.shape != (n, len(SEEDS), len(ARMS)) or active.shape != (n,)
            or active.dtype != np.bool_ or not np.isfinite(times).all()
            or not np.isfinite(logits).all() or np.any(times[1:] <= times[:-1])):
        raise ValueError("finite ordered times, aligned logits, and boolean active required")

    if disruptive:
        with np.errstate(over="ignore", invalid="ignore"):
            raw_lead = (down - times) * 1000
            leads = np.round(raw_lead, 9)
        if not np.isfinite(raw_lead).all() or not np.isfinite(leads).all():
            raise ValueError("lead times must be finite")
        eligible = raw_lead >= -POST_OUTCOME_TOLERANCE_MS
        positive = active & (leads > lead_ms) & (leads <= horizon_ms + HORIZON_TOLERANCE_MS)
        early = active & (leads > horizon_ms + HORIZON_TOLERANCE_MS)
        late = active & eligible & (leads <= lead_ms)
    else:
        leads = None
        eligible = np.ones(n, dtype=bool)
        positive = early = late = np.zeros(n, dtype=bool)

    logit_threshold = math.log(threshold) - math.log1p(-threshold)
    row = {
        "disruptive": disruptive, "down": down, "horizon_ms": horizon_ms,
        "threshold": threshold, "lead_ms": lead_ms, "n_points": n,
        "n_active": int(active.sum()), "seeds": {},
    }
    for i, seed in enumerate(SEEDS):
        safe_high = active & (_probability(logits[:, i, 0]) >= threshold - 0.15) & (not disruptive)
        masks = {"positive": positive, "early": early, "late": late,
                 "outside": early | late, "safe_high": safe_high}
        entries = {}
        for j, arm in enumerate(ARMS):
            scores = logits[:, i, j]
            crossings = np.flatnonzero(eligible & (scores >= logit_threshold))
            first = int(crossings[0]) if len(crossings) else None
            first_lead = float(leads[first]) if disruptive and first is not None else None
            raw_tp = disruptive and first_lead is not None and first_lead > lead_ms
            native_reasonable = raw_tp and first_lead <= 3000.0
            reasonable = native_reasonable and first_lead > 30.0
            peak = None
            if not disruptive:
                peak_index = int(np.argmax(scores))
                peak = {"index": peak_index, "time": float(times[peak_index]),
                        "logit": float(scores[peak_index]),
                        "probability": float(_probability(scores[peak_index]))}
            entries[arm] = {
                "first_index": first, "first_lead_ms": first_lead,
                "raw_tp": bool(raw_tp), "reasonable_tp": bool(reasonable),
                "native_reasonable_tp": bool(native_reasonable),
                "fp": bool(not disruptive and first is not None), "safe_peak": peak,
                "strata": {name: _stratum(scores, mask, name == "positive")
                           for name, mask in masks.items()},
            }
        row["seeds"][str(seed)] = entries
    return row


def prediction_diagnostics(maturity, common, norms, cos, offsets):
    """Summarize only common-support, prediction-only action measurements.

    Shapes are (M,), (M,), (M,3,3), (M,3), and (M,3).  Norm channels are
    old/point/block, and cosine is point versus block.  This is diagnostic
    accounting, not a replacement for the caller's independent seal validation.
    """
    try:
        maturity = np.asarray(maturity)
        common = np.asarray(common)
        norms = np.asarray(norms, dtype=np.float64)
        cos = np.asarray(cos, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid prediction diagnostic arrays") from exc
    if maturity.ndim != 1:
        raise ValueError("maturity must be a vector")
    m = len(maturity)
    if (common.shape != (m,) or (m and common.dtype != np.bool_)
            or (m and not np.issubdtype(maturity.dtype, np.integer))
            or norms.shape != (m, 3, 3) or cos.shape != (m, 3)
            or offsets.shape != (m, 3) or not np.isfinite(norms).all()
            or not np.isfinite(cos).all() or not np.isfinite(offsets).all()
            or (m and (np.any(maturity < 0) or np.any(maturity[1:] <= maturity[:-1])))):
        raise ValueError("invalid prediction diagnostic shapes, masks, or values")
    common = common.astype(bool, copy=False)
    result = {"n_maturity": m, "n_common": int(common.sum()),
              "common_maturity": [int(value) for value in maturity[common]],
              "per_seed": {}}
    for i, seed in enumerate(SEEDS):
        result["per_seed"][str(seed)] = {
            "mean_norm": {arm: _mean(norms[common, i, j])
                          for j, arm in enumerate(("old", "point", "block"))},
            "mean_point_block_cos": _mean(cos[common, i]),
            "mean_score_offset": _mean(offsets[common, i]),
            "mean_abs_score_offset": _mean(np.abs(offsets[common, i])),
        }
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


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    names = ("min", "p05", "p25", "median", "p75", "p90", "p95", "p99", "max")
    if not len(values):
        return {"count": 0, "mean": None, **{name: None for name in names}}
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .9, .95, .99, 1])
    return {"count": len(values), "mean": _mean(values),
            **{name: float(value) for name, value in zip(names, quantiles, strict=True)}}


def _stratum_summary(observations):
    shot_values = []
    seed_shots = points = 0
    for seed_entries in observations:
        valid = [entry for entry in seed_entries if entry["bce"] is not None]
        if valid:
            shot_values.append(_mean(entry["bce"] for entry in valid))
            seed_shots += len(valid)
            points += sum(entry["n_points"] for entry in valid)
    return {"n_shots": len(shot_values), "n_seed_shot_observations": seed_shots,
            "n_points": points, "bce": _mean(shot_values)}


def _bootstrap(differences):
    n = len(differences)
    result = {"n_shots": n, "mean_difference": _mean(differences), "ci_95": None,
              "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
              "unit": "shot; seed differences averaged within shot",
              "interpretation": "descriptive percentile interval"}
    if n:
        scale = max(abs(value) for value in differences)
        if scale == 0:
            result["ci_95"] = [0.0, 0.0]
        else:
            values = np.asarray(differences, dtype=np.float64) / scale
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
            for start in range(0, BOOTSTRAP_RESAMPLES, 128):
                end = min(start + 128, BOOTSTRAP_RESAMPLES)
                indices = rng.integers(0, n, size=(end - start, n))
                means[start:end] = values[indices].mean(axis=1)
            result["ci_95"] = [float(value * scale)
                               for value in np.quantile(means, [.025, .975])]
    return result


def _difference(row, control, metric):
    if metric in ("raw_tp", "reasonable_tp", "native_reasonable_tp", "positive_bce"):
        if not row["disruptive"]:
            return None
    elif metric in ("fp", "safe_peak_probability") and row["disruptive"]:
        return None
    differences = []
    for seed in SEEDS:
        arms = row["seeds"][str(seed)]
        block, other = arms["block"], arms[control]
        if metric == "positive_bce":
            left, right = block["strata"]["positive"]["bce"], other["strata"]["positive"]["bce"]
            if left is None or right is None:
                continue
        elif metric == "safe_peak_probability":
            left, right = block["safe_peak"]["probability"], other["safe_peak"]["probability"]
        elif metric == "net":
            left = int(block["raw_tp"]) - int(block["fp"])
            right = int(other["raw_tp"]) - int(other["fp"])
        else:
            left, right = int(block[metric]), int(other[metric])
        differences.append(left - right)
    return _mean(differences)


def _comparison(rows, control):
    per_seed = {}
    for seed in SEEDS:
        change = {key: [] for key in ("timely_gains", "timely_losses",
                                      "reasonable_gains", "reasonable_losses",
                                      "new_fp", "removed_fp")}
        for row in rows:
            arms = row["seeds"][str(seed)]
            block, other = arms["block"], arms[control]
            for metric, gained, lost in (("raw_tp", "timely_gains", "timely_losses"),
                                         ("reasonable_tp", "reasonable_gains", "reasonable_losses"),
                                         ("fp", "new_fp", "removed_fp")):
                if block[metric] and not other[metric]:
                    change[gained].append(row["shot"])
                elif other[metric] and not block[metric]:
                    change[lost].append(row["shot"])
        change["timely_delta"] = len(change["timely_gains"]) - len(change["timely_losses"])
        change["fp_delta"] = len(change["new_fp"]) - len(change["removed_fp"])
        change["net_delta"] = change["timely_delta"] - change["fp_delta"]
        per_seed[str(seed)] = change
    totals = {name: sum(len(per_seed[str(seed)][name]) for seed in SEEDS)
              for name in ("timely_gains", "timely_losses", "reasonable_gains",
                           "reasonable_losses", "new_fp", "removed_fp")}
    totals.update({name: sum(per_seed[str(seed)][name] for seed in SEEDS)
                   for name in ("timely_delta", "fp_delta", "net_delta")})
    boot = {}
    for metric in ("raw_tp", "reasonable_tp", "native_reasonable_tp", "fp",
                   "positive_bce", "safe_peak_probability", "net"):
        differences = [_difference(row, control, metric) for row in rows]
        boot[metric] = _bootstrap([value for value in differences if value is not None])
    return {"per_seed": per_seed, "totals": totals, "bootstrap": boot}


def source_gate(summary):
    """Apply the fixed source gate to whole-trajectory raw first-alarm counts."""
    by_seed = {}
    for seed in SEEDS:
        arms = summary["per_seed"][str(seed)]
        block, static = arms["block"], arms["static"]
        by_seed[str(seed)] = {
            "block_raw_tp": block["raw_tp"], "static_raw_tp": static["raw_tp"],
            "block_fp": block["fp"], "static_fp": static["fp"],
            "raw_tp_at_least_static": block["raw_tp"] >= static["raw_tp"],
            "fp_at_most_static": block["fp"] <= static["fp"],
            "raw_tp_strictly_above_static": block["raw_tp"] > static["raw_tp"],
        }
    totals = {arm: sum(summary["per_seed"][str(seed)][arm]["raw_tp"]
                       - summary["per_seed"][str(seed)][arm]["fp"] for seed in SEEDS)
              for arm in ARMS}
    raw_totals = {arm: sum(summary["per_seed"][str(seed)][arm]["raw_tp"] for seed in SEEDS)
                  for arm in ARMS}
    improved = sum(value["raw_tp_strictly_above_static"] for value in by_seed.values())
    criteria = {
        "has_disruptive_shots": summary["n_disruptive"] > 0,
        "has_safe_shots": summary["n_safe"] > 0,
        "all_seeds_raw_tp_at_least_static": all(v["raw_tp_at_least_static"] for v in by_seed.values()),
        "all_seeds_fp_at_most_static": all(v["fp_at_most_static"] for v in by_seed.values()),
        "aggregate_raw_tp_strictly_above_static": raw_totals["block"] > raw_totals["static"],
        "at_least_two_seeds_improve_raw_tp": improved >= 2,
        **{f"block_net_strictly_above_{control}": totals["block"] > totals[control]
           for control in ("point", "reverse", "score")},
    }
    return {"passed": all(criteria.values()), "criteria": criteria, "per_seed": by_seed,
            "n_seeds_improved": improved, "raw_tp_by_arm": raw_totals,
            "objective": "sum_over_seeds(raw_tp - fp)", "objective_by_arm": totals}


def summarize(rows):
    """Summarize rows produced by ``per_shot`` after the caller adds shot IDs."""
    checked, seen = [], set()
    threshold = lead_ms = None
    for row in rows:
        if "shot" not in row:
            raise ValueError("each row requires a shot")
        shot = _shot_id(row["shot"])
        identity = (type(shot), shot)
        if identity in seen:
            raise ValueError("duplicate shot")
        seen.add(identity)
        if threshold is None:
            threshold, lead_ms = row["threshold"], row["lead_ms"]
        if row["threshold"] != threshold:
            raise ValueError("row threshold does not match")
        if row["lead_ms"] != lead_ms:
            raise ValueError("row lead_ms does not match")
        checked.append({**row, "shot": shot})
    checked.sort(key=_shot_sort)
    n_disruptive = sum(row["disruptive"] for row in checked)
    n_safe = len(checked) - n_disruptive
    result = {"schema_version": 1, "threshold": threshold, "lead_ms": lead_ms,
              "horizons_ms": sorted({row["horizon_ms"] for row in checked}),
              "n_shots": len(checked), "n_disruptive": n_disruptive, "n_safe": n_safe,
              "per_seed": {}, "pooled_strata": {}, "pooled_safe_peak_probability": {},
              "mean_counts": {}, "paired": {}}
    for seed in SEEDS:
        seed_result = {}
        for arm in ARMS:
            entries = [row["seeds"][str(seed)][arm] for row in checked]
            raw = sum(entry["raw_tp"] for entry in entries)
            reasonable = sum(entry["reasonable_tp"] for entry in entries)
            native_reasonable = sum(entry["native_reasonable_tp"] for entry in entries)
            fp = sum(entry["fp"] for entry in entries)
            miss = sum(row["disruptive"] and entry["first_index"] is None
                       for row, entry in zip(checked, entries, strict=True))
            safe_peaks = [{"shot": row["shot"], **entry["safe_peak"]}
                          for row, entry in zip(checked, entries, strict=True) if not row["disruptive"]]
            seed_result[arm] = {
                "n_disruptive": n_disruptive, "n_safe": n_safe, "raw_tp": raw,
                "reasonable_tp": reasonable, "native_reasonable_tp": native_reasonable,
                "very_early_tp": raw - native_reasonable, "miss": miss,
                "tardy": n_disruptive - raw - miss, "fp": fp, "tn": n_safe - fp,
                "raw_tpr": _rate(raw, n_disruptive), "reasonable_tpr": _rate(reasonable, n_disruptive),
                "fpr": _rate(fp, n_safe), "safe_peaks": safe_peaks,
                "safe_peak_probability": _distribution([peak["probability"] for peak in safe_peaks]),
                "strata": {name: _stratum_summary([[entry["strata"][name]] for entry in entries])
                           for name in STRATA},
            }
        result["per_seed"][str(seed)] = seed_result
    for arm in ARMS:
        result["mean_counts"][arm] = {name: _mean(result["per_seed"][str(seed)][arm][name] for seed in SEEDS)
                                      for name in ("raw_tp", "reasonable_tp", "native_reasonable_tp", "fp")}
        result["pooled_safe_peak_probability"][arm] = _distribution([
            _mean(row["seeds"][str(seed)][arm]["safe_peak"]["probability"] for seed in SEEDS)
            for row in checked if not row["disruptive"]
        ])
    for name in STRATA:
        result["pooled_strata"][name] = {
            arm: _stratum_summary([[row["seeds"][str(seed)][arm]["strata"][name] for seed in SEEDS]
                                   for row in checked]) for arm in ARMS}
    for control in CONTROLS:
        result["paired"][f"block_minus_{control}"] = _comparison(checked, control)
    result["source_gate"] = source_gate(result)
    return result
