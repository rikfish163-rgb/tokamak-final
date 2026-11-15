"""Post-seal, outcome-stratified FP/TP matching and descriptive summaries.

This module does not load outcomes, predictions, models, or data. The caller
must provide one first-report anchor per shot, *after* prediction is sealed,
and is responsible for deriving class tags from the authoritative outcomes.
No outcome is used to generate a prediction. This module cannot verify the
prediction seal, outcome authority, future time point, or scalar parity.
The caller supplies base-minus-reverse deltas, and ``future_residual`` as the
future base-minus-reverse effect minus the anchor recent effect, using the same
scalar (or None if +24 is unavailable). Each rank split is matched anew.

An admissible edge has the same ``index < 320`` phase and passes all five
inclusive calipers. Its cost is the arithmetic mean of the five absolute
caliper-scaled distances, in [0, 1]. The assignment maximizes pair count first
and then minimizes the total cost. Shot IDs are sorted before assignment, so
input order does not affect the result. The bootstrap resamples fixed pairs;
it does not account for rematching or establish causal or prospective validity.

Rows missing valid main fields are reported and excluded. Missing/non-finite
optional inner or future fields only exclude the affected paired statistic.
Duplicate shots and invalid shot/rank identifiers raise rather than choosing
an order-dependent first row. See ``match_summary`` for the output contract.
"""

import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment


_CLASSES = ("fp", "reasonable_tp")
_SPLITS = (("full_after20", 21, 360), ("front_after20", 21, 190), ("back", 191, 360))
_MAIN_FIELDS = (
    "index", "logit", "coverage", "motion_recent", "motion_old",
    "delta_recent", "delta_old",
)
_OPTIONAL_FIELDS = ("delta_recent_inner", "delta_old_inner", "future_residual")
_BOOTSTRAPS = 10000
_SEED = 131


def _number(row, field):
    """Return a finite float or a reason; never coerce strings or booleans."""
    value = row.get(field)
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, "invalid_type"
    try:
        number = float(value)
    except OverflowError:
        return None, "nonfinite"
    if not math.isfinite(number):
        return None, "nonfinite"
    return number, None


def _main_values(row):
    values, problems = {}, {}
    for field in _MAIN_FIELDS:
        if field == "index":
            value = row.get(field)
            if value is None:
                problem = "missing"
            elif isinstance(value, bool) or not isinstance(value, Integral):
                problem = "invalid_type"
            elif value < 0:
                problem = "out_of_range"
            else:
                value, problem = int(value), None
        else:
            value, problem = _number(row, field)
            if problem is None and (
                (field == "coverage" and not 0 <= value <= 1)
                or (field in ("motion_recent", "motion_old") and value < 0)
            ):
                problem = "out_of_range"
        if problem is not None:
            problems[field] = problem
        else:
            values[field] = value
    return values, problems


def _check_rows(rows):
    checked, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each anchor row must be a dict")
        for field in ("shot", "rank"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{field} must be an integer")
        shot, rank = int(row["shot"]), int(row["rank"])
        if shot in seen:
            raise ValueError(f"duplicate shot: {shot}")
        seen.add(shot)
        checked.append((shot, rank, row))
    return sorted(checked, key=lambda entry: entry[0])


def _distance(tp, fp):
    if (tp["index"] < 320) != (fp["index"] < 320):
        return None

    logit = abs(tp["logit"] - fp["logit"])
    coverage = abs(tp["coverage"] - fp["coverage"])
    if logit > 0.25 or coverage > 0.05:
        return None

    a, b = tp["index"] + 128, fp["index"] + 128
    high, low = max(a, b), min(a, b)
    if high > 2 * low:
        return None
    index = math.log(high / low) / math.log(2)

    motion = []
    for field in ("motion_recent", "motion_old"):
        a, b = tp[field] + 0.01, fp[field] + 0.01
        high, low = max(a, b), min(a, b)
        log_ratio = math.log(high / low)
        if log_ratio > math.log(3):
            return None
        motion.append(log_ratio / math.log(3))

    return (logit / 0.25 + index + coverage / 0.05 + sum(motion)) / 5


def _match(tps, fps):
    if not tps or not fps:
        return []

    # Every valid cost is at most one. One more real match saves a penalty
    # greater than the largest possible difference in all real-edge costs.
    penalty = min(len(tps), len(fps)) + 1
    forbidden = penalty * (len(tps) + 1)
    costs = np.full((len(tps), len(fps) + len(tps)), float(penalty))
    costs[:, :len(fps)] = forbidden
    for i, tp in enumerate(tps):
        for j, fp in enumerate(fps):
            distance = _distance(tp, fp)
            if distance is not None:
                costs[i, j] = distance

    ti, fi = linear_sum_assignment(costs)
    return [
        (tps[int(i)], fps[int(j)], float(costs[i, j]))
        for i, j in zip(ti, fi)
        if j < len(fps) and costs[i, j] != forbidden
    ]


def _difference(a, b, name):
    value = a - b
    if not math.isfinite(value):
        raise ValueError(f"nonfinite derived {name}; input magnitudes overflowed")
    return value


def _describe(values, unavailable):
    result = {
        "n": len(values),
        "mean": None,
        "median": None,
        "ci95_mean": None,
        "tp_higher": 0,
        "tp_lower": 0,
        "ties": 0,
        "unavailable_pairs": unavailable,
    }
    if not values:
        return result

    x = np.asarray(values, dtype=float)
    n = len(x)
    ordered = sorted(values)
    if n % 2:
        median = ordered[n // 2]
    else:
        a, b = ordered[n // 2 - 1:n // 2 + 1]
        width = b - a
        median = a + width / 2 if math.isfinite(width) else a / 2 + b / 2

    # Scaling keeps both very large finite values and subnormals representable
    # through resampling; the mean of scaled observations remains within [-1, 1].
    scale = float(np.max(np.abs(x)))
    scaled = x / scale if scale else x
    mean = float(np.clip(np.mean(scaled), -1, 1) * scale)
    rng = np.random.default_rng(_SEED)
    bootstrap = np.empty(_BOOTSTRAPS, dtype=float)
    for start in range(0, _BOOTSTRAPS, 512):
        size = min(512, _BOOTSTRAPS - start)
        indices = rng.integers(0, n, size=(size, n))
        bootstrap[start:start + size] = np.clip(np.mean(scaled[indices], axis=1), -1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        ci = np.quantile(bootstrap, [0.025, 0.975]) * scale
    if not math.isfinite(mean) or not math.isfinite(median) or not np.isfinite(ci).all():
        raise ValueError("nonfinite derived summary; input magnitudes overflowed")
    result.update(
        mean=mean,
        median=float(median),
        ci95_mean=[float(ci[0]), float(ci[1])],
        tp_higher=int(np.count_nonzero(x > 0)),
        tp_lower=int(np.count_nonzero(x < 0)),
        ties=int(np.count_nonzero(x == 0)),
    )
    return result


def _paired_summaries(pairs):
    names = ("recent", "old", "contrast", "inner_contrast", "future_residual")
    values = {name: [] for name in names}
    unavailable = {name: {"missing": 0, "invalid": 0} for name in names}
    for tp, fp, _ in pairs:
        recent = _difference(tp["delta_recent"], fp["delta_recent"], "recent")
        old = _difference(tp["delta_old"], fp["delta_old"], "old")
        values["recent"].append(recent)
        values["old"].append(old)
        tp_contrast = _difference(tp["delta_recent"], tp["delta_old"], "TP contrast")
        fp_contrast = _difference(fp["delta_recent"], fp["delta_old"], "FP contrast")
        values["contrast"].append(_difference(tp_contrast, fp_contrast, "contrast"))

        for name, fields in (
            ("inner_contrast", ("delta_recent_inner", "delta_old_inner")),
            ("future_residual", ("future_residual",)),
        ):
            optional = [[record["optional"][field] for field in fields] for record in (tp, fp)]
            problems = [reason for group in optional for _, reason in group if reason]
            if problems:
                # A pair with both invalid and missing values is counted once as invalid.
                kind = "invalid" if any(reason != "missing" for reason in problems) else "missing"
                unavailable[name][kind] += 1
                continue
            if name == "inner_contrast":
                tp_inner = _difference(optional[0][0][0], optional[0][1][0], name)
                fp_inner = _difference(optional[1][0][0], optional[1][1][0], name)
                value = _difference(tp_inner, fp_inner, name)
            else:
                value = _difference(optional[0][0][0], optional[1][0][0], name)
            values[name].append(value)
    return {name: _describe(values[name], unavailable[name]) for name in names}


def _summarize_split(checked, low, high):
    groups = {name: [] for name in _CLASSES}
    excluded = {
        "other_class": 0,
        "invalid_class_tag": [],
        "invalid_main": {name: [] for name in _CLASSES},
    }
    for shot, rank, row in checked:
        if not low <= rank <= high:
            continue
        tag = row.get("class_tag")
        if not isinstance(tag, str):
            reason = "missing" if tag is None else "invalid_type"
            excluded["invalid_class_tag"].append({"shot": shot, "reason": reason})
            continue
        if tag not in groups:
            excluded["other_class"] += 1
            continue
        values, problems = _main_values(row)
        if problems:
            excluded["invalid_main"][tag].append({"shot": shot, "fields": problems})
            continue
        values.update(
            shot=shot,
            optional={field: _number(row, field) for field in _OPTIONAL_FIELDS},
        )
        groups[tag].append(values)

    pairs = _match(groups["reasonable_tp"], groups["fp"])
    count = len(pairs)
    return {
        "rank_range": [low, high],
        "n_eligible": {name: len(groups[name]) for name in _CLASSES},
        "n_pairs": count,
        "n_matched": {name: count for name in _CLASSES},
        "n_unmatched": {name: len(groups[name]) - count for name in _CLASSES},
        "excluded": excluded,
        "pairs": [
            {"tp_shot": tp["shot"], "fp_shot": fp["shot"], "cost": cost}
            for tp, fp, cost in pairs
        ],
        "paired_summary": _paired_summaries(pairs),
    }


def _gate(splits):
    full, front, back = (splits[name] for name in ("full_after20", "front_after20", "back"))
    full_contrast = full["paired_summary"]["contrast"]
    front_mean = front["paired_summary"]["contrast"]["mean"]
    back_mean = back["paired_summary"]["contrast"]["mean"]
    ci = full_contrast["ci95_mean"]
    checks = {
        "full_at_least_12_pairs": full["n_pairs"] >= 12,
        "front_at_least_5_pairs": front["n_pairs"] >= 5,
        "back_at_least_5_pairs": back["n_pairs"] >= 5,
        "full_mean_contrast_positive": full_contrast["mean"] is not None and full_contrast["mean"] > 0,
        "full_ci_lower_positive": ci is not None and ci[0] > 0,
        "front_mean_contrast_positive": front_mean is not None and front_mean > 0,
        "back_mean_contrast_positive": back_mean is not None and back_mean > 0,
    }
    enough = all(checks[name] for name in (
        "full_at_least_12_pairs", "front_at_least_5_pairs", "back_at_least_5_pairs"
    ))
    passed = all(checks.values())
    return {
        "status": "passed_tentative" if passed else ("not_passed" if enough else "insufficient_pairs"),
        "passed": passed,
        "promotes_method": False,
        "checks": checks,
    }


def match_summary(rows):
    """Return JSON-ready independent split matches and paired TP-minus-FP summaries.

    Required main fields are ``shot``, ``rank``, ``class_tag``, ``index``,
    ``logit``, ``coverage``, both motions, and both outer deltas. Missing or
    invalid optional fields are counted per pair, with invalid taking priority
    over missing if both occur. Each shot's contrast is recent minus old, and
    all paired differences are TP minus FP. Positive paired values mean TP is
    higher, not necessarily better. The future residual is diagnostic and is
    not gated. A one-pair percentile interval is necessarily degenerate.

    Outputs contain ``splits`` (with counts, exclusions, pair IDs and costs,
    and paired summaries), the bootstrap definition, and a tentative ``gate``.
    Derived numeric overflow raises ValueError instead of emitting non-JSON NaN
    or infinity. Each bootstrap uses 10,000 fixed-pair mean resamples and seed
    131, independently initialized for each split/statistic.
    """
    checked = _check_rows(rows)
    splits = {name: _summarize_split(checked, low, high) for name, low, high in _SPLITS}
    return {
        "splits": splits,
        "matching": {
            "assignment": "maximum cardinality, then minimum total normalized cost",
            "distance": "mean of the five absolute caliper-scaled distances",
            "phase": "same index < 320",
            "calipers": {
                "logit_abs": 0.25,
                "index_log_ratio_abs": math.log(2),
                "coverage_abs": 0.05,
                "motion_log_ratio_abs": math.log(3),
                "index_offset": 128,
                "motion_offset": 0.01,
            },
        },
        "bootstrap": {
            "replicates": _BOOTSTRAPS,
            "seed": _SEED,
            "confidence": 0.95,
            "method": "paired percentile of mean",
            "rematched": False,
            "interpretation": "descriptive, conditional on the fixed matches",
            "single_pair_ci": "degenerate at the observed difference",
        },
        "gate": _gate(splits),
    }
