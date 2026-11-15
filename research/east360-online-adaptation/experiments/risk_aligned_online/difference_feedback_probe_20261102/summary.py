"""CPU-only shot-weighted diagnostic statistics and preregistered gates.

Inputs are already evaluated records; this module never reads experiment data,
recomputes BCE, or performs model inference. These screening gates are not
statistical significance tests or independent validation.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real


ARMS = (
    "centered_real",
    "difference_real",
    "difference_reverse",
    "difference_hold",
)
GROUPS = ("class0", "class1", "safe")

POSITIVE_SHOTS_MIN = 6
SAFE_NEGATIVE_SHOTS_MIN = 8
IMPROVEMENT_THRESHOLD = -1e-5
HARM_TOLERANCE = 1e-6
COMPARATOR_MARGIN = 1e-6
HOLD_TOLERANCE = 1e-10


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    count = len(values)
    try:
        return math.fsum(values) / count
    except OverflowError:
        try:
            return math.fsum(value / count for value in values)
        except OverflowError:
            # A mean of finite numbers is finite, even when a sum of
            # nearly-maximal floats overflows due to rounding.
            scale = max(abs(value) for value in values)
            scaled = math.fsum((value / scale) / count for value in values)
            return scale * min(1.0, max(-1.0, scaled))


def _group_summary(by_shot: Mapping[int, list[float]]) -> dict:
    shot_means = [_mean(values) for values in by_shot.values()]
    return {
        "n_points": sum(len(values) for values in by_shot.values()),
        "n_shots": len(by_shot),
        "mean_delta_bce": _mean(shot_means),
    }


def _required_arm_deltas(row: Mapping) -> dict[str, float]:
    results = row.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("row results must be a mapping containing all four arms")

    deltas = {}
    for arm in ARMS:
        arm_result = results.get(arm)
        if not isinstance(arm_result, Mapping) or "delta_bce" not in arm_result:
            raise ValueError(f"missing {arm} arm or its delta_bce")
        value = arm_result["delta_bce"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{arm} delta_bce must be a finite number")
        try:
            delta = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"{arm} delta_bce must be a finite number") from None
        if not math.isfinite(delta):
            raise ValueError(f"{arm} delta_bce must be a finite number")
        deltas[arm] = delta
    return deltas


def summarize(rows: Iterable[Mapping]) -> dict:
    """Summarize four arms and return explicit, fail-closed screening gates.

    Each (shot, query) must occur once and have all four finite delta_bce
    values. A shot's disruptive flag must be consistent across its records.
    Every label group first averages points within a shot, then averages shot
    means. The balanced mean weights the two label means equally. The safe
    group is computed separately from non-disruptive, label-0 records.

    Coverage counts distinct disruptive label-1 and non-disruptive label-0
    shots. If coverage is insufficient, ``decision`` reports that fact even
    when another gate fails; every gate, including the pointwise hold check,
    remains individually visible. No missing group is treated as zero.
    """
    by_arm = {arm: {group: defaultdict(list) for group in GROUPS} for arm in ARMS}
    seen = set()
    shot_status = {}
    positive_disruptive_shots = set()
    negative_safe_shots = set()
    hold_zero = True

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a mapping")
        try:
            shot = row["shot"]
            query = row["query"]
            disruptive = row["disruptive"]
            label = row["label"]
        except KeyError as exc:
            raise ValueError(f"missing required row field {exc.args[0]}") from None
        if (
            isinstance(shot, bool)
            or not isinstance(shot, Integral)
            or isinstance(query, bool)
            or not isinstance(query, Integral)
        ):
            raise ValueError("shot and query must be integers")
        if not isinstance(disruptive, bool):
            raise ValueError("disruptive must be a bool")
        if (
            isinstance(label, bool)
            or not isinstance(label, Integral)
            or label not in (0, 1)
        ):
            raise ValueError("label must be integer 0 or 1")

        shot, query, label = int(shot), int(query), int(label)
        key = (shot, query)
        if key in seen:
            raise ValueError(f"duplicate (shot, query): {key}")
        if shot in shot_status and shot_status[shot] != disruptive:
            raise ValueError(f"inconsistent disruptive status for shot {shot}")
        seen.add(key)
        shot_status[shot] = disruptive

        deltas = _required_arm_deltas(row)
        group_names = [f"class{label}"]
        if disruptive and label == 1:
            positive_disruptive_shots.add(shot)
        if not disruptive and label == 0:
            group_names.append("safe")
            negative_safe_shots.add(shot)
        for arm, delta in deltas.items():
            for group in group_names:
                by_arm[arm][group][shot].append(delta)
        hold_zero = hold_zero and abs(deltas["difference_hold"]) <= HOLD_TOLERANCE

    arms = {}
    for arm in ARMS:
        arm_summary = {group: _group_summary(by_arm[arm][group]) for group in GROUPS}
        class0 = arm_summary["class0"]["mean_delta_bce"]
        class1 = arm_summary["class1"]["mean_delta_bce"]
        arm_summary["balanced_delta_bce"] = (
            _mean([class0, class1])
            if class0 is not None and class1 is not None
            else None
        )
        arms[arm] = arm_summary

    enough_positive = len(positive_disruptive_shots) >= POSITIVE_SHOTS_MIN
    enough_safe = len(negative_safe_shots) >= SAFE_NEGATIVE_SHOTS_MIN
    coverage = {
        "positive_disruptive_n_shots": len(positive_disruptive_shots),
        "negative_safe_n_shots": len(negative_safe_shots),
        "positive_disruptive": enough_positive,
        "negative_safe": enough_safe,
        "passed": enough_positive and enough_safe,
    }

    real = arms["difference_real"]
    class0 = real["class0"]["mean_delta_bce"]
    class1 = real["class1"]["mean_delta_bce"]
    safe = real["safe"]["mean_delta_bce"]
    balanced = real["balanced_delta_bce"]
    centered = arms["centered_real"]["balanced_delta_bce"]
    reverse = arms["difference_reverse"]["balanced_delta_bce"]
    gates = {
        "coverage": coverage["passed"],
        "hold_zero": bool(seen) and hold_zero,
        "class1": class1 is not None and class1 < IMPROVEMENT_THRESHOLD,
        "class0": class0 is not None and class0 <= HARM_TOLERANCE,
        "safe": safe is not None and safe <= HARM_TOLERANCE,
        "balanced": balanced is not None and balanced < IMPROVEMENT_THRESHOLD,
        "vs_centered_real": (
            balanced is not None
            and centered is not None
            and centered - balanced >= COMPARATOR_MARGIN
        ),
        "vs_difference_reverse": (
            balanced is not None
            and reverse is not None
            and reverse - balanced >= COMPARATOR_MARGIN
        ),
    }
    gates["passed"] = all(gates.values())

    if not coverage["passed"]:
        decision = "coverage_insufficient"
    elif not gates["hold_zero"]:
        decision = "implementation_check_failed"
    elif gates["passed"]:
        decision = "pass"
    else:
        decision = "screen_failed"

    return {"arms": arms, "coverage": coverage, "gates": gates, "decision": decision}
