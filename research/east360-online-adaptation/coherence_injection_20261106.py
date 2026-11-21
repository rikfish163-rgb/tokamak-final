"""Fixed synthetic diagnostic perturbations and paired raw-logit metrics.

These pure NumPy helpers neither read shots nor run a model.  The caller must
exclude raw trajectories shorter than N and report a None injection as missing
support; a missing observation is never imputed by this module.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


GROUPS = (("mir", (3, 4, 5)), ("sxr", (14, 15, 16, 17)), ("xuv", (19, 20, 21, 22)))
CASES = ("clean", "mir", "sxr", "xuv", "common", "staggered")
BLOCK = 32
ONSET = 256
OFF = 512
N = 640
AMP = 0.60
ARMS = ("static", "single", "coherent", "stale", "wrong_family", "scalar", "oracle")


def _case(case: str) -> None:
    if not isinstance(case, str) or case not in CASES:
        raise ValueError(f"unsupported case: {case!r}")


def _array(value: object, name: str) -> np.ndarray:
    # np.asarray would silently discard a masked array's own missingness.
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must not be a masked array")
    return np.asarray(value)


def injection(mask: np.ndarray, shot: int, case: str) -> np.ndarray | None:
    """Return the signed additive perturbation, or None for missing support.

    ``mask`` is a binary (N, 23) array, where one means observed.  Full
    availability is required only within the specific intervals and channels
    designated by the case.  A clean case designates no interval.
    """
    _case(case)
    if isinstance(shot, (bool, np.bool_)) or not isinstance(shot, Integral):
        raise TypeError("shot must be an integer")
    observed = _array(mask, "mask")
    if observed.shape != (N, 23) or observed.dtype.kind not in "biuf":
        raise ValueError("mask must be a binary array of shape (640, 23)")
    if not np.all((observed == 0) | (observed == 1)):
        raise ValueError("mask must contain only zero and one")

    regions = []
    for index, (name, channels) in enumerate(GROUPS):
        if case in (name, "common", "staggered"):
            start = ONSET + (index * BLOCK if case == "staggered" else 0)
            if not np.all(observed[start:OFF, channels] == 1):
                return None
            regions.append((start, channels))

    result = np.zeros((N, 23), dtype=np.float32)
    amplitude = AMP if shot % 2 == 0 else -AMP
    for start, channels in regions:
        result[start:OFF, channels] = amplitude
    return result


def _logits(value: object, name: str, length: int) -> np.ndarray:
    raw = _array(value, name)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric logit curve")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite raw logits")
    return result


def _distance(curve: np.ndarray, reference: np.ndarray, window: np.ndarray) -> float | None:
    if not np.any(window):
        return None
    # Accumulate in extended precision to avoid overflow of a finite sum.
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(curve[window].astype(np.longdouble) - reference[window])
        value = float(np.mean(difference, dtype=np.longdouble))
    if not np.isfinite(value):
        raise ValueError("mean absolute logit distance is not representable as a finite float")
    return value


def score_case(
    times: np.ndarray, clean: np.ndarray, curves: dict[str, np.ndarray], case: str
) -> dict:
    """Score aligned curves using absolute sample clocks, not array indices.

    A strictly increasing subset of clocks is allowed if it spans [320, 512)
    and contains at least one mature-window observation.  Counts are shared
    across arms.  Optional windows without samples return None for every arm.

    Single-family mature scores use the original clean native logits; clean,
    common, and staggered controls use the static logits for that same case.
    Pre-onset and recovery movement always use that case's static logits.
    """
    _case(case)
    clock = _array(times, "times")
    if clock.ndim != 1 or not len(clock) or clock.dtype.kind not in "iu":
        raise ValueError("times must be a nonempty one-dimensional integer clock")
    if (
        clock[0] < 0
        or clock[-1] >= N
        or np.any(clock[1:] <= clock[:-1])
    ):
        raise ValueError("times must be strictly increasing sample clocks in [0, 640)")
    mature = (clock >= ONSET + 2 * BLOCK) & (clock < OFF)
    if clock[0] > ONSET + 2 * BLOCK or clock[-1] < OFF - 1 or not np.any(mature):
        raise ValueError("times must span [320, 512) and contain a mature sample")
    if not isinstance(curves, dict) or set(curves) != set(ARMS):
        raise ValueError("curves must be a dict containing exactly the specified arms")
    clean_curve = _logits(clean, "clean", len(clock))
    arms = {arm: _logits(curves[arm], arm, len(clock)) for arm in ARMS}
    static = arms["static"]
    target = clean_curve if case in ("mir", "sxr", "xuv") else static
    windows = {
        "mature": mature,
        "nonanchor": mature & (clock % BLOCK != 0),
        "pre": clock < ONSET,
        "recovery": (clock >= OFF + BLOCK) & (clock < N),
    }
    scores = {
        name: {
            arm: _distance(curve, target if name in ("mature", "nonanchor") else static, window)
            for arm, curve in arms.items()
        }
        for name, window in windows.items()
    }
    scores["counts"] = {name: int(np.count_nonzero(window)) for name, window in windows.items()}
    return scores
