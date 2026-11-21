"""First-alarm block attribution and exact per-block arm reachability.

Times and disruptive-shot ``down`` are in seconds.  An alarm crosses when a
probability is greater than or equal to ``threshold``.  Blocks start at
``first, first + stride, ...``; samples before ``first`` are immutable.

``alarm_envelope`` takes a two-dimensional ``(arms, time_points)`` array.
An arm is selected independently for each complete or final partial block.
Safe shots must use ``down=None``.  A first-hit index of ``len(times)`` means
that no eligible alarm occurred.
"""

from numbers import Integral

import numpy as np


def _finite_array(values, name):
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _scalar(value, name):
    array = _finite_array(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return float(array)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _prepare(times, disruptive, down, threshold, first, stride):
    times = _finite_array(times, "times")
    if times.ndim != 1 or np.any(times[1:] <= times[:-1]):
        raise ValueError("times must be one-dimensional and strictly increasing")
    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError("disruptive must be a bool")
    threshold = _scalar(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    first = _integer(first, "first", 0)
    stride = _integer(stride, "stride", 1)
    n = len(times)
    if disruptive:
        down = _scalar(down, "down")
        try:
            with np.errstate(over="raise", invalid="raise"):
                lead = (down - times) * 1000.0
        except FloatingPointError as exc:
            raise ValueError("disruption lead is not finite") from exc
        eligible = lead >= -1e-6
    else:
        if down is not None:
            raise ValueError("safe shots require down=None")
        lead = None
        eligible = np.ones(n, dtype=bool)
    if first >= n:
        anchors = np.empty(0, dtype=np.int64)
    else:
        # A huge stride is equivalent to n and must not overflow int64.
        anchors = np.arange(first, n, min(stride, n), dtype=np.int64)
    return times, threshold, first, stride, anchors, lead, eligible


def _probabilities(values, n, dimensions, name):
    probs = _finite_array(values, name)
    if probs.ndim != dimensions or probs.shape[-1] != n:
        raise ValueError(f"{name} must have {dimensions} dimensions and {n} points")
    if dimensions == 2 and probs.shape[0] == 0:
        raise ValueError("at least one probability arm is required")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"{name} must lie between 0 and 1")
    return probs


def _prefix_before(block_hits, static_first):
    prefix = np.empty(len(block_hits), dtype=np.int64)
    if len(prefix):
        prefix[0] = static_first
        if len(prefix) > 1:
            prefix[1:] = np.minimum(static_first, np.minimum.accumulate(block_hits[:-1]))
    return prefix


def _suffix_after(block_hits, n):
    suffix = np.full(len(block_hits), n, dtype=np.int64)
    if len(suffix) > 1:
        suffix[:-1] = np.minimum.accumulate(block_hits[:0:-1])[::-1]
    return suffix


def _min3(a, b, c):
    return np.minimum(np.minimum(a, b), c)


def block_credit(times, native, adapted, *, disruptive, down, threshold,
                 first=135, stride=32):
    """Return strict and capped per-block first-alarm utility differences.

    Each block's telescoping difference replaces it after all earlier blocks
    have been adapted.  Isolated compares only this replacement to the native
    curve; leaveout compares the adapted curve to replacing only this block
    with native.  Work and memory are O(time_points + blocks).
    """
    times, threshold, first, stride, anchors, lead, eligible = _prepare(
        times, disruptive, down, threshold, first, stride
    )
    n = len(times)
    native = _probabilities(native, n, 1, "native")
    adapted = _probabilities(adapted, n, 1, "adapted")
    static_end = min(first, n)
    if not np.array_equal(native[:static_end], adapted[:static_end]):
        raise ValueError("native and adapted must match before first")

    indices = np.arange(n, dtype=np.int64)
    native_hits = np.where((native >= threshold) & eligible, indices, n)
    adapted_hits = np.where((adapted >= threshold) & eligible, indices, n)
    native_first = int(np.min(native_hits, initial=n))
    adapted_first = int(np.min(adapted_hits, initial=n))
    static_first = int(np.min(native_hits[:static_end], initial=n))

    if len(anchors):
        native_blocks = np.minimum.reduceat(native_hits, anchors)
        adapted_blocks = np.minimum.reduceat(adapted_hits, anchors)
    else:
        native_blocks = np.empty(0, dtype=np.int64)
        adapted_blocks = np.empty(0, dtype=np.int64)
    past_native = _prefix_before(native_blocks, static_first)
    past_adapted = _prefix_before(adapted_blocks, static_first)
    future_native = _suffix_after(native_blocks, n)
    future_adapted = _suffix_after(adapted_blocks, n)

    telescoping_before = _min3(past_adapted, native_blocks, future_native)
    telescoping_after = _min3(past_adapted, adapted_blocks, future_native)
    isolated_after = _min3(past_native, adapted_blocks, future_native)
    leaveout_before = _min3(past_adapted, native_blocks, future_adapted)

    if disruptive:
        # Clipping away from both decision boundaries avoids overflow in round
        # for otherwise finite, extreme clock values without changing either
        # comparison at 30 or 3000 ms.
        rounded = np.round(np.clip(lead, 0.0, 4000.0), 9)
        strict = np.r_[(rounded > 30.0).astype(np.int64), 0]
        capped = np.r_[((rounded > 30.0) & (rounded <= 3000.0)).astype(np.int64), 0]
    else:
        strict = np.r_[np.full(n, -1, dtype=np.int64), 0]
        capped = strict

    answer = {"anchor_indices": anchors, "native_first": native_first,
              "adapted_first": adapted_first}
    for name, utility in (("strict", strict), ("capped", capped)):
        answer[f"{name}_telescoping"] = (
            utility[telescoping_after] - utility[telescoping_before]
        )
        answer[f"{name}_isolated"] = utility[isolated_after] - utility[native_first]
        answer[f"{name}_leaveout"] = utility[adapted_first] - utility[leaveout_before]
    return answer


def alarm_envelope(times, probs, *, disruptive, down, threshold,
                   first=135, stride=32):
    """Return sorted feasible first hits under independent block-arm choices.

    Every arm must share its immutable prefix.  A later block can affect the
    result only when every earlier block has at least one silent arm.  For A
    arms and N points, time is O(A*N) and returned hits are already ordered.
    """
    times, threshold, first, stride, anchors, _lead, eligible = _prepare(
        times, disruptive, down, threshold, first, stride
    )
    n = len(times)
    probs = _probabilities(probs, n, 2, "probs")
    static_end = min(first, n)
    if not np.all(probs[:, :static_end] == probs[0, :static_end]):
        raise ValueError("all arms must match before first")
    static_hits = (probs[0, :static_end] >= threshold) & eligible[:static_end]
    if np.any(static_hits):
        return np.asarray([int(np.argmax(static_hits))], dtype=np.int64)

    result = []
    for start_index in anchors:
        start = int(start_index)
        end = min(start + stride, n)
        hits = (probs[:, start:end] >= threshold) & eligible[None, start:end]
        has_hit = np.any(hits, axis=1)
        if np.any(has_hit):
            offsets = np.argmax(hits[has_hit], axis=1)
            seen = np.zeros(end - start, dtype=bool)
            seen[offsets] = True
            result.extend((np.flatnonzero(seen) + start).tolist())
        if np.all(has_hit):
            return np.asarray(result, dtype=np.int64)
    result.append(n)
    return np.asarray(result, dtype=np.int64)
