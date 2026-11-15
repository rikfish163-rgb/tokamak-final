"""Post-reveal, whole-shot utilities for an independent finite patch menu.

The caller supplies the already-revealed HDF5 outcome: ``disruptive`` is the
verified IsDisrupt boolean and ``down`` is DownTime in seconds for disruptive
shots.  This module performs no I/O and is not an online prediction policy.
Times and down are seconds; horizon and reported leads are milliseconds.

An action changes only the at-most-32 samples *after* its release index.  The
single-edit targets are changes relative to the native whole shot; they are not
additive action values.  The menu oracle instead considers all independent
per-block choices and reports every attainable first crossing, with ``len(times)``
as the no-crossing sentinel.  It is an offline upper bound, not a deployable
selection rule.
"""

import math

import numpy as np


_WIDTH = 32
_TEMPERATURE = 0.05
_PRETERMINAL_TOLERANCE_MS = 1e-6
_HORIZON_TOLERANCE_MS = 1e-6


def _real_array(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real array")
    return array


def _as_float64(values, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable as finite float64")
    return result


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _integer_vector(values, name, length):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer vector") from exc
    if array.ndim != 1 or len(array) != length or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be an integer vector of length {length}")
    return array


def _same_values(left, right):
    """Compare mixed integer/floating inputs without lossy NumPy promotion."""
    lk, rk = left.dtype.kind, right.dtype.kind
    if lk in "iu" and rk in "iu" and lk != rk:
        return all(int(a) == int(b) for a, b in zip(left, right))
    if (lk in "iu" and rk == "f") or (lk == "f" and rk in "iu"):
        integers, floats = (left, right) if lk in "iu" else (right, left)
        for integer, value in zip(integers, floats):
            numerator, denominator = value.as_integer_ratio()
            if int(integer) * denominator != numerator:
                return False
        return True
    return np.array_equal(left, right)


def _computation_dtype(*arrays):
    dtype = np.result_type(*(a.dtype for a in arrays), np.float64)
    if any(a.dtype.kind in "iu" for a in arrays):
        dtype = np.result_type(dtype, np.longdouble)
    return dtype


def _menu(native, clock, curves, allowed):
    native = _real_array(native, "native")
    curves = _real_array(curves, "curves")
    allowed = np.asarray(allowed)
    if native.ndim != 1 or not len(native):
        raise ValueError("native must be a nonempty vector")
    if curves.ndim != 3 or curves.shape[2] != _WIDTH or curves.shape[1] < 1:
        raise ValueError("curves must have shape [blocks, actions, 32]")
    blocks, actions, _ = curves.shape
    if allowed.shape != (blocks, actions) or allowed.dtype != np.bool_:
        raise ValueError("allowed must be a boolean [blocks, actions] array")
    if not allowed[:, 0].all():
        raise ValueError("action 0 must always be available")
    clock = _integer_vector(clock, "clock", blocks)
    if np.any(clock < 0) or np.any(clock >= len(native)):
        raise ValueError("release indices must be inside the shot")
    clock = clock.astype(np.intp, copy=False)
    starts = clock + 1
    ends = np.minimum(starts + _WIDTH, len(native))
    order = np.argsort(starts, kind="stable")
    last_end = 0
    for b in order:
        start, end = int(starts[b]), int(ends[b])
        if start < last_end:
            raise ValueError("post-release blocks overlap")
        last_end = end
        if not _same_values(curves[b, 0, :end - start], native[start:end]):
            raise ValueError("action 0 must exactly reproduce the native block")
    return native, clock, curves, allowed, starts, ends, order


def replay(native, clock, curves, allowed, choices):
    """Copy native and splice the selected, available post-release blocks.

    Input block order may be arbitrary, but real post-release spans may not
    overlap.  Padding is never read as part of a block.  Action 0 is required
    to match native exactly on each real span, even for an empty final block.
    """
    native, _, curves, allowed, starts, ends, _ = _menu(native, clock, curves, allowed)
    choices = _integer_vector(choices, "choices", len(starts))
    if np.any(choices < 0) or np.any(choices >= curves.shape[1]):
        raise ValueError("choice outside the action menu")
    choices = choices.astype(np.intp, copy=False)
    if not allowed[np.arange(len(choices)), choices].all():
        raise ValueError("selected action is unavailable")
    dtype = np.result_type(native.dtype, curves.dtype)
    # Retain all int64/uint64 values when mixing integer and floating inputs.
    if dtype.kind == "f" and (native.dtype.kind in "iu" or curves.dtype.kind in "iu"):
        dtype = np.result_type(dtype, np.longdouble)
    result = np.array(native, dtype=dtype, copy=True)
    for b, (start, end) in enumerate(zip(starts, ends)):
        result[start:end] = curves[b, choices[b], :end - start]
    return result


def _clock_and_truth(times, length, disruptive, down, tau):
    times = _as_float64(_real_array(times, "times"), "times")
    if times.shape != (length,) or not length or np.any(times[1:] <= times[:-1]):
        raise ValueError("times must be a nonempty, strictly increasing aligned vector")
    if not isinstance(disruptive, (bool, np.bool_)):
        raise ValueError("disruptive must be a verified boolean")
    disruptive = bool(disruptive)
    tau = _finite_scalar(tau, "tau")
    if disruptive:
        down = _finite_scalar(down, "down")
        with np.errstate(over="ignore", invalid="ignore"):
            raw_leads = (down - times) * 1000.0
        if not np.isfinite(raw_leads).all():
            raise ValueError("lead times must be finite")
        eligible = raw_leads >= -_PRETERMINAL_TOLERANCE_MS
    else:
        raw_leads = None
        eligible = np.ones(length, dtype=bool)
    return times, disruptive, tau, raw_leads, eligible


def _rounded_near(raw_leads, boundary, rounded=None):
    """Round near a comparison boundary without overflowing huge leads."""
    if rounded is None:
        rounded = raw_leads.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        near = np.flatnonzero(np.abs(raw_leads - boundary) <= 2e-9)
    for i in near:
        rounded[i] = round(float(raw_leads[i]), 9)
    return rounded


def _score_first(first, raw_leads, disruptive, length):
    first = int(first) if first < length else None
    lead = round(float(raw_leads[first]), 9) if disruptive and first is not None else None
    return dict(first=first, lead_ms=lead, timely=bool(lead is not None and lead > 10),
                false_alarm=bool(not disruptive and first is not None))


def score(times, curve, disruptive, down, tau):
    """Score the first eligible crossing, independent of the soft horizon."""
    curve = _real_array(curve, "curve")
    if curve.ndim != 1 or not len(curve):
        raise ValueError("curve must be a nonempty vector")
    curve = curve.astype(_computation_dtype(curve), copy=False)
    _, disruptive, tau, leads, eligible = _clock_and_truth(times, len(curve), disruptive, down, tau)
    hits = np.flatnonzero((curve >= tau) & eligible)
    return _score_first(hits[0] if len(hits) else len(curve), leads, disruptive, len(curve))


def _cdf(running, tau):
    # logaddexp also handles finite inputs whose subtraction or scaling overflows.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = (running - tau) / _TEMPERATURE
        return np.exp(-np.logaddexp(0.0, -scaled))


def _utility(cdf, role, count, subtract_before):
    if role == "safe":
        return 1.0 - cdf[..., 0]
    mean = np.mean(cdf[..., :count], axis=-1)
    return mean - cdf[..., -1] if subtract_before else mean


def _running_at_queries(native, running, block, start, end, query):
    """Running maxima for all actions, evaluated only at utility coordinates."""
    values = np.broadcast_to(running[query], (len(block), len(query))).copy()
    if start == end:
        return values
    prefix = running[start - 1] if start else -np.inf
    block_running = np.maximum.accumulate(block[:, :end - start], axis=1)
    inside = (query >= start) & (query < end)
    if inside.any():
        values[:, inside] = np.maximum(prefix, block_running[:, query[inside] - start])
    after = query >= end
    if after.any():
        tail = np.maximum.accumulate(native[end:int(query[after].max()) + 1])
        values[:, after] = np.maximum(
            np.maximum(prefix, block_running[:, -1])[:, None], tail[query[after] - end])
    return values


def outcomes(native, times, clock, curves, allowed, *, tau, disruptive, down, horizon):
    """Return single-edit hard/soft targets and the exact independent-menu oracle.

    Hard reward is the change in timely detection for disruptive shots or the
    reduction in false alarms for safe shots.  Soft reward is the change in
    first-passage utility, using the role fixed by the *native* shot.  The
    canonical recovery interval uses rounded leads with
    10 < lead <= horizon + 1e-6 ms;
    a missing pre-window sample contributes zero, and an empty interval has
    zero utility.  Unavailable actions and all identity actions have zero targets.

    The oracle fields are outcome-specific: ``oracle_timely`` is false on safe
    shots, and ``oracle_no_false_alarm`` is false on disruptive shots.
    """
    native, _, curves, allowed, starts, ends, order = _menu(native, clock, curves, allowed)
    dtype = _computation_dtype(native, curves)
    native = native.astype(dtype, copy=False)
    curves = curves.astype(dtype, copy=False)
    length = len(native)
    _, disruptive, tau, leads, eligible = _clock_and_truth(times, length, disruptive, down, tau)
    horizon = _finite_scalar(horizon, "horizon")
    native_hits = (native >= tau) & eligible
    next_hit = np.full(length + 1, length, dtype=np.intp)
    next_hit[:-1] = np.minimum.accumulate(
        np.where(native_hits, np.arange(length), length)[::-1])[::-1]
    baseline = _score_first(next_hit[0], leads, disruptive, length)

    timely_at = np.zeros(length + 1, dtype=bool)
    if disruptive:
        rounded = _rounded_near(leads, 10.0)
        timely_at[:-1] = rounded > 10.0
    else:
        rounded = None

    if not disruptive:
        role, query, count, subtract_before = "safe", np.array([length - 1]), 1, False
    elif baseline["timely"]:
        first = baseline["first"]
        query = np.arange(max(0, first - 8), first + 1)
        role, count, subtract_before = "protect", len(query), False
    else:
        # Re-evaluate the other comparison boundary with the same rounding rule.
        upper = horizon + _HORIZON_TOLERANCE_MS
        rounded = _rounded_near(leads, upper, rounded)
        canonical = np.flatnonzero((rounded > 10) & (rounded <= upper))
        count = len(canonical)
        subtract_before = bool(count and canonical[0] > 0)
        query = (np.append(canonical, canonical[0] - 1) if subtract_before else canonical)
        role = "recover"

    running = np.maximum.accumulate(native)
    base_utility = (_utility(_cdf(running[query], tau), role, count, subtract_before)
                    if len(query) else 0.0)
    hard = np.zeros(allowed.shape, dtype=np.int8)
    soft = np.zeros(allowed.shape, dtype=np.float64)
    block_firsts = []
    for b, (start, end) in enumerate(zip(starts, ends)):
        start, end = int(start), int(end)
        if start == end:
            block_firsts.append(np.full(curves.shape[1], length, dtype=np.intp))
            continue
        crossings = (curves[b, :, :end - start] >= tau) & eligible[start:end]
        firsts = np.where(crossings.any(axis=1), start + crossings.argmax(axis=1), length)
        block_firsts.append(firsts)
        prefix_first = next_hit[0] if next_hit[0] < start else length
        whole_first = np.minimum(firsts, min(prefix_first, next_hit[end]))
        if disruptive:
            change = timely_at[whole_first].astype(np.int8) - int(baseline["timely"])
        else:
            change = int(baseline["false_alarm"]) - (whole_first < length).astype(np.int8)
        hard[b, allowed[b]] = change[allowed[b]]
        if len(query):
            sampled = _running_at_queries(native, running, curves[b], start, end, query)
            change_soft = _utility(_cdf(sampled, tau), role, count, subtract_before) - base_utility
            soft[b, allowed[b]] = change_soft[allowed[b]]
    hard[:, 0] = 0
    soft[:, 0] = 0.0

    reachable = set()
    cursor = 0
    alive = True
    for b in order:
        start, end = int(starts[b]), int(ends[b])
        if next_hit[cursor] < start:
            reachable.add(int(next_hit[cursor]))
            alive = False
            break
        firsts = block_firsts[b][allowed[b]]
        reachable.update(int(i) for i in firsts if i < length)
        if not np.any(firsts == length):
            alive = False
            break
        cursor = end
    if alive:
        reachable.add(int(next_hit[cursor]))
    reachable = sorted(reachable)
    return dict(baseline=baseline, hard=hard, soft=soft, target=0.75 * hard + 0.25 * soft,
                reachable_first=reachable,
                oracle_timely=bool(disruptive and any(timely_at[i] for i in reachable)),
                oracle_no_false_alarm=bool(not disruptive and length in reachable))
