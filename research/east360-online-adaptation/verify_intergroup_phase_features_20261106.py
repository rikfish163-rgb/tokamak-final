"""Independent scalar reference for synthetic/archived-prefix verification.

Inputs are time-major ``values[sample, channel]`` and boolean
``available[sample, channel]``, plus per-channel ``center`` and ``scale``.
The caller supplies ``m >= 63``, at least 23 channels, valid per-channel
nonzero scales, and finite observed values. Only the inclusive 64-sample prefix
ending at ``m`` is read; unobserved channel values are never read.
"""

import math

import numpy as np


_GROUPS = ((3, 4, 5), (14, 15, 16, 17), (19, 20, 21, 22))
_PAIRS = ((0, 1), (0, 2), (1, 2))
_LAGS = (4, 8, 16)
_SHIFTS = (7, 11, 13)
_MIN_NORM = 1e-12


def _latest_longest_run(flags):
    """Return (start, length), preferring the latest start for equal lengths."""
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for index, flag in enumerate(flags):
        if flag:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length >= best_length:
                best_start = current_start
                best_length = current_length
        else:
            current_length = 0
    return best_start, best_length


def _rotate_right(sequence, shift):
    """Return an exact right rotation without mutating or changing the values."""
    length = len(sequence)
    if length == 0:
        return []
    return [sequence[(index - shift) % length] for index in range(length)]


def _center_and_norm(sequence):
    mean = math.fsum(sequence) / len(sequence)
    centered = [value - mean for value in sequence]
    norm = math.sqrt(math.fsum(value * value for value in centered))
    return centered, norm


def _correlation(centered_a, norm_a, centered_b, norm_b):
    numerator = math.fsum(a * b for a, b in zip(centered_a, centered_b))
    return numerator / (norm_a * norm_b)


def reference(values, available, center, scale, m):
    """Compute within-group and asymmetric cross-phase features for one prefix.

    Return float64 arrays ``within``, ``real``, and ``shifted`` and a boolean
    ``supported`` array, each of length nine, plus integer ``common``. A phase
    coordinate is unsupported and stays zero when the longest common run is
    shorter than 49 increments or any required centered norm is <= 1e-12.
    """
    within = np.zeros(9, dtype=np.float64)
    real = np.zeros(9, dtype=np.float64)
    shifted = np.zeros(9, dtype=np.float64)
    supported = np.zeros(9, dtype=bool)
    window_start = m - 63
    # None signifies an unavailable increment, not a zero-activity increment.
    activities = [[None for _ in range(63)] for _ in _GROUPS]

    for group_index, channels in enumerate(_GROUPS):
        for increment in range(63):
            left = window_start + increment
            right = left + 1
            observed = True
            for channel in channels:
                if not available[left, channel] or not available[right, channel]:
                    observed = False
                    break
            if not observed:
                continue

            squared_increments = []
            for channel in channels:
                channel_center = float(center[channel])
                channel_scale = float(scale[channel])
                before = math.asinh((float(values[left, channel]) - channel_center) / channel_scale)
                after = math.asinh((float(values[right, channel]) - channel_center) / channel_scale)
                difference = after - before
                squared_increments.append(difference * difference)
            activities[group_index][increment] = math.sqrt(
                math.fsum(squared_increments) / len(channels)
            )

        observed_activity = []
        for activity in activities[group_index]:
            if activity is not None:
                observed_activity.append(activity)
        count = len(observed_activity)
        if count:
            mean = math.fsum(observed_activity) / count
            rms = math.sqrt(math.fsum(value * value for value in observed_activity) / count)
            population_sd = math.sqrt(
                math.fsum((value - mean) * (value - mean) for value in observed_activity) / count
            )
            within[3 * group_index] = math.log1p(rms)
            within[3 * group_index + 1] = math.log1p(population_sd)
            within[3 * group_index + 2] = count / 63.0

    common_flags = []
    for increment in range(63):
        common_flags.append(all(activities[group][increment] is not None for group in range(3)))
    run_start, common = _latest_longest_run(common_flags)
    result = {
        "within": within,
        "real": real,
        "shifted": shifted,
        "supported": supported,
        "common": common,
    }
    if common < 49:
        return result

    for pair_index, (group_a, group_b) in enumerate(_PAIRS):
        for lag_index, lag in enumerate(_LAGS):
            x = []
            plus = []
            minus = []
            for local_index in range(16, common - 16):
                index = run_start + local_index
                x.append(activities[group_a][index])
                plus.append(activities[group_b][index + lag])
                minus.append(activities[group_b][index - lag])

            x_centered, x_norm = _center_and_norm(x)
            plus_centered, plus_norm = _center_and_norm(plus)
            minus_centered, minus_norm = _center_and_norm(minus)
            if x_norm <= _MIN_NORM or plus_norm <= _MIN_NORM or minus_norm <= _MIN_NORM:
                continue

            index = pair_index * len(_LAGS) + lag_index
            plus_correlation = _correlation(x_centered, x_norm, plus_centered, plus_norm)
            minus_correlation = _correlation(x_centered, x_norm, minus_centered, minus_norm)
            real[index] = (plus_correlation - minus_correlation) / 2.0

            rotated_plus = _rotate_right(plus, _SHIFTS[pair_index])
            rotated_minus = _rotate_right(minus, _SHIFTS[pair_index])
            shifted_plus_centered, shifted_plus_norm = _center_and_norm(rotated_plus)
            shifted_minus_centered, shifted_minus_norm = _center_and_norm(rotated_minus)
            shifted_plus = _correlation(x_centered, x_norm, shifted_plus_centered, shifted_plus_norm)
            shifted_minus = _correlation(x_centered, x_norm, shifted_minus_centered, shifted_minus_norm)
            shifted[index] = (shifted_plus - shifted_minus) / 2.0
            supported[index] = True

    return result
