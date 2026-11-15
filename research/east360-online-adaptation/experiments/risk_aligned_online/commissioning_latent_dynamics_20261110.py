"""Causal, diagonal latent-trend fitting and per-shot adaptation.

All indices are zero-based. A block at q contains q+1 through q+25; it is
complete at q+25, and its update may first affect the feature at q+26. The
last incomplete block is never fitted or scored. The functions use only
the supplied encoder states and observation masks, and keep no shot state.
"""

import numpy as np


LAG = 32
HORIZON = 25
START = 127
STRIDE = 32

_RIDGE = 0.001
_DECAY = 2.0 ** (-1.0 / 4.0)
_FUTURE_SCALE = (np.arange(1, HORIZON + 1, dtype=np.float64) / LAG)[:, None]


def _shot_arrays(hidden, observed):
    hidden = np.asarray(hidden, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if hidden.ndim != 2 or hidden.shape[1] == 0:
        raise ValueError("hidden must have shape (N, D) with D > 0")
    if observed.ndim != 1 or len(observed) != len(hidden):
        raise ValueError("observed must have shape (N,)")
    return hidden, observed


def _block_terms(hidden, observed, origin, reverse=False):
    """Return per-observed-target mean x*x and x*y for one complete block."""
    future = hidden[origin + 1 : origin + HORIZON + 1]
    mask = observed[origin + 1 : origin + HORIZON + 1]
    if reverse:
        future = future[::-1]
        mask = mask[::-1]
    design = _FUTURE_SCALE[mask] * (hidden[origin] - hidden[origin - LAG])
    target = future[mask] - hidden[origin]
    return np.mean(design * design, axis=0), np.mean(design * target, axis=0)


def fit_prior(shots):
    """Fit the frozen (gain, precision) prior from ended encoder-state shots.

    A shot contributes the mean of its eligible, complete block statistics;
    such shots then have equal mass regardless of duration. Shots with no
    eligible blocks contribute no mass. An eligible block has observed origin
    and lagged origin and at least one observed target. If none are eligible,
    the prior is zero gain and ridge precision. At least one shot is required
    to establish the feature dimension.
    """
    dimension = None
    energy_sum = None
    cross_sum = None
    usable_shots = 0
    for hidden, observed in shots:
        hidden, observed = _shot_arrays(hidden, observed)
        if dimension is None:
            dimension = hidden.shape[1]
            energy_sum = np.zeros(dimension, dtype=np.float64)
            cross_sum = np.zeros(dimension, dtype=np.float64)
        elif hidden.shape[1] != dimension:
            raise ValueError("all shots must have the same feature dimension")

        shot_energy = np.zeros(dimension, dtype=np.float64)
        shot_cross = np.zeros(dimension, dtype=np.float64)
        block_count = 0
        for origin in range(START, len(hidden) - HORIZON, STRIDE):
            if not (observed[origin] and observed[origin - LAG]):
                continue
            if not np.any(observed[origin + 1 : origin + HORIZON + 1]):
                continue
            energy, cross = _block_terms(hidden, observed, origin)
            shot_energy += energy
            shot_cross += cross
            block_count += 1
        if block_count:
            energy_sum += shot_energy / block_count
            cross_sum += shot_cross / block_count
            usable_shots += 1

    if dimension is None:
        raise ValueError("at least one shot is needed to establish D")
    if usable_shots:
        energy_sum /= usable_shots
        cross_sum /= usable_shots
    precision = energy_sum + _RIDGE
    gain = np.clip(cross_sum / precision, -2.0, 2.0)
    return gain, precision


def causal_drift(hidden, observed, prior, *, mode):
    """Return float32 (N, D) drift features and forecast/update statistics.

    ``mode`` is ``static``, ``real``, or ``reverse``. In reverse mode, each
    completed block's future states and masks are reversed together for the
    update; forecast errors always use the actual, unreversed future. The
    gain at every origin issues the complete 25-step forecast before any of
    that block's targets are used. Accumulated update terms decay only when a
    new eligible block is incorporated; the prior itself never decays.

    ``mature_blocks`` counts complete blocks with both origin observations
    and at least one observed target, even if the shot ends at q+25 and no
    later feature can use the block. Both squared-error sums score these same
    blocks against the real order, summing over observed targets and channels;
    ``forecast_weight`` is the number of those scalar target values. Their
    ratios to the weight are per-scalar MSEs when the weight is nonzero.
    ``changed_steps`` counts output rows differing from the static output
    after float32 conversion. The prior and all per-shot state remain local.
    """
    if mode not in ("static", "real", "reverse"):
        raise ValueError("mode must be 'static', 'real', or 'reverse'")
    hidden, observed = _shot_arrays(hidden, observed)
    prior_gain = np.asarray(prior[0], dtype=np.float64)
    precision = np.asarray(prior[1], dtype=np.float64)
    dimension = hidden.shape[1]
    if prior_gain.shape != (dimension,) or precision.shape != (dimension,):
        raise ValueError("prior gain and precision must have shape (D,)")
    if not np.all(np.isfinite(prior_gain)) or not np.all(np.isfinite(precision)) or np.any(precision <= 0):
        raise ValueError("prior gain must be finite and precision must be finite and positive")

    length = len(hidden)
    difference = np.zeros_like(hidden)
    if length > LAG:
        valid = observed[LAG:] & observed[:-LAG]
        difference[LAG:][valid] = hidden[LAG:][valid] - hidden[:-LAG][valid]

    drift = np.empty_like(hidden)
    gain = prior_gain.copy()
    accumulated_xx = np.zeros(dimension, dtype=np.float64)
    accumulated_xy = np.zeros(dimension, dtype=np.float64)
    segment_start = 0
    stats = {
        "changed_steps": 0,
        "mature_blocks": 0,
        "squared_forecast_error_sum": 0.0,
        "squared_hold_error_sum": 0.0,
        "forecast_weight": 0.0,
    }

    for origin in range(START, length - HORIZON, STRIDE):
        if not (observed[origin] and observed[origin - LAG]):
            continue
        target_mask = observed[origin + 1 : origin + HORIZON + 1]
        count = int(np.count_nonzero(target_mask))
        if not count:
            continue

        origin_state = hidden[origin]
        future = hidden[origin + 1 : origin + HORIZON + 1]
        prediction = origin_state + _FUTURE_SCALE * difference[origin] * gain
        actual = future[target_mask]
        forecast_error = prediction[target_mask] - actual
        hold_error = origin_state - actual
        stats["mature_blocks"] += 1
        stats["squared_forecast_error_sum"] += float(np.sum(forecast_error * forecast_error))
        stats["squared_hold_error_sum"] += float(np.sum(hold_error * hold_error))
        stats["forecast_weight"] += float(count * dimension)

        release = origin + HORIZON + 1
        if mode != "static" and release < length:
            drift[segment_start:release] = (HORIZON / LAG) * difference[segment_start:release] * gain
            energy, cross = _block_terms(hidden, observed, origin, reverse=(mode == "reverse"))
            accumulated_xx = _DECAY * accumulated_xx + energy
            accumulated_xy = _DECAY * accumulated_xy + cross
            gain = np.clip((precision * prior_gain + accumulated_xy) / (precision + accumulated_xx), -2.0, 2.0)
            segment_start = release

    drift[segment_start:] = (HORIZON / LAG) * difference[segment_start:] * gain
    drift = drift.astype(np.float32)
    if mode != "static":
        static_drift = ((HORIZON / LAG) * difference * prior_gain).astype(np.float32)
        stats["changed_steps"] = int(np.count_nonzero(np.any(drift != static_drift, axis=1)))
    return drift, stats
