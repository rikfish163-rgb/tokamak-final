"""Causal, per-channel roughness references for one shot at a time.

Indices are zero-based. A block at q=127+32k contains q+1 through q+25;
its target is available after row q+25 and can first affect row q+26.
References are fitted only from the reference shots supplied by the caller.
"""

import numpy as np


_START = 127
_STRIDE = 32
_HORIZON = 25
_FLOOR = 0.001
_HUBER_TRANSITION = np.log(2.0)
_MAX_DISPLACEMENT = np.log(4.0)


def _arrays(z, mask):
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if z.ndim != 2 or z.shape[1] == 0 or mask.shape != z.shape:
        raise ValueError("z and mask must have the same shape (N, C), C > 0")
    mask = mask & np.isfinite(z)
    return np.where(mask, z, 0.0), mask


def coordinates(values, available, center, scale):
    """Return asinh source coordinates and their finite observation mask.

    Inputs have shape (N, C), with source center and positive scale of shape
    (C,). Unavailable or nonfinite observations are masked and set to zero;
    the caller is responsible for marking unavailable channels.
    """
    values, mask = _arrays(values, available)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if center.shape != (values.shape[1],) or scale.shape != center.shape:
        raise ValueError("center and scale must have shape (C,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("center must be finite and scale must be finite and positive")
    normalized = np.zeros_like(values)
    np.subtract(values, center, out=normalized, where=mask)
    np.divide(normalized, scale, out=normalized, where=mask)
    return np.arcsinh(normalized), mask


def _block_log_roughness(z, mask, origin):
    block = z[origin + 1:origin + _HORIZON + 1]
    valid = mask[origin + 1:origin + _HORIZON + 1]
    triplets = valid[:-2] & valid[1:-1] & valid[2:]
    second = np.abs(block[2:] - 2.0 * block[1:-1] + block[:-2])
    observed = np.count_nonzero(triplets, axis=0) >= 12
    result = np.zeros(z.shape[1], dtype=np.float64)
    for channel in np.flatnonzero(observed):
        result[channel] = np.log(max(_FLOOR, np.median(second[triplets[:, channel], channel])))
    return result, observed


def fit_prior(shots):
    """Fit (log_ref, supported) from supplied safe reference shots.

    Each complete future block contributes a channel's log median absolute
    second difference only if at least 12 of its 23 triplets are observed.
    Its amplitude is floored at 0.001. Each shot contributes its median log
    statistic per channel, then shots receive equal weight in a second median.
    At least two shots must contribute to support a channel; unsupported
    channels have a zero reference and contribute no feature. Each input entry
    must represent a distinct, safe shot: this API has no shot IDs or outcome
    data with which to verify that condition. An empty list cannot establish
    the channel dimension.
    """
    by_channel = None
    for z, mask in shots:
        z, mask = _arrays(z, mask)
        if by_channel is None:
            by_channel = [[] for _ in range(z.shape[1])]
        elif len(by_channel) != z.shape[1]:
            raise ValueError("all shots must have the same channel dimension")
        within_shot = [[] for _ in by_channel]
        for origin in range(_START, len(z) - _HORIZON, _STRIDE):
            target, observed = _block_log_roughness(z, mask, origin)
            for channel in np.flatnonzero(observed):
                within_shot[channel].append(target[channel])
        for channel, targets in enumerate(within_shot):
            if targets:
                by_channel[channel].append(np.median(targets))

    if by_channel is None:
        raise ValueError("at least one reference shot is needed to establish C")
    supported = np.array([len(targets) >= 2 for targets in by_channel], dtype=bool)
    log_ref = np.zeros(len(by_channel), dtype=np.float64)
    for channel in np.flatnonzero(supported):
        log_ref[channel] = np.median(by_channel[channel])
    return log_ref, supported


def _contrast(z, mask):
    length, channels = z.shape
    sums = np.concatenate((np.zeros((1, channels)), np.cumsum(z, axis=0)), axis=0)
    counts = np.concatenate((np.zeros((1, channels), dtype=np.int64),
                             np.cumsum(mask, axis=0)), axis=0)
    end = np.arange(1, length + 1)
    recent_start = np.maximum(end - 16, 0)
    past_start = np.maximum(end - 64, 0)
    recent_count = counts[end] - counts[recent_start]
    past_count = counts[recent_start] - counts[past_start]
    recent = np.divide(sums[end] - sums[recent_start], recent_count,
                       out=np.zeros_like(z), where=recent_count > 0)
    past = np.divide(sums[recent_start] - sums[past_start], past_count,
                     out=np.zeros_like(z), where=past_count > 0)
    return np.where((recent_count >= 8) & (past_count >= 24), recent - past, 0.0)


def summarize(z, mask, prior, *, mode):
    """Return causal float32 (N, C) contrast features and scalar counters.

    Each score is asinh((recent mean - preceding mean) / exp(log_state)).
    Recent and preceding windows have up to 16 and 48 rows and need at least
    8 and 24 observed values. Unsupported channels always contribute zero.
    The state resets to the supplied prior on every call. ``static`` never
    updates; ``real`` uses each observed block; ``first`` uses only the first
    observed block independently per channel; ``lag`` uses the immediately
    preceding block, only if both that block and the current block have an
    observed target in the channel.

    ``mature_blocks`` counts temporally complete 25-point blocks, including
    unestimable ones. ``update_blocks`` counts complete blocks at whose
    release at least one supported channel receives an informed update,
    even if the update is numerically zero or the shot ends on that block.
    ``changed_rows`` counts rows whose float32 feature differs from static in
    at least one channel. These are bookkeeping counts, not alarm utility.
    """
    if mode not in ("static", "real", "first", "lag"):
        raise ValueError("mode must be 'static', 'real', 'first', or 'lag'")
    z, mask = _arrays(z, mask)
    log_ref = np.asarray(prior[0], dtype=np.float64)
    supported = np.asarray(prior[1], dtype=bool)
    if log_ref.shape != (z.shape[1],) or supported.shape != log_ref.shape:
        raise ValueError("prior arrays must have shape (C,)")
    if not np.all(np.isfinite(log_ref[supported])):
        raise ValueError("supported log references must be finite")
    log_ref = np.where(supported, log_ref, 0.0)
    contrast = _contrast(z, mask)
    contrast[:, ~supported] = 0.0
    static = np.arcsinh(contrast / np.exp(log_ref)).astype(np.float32)
    features = static.copy()
    state = log_ref.copy()
    first_done = np.zeros(z.shape[1], dtype=bool)
    previous_target = np.zeros(z.shape[1], dtype=np.float64)
    previous_observed = np.zeros(z.shape[1], dtype=bool)
    stats = {"update_blocks": 0, "changed_rows": 0, "mature_blocks": 0}

    for origin in range(_START, len(z) - _HORIZON, _STRIDE):
        stats["mature_blocks"] += 1
        if mode == "static":
            continue
        target, observed = _block_log_roughness(z, mask, origin)
        eligible = supported & observed
        if mode == "first":
            eligible &= ~first_done
        elif mode == "lag":
            eligible &= previous_observed
        used_target = previous_target if mode == "lag" else target
        if np.any(eligible):
            gradient = np.clip(state[eligible] - used_target[eligible],
                               -_HUBER_TRANSITION, _HUBER_TRANSITION)
            gradient += 0.1 * (state[eligible] - log_ref[eligible])
            state[eligible] = np.clip(state[eligible] - 0.25 * gradient,
                                      log_ref[eligible] - _MAX_DISPLACEMENT,
                                      log_ref[eligible] + _MAX_DISPLACEMENT)
            first_done[eligible] = True
            stats["update_blocks"] += 1
        previous_target, previous_observed = target, observed
        release = origin + _HORIZON + 1
        end = min(release + _STRIDE, len(z))
        features[release:end] = np.arcsinh(contrast[release:end] / np.exp(state)).astype(np.float32)

    stats["changed_rows"] = int(np.count_nonzero(np.any(features != static, axis=1)))
    return features, stats
