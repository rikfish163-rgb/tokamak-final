"""Causal two-channel roughness controls with a common observation schedule.

The caller supplies source-coordinate pairs and safe reference shots.  Blocks
start at q=127+32k, contain rows q+1 through q+25, and first affect q+26.
All state is temporary to a single :func:`summarize_pair` call.
"""

import numpy as np

from . import commissioning_roughness_20261110 as roughness


def _pair(z, mask):
    z, mask = roughness._arrays(z, mask)
    if z.shape[1] != 2:
        raise ValueError("z and mask must have shape (N, 2)")
    return z, mask


def _reference_scale(log_ref):
    log_ref = np.asarray(log_ref, dtype=np.float64)
    if log_ref.shape != (2,) or not np.all(np.isfinite(log_ref)):
        raise ValueError("log_ref must be a finite array of shape (2,)")
    with np.errstate(over="ignore", under="ignore"):
        scale = np.exp(log_ref)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("exp(log_ref) must be finite and positive")
    return log_ref, scale


def _crossfit_residual(values):
    residual = np.empty_like(values)
    for target in range(2):
        predictor = 1 - target
        for parity in (0, 1):
            train = values[parity::2]
            test = values[1 - parity::2]
            x, y = train[:, predictor], train[:, target]
            coefficient = np.dot(x, y) / (np.dot(x, x) + 1.0)
            residual[1 - parity::2, target] = test[:, target] - coefficient * test[:, predictor]
    return residual


def _block_targets(z, mask, q, scale):
    block = z[q + 1:q + roughness._HORIZON + 1]
    valid = mask[q + 1:q + roughness._HORIZON + 1]
    triplets = valid[:-2] & valid[1:-1] & valid[2:]
    joint = np.all(triplets, axis=1)
    if np.count_nonzero(joint) < 12:
        return None

    curvature = (block[2:] - 2.0 * block[1:-1] + block[:-2])[joint]
    standardized = curvature / scale
    shifted = standardized.copy()
    shifted[:, 1] = np.roll(shifted[:, 1], len(shifted) // 2)
    amplitudes = np.stack((
        np.median(np.abs(curvature), axis=0),
        np.median(np.abs(_crossfit_residual(standardized) * scale), axis=0),
        np.median(np.abs(_crossfit_residual(shifted) * scale), axis=0),
    ))
    targets = np.log(np.maximum(amplitudes, roughness._FLOOR))
    left, right = standardized.T
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    coherence = float(abs(np.dot(left, right)) / denominator) if denominator > 0 else 0.0
    return targets, coherence


def fit_reference(shots, log_ref):
    """Fit three two-channel log offsets from caller-provided safe shots.

    ``shots`` contains distinct ``(z, mask)`` pairs, each with shape ``(N, 2)``;
    masked or nonfinite values are unavailable.  The caller must ensure that
    every entry is a distinct safe shot: these arrays carry no shot ID or
    outcome. Repeated uses of the identical ``z`` object are rejected when
    eligible, but copied or separately loaded duplicates cannot be detected.

    A complete block contributes only when at least 12 of its 23 second-
    difference rows have all three samples observed in both channels. Raw
    amplitude is the median absolute signed curvature. For each residual
    amplitude, curvature is divided by ``exp(log_ref)``; each channel is
    predicted from the other by a through-origin ridge fit with penalty 1.0,
    training on even joint rows and predicting odd rows, and conversely.
    The shifted control first rolls the second standardized column by half
    the number of joint rows. Residuals are converted back to source units;
    all amplitudes are floored at 0.001 before taking logarithms.

    The result has shape ``(3, 2)`` in raw, aligned-residual, shifted-residual
    order: its entries are medians over shots of each shot's median block estimates.
    At least two distinct provided shots must contribute.  The JSON-safe
    receipt counts ``provided_shots``, ``eligible_shots``, ``eligible_blocks``,
    and all temporally complete ``mature_blocks``.
    """
    _, scale = _reference_scale(log_ref)
    per_shot = []
    contributing_inputs = []
    receipt = {"provided_shots": 0, "eligible_shots": 0,
               "eligible_blocks": 0, "mature_blocks": 0}
    for original_z, original_mask in shots:
        receipt["provided_shots"] += 1
        z, mask = _pair(original_z, original_mask)
        block_targets = []
        for q in range(roughness._START, len(z) - roughness._HORIZON, roughness._STRIDE):
            receipt["mature_blocks"] += 1
            block = _block_targets(z, mask, q, scale)
            if block is not None:
                block_targets.append(block[0])
        if block_targets:
            if any(original_z is previous for previous in contributing_inputs):
                raise ValueError("reference shots must be distinct; duplicate z object")
            contributing_inputs.append(original_z)
            receipt["eligible_shots"] += 1
            receipt["eligible_blocks"] += len(block_targets)
            per_shot.append(np.median(np.stack(block_targets), axis=0))
    if len(per_shot) < 2:
        raise ValueError("at least two distinct safe shots with eligible blocks are required")
    return np.median(np.stack(per_shot), axis=0), receipt


def summarize_pair(z, mask, log_ref, offsets):
    """Return causal features and JSON-safe counters for a two-channel shot.

    ``log_ref`` is the original log scale, shape ``(2,)``; ``offsets`` is the
    ``(3, 2)`` output of :func:`fit_reference`. The returned float32 ``(N, 2)``
    arrays are keyed by ``static``, ``raw_matched``, ``aligned``, ``shifted``,
    and ``stale``. All retain the original signed recent-minus-past numerator.
    Each feature is ``asinh(contrast / exp(state))``, with the old 16-row
    recent and preceding 48-row means and their respective 8/24 minimum
    observation counts. The latter four arms start at ``log_ref`` on each
    call and update together on the same jointly observed blocks. Each target
    is recentered as ``log_ref + target - offset``. The explicit Huber gradient
    uses transition ``log(2)``, regularizer 0.1, step size 0.25, and a final
    state bound of ``log_ref +/- log(4)``. Stale uses the aligned offset and
    the aligned target on the first eligible block, then the immediately
    preceding eligible observed target thereafter, including across gaps.

    ``mature_blocks`` counts every complete block, ``eligible_blocks`` counts
    those with at least 12 jointly observed curvature triplets, and
    ``emitted_updates`` counts eligible updates with a q+26 row in this shot.
    The median raw/aligned/shifted scales are componentwise medians of the
    floored block amplitudes in source units. ``median_abs_curvature_coherence``
    is the median absolute uncentered cosine of the two standardized curvature
    columns (zero when either is identically zero). These diagnostics are
    ``None`` when no block is eligible; they do not establish independence.
    """
    z, mask = _pair(z, mask)
    log_ref, scale = _reference_scale(log_ref)
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (3, 2) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets must be a finite array of shape (3, 2)")

    contrast = roughness._contrast(z, mask)
    static = np.arcsinh(contrast / scale).astype(np.float32)
    names = ("raw_matched", "aligned", "shifted", "stale")
    features = {"static": static, **{name: static.copy() for name in names}}
    states = np.broadcast_to(log_ref, (4, 2)).copy()
    previous_aligned = None
    all_targets, coherences = [], []
    mature_blocks = emitted_updates = 0

    for q in range(roughness._START, len(z) - roughness._HORIZON, roughness._STRIDE):
        mature_blocks += 1
        block = _block_targets(z, mask, q, scale)
        release = q + roughness._HORIZON + 1
        if block is not None:
            targets, coherence = block
            all_targets.append(targets)
            coherences.append(coherence)
            stale = targets[1] if previous_aligned is None else previous_aligned
            active_targets = np.vstack((targets, stale))
            active_offsets = offsets[[0, 1, 2, 1]]
            recentered = log_ref + active_targets - active_offsets
            gradient = np.clip(states - recentered,
                               -roughness._HUBER_TRANSITION, roughness._HUBER_TRANSITION)
            gradient += 0.1 * (states - log_ref)
            states = np.clip(states - 0.25 * gradient,
                             log_ref - roughness._MAX_DISPLACEMENT,
                             log_ref + roughness._MAX_DISPLACEMENT)
            previous_aligned = targets[1].copy()
            emitted_updates += int(release < len(z))

        end = min(release + roughness._STRIDE, len(z))
        for i, name in enumerate(names):
            features[name][release:end] = np.arcsinh(
                contrast[release:end] / np.exp(states[i])).astype(np.float32)

    stats = {"mature_blocks": mature_blocks, "eligible_blocks": len(all_targets),
             "emitted_updates": emitted_updates,
             "median_abs_curvature_coherence": None,
             "median_raw_scale": None, "median_aligned_scale": None,
             "median_shifted_scale": None}
    if all_targets:
        median_scales = np.median(np.exp(np.stack(all_targets)), axis=0)
        stats["median_abs_curvature_coherence"] = float(np.median(coherences))
        for name, row in zip(("median_raw_scale", "median_aligned_scale", "median_shifted_scale"),
                             median_scales):
            stats[name] = row.tolist()
    return features, stats
