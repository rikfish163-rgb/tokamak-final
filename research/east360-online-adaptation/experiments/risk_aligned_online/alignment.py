"""Label-free, ended-shot robust statistics for a diagnostic alignment arm.

This estimates a numerical distribution, not a physical unit conversion.
Constant/missing calibration channels are explicitly marked unusable. Their
placeholder center=0/scale=1 is never evidence of successful calibration;
the consumer must mask these channels at inference.
"""
import numpy as np


def fit_ended_shot_statistics(shots, *, ended_ids):
    shots = list(shots)
    ids = [int(s['shot']) for s in shots]
    if not shots or not set(ids).issubset(set(ended_ids)):
        raise ValueError('only explicitly ended shots may fit target statistics')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate calibration shot')
    for s in shots:
        if s['device'] != 'EAST' or s['values'].ndim != 2 or s['values'].shape[1] != 23:
            raise ValueError('EAST flat23 signals required')
        if s['available'].shape != s['values'].shape:
            raise ValueError('matching availability required')
    center = np.zeros(23, np.float32)
    scale = np.ones(23, np.float32)
    usable = np.zeros(23, bool)
    counts = np.zeros(23, np.int64)
    for c in range(23):
        samples = []
        for s in shots:
            mask = s['available'][:, c] & np.isfinite(s['values'][:, c])
            v = s['values'][mask, c]
            if len(v):
                # Match the existing source scaler's approximate per-shot
                # sampling density; long discharges must not dominate.
                samples.append(v[::max(1, len(v)//128)])
        if not samples:
            continue
        v = np.concatenate(samples)
        counts[c] = len(v)
        q25, median, q75 = np.quantile(v, [.25, .5, .75])
        spread = q75-q25
        # No supported scale can be inferred from a zero interquartile range.
        # Report non-identifiability rather than amplify numerical noise.
        # Do not impose an absolute epsilon: channels have different physical
        # units, so a very small but representable IQR remains informative.
        if spread <= 0:
            continue
        if not np.isfinite(np.float32(spread)) or np.float32(spread) <= 0:
            raise ValueError(f'channel {c}: scale not representable in float32')
        center[c], scale[c], usable[c] = median, spread, True
    return dict(center=center, scale=scale, usable=usable, counts=counts,
                calibration_ids=ids)
