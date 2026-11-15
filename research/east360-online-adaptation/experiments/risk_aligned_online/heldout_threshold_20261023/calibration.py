"""Whole-safe-shot order statistic; no asserted guarantee under domain drift."""
from fractions import Fraction
import math

import numpy as np


def safe_threshold(peaks, alpha):
    x = np.asarray(peaks, dtype=np.float64)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all()
            or np.any((x < 0) | (x > 1)) or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError('nonempty finite probability peaks and 0<alpha<1 required')
    rank = math.ceil((len(x)+1) * (1-Fraction(str(alpha))))
    result = dict(safe_count=len(x), alpha=float(alpha), rank=rank,
                  status='FITTED', order_statistic=None, threshold=None)
    if rank > len(x):
        # The finite-sample rank is outside the observed order statistics.
        # Report no usable threshold, not zero-FPR success from never alarming.
        result['status'] = 'INSUFFICIENT_FOR_NONTRIVIAL_THRESHOLD'
        return result
    q = float(np.sort(x)[rank-1])
    # Existing scorer uses >= and float32 curves. Move one float32 ULP so a
    # tied peak is NOT an alarm; a float64-only ULP would cast back to the tie.
    threshold = float(np.nextafter(np.float32(q), np.float32(np.inf)))
    assert threshold > q
    result.update(order_statistic=q, threshold=threshold)
    return result
