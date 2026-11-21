"""Prespecified numeric interventions, not physical calibration or a policy.

All variants keep the same unavailable-channel mask. The floor is copied
from simple_data.fit_source_scaler, not selected on target performance.
Keeping Bt/ip tests preservation of operating levels; density_only tests
one suspect coordinate without asserting that its physical units are known.
"""
import numpy as np

from .io23 import RAW23


POLICIES = ('target_robust_scale', 'source_floor',
            'source_floor_preserve_controls', 'density_only')


def scale_variant(source_center, source_scale, statistics, policy):
    if policy not in POLICIES:
        raise ValueError('an explicit prespecified scale policy is required')
    arrays = [np.asarray(v, dtype=np.float32).copy() for v in
              (source_center, source_scale, statistics['center'], statistics['scale'])]
    if any(v.shape != (23,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('four finite flat23 center/scale vectors required')
    if any((v <= 0).any() for v in (arrays[1], arrays[3])):
        raise ValueError('positive source and target scales required')
    usable = np.asarray(statistics['usable']).copy()
    if usable.shape != (23,) or usable.dtype != bool:
        raise ValueError('boolean flat23 identifiability mask required')
    source_center, source_scale, target_center, target_scale = arrays
    if policy == 'target_robust_scale':
        center, scale = target_center, target_scale
    else:
        floored = np.maximum.reduce((target_scale, abs(target_center)*.01,
                                     np.full(23, 1e-6, np.float32)))
        if policy == 'density_only':
            center, scale = source_center, source_scale
            c = RAW23.index('ne_nG')
            center[c], scale[c] = target_center[c], floored[c]
        else:
            center, scale = target_center, floored
            if policy == 'source_floor_preserve_controls':
                for name in ('Bt', 'ip'):
                    c = RAW23.index(name)
                    center[c], scale[c] = source_center[c], source_scale[c]
    return dict(center=center, scale=scale, usable=usable)
