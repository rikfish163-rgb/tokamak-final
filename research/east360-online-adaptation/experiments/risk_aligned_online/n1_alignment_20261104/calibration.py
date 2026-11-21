"""Safe-shot numerical n=1 amplitude calibration; not a physical unit conversion."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import h5py
import numpy as np

from .. import io23

FEATURE = 'n=1 amplitude'
CHANNEL = io23.RAW23.index(FEATURE)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def source_reference(train_ids):
    """Read every TRAIN outcome, then only the n=1 signal of HDF-safe shots.

    The source estimate is the median of per-safe-shot medians over all finite
    stored samples, including zeros. No source/target normalization is fitted.
    """
    if not isinstance(train_ids, list):
        raise TypeError('source TRAIN IDs must be a list in canonical order')
    canonical = io23.source_roles()['train']
    if (len(canonical) != 1353 or len(set(canonical)) != 1353
            or any(type(sid) is not int for sid in train_ids) or train_ids != canonical):
        raise ValueError('exact canonical 1353-shot source TRAIN list required')
    safe_ids, medians = [], {}
    for sid in train_ids:
        path = io23.path_for('J-TEXT', sid)
        if not path.is_file():
            raise FileNotFoundError(f'missing source TRAIN HDF file: {path}')
        disruptive, _ = io23.read_outcome('J-TEXT', sid)
        if disruptive:
            continue
        with h5py.File(path, 'r') as handle:
            values = np.asarray(handle[f'data/{FEATURE}'][()])
        if (values.ndim != 1 or not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.complexfloating)):
            raise ValueError(f'source shot {sid} must have a one-dimensional real signal')
        finite = values[np.isfinite(values)].astype(np.float64, copy=False)
        if not finite.size:
            raise ValueError(f'source safe shot {sid} has no finite {FEATURE} observations')
        median = float(np.median(finite))
        if not math.isfinite(median):
            raise ValueError(f'source safe shot {sid} has a nonfinite median')
        safe_ids.append(sid)
        medians[sid] = median
    if not safe_ids:
        raise ValueError('no HDF-safe source TRAIN shots')
    value = _positive(float(np.median(list(medians.values()))), 'source reference')
    return dict(value=value, source_safe_ids=safe_ids, per_shot_medians=medians,
                source_train_ids=train_ids.copy(), source_data_feature=FEATURE)


def fit_gain(reference, past, truth, allowed_ids):
    """Compute a safe-only gain from exactly 20 caller-revealed ended shots.

    ``truth[sid]`` is the existing ``io23.read_outcome`` pair (bool, down).
    The caller must have sealed each shot and established that ``allowed_ids``
    are the locked first 20. This function performs no file or outcome access.
    """
    if not isinstance(reference, Mapping) or 'value' not in reference:
        raise ValueError('source reference with a value is required')
    source = _positive(reference['value'], 'source reference')
    if (not isinstance(allowed_ids, (list, tuple)) or len(allowed_ids) != 20
            or any(type(sid) is not int for sid in allowed_ids)
            or len(set(allowed_ids)) != 20):
        raise ValueError('exactly twenty distinct integer calibration IDs required')
    ids = list(allowed_ids)
    if (not isinstance(past, Mapping) or not isinstance(truth, Mapping)
            or any(type(sid) is not int for sid in past)
            or any(type(sid) is not int for sid in truth)
            or set(past) != set(ids) or set(truth) != set(ids)):
        raise ValueError('past and truth must match precisely the allowed twenty IDs')
    safe_ids, medians = [], {}
    for sid in ids:
        record, outcome = past[sid], truth[sid]
        if (not isinstance(outcome, (tuple, list)) or len(outcome) != 2
                or type(outcome[0]) is not bool):
            raise ValueError(f'shot {sid} requires a boolean HDF outcome and down time')
        disruptive, down = outcome
        if (disruptive and down is None) or (down is not None and
                (isinstance(down, (bool, np.bool_)) or not isinstance(down, Real)
                 or not math.isfinite(float(down)))):
            raise ValueError(f'shot {sid} requires a finite disruption time when disruptive')
        if (not isinstance(record, Mapping) or 'values' not in record or 'available' not in record
                or ('device' in record and record['device'] != 'EAST')
                or ('shot' in record and (type(record['shot']) is not int or record['shot'] != sid))):
            raise ValueError(f'shot {sid} requires the matching EAST input record')
        values, available = record['values'], record['available']
        if (not isinstance(values, np.ndarray) or not isinstance(available, np.ndarray)
                or values.ndim != 2 or values.shape[1:] != (23,) or not len(values)
                or values.dtype != np.float32 or available.shape != values.shape
                or available.dtype != np.bool_ or not np.isfinite(values[available]).all()):
            raise ValueError(f'shot {sid} requires finite observed float32 N x 23 inputs and boolean mask')
        if disruptive:
            continue
        observed = values[available[:, CHANNEL], CHANNEL].astype(np.float64, copy=False)
        if not observed.size:
            raise ValueError(f'safe shot {sid} has no observed {FEATURE} values')
        median = float(np.median(observed))
        if not math.isfinite(median):
            raise ValueError(f'safe shot {sid} has a nonfinite median')
        safe_ids.append(sid)
        medians[sid] = median
    if len(safe_ids) < 2:
        raise ValueError('at least two HDF-safe ended shots required')
    target = _positive(float(np.median(list(medians.values()))), 'target reference')
    gain = _positive(source / target, 'gain')
    return dict(gain=gain, source_reference=source, target_reference=target,
                target_safe_ids=safe_ids, calibration_ids=ids,
                per_shot_target_medians=medians,
                method='ratio_of_equal_shot_safe_medians',
                physical_unit_conversion_claim=False)
