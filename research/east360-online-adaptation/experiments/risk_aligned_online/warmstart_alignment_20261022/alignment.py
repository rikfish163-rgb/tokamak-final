"""Signal-only affine transport into the frozen source model's raw coordinates.

Unlike replacing asinh by z-score at inference, this leaves the trained
preprocessor unchanged. Mean/std transport implements the statistical idea
of paper131 EFD, not its PGFE-U feature extraction or source training.
MNM and mode_number_n (uncertified extraction equivalence across devices) are
never rescaled. Weighted-average mode numbers need not be categorical.
An unidentifiable scale leaves the existing input untouched
and is explicitly reported: a constant calibration channel is NOT missing.
"""
import numpy as np

from ..io23 import RAW23

POLICIES = ('moments_all', 'robust_all', 'density_moments', 'density_gain')
SIGNAL_KEYS = {'shot', 'device', 'values', 'available'}
EXCLUDED = {'MNM', 'mode_number_n'}


def _arrays(record):
    x, m = np.asarray(record['values']), np.asarray(record['available'])
    if (x.ndim != 2 or x.shape[1] != 23 or not len(x)
            or x.dtype != np.float32 or m.shape != x.shape or m.dtype != bool):
        raise ValueError('nonempty float32 N-by-23 signals and boolean mask required')
    if not np.isfinite(x[m]).all():
        raise ValueError('available observations must be finite')
    return x, m


def _weighted_quantile(values, weights, probabilities):
    order = np.argsort(values, kind='stable')
    v, w = values[order], weights[order]
    midpoints = (np.cumsum(w) - .5*w) / w.sum()
    return np.interp(probabilities, midpoints, v)


def fit_statistics(records, *, device, allowed_ids):
    """Each observed shot has equal weight, regardless of its duration.

Up to 128 uniformly spaced AVAILABLE observations per shot and channel are
used. Each such observation receives weight 1/n_observed_in_that_shot.
This is a deterministic statistical estimate, not 128 independent examples.
No label-based time crop, shot selection, or future-stream access occurs.
"""
    rows = list(records)
    if device not in ('J-TEXT', 'EAST') or not rows:
        raise ValueError('explicit device and nonempty signal records required')
    if any(set(r) != SIGNAL_KEYS for r in rows):
        raise ValueError('signals only; outcome, validity and event fields forbidden')
    ids = [r['shot'] for r in rows]
    if (any(type(s) is not int for s in ids) or len(ids) != len(set(ids))
            or not set(ids) <= set(allowed_ids)
            or any(r['device'] != device for r in rows)):
        raise ValueError('distinct authorized device records required')
    arrays = [_arrays(r) for r in rows]
    result = dict(device=device, shot_ids=ids, sample_cap_per_shot_channel=128,
                  weighting='equal total weight per observed shot and channel',
                  mean=[], std=[], median=[], iqr=[], record_counts=[], sample_counts=[])
    for c in range(23):
        samples, weights = [], []
        for x, m in arrays:
            v = x[m[:, c], c].astype(np.float64)
            if not len(v): continue
            v = v[np.linspace(0, len(v)-1, min(128, len(v)), dtype=int)]
            samples.append(v)
            weights.append(np.full(len(v), 1./len(v)))
        result['record_counts'].append(len(samples))
        result['sample_counts'].append(sum(map(len, samples)))
        if not samples:
            for key in ('mean', 'std', 'median', 'iqr'): result[key].append(None)
            continue
        v, w = np.concatenate(samples), np.concatenate(weights)
        w /= w.sum()
        mean = float(w @ v)
        std = float(np.sqrt(w @ np.square(v-mean)))
        q25, median, q75 = _weighted_quantile(v, w, [.25, .5, .75])
        # Exact constants should not acquire a spurious 1e-15 scale from
        # weighted-mean roundoff. No unitful epsilon is used.
        if v.min() == v.max(): std = 0.
        for key, val in (('mean', mean), ('std', std), ('median', median), ('iqr', q75-q25)):
            if not np.isfinite(val): raise FloatingPointError('nonfinite fitted statistic')
            result[key].append(float(val))
    return result


def make_map(source, target, policy):
    if policy not in POLICIES: raise ValueError('explicit preregistered mapping required')
    if source['device'] != 'J-TEXT' or target['device'] != 'EAST':
        raise ValueError('source TRAIN to past EAST statistics required')
    gains, offsets, channels = [], [], []
    for c, name in enumerate(RAW23):
        selected = name not in EXCLUDED and (policy.endswith('_all') or name == 'ne_nG')
        gain, offset, status = 1., 0., 'not_selected_identity'
        if selected:
            if not source['record_counts'][c] or not target['record_counts'][c]:
                status = 'missing_unidentifiable_identity'
            elif policy == 'density_gain':
                s, t = source['median'][c], target['median'][c]
                if s <= 0 or t <= 0:
                    raise ValueError('positive density medians required for zero-preserving gain')
                gain, status = s/t, 'mapped'
            else:
                center, spread = ('median', 'iqr') if policy == 'robust_all' else ('mean', 'std')
                sc, tc = source[center][c], target[center][c]
                ss, ts = source[spread][c], target[spread][c]
                if ss <= 0 or ts <= 0:
                    # Sparse real activity can have IQR=0 with positive
                    # variance (observed in EAST Mirnov channels). Identity
                    # is still the declared unidentifiable-scale operation,
                    # but its cause must not be mislabeled as a constant.
                    status = ('constant_unidentifiable_identity'
                              if source['std'][c] <= 0 or target['std'][c] <= 0
                              else 'zero_iqr_unidentifiable_identity')
                else:
                    gain = ss/ts
                    offset = sc-gain*tc
                    status = 'mapped'
        if not np.isfinite([gain, offset]).all() or gain <= 0:
            raise FloatingPointError('finite positive affine mapping required')
        gains.append(gain); offsets.append(offset)
        channels.append(dict(name=name, status=status, gain=gain, offset=offset))
    return dict(policy=policy, gain=gains, offset=offsets, channels=channels,
                calibration_ids=target['shot_ids'], source_statistics=source,
                target_statistics=target, physical_calibration_certified=False,
                neural_parameter_updates=0, availability_changes=0)


def fit_prefix_map(source, records, *, order, count, policy):
    rows = list(records)
    if (type(count) is not int or not 0 < count < len(order)
            or len(order) != len(set(order))
            or [r['shot'] for r in rows] != list(order[:count])):
        raise ValueError('only the exact ended stream prefix may estimate target statistics')
    target = fit_statistics(rows, device='EAST', allowed_ids=order[:count])
    mapping = make_map(source, target, policy)
    mapping['first_affected_index'] = count
    return mapping


def transform(signal, mapping):
    if set(signal) != {'values', 'available'}:
        raise ValueError('signals only, no outcome/event fields')
    x, m = _arrays(signal)
    gain, offset = np.asarray(mapping['gain']), np.asarray(mapping['offset'])
    if (gain.shape != (23,) or offset.shape != (23,)
            or not np.isfinite(gain).all() or not np.isfinite(offset).all()
            or (gain <= 0).any()):
        raise ValueError('finite positive flat23 affine map required')
    out = x.copy()
    for c in np.flatnonzero((gain != 1.) | (offset != 0.)):
        valid = m[:, c]
        mapped = x[valid, c].astype(np.float64)*gain[c] + offset[c]
        if not np.isfinite(mapped).all() or (abs(mapped) > np.finfo(np.float32).max).any():
            raise FloatingPointError('mapped observation is not representable')
        out[valid, c] = mapped.astype(np.float32)
    return dict(values=out, available=m.copy())
