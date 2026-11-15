"""CPU-only selection of source teaching episodes from already-scored shots.

Population counts are per shot. ``origins`` counts every nominal 127+25k
origin present in the shot, including those too late to have a query. An
origin is counted as ``no_query`` before feedback support is considered.
The ``before`` group counts are the supported episodes before the cap;
``after`` counts the returned episodes. The caller attaches the shot ID to the
summary; each returned record already contains it. No signal arrays are stored.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


_GROUPS = ('safe', 'danger', 'early')
_REQUIRED = ('shot', 'device', 'values', 'available', 'times', 'y',
             'valid', 'is_disrupt', 'down')


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _array(value, name, kinds):
    """Convert a CPU array or tensor without silently moving an accelerator tensor."""
    device = getattr(value, 'device', None)
    if device is not None and str(device) != 'cpu':
        raise ValueError(f'{name} must be on CPU')
    if np.ma.isMaskedArray(value):
        raise ValueError(f'{name} must not be a masked array')
    try:
        if callable(getattr(value, 'detach', None)):
            value = value.detach()
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'invalid {name} array') from exc
    if array.dtype.kind not in kinds:
        raise ValueError(f'invalid {name} dtype')
    return array


def _positive_count(value, name, maximum=None):
    if not _integer(value) or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f'invalid {name}')
    return int(value)


def _validated_shot(shot, logits):
    if not isinstance(shot, Mapping) or any(key not in shot for key in _REQUIRED):
        raise ValueError('labeled source shot fields required')
    if shot['device'] != 'J-TEXT' or not _integer(shot['shot']):
        raise ValueError('integer shot ID and J-TEXT device required')
    if not isinstance(shot['is_disrupt'], (bool, np.bool_)):
        raise ValueError('boolean disruption flag required')
    disruptive = bool(shot['is_disrupt'])
    down = shot['down']
    if disruptive:
        if isinstance(down, (bool, np.bool_)) or not isinstance(down, Real) or not np.isfinite(down):
            raise ValueError('finite disruptive terminal time required')
        down = float(down)
    elif down is not None:
        raise ValueError('non-disruptive shot cannot have a terminal time')

    times = _array(shot['times'], 'times', 'iuf')
    values = _array(shot['values'], 'values', 'iuf')
    available = _array(shot['available'], 'available', 'b')
    valid = _array(shot['valid'], 'valid', 'b')
    labels = _array(shot['y'], 'y', 'biuf')
    scores = _array(logits, 'logits', 'iuf')
    if times.ndim != 1:
        raise ValueError('times must be one-dimensional')
    n = len(times)
    if (values.shape != (n, 23) or available.shape != (n, 23)
            or valid.shape != (n,) or labels.shape != (n,) or scores.shape != (n,)):
        raise ValueError('shot and logit shapes do not match')
    times = np.asarray(times, dtype=np.float64)
    if (not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or not np.isfinite(scores).all() or not np.isfinite(labels).all()
            or not np.isin(labels, (0, 1)).all()):
        raise ValueError('finite, ordered times, finite logits and binary labels required')
    if not disruptive and np.any(labels != 0):
        raise ValueError('non-disruptive shot has positive labels')
    if disruptive:
        with np.errstate(over='ignore', invalid='ignore'):
            leads = np.round((down - times) * 1000., 9)
        if not np.isfinite(leads).all():
            raise ValueError('nonfinite disruption lead')
    else:
        leads = None
    return times, values, available, valid, labels, scores, disruptive, leads


def select_records(shot, logits, max_per_group=8):
    """Select frozen-source peak queries and at most eight episodes per group.

    Queries are selected among valid points in ``origin+26..origin+50``
    (clipped at the end of the shot); disruptive points also require rounded
    lead >10 ms. Unsupported feedback is excluded. Caps use evenly spaced
    positions in origin order, independently for each group, never peak rank.
    """
    cap = _positive_count(max_per_group, 'max_per_group', maximum=8)
    times, values, available, valid, labels, scores, disruptive, leads = _validated_shot(shot, logits)
    groups = {group: [] for group in _GROUPS}
    excluded = dict(no_query=0, unsupported=0)
    origins = range(127, len(times), 25)
    for origin in origins:
        start, stop = origin + 26, min(origin + 51, len(times))
        if start >= stop:
            excluded['no_query'] += 1
            continue
        eligible = valid[start:stop].copy()
        if disruptive:
            eligible &= leads[start:stop] > 10.
        candidates = np.flatnonzero(eligible) + start
        if not len(candidates):
            excluded['no_query'] += 1
            continue
        query = int(candidates[np.argmax(scores[candidates])])

        origin_support = available[origin] & np.isfinite(values[origin])
        origin_support[10] = False
        arrival = slice(origin + 1, origin + 26)
        supported = bool(np.any(available[arrival] & np.isfinite(values[arrival]) & origin_support))
        if not supported:
            excluded['unsupported'] += 1
            continue

        label = float(labels[query])
        group = 'safe' if not disruptive else ('danger' if label == 1. else 'early')
        groups[group].append(dict(
            shot=int(shot['shot']), origin=origin, maturity=origin + 25, query=query,
            group=group, label=label, lead_ms=float(leads[query]) if disruptive else None,
            disruptive=disruptive, feedback_observed=True,
        ))

    selected = []
    for rows in groups.values():
        indices = np.rint(np.linspace(0, len(rows) - 1, min(cap, len(rows)))).astype(int)
        selected.extend(rows[index] for index in indices)
    selected.sort(key=lambda row: row['origin'])
    population = dict(
        samples=len(times), origins=len(origins),
        before={group: len(groups[group]) for group in _GROUPS},
        after={group: min(cap, len(groups[group])) for group in _GROUPS},
        excluded=excluded,
    )
    return selected, population


def sample_indices(records, rng, per_class=6):
    """Draw safe then danger indices, uniformly by distinct shot, with replacement."""
    count = _positive_count(per_class, 'per_class')
    if not isinstance(records, (list, tuple)) or not callable(getattr(rng, 'choice', None)):
        raise ValueError('records and an RNG with choice are required')
    by_group = {'safe': {}, 'danger': {}}
    shot_groups = {}
    for index, record in enumerate(records):
        if (not isinstance(record, Mapping) or record.get('group') not in _GROUPS
                or not _integer(record.get('shot'))):
            raise ValueError('record requires a valid group and integer shot ID')
        group, sid = record['group'], int(record['shot'])
        previous = shot_groups.setdefault(sid, set())
        previous.add(group)
        if 'safe' in previous and len(previous) > 1:
            raise ValueError('a shot cannot be both safe and disruptive')
        if group != 'early':
            by_group[group].setdefault(sid, []).append(index)
    if any(not by_group[group] for group in ('safe', 'danger')):
        raise ValueError('both safe and danger populations are required')
    result = []
    for group in ('safe', 'danger'):
        shots = list(by_group[group])
        for _ in range(count):
            sid = rng.choice(shots)
            result.append(int(rng.choice(by_group[group][sid])))
    return result
