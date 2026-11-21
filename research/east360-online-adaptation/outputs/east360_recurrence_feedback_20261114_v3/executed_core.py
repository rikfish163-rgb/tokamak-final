"""Causal, within-shot matching against already matured historical windows."""

import numpy as np


_NGROUPS = 4
_TIE_RTOL = 1e-12


def find_precedents(
    q: np.ndarray,
    key: np.ndarray,
    group_id: np.ndarray,
    ok: np.ndarray,
    *,
    stride: int,
    horizon: int,
    recent_slots: int,
    weights: np.ndarray | None = None,
    min_separation: int = 0,
    query_ok: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return matched, condition-only and displaced indices, plus match distances.

    Rows correspond to diagnostic groups 0 through 3, columns to query rows.
    Each index addresses the same input arrays. Donors must be marked by ``ok``,
    have matured by the query, and fall inside both the recent window and the
    minimum separation. A query is emitted only if it has at least two donors
    and passes ``query_ok``; when omitted, ``query_ok`` defaults to ``ok``.

    Matching uses weighted mean squared distance over the condition and all
    diagnostic columns except the row's held-out group. The condition-only
    match uses the same donor set. Distances within 1e-12 relative to the minimum
    tie in favor of the latest donor, with no absolute tolerance. Displacement
    rotates the matched donor forward by floor(candidate_count / 2) in the
    chronological candidate list.

    ``full_distance`` is the matched distance on the actual, held-out-excluded
    search key; ``own_distance`` is its distance on the held-out group, computed
    only after selection. Both are weighted means. If a search key has no columns,
    all donors tie at zero; if the held-out group has no columns, its diagnostic
    distance is undefined and reported as +inf. Missing records have index -1
    and both distances +inf.
    """
    q = np.asarray(q)
    key = np.asarray(key, dtype=np.float64)
    group_id = np.asarray(group_id)
    ok = np.asarray(ok, dtype=bool)
    if q.ndim != 1 or not np.issubdtype(q.dtype, np.integer):
        raise ValueError('q must be a one-dimensional integer array')
    n = q.size
    if n > 1 and np.any(q[1:] <= q[:-1]):
        raise ValueError('q must be strictly increasing')
    if key.ndim != 2 or key.shape[0] != n or not np.all(np.isfinite(key)):
        raise ValueError('key must be a finite [N, D] array')
    d = key.shape[1]
    if (group_id.shape != (d,) or not np.issubdtype(group_id.dtype, np.integer)
            or np.any((group_id < -1) | (group_id >= _NGROUPS))):
        raise ValueError('group_id must contain -1 or diagnostic groups 0 through 3')
    if ok.shape != (n,):
        raise ValueError('ok must have shape [N]')
    if query_ok is None:
        query_ok = ok
    else:
        query_ok = np.asarray(query_ok, dtype=bool)
        if query_ok.shape != (n,):
            raise ValueError('query_ok must have shape [N]')
    for name, value, minimum in (
        ('stride', stride, 1), ('horizon', horizon, 0),
        ('recent_slots', recent_slots, 0), ('min_separation', min_separation, 0),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if weights is None:
        weights = np.ones(d, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (d,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError('weights must be finite, positive and have shape [D]')

    def columns_and_weights(mask):
        columns = np.flatnonzero(mask)
        if columns.size == 0:
            return columns, np.empty(0, dtype=np.float64)
        w = weights[columns]
        w = w / w.max()
        return columns, w / w.sum()

    condition = columns_and_weights(group_id == -1)
    search_keys = [columns_and_weights(group_id != group) for group in range(_NGROUPS)]
    own_keys = [columns_and_weights(group_id == group) for group in range(_NGROUPS)]

    def distances(squared, selection):
        columns, normalized_weights = selection
        if columns.size == 0:
            return np.zeros(squared.shape[0], dtype=np.float64)
        return np.sum(squared[:, columns] * normalized_weights, axis=1)

    def nearest(distance):
        best = distance.min()
        return np.flatnonzero(distance <= best + _TIE_RTOL * best)[-1]

    shape = (_NGROUPS, n)
    matched = np.full(shape, -1, dtype=np.int64)
    blind = np.full(shape, -1, dtype=np.int64)
    displaced = np.full(shape, -1, dtype=np.int64)
    full_distance = np.full(shape, np.inf, dtype=np.float64)
    own_distance = np.full(shape, np.inf, dtype=np.float64)
    max_age = int(recent_slots) * int(stride)
    min_age = max(int(horizon) + 1, int(min_separation))

    for i in np.flatnonzero(query_ok):
        t = int(q[i])
        first = np.searchsorted(q, t - max_age, side='left')
        last = np.searchsorted(q, t - min_age, side='right')
        candidates = np.flatnonzero(ok[first:last]) + first
        count = candidates.size
        if count < 2:
            continue
        difference = key[candidates] - key[i]
        squared = difference * difference
        blind_index = candidates[nearest(distances(squared, condition))]
        for group in range(_NGROUPS):
            distance = distances(squared, search_keys[group])
            position = nearest(distance)
            matched[group, i] = candidates[position]
            blind[group, i] = blind_index
            displaced[group, i] = candidates[(position + count // 2) % count]
            full_distance[group, i] = distance[position]
            if own_keys[group][0].size:
                own_distance[group, i] = distances(squared[position:position + 1],
                                                   own_keys[group])[0]

    return matched, blind, displaced, full_distance, own_distance
