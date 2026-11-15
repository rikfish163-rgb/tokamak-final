"""Balanced endpoint replay using already sealed, potentially stale errors.

This is an optional sampling helper, not an updater, risk head, or live hook.
It never loads outcomes, predicts risks, constructs windows, or retains state.
"""
import numpy as np

from .inter import MAX_REPLAY_SHOTS, _pools, _sample as _uniform_sample


def _validate(records):
    """Fail closed before RNG use if supplied pools escape the revealed strata."""
    seen, current_count = set(), 0
    for record in records:
        required = {'shot', 'positive', 'negative', 'counts', 'current'}
        if not required <= record.keys() or not isinstance(record['current'], (bool, np.bool_)):
            raise ValueError('revealed records require actual pools/counts and a boolean current flag')
        positive, negative, counts = _pools(record['shot'])
        if (not np.array_equal(record['positive'], positive)
                or not np.array_equal(record['negative'], negative)):
            raise ValueError('candidate pools differ from the actual 125 ms timely/safe strata')
        if record['counts'] != counts:
            raise ValueError('candidate counts differ from the revealed label strata')
        sid = int(record['shot']['shot'])
        if sid in seen:
            raise ValueError('duplicate revealed shot identity in replay')
        seen.add(sid)
        current_count += int(record['current'])
    if current_count > 1:
        raise ValueError('at most one current revealed shot is permitted')
    if len(records) - current_count > MAX_REPLAY_SHOTS:
        raise ValueError('replay must be bounded to at most 32 historical shots')


def sample(records, rng):
    """Return ``selected, counts, audit`` with the frozen sampler's quotas.

    Records have the same layout as ``inter._prepare_records`` returns. They
    must be already sealed/revealed EAST records with exact canonical pools;
    seal durability and historical chronology remain caller responsibilities.

    In each actual-label/current-or-history group, replace up to half of the
    baseline uniform draws with distinct endpoints having the largest
    ``abs(label - shot['_sealed_scores'][end])``. Ties follow input pool order.
    Remaining draws retain their baseline uniform sampling distribution.
    Small priority pools fall back to uniform, not weighted replacement.
    Errors are explicitly SEALED errors, not current-model residuals: an old
    error can remain prioritized even after neural adaptation has repaired it.
    """
    records = list(records)
    _validate(records)
    # Bind the original helper at import time: a future dispatch or isolated
    # test hook replacing inter._sample cannot recurse back into this function.
    baseline, counts, audit = _uniform_sample(records, rng)
    audit.update(replay_policy='error_aware', priority_source='sealed_scores',
                 priority_rule='largest_absolute_label_error',
                 requested_priority_draws=0, priority_draws=0, uniform_draws=0,
                 priority_shortfall=0, groups=[], selected_endpoints=[])
    selected = []
    groups = dict.fromkeys((label, origin) for _, _, label, origin in baseline)
    for label, origin in groups:
        draws = [row for row in baseline if row[2:] == (label, origin)]
        pool = [(record['shot'], int(end), label, origin)
                for record in records
                if ('current' if record['current'] else 'history') == origin
                for end in record['positive' if label else 'negative']]
        errors = np.asarray([abs(label - float(s['_sealed_scores'][end]))
                             for s, end, _, _ in pool])
        requested = len(draws) // 2
        n_priority = min(requested, len(pool))
        ranked = np.argsort(-errors, kind='stable')[:n_priority]
        priority = [pool[int(index)] for index in ranked]
        uniform = draws[n_priority:]
        selected.extend(priority + uniform)
        audit['requested_priority_draws'] += requested
        audit['priority_draws'] += n_priority
        audit['uniform_draws'] += len(uniform)
        audit['priority_shortfall'] += requested - n_priority
        audit['groups'].append(dict(label=label, origin=origin, available=len(pool),
                                    draws=len(draws), requested_priority_draws=requested,
                                    priority_draws=n_priority, uniform_draws=len(uniform),
                                    priority_shortfall=requested - n_priority))
        for rows, role in ((priority, 'priority'), (uniform, 'uniform')):
            audit['selected_endpoints'].extend(
                dict(shot=int(s['shot']), end=end, label=label, origin=origin, role=role,
                     sealed_score=float(s['_sealed_scores'][end]),
                     sealed_error=abs(label - float(s['_sealed_scores'][end])))
                for s, end, _, _ in rows)
    audit['unique_windows'] = len({(int(s['shot']), end) for s, end, _, _ in selected})
    # Preserve the baseline field's meaning: duplicate draws, including a
    # possible overlap of deterministic priority and independent uniform draws.
    audit['replacement_draws'] = len(selected) - audit['unique_windows']
    return selected, counts, audit
