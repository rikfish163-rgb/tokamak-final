"""Retrospective historical controls, matched on shot order, not test outcomes."""


def matched_controls(records, disruptive_ids, *, test_ids):
    lookup = {r['shot']: r for r in records}
    if len(lookup) != len(records):
        raise ValueError('distinct historical records required')
    if set(lookup) & set(test_ids):
        raise ValueError('training/test overlap')
    if any(r['is_disrupt'] != r['shotlist_is_disrupt'] for r in records):
        raise ValueError('HDF5 and shotlist disagree')
    cases = sorted(disruptive_ids)
    if (len(cases) != 4 or len(set(cases)) != 4
            or any(s not in lookup or not lookup[s]['is_disrupt'] for s in cases)):
        raise ValueError('exactly four disruptive historical identities required')
    safe = {r['shot'] for r in records if not r['is_disrupt']}
    used, pairs = set(), []
    # Round-robin nearest available controls. No replacement, no outcome-driven
    # reordering, no claim of globally optimal or physical-condition matching.
    for turn, side in enumerate(('before', 'after', 'before', 'after')):
        for case in cases:
            options = [sid for sid in safe-used if (sid < case if side == 'before' else sid > case)]
            if not options:
                raise ValueError(f'no unused safe control {side} shot {case}')
            sid = min(options, key=lambda s: (abs(s-case), s))
            used.add(sid)
            pairs.append(dict(round=turn, disruptive=case, safe=sid, side=side, distance=abs(sid-case)))
    selected = [lookup[s] for s in sorted(set(cases)|used)]
    assert len(selected) == 20 and len(used) == 16
    return selected, pairs
