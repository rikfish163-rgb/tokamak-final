"""Locked full-shot estimands; no retrospective threshold selection."""
import math

VARIANTS = ('origin', 'mature')
ARMS = ('real', 'reverse', 'mean_feedback', 'donor')
NAMES = ('cold', 'identical', *(v+'__'+a for v in VARIANTS for a in ARMS))


def operational_status(results):
    cold = results['cold']['full360']
    return {v: dict(
        operational_improvement=(results[v+'__real']['full360']['J'] > cold['J']
            and results[v+'__real']['full360']['tpr'] >= cold['tpr']
            and results[v+'__real']['full360']['fpr'] <= .20),
        real_exceeds_controls=all(results[v+'__real']['full360']['J'] >
                                 results[v+'__'+a]['full360']['J'] for a in ('reverse', 'donor')),
        original_component_screen='FAILED', independent_confirmation=False)
        for v in VARIANTS}


def validate_receipts(receipts, length):
    events = list(range(152, length, 25))
    if len(receipts) != len(events):
        raise ValueError('one receipt per matured feedback event required')
    for event, row in zip(events, receipts, strict=True):
        if ((row['origin'], row['maturity'], row['first_affected'], row['apply_count']) !=
                (event-25, event, event+1, min(25, length-event-1))
                or set(row['state_norm']) != set(NAMES)):
            raise ValueError('changed update clock or branch set')
        norms = row['state_norm']
        if (any(not math.isfinite(v) or not 0 <= v <= .100001 for v in norms.values())
                or not math.isclose(row['max_bound'], max(norms.values()), abs_tol=1e-6)
                or any(norms[n] != 0 for n in ('cold', 'identical', 'origin__mean_feedback',
                                               'mature__mean_feedback'))
                or (not row['feedback_supported'] and any(norms.values()))):
            raise ValueError('invalid action bound or nonzero reference action')
