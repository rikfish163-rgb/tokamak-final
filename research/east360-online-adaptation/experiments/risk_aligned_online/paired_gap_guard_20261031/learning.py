"""Guard a persistent proposal using only already-available replay and source."""

from numbers import Integral

from ..lifecycle_stream_20261025 import learning as legacy
from . import rules


def update_with_guards(model, current, replay, source, *, rule, eligible_so_far=0,
                       policy='disruptive_positive_only', steps=2):
    """Return candidate losses even on rollback; guards never use ``current``.

    The caller carries the returned eligible count within each rule's own
    trajectory. Only proposals passing the original ON guard increment it.
    """
    if not isinstance(rule, str) or rule not in ('original', 'dual', 'paired', 'half'):
        raise ValueError('unknown persistent-update guard rule')
    if isinstance(eligible_so_far, bool) or not isinstance(eligible_so_far, Integral) or eligible_so_far < 0:
        raise ValueError('eligible_so_far must be a nonnegative integer')
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError('steps must be a positive integer')

    before = model.adapted_state()
    try:
        before_on = legacy.guard_losses(model, replay, source, True, policy)
        before_off = legacy.guard_losses(model, replay, source, False, policy)
        result = legacy.update(model, current, replay, source, fast=True, steps=steps,
                               guarded=False, policy=policy)
        after_on = legacy.guard_losses(model, replay, source, True, policy)
        after_off = legacy.guard_losses(model, replay, source, False, policy)
        checks = rules.evaluate_guards(before_on, before_off, after_on, after_off)
        count_before = int(eligible_so_far)
        count_after = count_before + int(checks['original_pass'])
        if rule == 'original':
            accepted = checks['original_pass']
        elif rule == 'dual':
            accepted = checks['dual_absolute_pass']
        elif rule == 'paired':
            accepted = checks['paired_pass']
        else:
            accepted = checks['original_pass'] and count_after % 2 == 1

        if not accepted:
            model.restore(before)
        result.update(accepted=accepted, guard_before=before_on, guard_after=after_on,
                      retained_delta=result['candidate_delta'] if accepted else
                      {key: 0. for key in result['candidate_delta']})
        result['paired_guard'] = dict(rule=rule, eligible_count_before=count_before,
                                      eligible_count_after=count_after,
                                      before_off=before_off, after_off=after_off,
                                      checks=checks)
        return result
    except BaseException:
        model.restore(before)
        raise
