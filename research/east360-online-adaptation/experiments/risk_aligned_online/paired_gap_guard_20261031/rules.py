"""Pure comparison rules; these say nothing about future-shot generalization."""
import math
from numbers import Real


def evaluate_guards(before_on, before_off, after_on, after_off, *, tolerance=1e-7):
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('finite nonnegative tolerance required')
    records = (before_on, before_off, after_on, after_off)
    if any(not isinstance(row, dict) for row in records):
        raise ValueError('four explicit group-loss dictionaries required')
    keys = set(before_on)
    if ('source_event' not in keys or not keys <= {'source_event', 'historical_D', 'historical_S'}
            or any(set(row) != keys for row in records)):
        raise ValueError('identical observed guard groups with source_event required')
    for row in records:
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value < 0 for value in row.values()):
            raise ValueError('finite nonnegative measured losses required')
    groups = {}
    on_bad, off_bad, gap_bad = [], [], []
    for key in sorted(keys):
        gap_before = before_on[key]-before_off[key]
        gap_after = after_on[key]-after_off[key]
        groups[key] = dict(on_before=float(before_on[key]), on_after=float(after_on[key]),
                           off_before=float(before_off[key]), off_after=float(after_off[key]),
                           gap_before=float(gap_before), gap_after=float(gap_after),
                           gap_change=float(gap_after-gap_before))
        if after_on[key] > before_on[key]+tolerance:
            on_bad.append(key)
        if after_off[key] > before_off[key]+tolerance:
            off_bad.append(key)
        if gap_after > gap_before+tolerance:
            gap_bad.append(key)
    original = not on_bad
    dual = original and not off_bad
    return dict(original_pass=original, dual_absolute_pass=dual,
                paired_pass=dual and not gap_bad, tolerance=float(tolerance),
                on_violations=on_bad, off_violations=off_bad,
                gap_violations=gap_bad, groups=groups)
