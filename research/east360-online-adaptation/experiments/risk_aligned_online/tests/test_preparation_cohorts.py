import pytest

from experiments.risk_aligned_online.shot_objective_alignment_20261023.cohorts import (
    class_ranges, consecutive_cohorts,
)


def test_class_ranges_flags_separation_but_not_causation():
    rows = [dict(shot=i, is_disrupt=i > 2) for i in range(1, 5)]
    result = class_ranges(rows)
    assert result['all_safe_earlier_than_all_disruptive']
    assert result['safe'] == dict(count=2, minimum=1, maximum=2)
    rows[1]['is_disrupt'], rows[2]['is_disrupt'] = True, False
    assert not class_ranges(rows)['all_safe_earlier_than_all_disruptive']


def test_cohorts_count_early_errors_without_reclassifying_late_alarms():
    rows = [dict(shot=1, is_disrupt=False, alarm=True, timely=False),
            dict(shot=2, is_disrupt=True, alarm=True, timely=False),
            dict(shot=3, is_disrupt=True, alarm=True, timely=True)]
    result = consecutive_cohorts(rows, [1], size=2)
    assert result[0] == dict(first=1, last=2, n_disrupt=1, n_safe=1,
                            tp=0, fp=1, early_fp=1)
    assert result[1]['n_safe'] == 0 and result[1]['tp'] == 1
    with pytest.raises(ValueError, match='ordered'):
        consecutive_cohorts(rows[::-1], [], size=2)
    with pytest.raises(ValueError, match='safe alarm'):
        consecutive_cohorts(rows, [2], size=2)
