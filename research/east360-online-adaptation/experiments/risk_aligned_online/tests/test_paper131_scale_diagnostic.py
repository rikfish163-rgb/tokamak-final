import pytest

from experiments.risk_aligned_online.paper131_scale_diagnostic import split_ids


def test_calibration_overlap_removed_without_class_or_score_selection():
    order = [dict(shot=sid) for sid in range(1, 51)]
    calibration, held = split_ids(order, [2, 30, 31, 40, 50])
    assert calibration == list(range(1, 31))
    assert held == [31, 40, 50]
    assert not set(calibration) & set(held)


def test_locked_order_preserved_not_sorted_again():
    order = [dict(shot=sid) for sid in (8, 2, 5, 4, 1)]
    assert split_ids(order, [1, 2, 4], count=2) == ([8, 2], [4, 1])


@pytest.mark.parametrize('order,cohort,count', [
    ([dict(shot=1),dict(shot=1)], [1], 1),
    ([dict(shot=1),dict(shot=2)], [3], 1),
    ([dict(shot=1),dict(shot=2)], [2,2], 1),
    ([dict(shot=1),dict(shot=2)], [2], True),
    ([dict(shot=1),dict(shot=2)], [1], 1),
])
def test_invalid_or_empty_comparison_rejected(order,cohort,count):
    with pytest.raises(ValueError):
        split_ids(order,cohort,count=count)
