import pytest

from experiments.risk_aligned_online.mask_repair_20261104.local_probe import (
    classify_endpoint, queries,
)


def test_queries_are_fixed_grid_with_real_future_and_no_short_record_fabrication():
    assert queries(155) == []
    assert queries(156) == [129]
    for n in (200, 1000):
        q = queries(n)
        assert len(q) <= 8 and q == sorted(set(q))
        assert all(t >= 129 and (t - 25) % 26 == 0 and t + 26 < n for t in q)


def test_groups_use_original_source_window_and_strict_ten_ms():
    assert classify_endpoint(False, None, 25, 0.9) == ('safe', 0)
    assert classify_endpoint(True, 1.0, 25, 0.98) == ('timely_positive', 1)
    assert classify_endpoint(True, 1.0, 25, 0.99) == ('late_positive', 1)
    assert classify_endpoint(True, 1.0, 25, 0.97) == ('earlier_disruption', 0)
    assert classify_endpoint(True, 1.0, 40, 0.97) == ('timely_positive', 1)
    assert classify_endpoint(True, 1.0, 25, 1.001) == ('postterminal', None)


def test_bad_length_is_not_a_valid_schedule():
    with pytest.raises(ValueError):
        queries(-1)
