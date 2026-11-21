import numpy as np

from source_view_coverage_20261104 import paired_views


def test_view_pair_keeps_old_targets_and_counts_each_source_context_twice():
    x = np.array([[1, 2], [3, 4]], float)
    y = np.array([[1, 0], [0, 1]], float)
    nx = x + 10
    ny = y + 2
    records = [
        dict(shot=1, group="safe", family=-1),
        dict(shot=2, group="danger", family=0),
    ]
    a, b, r = paired_views(x, y, nx, ny, records)
    np.testing.assert_array_equal(a[:2], x)
    np.testing.assert_array_equal(b[:2], y)
    np.testing.assert_array_equal(a[2:], nx)
    assert [v["view"] for v in r] == ["no_MNM_CIII"] * 2 + ["no_MNM"] * 2
    assert [v["shot"] for v in r] == [1, 2, 1, 2]
