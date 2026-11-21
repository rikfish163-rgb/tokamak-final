import numpy as np

from experiments.risk_aligned_online.simple_data import window
from experiments.risk_aligned_online.causal_span_20261101.core import span_windows


def signal():
    x = np.arange(500 * 23, dtype=np.float32).reshape(500, 23)
    a = np.ones_like(x, dtype=bool)
    a[100:105, 2] = False
    return dict(values=x, available=a)


def test_d1_is_exact_original_at_all_prefix_boundaries():
    s = signal()
    ends = [0, 15, 127, 128, 253, 254, 499]
    x, a, eligible = span_windows(s, ends, 1)
    for row, end in enumerate(ends):
        xx, aa = window(s, end, 128)
        assert np.array_equal(x[row], xx) and np.array_equal(a[row], aa)
    assert eligible.tolist() == [False, False, True, True, True, True, True]


def test_d2_uses_exact_past_indices_and_current_observation():
    s = signal()
    x, a, eligible = span_windows(s, [254, 255, 499], 2)
    for row, end in enumerate([254, 255, 499]):
        index = end - 2 * np.arange(127, -1, -1)
        assert np.array_equal(x[row], s["values"][index])
        assert np.array_equal(a[row], s["available"][index])
        assert np.array_equal(x[row, -1], s["values"][end])
    assert eligible.all()


def test_insufficient_history_keeps_original_without_suppressing_scores():
    s = signal()
    x, a, eligible = span_windows(s, [0, 127, 253], 2)
    for row, end in enumerate([0, 127, 253]):
        xx, aa = window(s, end, 128)
        assert np.array_equal(x[row], xx) and np.array_equal(a[row], aa)
    assert not eligible.any()


def test_future_suffix_and_truncation_cannot_change_windows():
    s = signal()
    end = 300
    truncated = {k: v[: end + 1].copy() for k, v in s.items()}
    changed = {k: v.copy() for k, v in s.items()}
    changed["values"][end + 1 :] = -9999
    changed["available"][end + 1 :] = False
    expected = span_windows(s, [end], 2)
    for other in (truncated, changed):
        assert all(
            np.array_equal(x, y)
            for x, y in zip(expected, span_windows(other, [end], 2))
        )
