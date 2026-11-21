import numpy as np

from prealarm_fullstream_support_20261106 import block_clock, opportunities


def test_block_state_only_changes_on_completed_release_boundaries():
    support = np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    np.testing.assert_array_equal(block_clock(support, 4), [False] * 4 + [True] * 4)


def test_later_support_cannot_repair_an_earlier_first_false_alarm():
    mask = np.ones((130, 23), dtype=bool)
    t = np.arange(len(mask)) / 1000
    p = np.zeros(len(t))
    p[50] = 0.99
    y = opportunities(t, p, mask, False, None, 0.72)
    assert y["status"] == "fp"
    assert not y["geometry"]["pair32_block"]["at_original_first"]
    assert y["geometry"]["pair16_block"]["at_original_first"]


def test_missing_static_alarm_can_have_later_causal_risk_support():
    mask = np.ones((200, 23), dtype=bool)
    t = np.arange(len(mask)) / 1000
    p = np.zeros(len(t))
    x = opportunities(t, p, mask, True, 0.2, 0.72)
    assert x["status"] == "miss"
    assert x["geometry"]["pair32_block"]["timely_positive"]
    assert not x["geometry"]["extra4x16"]["timely_positive"]
    assert not x["geometry"]["pair32_block"]["at_original_first"]
