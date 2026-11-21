import numpy as np

from deep_miss_source_horizon_audit_20261106 import training_support


def test_source_25_ms_marks_target_like_leads_negative_without_changing_the_contract():
    t = np.arange(1500, dtype=np.float64) / 1000.0
    a = np.ones((len(t), 23), dtype=bool)
    low = training_support(t, a, True, 1.5, 25.0)
    high = training_support(t, a, True, 1.5, 90.0)
    assert low["all_east_like_y1"] == 0 and low["all_east_like_y0"] == 95
    assert high["all_east_like_y1"] == 60 and high["all_east_like_y0"] == 35
    assert low["sampled_east_like_y1"] == 0
    assert high["sampled_east_like_y1"] > 0
    assert sum(low["common"]) == sum(low["counts"]) - len(range(127, 1151, 64))
