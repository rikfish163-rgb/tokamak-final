import numpy as np

from alarm_supervision_mismatch_20261105 import lead_accounting


def test_early_valid_crossing_is_not_equivalent_to_125ms_positive_window():
    lead = np.array([4000.0, 150.0, 125.0, 40.0, 30.0, 20.0, -5.0])
    scores = np.array([0.1, 0.9, 0.2, 0.2, 0.95, 0.95, 0.95])
    r = lead_accounting(lead, scores, 0.8)
    assert r == dict(
        first=1,
        first_lead=150.0,
        bin="125_to_3000",
        timely=True,
        window_crossings=0,
        beyond_window_only=True,
        capped_crossings=1,
    )


def test_30_is_strict_125_inclusive_and_later_alternative_is_counted():
    lead = np.array([3001.0, 3000.0, 125.0, 30.0, -1.0])
    r = lead_accounting(lead, np.full(5, 0.9), 0.8)
    assert r["bin"] == "over3000"
    assert r["window_crossings"] == 1 and r["capped_crossings"] == 2
    assert not r["beyond_window_only"]
    late = lead_accounting(np.array([30.0, 0.0, -1.0]), np.full(3, 0.9), 0.8)
    assert late["bin"] == "tardy" and not late["timely"]
