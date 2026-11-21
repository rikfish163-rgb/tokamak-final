import numpy as np
from experiments.risk_aligned_online.task_geometry_20261023.timing import feedback_timing


def test_arrived_positive_window_must_precede_strict_warning_deadline():
    times = np.arange(1000)*.002
    counts = list(range(256,1000,128))
    empty = feedback_timing(times,2.,counts)
    assert empty['within_positive_window_before_deadline'] == []
    assert empty['median_sample_ms'] == 2.
    assert empty['median_update_ms'] == 256.
    one = feedback_timing(times,1.9,counts)
    assert one['within_positive_window_before_deadline'] == [896]
    # First prediction at 1.792s is exactly30ms before1.822s; strict >30 excludes it.
    boundary = feedback_timing(times,1.822,counts)
    assert boundary['within_positive_window_before_deadline'] == []
