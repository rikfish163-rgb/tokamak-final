import numpy as np

from experiments.risk_aligned_online.commissioning_source_dynamics_20261110 import selected_origins


def test_positive_fit_uses_at_most_one_origin_per_family_with_full_timely_future():
    q = np.arange(127, 244, 32)
    times = np.arange(250) / 1000
    f = np.full(250, -1)
    f[q] = [1, 1, 2, 2]
    # q=191 is excluded: its 25th target is exactly 30 ms before disruption.
    np.testing.assert_array_equal(selected_origins(q, times, f, [1]*4, True, .246), [1])
    # Safe tracks need an event observation but do not have a disruption clock.
    np.testing.assert_array_equal(selected_origins(q, times, f, [1, 0, 1, 0], False, None), [0, 2])
