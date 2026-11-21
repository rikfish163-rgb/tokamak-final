import numpy as np

from fit_layer0_arrival_motion_20261106 import source_origins, past_arrays


def test_source_origins_keep_full_past_and_future_windows():
    assert not len(source_origins(152))
    np.testing.assert_array_equal(source_origins(153), [127])
    p = source_origins(1001)
    assert len(p) == 16 and p[0] == 127 and p[-1] == 975
    assert len(set(p)) == len(p)


def test_padded_past_read_has_no_future_and_preserves_mask():
    x = np.broadcast_to(np.arange(200, dtype=np.float32)[:, None], (200, 23)).copy()
    a = np.ones_like(x, dtype=bool)
    a[10, 7] = False
    xx, aa = past_arrays(x, a, np.array([10, 130]))
    assert not aa[0, :-11].any()
    assert xx[0, -1, 0] == 10 and xx[1, -1, 0] == 130
    assert not aa[0, -1, 7]
    x[131:] = -99
    np.testing.assert_array_equal(xx, past_arrays(x, a, np.array([10, 130]))[0])
