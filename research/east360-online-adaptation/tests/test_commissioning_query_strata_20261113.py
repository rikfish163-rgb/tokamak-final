import numpy as np

from experiments.risk_aligned_online import commissioning_query_strata_20261113 as replay


def record(rank, available_from, disruptive, bin_id):
    return dict(rank=rank, available_from=available_from, disruptive=disruptive, bin=bin_id, points=[0])


def test_matching_replaces_only_the_safe_slot_after_reveal():
    entries = [record(35, 41, True, -1), record(165, 181, False, 1), record(190, 201, False, -1),
               record(222, 241, False, 1)]
    assert replay.select(entries, 180) == (0, None, None)
    assert replay.select(entries, 201) == (0, 2, 1)
    assert replay.select(entries, 241) == (0, 3, 3)


def test_prefix_key_uses_only_first_100_points_and_original_missingness():
    a = np.zeros((250,23),np.float32)
    a[:,replay.field.BT] = .025
    m = np.ones_like(a,bool)
    s = dict(values=a,available=m)
    assert replay.stratum(s,.19)
    a[100:,replay.field.BT] = 3.
    m[100:,1] = False
    assert replay.stratum(s,.19)
    assert not replay.stratum(dict(values=a[:99],available=m[:99]),.19)
    m[99,1] = False
    assert not replay.stratum(s,.19)


def test_fitting_windows_and_protected_prefix():
    t = np.arange(300) / 1000.
    base = np.full((2, len(t)), .5, np.float32)
    d = replay.teaching_points(t, base.max(0), True, .250)
    s = replay.teaching_points(t, base.max(0), False, None)
    assert len(np.unique(d)) == len(np.unique(s)) == replay.POINTS
    assert t[d[0]] == .125 and t[d[-1]] == .250
    assert np.any(t[d] >= .220)
    assert np.all(s >= replay.N-1)
    hidden = np.ones((2, len(t), 1), np.float32)
    params = {'matched_supported':np.ones(1)}
    off = replay.predict(base, hidden, -1, params)
    on = replay.predict(base, hidden, 1, params)
    for k in off:
        np.testing.assert_array_equal(off[k], base[0])
        np.testing.assert_array_equal(on[k][:replay.N-1], base[0, :replay.N-1])
        assert np.all(on[k][replay.N-1:] > base[0, replay.N-1:])
