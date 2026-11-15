import numpy as np

from experiments.risk_aligned_online import commissioning_voltage_response_20261112 as probe


def signal():
    rng = np.random.default_rng(34)
    n = 420
    current = 4e5 + np.cumsum(rng.normal(0., 2100., n))
    voltage = .4 + 4e-5*np.diff(current, prepend=current[0])
    return np.c_[current, np.full(n, 2.), voltage, rng.normal(size=(n, 2))]


def test_current_coupling_cannot_use_voltage_of_the_bin_being_predicted():
    x = signal()
    r = np.array([153, 185, 217, 249, 281, 313, 345])
    a = probe.observe_frames(x, np.ones_like(x, bool), r)
    changed = x.copy()
    changed[344, 2] += 1.0
    b = probe.observe_frames(changed, np.ones_like(x, bool), r)
    # This is the last sample of the last target bin: only the measurement, not
    # its estimate from older voltage and already delivered current, can move.
    np.testing.assert_array_equal(a['predictions'][-1], b['predictions'][-1])
    assert b['loss'][-1, 1] > a['loss'][-1, 1]
    for k in a:
        np.testing.assert_array_equal(a[k][:-1], b[k][:-1])


def test_truncated_and_missing_raw_measurements_cannot_change_the_common_support():
    x = signal()
    r = np.array([153, 185, 217, 249, 281, 313, 345])
    mask = np.ones_like(x, bool)
    a = probe.observe_frames(x, mask, r)
    b = probe.observe_frames(x[:282], mask[:282], r[:5])
    for k in a:
        np.testing.assert_array_equal(a[k][:5], b[k])
    mask[313-2, 4] = False
    c = probe.observe_frames(x, mask, r)
    assert 313 not in c['release']


def test_conditioned_lead_and_wrong_order_share_the_same_radiation_marginals():
    v = np.array([[.4, -.7, -.6, .9, .5]])
    rad = np.array([[[0., 0.], [2., 0.], [-1., 1.], [4., -2.], [.5, 0.]]])
    true, wrong = probe.aligned_scores(v, rad)
    np.testing.assert_allclose(wrong, probe.aligned_scores(v, np.roll(rad, 2, axis=1))[0])
    assert not np.allclose(true, wrong)
