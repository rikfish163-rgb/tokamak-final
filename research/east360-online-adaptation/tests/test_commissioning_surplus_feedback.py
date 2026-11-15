import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_surplus_feedback_20261111 import (
    GROUPS, HORIZON, innovations, physical_keys, raw_feedback,
)


def test_raw_linear_trend_and_known_signed_surplus():
    q = 40
    t = np.arange(100, dtype=np.float32)
    z = t[:, None] * np.linspace(-0.1, 0.1, 10)[None, :]
    mask = np.ones_like(z, dtype=bool)
    raw, observed = raw_feedback(z, mask, [q])
    np.testing.assert_allclose(raw, 0, atol=1e-6)
    assert observed.all() and raw.dtype == np.float32 and observed.dtype == bool
    expected = np.linspace(-2, 2, 10)
    for k in range(5):
        z[q + 1 + 5*k:q + 6 + 5*k] += expected * (k / 4)
    raw, observed = raw_feedback(z, mask, [q])
    np.testing.assert_allclose(raw[0], expected, atol=1e-6)
    assert HORIZON == 25 and GROUPS == (0, 0, 0, 1, 1, 3, 2, 2, 2, 2)


def test_raw_masks_finite_counts_bounds_and_truncation():
    z = np.zeros((100, 10), np.float32)
    mask = np.ones_like(z, dtype=bool)
    q = 31
    mask[0:9, 0] = False  # only seven in prior history
    mask[q+1:q+4, 1] = False  # only two in first future bin
    z[q+6:q+9, 2] = np.nan  # finite values also count toward availability
    z[q+10, 3] = np.inf  # one bad sample is tolerated
    full, obs = raw_feedback(z, mask, [q, 60])
    short, short_obs = raw_feedback(z[:q+26], mask[:q+26], [q])
    np.testing.assert_array_equal(full[:1], short)
    np.testing.assert_array_equal(obs[:1], short_obs)
    np.testing.assert_array_equal(obs[0], [False, False, False, True, True, True, True, True, True, True])
    assert np.isfinite(full).all() and (full[~obs] == 0).all()
    with pytest.raises(ValueError):
        raw_feedback(z, mask, [30])
    with pytest.raises(ValueError):
        raw_feedback(z, mask, [75])
    empty, available = raw_feedback(z, mask, [])
    assert empty.shape == available.shape == (0, 10)


def test_raw_exact_minimum_counts_and_weighted_slope():
    q = 31
    z = np.zeros((57, 10), np.float32)
    mask = np.zeros_like(z, bool)
    mask[0:8] = True
    mask[16:24] = True
    residuals = [0, 1, -2, 3, -4]
    for k, value in enumerate(residuals):
        start = q + 1 + 5*k
        mask[start:start+3] = True
        z[start:start+5] = value
    raw, observed = raw_feedback(z, mask, [q])
    assert observed.all()
    np.testing.assert_allclose(raw, -4/3, rtol=1e-6)
    mask[1, 0] = False
    mask[q+1, 1] = False
    raw, observed = raw_feedback(z, mask, [q])
    assert not observed[0, 0] and not observed[0, 1] and observed[0, 2:].all()
    assert raw[0, 0] == raw[0, 1] == 0


def test_asymmetric_missingness_does_not_turn_linear_motion_into_surplus():
    q = 40
    t = np.arange(80, dtype=np.float64)
    z = t[:, None] * np.linspace(-.2, .2, 10)[None,:]
    mask = np.ones_like(z, bool)
    mask[q-31:q-23, 0:3] = False
    mask[q-7:q+1, 0:3] = False
    mask[q+1:q+3, 0:3] = False
    mask[q+24:q+26, 0:3] = False
    raw, observed = raw_feedback(z, mask, [q])
    assert observed.all()
    np.testing.assert_allclose(raw, 0, atol=1e-6)


def test_physical_keys_are_causal_and_handle_partial_bins():
    context = np.column_stack((np.arange(48) / 16, np.full(48, 2.))).astype(np.float32)
    mask = np.ones_like(context, bool)
    keys = physical_keys(context, mask)
    np.testing.assert_allclose(keys[31], .5 * np.array([1, np.tanh(23.5/16), np.tanh(2), np.tanh(4)]), atol=1e-6)
    assert keys.shape == (48, 4) and keys.dtype == np.float32
    short = physical_keys(context[:26], mask[:26])
    np.testing.assert_array_equal(keys[:26], short)
    mask[:9, 0] = False
    context[20:29, 1] = np.nan
    keys = physical_keys(context, mask)
    assert keys[31, 1] != 0 and keys[31, 2] == 0 and keys[31, 3] == 0
    assert np.isfinite(keys).all() and keys[0, 0] == .5
    assert physical_keys(np.empty((0, 2)), np.empty((0, 2), bool)).shape == (0, 4)


def test_innovations_exact_features_and_group_eligibility():
    r = np.zeros((4, 10), np.float32)
    r[:, 0] = [2, 1, -1, 3]
    r[:, 3] = [2, 4, 4, 4]
    obs = np.zeros_like(r, bool)
    obs[:, [0, 3]] = True
    obs[2, 3] = False  # ineligible, hence no features, update or continuity
    scales = np.ones(10, np.float32)
    scales[0] = 2
    arms, eligible, audit = innovations(r, obs, scales)
    assert set(arms) == {'no_background', 'intra', 'lag'}
    assert all(v.shape == (4, 28) and v.dtype == np.float32 for v in arms.values())
    np.testing.assert_array_equal(eligible, [True, True, False, True])
    assert audit['eligible_blocks'] == 3
    assert audit['background_updates'] == 3
    assert (arms['intra'][2] == 0).all()
    first = arms['no_background'][0]
    np.testing.assert_allclose(first[[0, 3, 20, 21]], np.tanh([1, 2, 1, 2]))
    np.testing.assert_allclose(arms['no_background'][1, 10], np.tanh(.5))
    assert arms['no_background'][3, 10] == 0
    np.testing.assert_allclose(arms['intra'][3, 0], np.tanh(1.5 - .0768), rtol=1e-6)
    np.testing.assert_array_equal(arms['lag'][0], arms['intra'][0])
    np.testing.assert_array_equal(arms['lag'][1], arms['intra'][0])
    np.testing.assert_array_equal(arms['lag'][3], arms['intra'][1])


def test_innovations_persistence_masks_and_signed_group_rms():
    r = np.zeros((3, 10), np.float32)
    r[:, 0] = [-2, -3, 3]
    r[:, 1] = [4, 1, 5]
    r[:, 3] = 1
    obs = np.zeros_like(r, bool)
    obs[:, [0, 1, 3]] = True
    obs[1, 1] = False
    arms, eligible, _ = innovations(r, obs, np.ones(10))
    x = arms['no_background']
    assert eligible.all()
    np.testing.assert_allclose(x[0, 20], np.tanh(np.sqrt(10)))
    np.testing.assert_allclose(x[1, 10], np.tanh(-2))
    np.testing.assert_allclose(x[1, 24], np.tanh(2))
    assert x[2, 10] == 0 and x[2, 11] == 0


def test_background_is_robust_post_release_and_resets_per_call():
    r = np.zeros((5, 10), np.float32)
    r[:, [0, 3]] = [[1, 2], [1, 2], [100, 2], [1, 2], [1, 2]]
    obs = np.zeros_like(r, bool)
    obs[:, [0, 3]] = True
    arms, eligible, audit = innovations(r, obs, np.ones(10))
    assert eligible.all() and audit['background_updates'] == 4
    np.testing.assert_allclose(arms['intra'][0, 0], np.tanh(1))
    np.testing.assert_allclose(arms['intra'][1, 0], np.tanh(.96))
    np.testing.assert_allclose(arms['intra'][3, 0], np.tanh(.92), atol=1e-7)
    np.testing.assert_allclose(arms['intra'][:, 3], np.tanh(2))
    short, _, _ = innovations(r[:3], obs[:3], np.ones(10))
    for key in arms:
        np.testing.assert_array_equal(short[key], arms[key][:3])
    reset, _, _ = innovations(r[3:], obs[3:], np.ones(10))
    np.testing.assert_allclose(reset['intra'][0, 0], np.tanh(1))


def test_background_bounds_gate_and_non_unit_clipping():
    r = np.zeros((200, 10), np.float32)
    r[:, 0] = 1.5
    r[:, 3] = -1.5
    obs = np.zeros_like(r, bool)
    obs[:, [0, 3]] = True
    arms, eligible, audit = innovations(r, obs, np.ones(10))
    assert eligible.all() and audit['background_updates'] == 400
    np.testing.assert_allclose(arms['intra'][-1, [0, 3]], np.tanh([.5, -.5]), atol=1e-6)
    r[:2, 0] = [20, 2]
    r[:2, 3] = [4, 4]
    arms, _, _ = innovations(r[:2], obs[:2], np.ones(10))
    np.testing.assert_allclose(arms['no_background'][0, [0, 3]], np.tanh([8, 4]))
    np.testing.assert_allclose(arms['intra'][1, [0, 3]], np.tanh([2, 4]))


def test_invalid_input_contracts_and_unobserved_nonfinite():
    z = np.zeros((60, 10))
    with pytest.raises(ValueError):
        raw_feedback(z, np.ones_like(z), [31])
    with pytest.raises(ValueError):
        raw_feedback(z[:, :9], np.ones((60, 9), bool), [31])
    with pytest.raises(ValueError):
        physical_keys(np.zeros((5, 3)), np.ones((5, 3), bool))
    with pytest.raises(ValueError):
        innovations(np.zeros((1, 10)), np.ones((1, 10), bool), np.zeros(10))
    with pytest.raises(ValueError):
        innovations(np.zeros((1, 9)), np.ones((1, 9), bool), np.ones(10))
    r = np.zeros((2, 10))
    r[:, 0] = np.nan
    r[:, 3] = 1
    obs = np.ones_like(r, bool)
    arms, eligible, _ = innovations(r, obs, np.ones(10))
    assert eligible.all() and all(np.isfinite(x).all() for x in arms.values())
    assert (arms['intra'][:, 0] == 0).all()
