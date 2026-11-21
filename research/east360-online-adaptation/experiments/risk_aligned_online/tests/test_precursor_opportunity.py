import numpy as np
import pytest

from experiments.risk_aligned_online.precursor_opportunity_20261023.diagnostic import (
    changes, fit_reference, novelty, first_sustained, next_feedback,
)


def signal(n=400):
    return {'values': np.random.default_rng(8).normal(size=(n, 23)),
            'available': np.ones((n, 23), bool)}


def test_changes_are_two_observed_blocks_and_future_invariant():
    s = signal(); c = np.zeros(23); scale = np.ones(23)
    d, valid = changes(s, c, scale)
    q = np.arcsinh(s['values'])
    np.testing.assert_allclose(d[49], q[25:50].mean(0)-q[:25].mean(0), atol=1e-14)
    assert not valid[:49].any() and not valid[:, 10].any()
    t = {k: v.copy() for k, v in s.items()}; t['values'][200:] = 999
    e, ev = changes(t, c, scale)
    np.testing.assert_array_equal(d[:200], e[:200])
    np.testing.assert_array_equal(valid[:200], ev[:200])
    with pytest.raises(ValueError): changes(dict(s, y=1), c, scale)


def test_missing_channel_never_becomes_zero_observation():
    s = signal(); s['available'][30, 3] = False
    _, v = changes(s, np.zeros(23), np.ones(23))
    assert not v[49:80, 3].any() and v[80, 3]
    empty = signal(); empty['available'][:] = False
    d, v = changes(empty, np.zeros(23), np.ones(23))
    with pytest.raises(ValueError): fit_reference([(d, v)], [1])


def test_reference_constant_channels_excluded_and_two_channels_required():
    s = signal(); s['values'][:, 0] = 3.
    d, v = changes(s, np.zeros(23), np.ones(23))
    r = fit_reference([(d, v)], [1]); assert not r['active'][0]
    score, ratios = novelty(d, v, r)
    assert np.isfinite(score[49:]).all() and np.isnan(score[:49]).all()
    one = np.zeros_like(v); one[:, 3] = v[:, 3]
    a, _ = novelty(d, one, r); assert np.isnan(a).all()
    with pytest.raises(ValueError): fit_reference([(d, v), (d, v)], [1, 1])


def test_trigger_is_declared_at_third_point_and_feedback_is_causal():
    assert first_sustained(np.array([0., 2., 2., 2., 0.]), 1.) == 3
    assert first_sustained(np.array([2., np.nan, 2., 2.]), 1.) is None
    assert next_feedback(255, 1000) == 256
    assert next_feedback(256, 1000) == 384
    assert next_feedback(255, 1000, full_after=True) == 384
    assert next_feedback(999, 1000) is None


def test_independent_arithmetic_matches_on_masked_and_constant_channels():
    from experiments.risk_aligned_online.precursor_opportunity_20261023.audit import independent_change, independent_score
    s = signal(1000); s['values'][:, 0] = 100
    s['available'][70:90, 1] = False
    c, scale = np.zeros(23), np.ones(23)
    d, v = changes(s, c, scale); e, ev = independent_change(s, c, scale)
    np.testing.assert_array_equal(v, ev)
    np.testing.assert_allclose(d, e, atol=1e-12)
    ref = fit_reference([(d, v)], [1])
    score, ratios = novelty(d, v, ref); other, other_ratios = independent_score(e, ev, ref)
    np.testing.assert_allclose(score, other, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(ratios, other_ratios, atol=1e-10, equal_nan=True)
