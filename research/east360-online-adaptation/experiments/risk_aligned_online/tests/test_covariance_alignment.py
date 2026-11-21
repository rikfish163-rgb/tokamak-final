import copy

import numpy as np
import pytest

from experiments.risk_aligned_online.covariance_alignment_20261022.alignment import (
    BLOCKS, summarize, combine, make_block_map, fit_maps, transform,
)
from experiments.risk_aligned_online.io23 import RAW23


def record(sid=1, device='J-TEXT', n=50):
    rng = np.random.default_rng(sid)
    x = rng.normal(size=(n, 23)).astype(np.float32)
    return dict(shot=sid, device=device, values=x, available=np.ones_like(x, bool))


def summary(r):
    return summarize(r, center=np.zeros(23), scale=np.ones(23))


def test_sampling_uses_common_timestamps_not_separate_channel_lists():
    r = record(n=200)
    cols = [RAW23.index(n) for n in BLOCKS['sxr']]
    r['available'][:50, cols[0]] = False
    r['available'][150:, cols[1]] = False
    s = summary(r)['blocks']['sxr']
    assert s['observed_rows'] == 100 and s['sample_rows'] == 100
    z = np.arcsinh(r['values'][50:150, cols].astype(float))
    np.testing.assert_allclose(s['mean'], z.mean(0))
    np.testing.assert_allclose(s['within_cov'], np.cov(z, rowvar=False, bias=True), atol=1e-15)


def test_total_covariance_decomposes_into_equal_shot_within_and_between():
    a, b = record(n=5), record(2, n=500)
    a['values'][:] = 0.; b['values'][:] = np.sinh(2.)
    stats = combine([summary(a), summary(b)], device='J-TEXT', allowed_ids=[1, 2])
    s = stats['blocks']['sxr']
    np.testing.assert_allclose(s['mean'], 1., atol=1e-7)
    np.testing.assert_allclose(s['within_cov'], 0., atol=1e-14)
    np.testing.assert_allclose(s['total_cov'], np.ones((4, 4)), atol=1e-6)
    assert s['observed_shots'] == 2


def stats_from(mu, cov):
    return dict(mean=np.asarray(mu).tolist(), total_cov=np.asarray(cov).tolist(),
                within_cov=np.asarray(cov).tolist(), observed_shots=10)


def test_full_coral_transports_regularized_correlation_with_correct_orientation():
    cs = np.array([[4., 1.], [1., 1.]])
    ct = np.array([[1., -.9], [-.9, 9.]])
    m = make_block_map(stats_from([0., 0.], cs), stats_from([1., 2.], ct), 'coral')
    a = np.array(m['matrix'])
    # Row-vector covariance is A.T @ C @ A. Ridge=1 is added in
    # correlation coordinates, equivalent to adding each covariance diagonal.
    np.testing.assert_allclose(a.T @ (ct + np.diag(np.diag(ct))) @ a,
                               cs + np.diag(np.diag(cs)), atol=1e-12)
    assert m['regularized_residual_relative'] < 1e-12
    assert np.max(np.abs(a - np.diag(np.diag(a)))) > .01


def test_diagonal_and_mean_controls_are_not_full_covariance_alignment():
    s = stats_from([2., 3.], [[4., 1.], [1., 9.]])
    t = stats_from([1., -2.], [[1., -.5], [-.5, 4.]])
    d = make_block_map(s, t, 'diagonal')
    np.testing.assert_allclose(d['matrix'], np.diag([2., 1.5]))
    np.testing.assert_array_equal(make_block_map(s, t, 'mean')['matrix'], np.eye(2))


def test_singular_correlations_are_regularized_without_inverting_zero_modes():
    c = np.ones((4, 4))
    m = make_block_map(stats_from(np.zeros(4), c), stats_from(np.ones(4), c), 'coral')
    np.testing.assert_allclose(m['matrix'], np.eye(4), atol=1e-12)
    assert m['target_condition_regularized'] == pytest.approx(5.)
    bad = make_block_map(stats_from([0., 0.], np.eye(2)),
                         stats_from([0., 0.], np.diag([1., 0.])), 'coral')
    assert bad['status'] == 'unidentifiable_constant_block_identity'


def test_covariance_cannot_be_fitted_with_labels_or_wrong_prefix():
    r = record(); r['y'] = np.zeros(50)
    with pytest.raises(ValueError, match='signal'): summary(r)
    source = combine([summary(record())], device='J-TEXT', allowed_ids=[1])
    target = [summary(record(10, 'EAST')), summary(record(11, 'EAST'))]
    with pytest.raises(ValueError, match='prefix'):
        fit_maps(source, target[::-1], order=[10, 11, 12], count=2, policy='coral')
    with pytest.raises(ValueError, match='authorized'):
        combine(target, device='EAST', allowed_ids=[10])


def test_missing_block_is_unchanged_not_imputed_and_never_changes_availability():
    src = combine([summary(record())], device='J-TEXT', allowed_ids=[1])
    tar = [summary(record(10, 'EAST'))]
    m = fit_maps(src, tar, order=[10, 11], count=1, policy='coral')
    r = record(20, 'EAST'); cols = [RAW23.index(n) for n in BLOCKS['sxr']]
    r['available'][0, cols[0]] = False
    signal = {k:r[k].copy() for k in ('values', 'available')}
    original = copy.deepcopy(signal)
    out, audit = transform(signal, m, center=np.zeros(23), scale=np.ones(23))
    np.testing.assert_array_equal(out['values'][0, cols], original['values'][0, cols])
    np.testing.assert_array_equal(out['available'], original['available'])
    np.testing.assert_array_equal(signal['values'], original['values'])
    assert audit['sxr']['incomplete_rows_identity'] == 1
    selected = {RAW23.index(n) for block in BLOCKS.values() for n in block}
    for c in set(range(23)) - selected:
        np.testing.assert_array_equal(out['values'][:, c], original['values'][:, c])
    signal['values'][25:] *= 10.
    changed, _ = transform(signal, m, center=np.zeros(23), scale=np.ones(23))
    np.testing.assert_array_equal(changed['values'][:25], out['values'][:25])


def test_mean_only_has_effect_even_if_covariance_unidentifiable():
    m = make_block_map(stats_from([1., 2.], np.zeros((2, 2))),
                       stats_from([3., 4.], np.zeros((2, 2))), 'mean')
    assert m['status'] == 'mapped'
    np.testing.assert_array_equal(m['matrix'], np.eye(2))


def test_unknown_policy_and_duplicate_shots_rejected():
    s = summary(record())
    with pytest.raises(ValueError): combine([s, s], device='J-TEXT', allowed_ids=[1])
    st = stats_from([0., 0.], np.eye(2))
    with pytest.raises(ValueError): make_block_map(st, st, 'tuned_on_test')


def test_replay_detects_wrong_arm_and_even_tiny_threshold_crossing_changes():
    from experiments.risk_aligned_online.covariance_alignment_20261022.audit import compare_replay
    assert compare_replay(np.array([.2, .8]), np.array([.2, .8]), .5) == 0.
    with pytest.raises(AssertionError, match='map-to-prediction'):
        compare_replay(np.array([.2, .8]), np.array([.21, .8]), .5)
    with pytest.raises(AssertionError, match='map-to-prediction'):
        compare_replay(np.array([.4999999]), np.array([.5000001]), .5)
