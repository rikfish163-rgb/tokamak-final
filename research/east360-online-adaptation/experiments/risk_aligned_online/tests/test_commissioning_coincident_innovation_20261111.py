import numpy as np

from experiments.risk_aligned_online import commissioning_coincident_innovation_20261111 as joint


def test_identical_activity_margins_can_have_different_coincidence():
    a = np.array([0., 1., 4., 2., 0.])
    activity = np.repeat(a[None, :, None], 4, axis=2)
    single, real, shifted = joint.signatures(activity)
    np.testing.assert_allclose(real, np.ones((1, 6)))
    assert np.all(shifted < .99)
    np.testing.assert_allclose(single, joint.signatures(activity + 0.)[0])
    rotated = joint.rotate_families(activity)
    np.testing.assert_array_equal(np.sort(rotated, axis=1), np.sort(activity, axis=1))


def test_flat_envelopes_have_no_evidence_of_coincidence():
    _, a, b = joint.signatures(np.ones((3, 5, 4)))
    np.testing.assert_array_equal(a, 0.)
    np.testing.assert_array_equal(b, 0.)


def test_feedback_uses_only_the_completed_block_and_fixed_prefix():
    rng = np.random.default_rng(31)
    z = rng.normal(size=(245, 10))
    observed = np.ones_like(z, dtype=bool)
    context = rng.normal(size=(len(z), 2))
    valid = np.ones_like(context, dtype=bool)
    full = joint.observed_features(z, observed, context, valid)
    prefix = joint.observed_features(z[:186], observed[:186], context[:186], valid[:186])
    assert prefix['release'].tolist() == [153, 185]
    for name in prefix:
        np.testing.assert_array_equal(prefix[name], full[name][full['release'] < 186])
    observed[150, 3] = False
    missing = joint.observed_features(z, observed, context, valid)
    assert 153 not in missing['release']


def test_outcome_support_strict_deadline_and_first_false_alarm():
    t = np.arange(300) * .001
    r = np.array([153,185,217,249])
    p = np.zeros(300)
    p[235:] = .8
    d = joint.case_support(t, p, r, True, .247, .5)
    assert d['late'] and d['positive'].tolist() == [True, True, False, False]
    safe = joint.case_support(t, p, r, False, None, .5)
    assert safe['fp'] and safe['safe'].tolist() == [True, True, True, False]
    early = joint.case_support(t, np.ones(300), r, False, None, .5)
    assert early['pre_feedback_fp'] and not early['safe'].any()


def test_equal_budget_broad_history_uses_most_recent_disjoint_pairs():
    from experiments.risk_aligned_online import commissioning_coincident_support_audit_20261111 as audit

    rank = np.array([20, 21, 80, 44, 100, 72])
    pairs = np.array([[0,1],[2,3],[4,5]])
    np.testing.assert_array_equal(audit.recent_pairs(rank, pairs, 2), [[4,5],[2,3]])
