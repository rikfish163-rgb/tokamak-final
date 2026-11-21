import numpy as np

from experiments.risk_aligned_online import commissioning_radiation_shape_20261114 as audit
from experiments.risk_aligned_online.io23 import RAW23


def test_shape_keeps_the_signed_moment_gap_and_uses_only_the_prefix():
    x = np.ones((70, 23), dtype=np.float32)
    valid = np.ones_like(x, dtype=bool)
    x[:35, RAW23.index('xuv_skew')] = np.sqrt(5)
    long, good = audit.measure(x, valid, np.array([33, 65]))
    short, ok = audit.measure(x[:34], valid[:34], np.array([33]))
    np.testing.assert_array_equal(good[:1], ok)
    np.testing.assert_array_equal(long[:1], short)
    np.testing.assert_allclose(short[0, 2], -2., atol=2e-6)


def test_missing_values_do_not_turn_into_low_or_zero_shape():
    x = np.ones((40, 23), dtype=np.float32)
    valid = np.ones_like(x, dtype=bool)
    valid[15, RAW23.index('xuv_kurt')] = False
    a, good = audit.measure(x, valid, np.array([39]))
    assert not good[0]
    assert not np.isfinite(a[0]).any()


def test_rematch_can_change_the_partner_without_sorting_by_shape_score():
    def row(shot, ctx, when, score):
        return dict(shot=shot, context=np.asarray(ctx),
                    end=when, score=np.asarray(score))
    pos = [row(11, [0.,0.], 120, [-3.,3.])]
    safe = [row(21, [.05,.9], 120, [-12.,-12.]),
            row(25, [.30,.10], 124, [9.,-9.])]
    got = audit.rematch(pos,safe,np.array([.5,.5]))
    assert got==[(0,1)]
