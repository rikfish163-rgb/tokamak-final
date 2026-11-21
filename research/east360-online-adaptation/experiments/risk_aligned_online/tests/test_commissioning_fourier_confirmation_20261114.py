import numpy as np

from experiments.risk_aligned_online import commissioning_fourier_confirmation_20261114 as probe


def test_diagnostic_trends_are_causal_and_do_not_fill_missing_values():
    t = np.arange(226)
    x = np.zeros((len(t), 23))
    x[:, probe.COLS] = np.column_stack((-400. + t, np.full_like(t, 2.), 1.+t*.01))
    known = np.ones_like(x, bool)
    known[110, probe.COLS[1]] = False
    trail = [126, 127, 190]
    vals, ok = probe.features(dict(values=x, available=known), np.asarray(trail))
    cut, cut_ok = probe.features(dict(values=x[:191], available=known[:191]), np.asarray(trail))
    np.testing.assert_array_equal(vals, cut)
    np.testing.assert_array_equal(ok, cut_ok)
    assert not ok[0].any() and np.array_equal(ok[1], [True, False, True])
    assert vals[1, 0].max() < 0 and vals[1, 2].min() > 0
    assert not vals[1, 1].any()


def test_gate_can_only_admit_positive_old_increments_and_never_delays_original():
    parent = np.array([.30, .48, .75, .51, .46], np.float32)
    previous = np.array([.63, .54, .42, .51, .60], np.float32)
    evidence = np.array([.5, -.1, .9, .2, .0])
    out = probe.gate(parent, previous, evidence, np.array([False, True, True, True, True]), 0.)
    np.testing.assert_array_equal(out, parent)
    out = probe.gate(parent, previous, evidence, np.ones(5, bool), 0.)
    assert out[0] == previous[0]
    np.testing.assert_array_equal(out[1:], parent[1:])


def test_safety_cutoff_controls_whole_shot_proposals_even_with_ties():
    score = np.array([.3, .8, .8, .8, -.2, .7, .5, 1.3])
    cut = probe.safety_cut(score)
    assert cut == .8
    assert np.count_nonzero(score > cut) <= len(score)//4
