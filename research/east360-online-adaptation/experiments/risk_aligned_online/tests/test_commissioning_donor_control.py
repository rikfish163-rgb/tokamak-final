import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_donor_control_20261111 as donor


def test_match_uses_only_ended_other_shots_with_same_support_and_near_context():
    query = dict(context=np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]]),
                 group=np.array([31, 33]))
    bank = dict(context=np.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                  [0., 0., 0., 0.], [.4, 0., 0., 0.],
                                  [0., 0., 0., 1.1], [.5, 0., 0., 0.]]),
                group=np.array([31, 31, 32, 31, 33, 31]),
                rank=np.array([5, 6, 3, 4, 4, 1]))
    chosen, distance = donor.select(query, bank, rank=5)
    np.testing.assert_array_equal(chosen, [3, -1])
    np.testing.assert_allclose(distance, [.4, np.inf])


def test_context_cannot_see_current_point_or_the_future():
    values = np.tile([300., -2.], (90, 1))
    observed = np.ones_like(values, bool)
    releases = np.array([36, 68])
    values[20:36, 0] = 306.
    a = donor.match_context(values, observed, releases)
    b = donor.match_context(values[:36], observed[:36], releases[:1])
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x[:1], y)
    assert a[1].tolist() == [False, False]
    assert a[2].tolist() == [True, True]
    np.testing.assert_allclose(a[0][0, 2], (306. - 300.) / (306. * .02))


def test_context_requires_observed_recent_field_and_both_current_halves():
    values = np.tile([300., 2.], (100, 1))
    observed = np.ones_like(values, bool)
    observed[10:19, 0] = False
    observed[62:71, 1] = False
    _, _, valid = donor.match_context(values, observed, np.array([42, 74]))
    np.testing.assert_array_equal(valid, [False, False])


def test_mapping_changes_direction_not_magnitude_and_has_no_unmatched_action():
    own = torch.tensor([[3., 4.], [0., 0.], [0., 7.], [0., 2.]])
    other = torch.tensor([[0., -2.], [1., 0.], [4., 0.], [0., 0.]])
    moved, applied = donor.equalized_delta(own, other, np.array([True, True, False, True]))
    assert applied.tolist() == [True, False, False, False]
    torch.testing.assert_close(moved, torch.tensor([[0., -5.], [0., 0.], [0., 0.], [0., 0.]]))


def test_donor_state_starts_at_release_never_fills_unmatched_blocks():
    release = np.array([217, 249, 313])
    selected = np.array([1, -1, 0])
    pool = np.array([[1., 2.], [3., 4.]], np.float32)
    states, support = donor.release_states(318, release, selected, pool)
    assert not support[:217].any()
    assert support[217:249].all()
    assert not support[249:313].any()
    assert support[313:].all()
    np.testing.assert_array_equal(states[217], pool[1])
    np.testing.assert_array_equal(states[-1], pool[0])
