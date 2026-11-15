import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_first_feedback_20261112 as first
from experiments.risk_aligned_online import commissioning_innovation_metric_20261112 as metric


def test_first_stage_starts_on_complete_release_and_ends_after_one_handoff():
    releases = np.arange(153, 410, 32)
    eligible = np.array([False, True, True, True, True, True, False, True, True])
    common = np.array([False, False, False, False, True, True, False, False, True])
    early = first.bootstrap_slots(eligible, common)
    np.testing.assert_array_equal(np.flatnonzero(early), [1, 2, 3])
    affected = first.bootstrap_endpoints(releases, early, common, 445, 8)
    np.testing.assert_array_equal(np.flatnonzero(affected), np.arange(185, 289))
    cut = 249
    have = releases < cut
    prefix = first.bootstrap_slots(eligible[have], common[have])
    np.testing.assert_array_equal(
        first.bootstrap_endpoints(releases[have], prefix, common[have], cut, 8),
        affected[:cut],
    )


def test_missing_feedback_cannot_create_or_restart_the_early_stage():
    releases = np.array([153, 185, 217, 249, 281])
    eligible = np.array([False, True, False, True, True])
    common = np.array([False, False, False, True, False])
    early = first.bootstrap_slots(eligible, common)
    np.testing.assert_array_equal(np.flatnonzero(early), [1])
    affected = first.bootstrap_endpoints(releases, early, common, 345, 8)
    np.testing.assert_array_equal(np.flatnonzero(affected), np.arange(185, 217))
    assert not first.bootstrap_endpoints(np.array([], int), np.array([], bool),
                                        np.array([], bool), 100, 8).any()


def test_source_risk_mass_reserves_half_for_safe_and_a_quarter_for_hard_misses():
    rows = [dict(stratum=k) for k in ('hard', 'easy', 'safe', 'safe', 'easy', 'safe')]
    weight = first.stratum_mass(rows)
    np.testing.assert_allclose(weight, [.25, .125, 1/6, 1/6, .125, 1/6])
    with pytest.raises(ValueError, match='hard'):
        first.stratum_mass(rows[1:])
    scores = torch.tensor([-1., 1., .3], requires_grad=True)
    labels = torch.tensor([True, True, False])
    mass = torch.tensor([.25, .25, .50])
    loss = metric.shot_loss(scores, labels, 0., mass=mass)
    expected = (torch.nn.functional.softplus(torch.tensor([1., -1., .3])/metric.TEMPERATURE)
                * mass).sum()
    torch.testing.assert_close(loss, expected)
