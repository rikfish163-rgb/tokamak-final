import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_lag_guard_20261111 as guard
from experiments.risk_aligned_online.commissioning_safe_support_20261111 import (
    extra_safe_loss, match_safety, reachable_anchor,
)


def test_near_threshold_point_must_be_reachable_by_a_completed_update():
    z = np.full(90, -2.)
    z[[9, 27, 45, 76]] = [-.02, -.09, -.37, -.21]
    available = np.ones(90, dtype=bool)
    available[76] = False
    # q=3 can first act at 29, so the first two peaks cannot train a guard.
    assert reachable_anchor(z, np.array([3, 35]), available) == (3, 45)


def test_pre_update_false_alarm_excludes_shot_from_protection_pool():
    z = np.full(90, -.4)
    z[0] = 0.
    assert reachable_anchor(z, np.array([3, 35]), np.ones(90,bool)) is None
    assert reachable_anchor(np.full(90,-.4), np.array([],int), np.ones(90,bool)) is None


def candidate(shot, margin, key):
    return dict(shot=shot, live_margin=margin, key=key)


def test_near_and_far_arms_add_same_distinct_operating_matched_shots():
    key = [0., 0., 0., 9., 1.]
    rows = [candidate(1,-.20,key), candidate(2,-.50,key),
            candidate(3,-.75,key), candidate(4,-1.0,[0., .8, 0., 9., 1.]),
            candidate(5,-.10,[0., -.8, 0., 9., 1.])]
    pairs = match_safety(rows, {1,2,3,4,5}, set())
    assert len(pairs) == 1
    n, f = pairs[0]
    assert n['shot'] in (1,2) and f['shot'] == 3
    assert n['live_margin'] >= -.50 > f['live_margin']


def test_selector_refuses_validation_or_reused_pair_data():
    rows = [candidate(7,-.2,[0.,0.,0.,9.,1.])]
    with pytest.raises(ValueError, match='role'):
        match_safety(rows, {8}, set())
    with pytest.raises(ValueError, match='role'):
        match_safety(rows, {7}, {7})


def test_total_safety_loss_weight_is_unchanged_when_repeating_same_support():
    d = torch.tensor([[.10,.05],[-.03,.06]])
    m, change = torch.tensor([-.02,.03]), torch.tensor([.04,-.02])
    expected = guard.guarded_loss(d,m,change)
    torch.testing.assert_close(extra_safe_loss(d,m,change), expected)
    torch.testing.assert_close(extra_safe_loss(d,m.repeat(2),change.repeat(2)), expected)


def test_additional_correctly_safe_crossing_has_nonzero_protection_gradient():
    d = torch.tensor([[.10,-.02]])
    vals = torch.tensor([-.02,.05], requires_grad=True)
    loss = extra_safe_loss(d,torch.tensor([-.03,-.02]),vals)
    loss.backward()
    assert vals.grad[1] > vals.grad[0] > 0.
