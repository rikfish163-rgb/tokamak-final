import numpy as np
import torch
import pytest

from .policy import plan, history_partition, guard
from .context import ACTIONS


def signals(n=70):
    return dict(device='EAST', values=np.zeros((n, 23), np.float32),
                available=np.ones((n, 23), bool), times=np.arange(n)*.001)


def test_plan_zero_rule_is_identity_and_intervention_only_future():
    s = signals(); b = torch.zeros(3, 117, dtype=torch.float64)
    q, events = plan(s, b, torch.zeros(23), torch.ones(23))
    assert not q.any() and len(events) == 5
    modified, changed = plan(s, b, torch.zeros(23), torch.ones(23), intervention=(1, 1))
    assert not modified[:26].any()
    np.testing.assert_allclose(modified[26:], .08)
    assert events[1]['context'] == changed[1]['context']


@pytest.mark.parametrize('arm', ['H', 'P', 'L'])
def test_online_plan_cannot_see_outcome_or_future_suffix(arm):
    s = signals(); b = torch.zeros(3, 117, dtype=torch.float64)
    b[ACTIONS.index(1), -1] = 1
    q, events = plan(s, b, torch.zeros(23), torch.ones(23), arm=arm)
    prefix = {k: v[:40] if isinstance(v, np.ndarray) else v for k, v in s.items()}
    qp, ep = plan(prefix, b, torch.zeros(23), torch.ones(23), arm=arm)
    np.testing.assert_array_equal(q[:40], qp)
    assert [e for e in events if e['end'] < 39] == ep


def test_history_partition_is_by_shot_not_label():
    a = history_partition([dict(shot=i, is_disrupt=False) for i in range(30)])
    b = history_partition([dict(shot=i, is_disrupt=True) for i in range(30)])
    assert [[r['shot'] for r in rows] for rows in a] == [[r['shot'] for r in rows] for rows in b]
    assert set(r['shot'] for r in a[0]).isdisjoint(r['shot'] for r in a[1])


def test_guard_requires_no_tradeoff_and_both_class_risks():
    old = dict(tp=1, fp=1, risks={'0': .2, '1': 1.})
    assert guard(dict(tp=1, fp=0, risks={'0': .1, '1': 1.}), old)
    assert not guard(dict(tp=0, fp=0, risks={'0': .1, '1': 1.}), old)
    assert not guard(dict(tp=1, fp=0, risks={'0': .1, '1': 1.1}), old)
    assert not guard(old, old)
