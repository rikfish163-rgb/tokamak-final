"""Reject inconsistent training attribution; retain the strict timely boundary."""
import copy

import pytest

from ..credit_horizon_20261023.audit import check_training
from ..horizon_memory_20261023.audit import independently_score


def fixture():
    entry = dict(step=0, episodes=[dict(shot=1, disruptive=False, count=512, ends=[512, 520]),
                                   dict(shot=2, disruptive=True, count=384, ends=[400])])
    row = dict(step=0)
    for arm, tail in [('tail2', 2), ('tail8', 8)]:
        episodes = []
        for e in entry['episodes']:
            counts = list(range(256, e['count']+1, 128))
            episodes.append(dict(**e, risk_before=.3, risk_after=.299, forecast=.2,
                                 objective=.2+(.299-.3)/.001, all_counts=counts,
                                 differentiated=counts[-tail:]))
        row[arm] = dict(episodes=episodes, outer_gradient_norm=.5)
    return [entry], [row]


def test_training_tail_and_preparation_identity_are_explicit():
    entries, history = fixture()
    assert check_training(entries, history, {1, 2}) == dict(episodes=4, forward_steps=10,
                                                          differentiated_steps=9)
    with pytest.raises(AssertionError):
        check_training(entries, history, {1})
    bad = copy.deepcopy(history)
    bad[0]['tail2']['episodes'][0]['differentiated'] = [256, 384, 512]
    with pytest.raises(AssertionError):
        check_training(entries, bad, {1, 2})
    bad = copy.deepcopy(history)
    bad[0]['tail8']['episodes'][0]['objective'] += 1.
    with pytest.raises(AssertionError):
        check_training(entries, bad, {1, 2})


def test_exact_30ms_is_not_timely():
    assert not independently_score(1, True, 1., [.970], [.8], .7)['timely']
    assert independently_score(1, True, 1., [.969], [.8], .7)['timely']
