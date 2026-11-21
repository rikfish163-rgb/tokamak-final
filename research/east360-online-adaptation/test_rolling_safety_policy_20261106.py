import itertools

import numpy as np
import pytest
import torch

from rolling_safety_policy_20261106 import (
    make_teacher,
    no_alarm_log,
    positive_probability,
    safe_weights,
    splice,
)


def test_first_in_interval_matches_enumerated_block_choices():
    # First block has an early hit only if writing. Second can alert in time
    # only if writing. The same action is held within a block.
    g = torch.tensor([.4, .25], dtype=torch.float64)
    early = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64)
    until = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
    total = 0.
    for a in itertools.product((0, 1), repeat=2):
        weight = np.prod([float(g[i] if k else (1-g[i])) for i,k in enumerate(a)])
        total += weight * (np.prod([early[i,k] for i,k in enumerate(a)]) -
                           np.prod([until[i,k] for i,k in enumerate(a)]))
    out = positive_probability(g, early, until, torch.ones(2, dtype=torch.float64))
    assert float(out) == pytest.approx(total)
    assert float(out) == pytest.approx(.15)
    assert float(no_alarm_log(g, early).exp()) == pytest.approx(.6)


def test_immutable_prefix_can_make_a_timely_alarm_impossible_or_already_true():
    g = torch.tensor([.5], dtype=torch.float64)
    before = torch.ones((1,2), dtype=torch.float64)
    within = torch.tensor([[1., 0.]], dtype=torch.float64)
    assert float(positive_probability(g, before, within, torch.tensor([0., 0.]))) == 0.
    assert float(positive_probability(g, before, within, torch.tensor([1., 0.]))) == 1.


def test_teacher_excludes_exactly_30_ms_and_counts_exactly_3000_as_timely():
    times = np.array([.5,1.,2.,3.9,3.97,3.99])
    native = np.array([.1,.2,.1,.1,.99,.99])
    adapted = np.array([.1,.8,.1,.1,.99,.99])
    teacher = make_teacher(times, native, adapted, 4., .5, first=1, stride=2)
    assert len(teacher.until) == 3
    np.testing.assert_allclose(teacher.early, 1.)
    assert teacher.until[0,1] < .01
    assert teacher.until[0,0] > .99
    assert teacher.until[1,1] > .99
    assert teacher.until[2,1] == 1.


def test_safety_rolling_weights_include_all_blocks_without_more_total_shot_weight():
    q = np.array([[.98,.40],[.55,.52],[.89,.2]])
    rolling = safe_weights(q,"rolling")
    assert np.all(rolling > 0.)
    assert float(rolling.sum()) == pytest.approx(1.)
    np.testing.assert_array_equal(safe_weights(q,"peak1"), [0.,1.,0.])
    assert rolling[-1] > rolling[0]
    assert float(safe_weights(q,"peak8").sum()) == pytest.approx(1.)


def test_splice_keeps_past_and_changes_only_selected_update_blocks():
    a,b = np.arange(10),100+np.arange(10)
    np.testing.assert_array_equal(splice(a,b,np.array([False,True,False]),first=2,stride=3),
                                  [0,1,2,3,4,105,106,107,8,9])
    with pytest.raises(ValueError):
        splice(a,b,np.array([True]),first=2,stride=3)
