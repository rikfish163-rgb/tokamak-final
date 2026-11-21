import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.conditional_weights_20261030.run import (
    collect, draw_balanced, screen, training_pools,
)


def test_pools_exclude_held_early_and_late():
    rows = [dict(shot=s, group=g) for s, g in
            [(1, 'safe'), (1, 'safe'), (2, 'danger'), (2, 'early'), (2, 'late'),
             (3, 'safe'), (4, 'danger')]]
    p = training_pools(rows, [1, 2])
    assert p == {0: {1: [0, 1]}, 1: {2: [2]}}
    draws = draw_balanced(p, np.random.default_rng(131))
    assert len(draws) == 24
    assert sum(i == 2 for i in draws) == 12
    assert set(draws) <= {0, 1, 2}


def test_missing_class_rejected():
    with pytest.raises(ValueError, match='both'):
        training_pools([dict(shot=1, group='safe')], [1])


def test_deterministic_draws():
    p = {0: {1: [0, 1]}, 1: {2: [2], 3: [3, 4]}}
    assert np.array_equal(draw_balanced(p, np.random.default_rng(131)),
                          draw_balanced(p, np.random.default_rng(131)))


def test_screen_needs_actual_and_feedback_benefit():
    groups = {a: {g: dict(mean_delta_bce=-.01 if a == 'real' else 0.,
                          improved=3 if g == 'safe' else 10, shots=3 if g == 'safe' else 17)
                  for g in ('safe', 'danger')}
              for a in ('real', 'reverse', 'mean_feedback', 'constant')}
    assert screen(groups, .5, .8)['passed']
    groups['mean_feedback']['safe']['mean_delta_bce'] = -.02
    assert not screen(groups, .5, .8)['passed']
    assert not screen(groups, 1., 0.)['passed']


def test_screen_rejects_no_direction():
    groups = {a: {g: dict(mean_delta_bce=0., improved=0, shots=3 if g == 'safe' else 17)
                  for g in ('safe', 'danger')}
              for a in ('real', 'reverse', 'mean_feedback', 'constant')}
    assert not screen(groups, None, 0.)['passed']


def test_arrived_context_does_not_see_query_or_future_signals():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    torch.set_num_threads(1)
    torch.manual_seed(131)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    signal = dict(values=np.random.default_rng(131).normal(size=(200, 23)).astype(np.float32),
                  available=np.ones((200, 23), dtype=bool))
    row = dict(origin=127, maturity=152, query=177)
    a = collect(net, signal, [row])
    changed = {k: v.copy() for k, v in signal.items()}
    changed['values'][153:] *= -7
    changed['available'][153:] = False
    b = collect(net, changed, [row])
    for key in ('x', 'mask', 'context', 'support'):
        assert torch.equal(a[key], b[key])
    assert not torch.equal(a['later_mask'], b['later_mask'])
