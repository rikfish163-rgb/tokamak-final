import numpy as np
import torch

from experiments.risk_aligned_online import commissioning_weak_shape_source_20261111 as source
from experiments.risk_aligned_online.commissioning_lag_state_20261110 import LagState, replay
from experiments.risk_aligned_online.commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE


def test_source_pairs_match_only_eligible_safe_windows_and_respect_strata():
    # 0 is timely; 1 is exactly 10 ms; 2 is too early to be in the training
    # window; 3 is a different disruptive shot. None of 1:4 is a safe donor.
    rows = dict(
        risk=np.array([1, 1, 0, 1, 0, 0, 0, 0], np.float32),
        disruptive=np.array([1, 1, 1, 1, 0, 0, 0, 0], bool),
        lead=np.array([30, 10, 100, 14, 0, 0, 0, 0], float),
        matchable=np.array([1, 1, 1, 0, 1, 1, 1, 0], bool),
        terminal=np.array([0, 0, 0, 0, 1, 1, 0, 1], bool),
        pair_context=np.broadcast_to([[0], [0], [0], [0], [2], [.25], [1], [0]], (8, 7)),
    )
    pos, neg, tail, distance = source.make_pairs(rows, np.ones(7))
    np.testing.assert_array_equal(pos, [0, 0])
    np.testing.assert_array_equal(neg, [5, 6])
    np.testing.assert_array_equal(tail, [True, False])
    np.testing.assert_allclose(distance, [.25, 1.])
    rand = source.make_pairs(rows, np.ones(7), random=np.random.default_rng(131))
    np.testing.assert_array_equal(rand[0], pos)
    np.testing.assert_array_equal(rows['terminal'][rand[1]], tail)
    assert set(rand[1]) <= {4, 5, 6}


def test_weak_shape_state_waits_for_complete_window_and_resets():
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    model = LagState(11, 15).eval()
    x = rng.normal(size=(280, 11)).astype(np.float32)
    origins = np.arange(FIRST, len(x)-HORIZON, STRIDE)
    feedback = rng.normal(size=(len(origins), 15)).astype(np.float32)
    eligible = np.ones(len(origins), bool)
    out, audit = replay(model, x, origins, feedback, -feedback, eligible)
    first = FIRST + HORIZON + 1
    for arm in ('real', 'shifted'):
        np.testing.assert_array_equal(out[arm][:first], out['fixed'][:first])
        assert not np.array_equal(out[arm][first:], out['fixed'][first:])
        assert audit['updates'][arm] == len(origins)
    prefix, _ = replay(model, x[:first], origins[:1], feedback[:1], -feedback[:1], eligible[:1])
    again, _ = replay(model, x, origins, feedback, -feedback, eligible)
    for arm in out:
        np.testing.assert_allclose(prefix[arm], out[arm][:first], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(again[arm], out[arm])
