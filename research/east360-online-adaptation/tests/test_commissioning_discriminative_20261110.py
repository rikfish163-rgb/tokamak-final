import numpy as np
import torch

from experiments.risk_aligned_online.commissioning_discriminative_20261110 import (
    Episode, Rule, causal_history, exact_replay, fit_rule, make_folds, project_logits,
)
from experiments.risk_aligned_online.commissioning_transfer_20261108 import predict
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_history_writes_only_after_matured_block_and_resets():
    x = np.zeros((2, 46), dtype=np.float32)
    x[0, 0], x[1, 0] = 1, -3
    full, index = causal_history(x, 200)
    np.testing.assert_array_equal(full[index[:153]], np.zeros((153, 46)))
    assert index[153] == 1 and index[184] == 1 and index[185] == 2
    assert full[1, 0] > 0 and full[2, 0] < 0
    head, head_index = causal_history(x[:1], 179)
    np.testing.assert_array_equal(head[head_index], full[index[:179]])
    no_updates, idx = causal_history(x[:0], 153)
    assert not no_updates[idx].any()


def test_stratified_folds_have_complete_disjoint_shot_support():
    ids = list(range(20))
    outcomes = {s: (s % 5 < 2, None) for s in ids}
    pairs = make_folds(ids, outcomes)
    assert sorted(s for _, held in pairs for s in held) == ids
    for fit, held in pairs:
        assert len(fit) == 15 and len(held) == 5
        assert set(fit).isdisjoint(held)
        assert sum(outcomes[s][0] for s in held) == 2


def test_rule_learns_opposite_updates_without_score_offsets_on_zero_evidence():
    def make(sid, positive):
        history = torch.zeros(2, 46)
        history[1, 0] = 1. if positive else -1.
        jacobian = torch.zeros(3, 8)
        jacobian[1:, 0] = 1.
        return Episode(sid, positive, history, torch.tensor([0, 1, 1]), torch.zeros(3),
                       jacobian, torch.tensor([False, True, True]))

    a, b = make(1, True), make(2, False)
    rule = fit_rule([a, b], .5)
    up, down = project_logits(a, rule), project_logits(b, rule)
    assert up[0] == 0 and down[0] == 0
    assert (up[1:] > 0).all() and (down[1:] < 0).all()
    assert rule.states(a.history).norm(dim=-1).max() <= 1.000001


def test_exact_feedback_only_changes_post_arrival_prefix_and_is_repeatable():
    torch.manual_seed(131)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(dim=8, heads=2, layers=2, fast_site="tokens"),
                    np.zeros(23), np.ones(23)).eval().requires_grad_(False)
    rng = np.random.default_rng(131)
    values = rng.normal(size=(235, 23)).cumsum(0).astype(np.float32)
    available = np.ones_like(values, bool)
    coef = torch.randn(8 + 23 + 1) / 5.
    rule = Rule(torch.ones(46), torch.eye(8)[:, :4], torch.randn(4, 46))

    def replay(n):
        signal = dict(values=values[:n], available=available[:n])
        static = predict(base, signal, {"cold": coef}, "cpu")["cold"]
        return static, exact_replay(base, coef, signal, rule, static)

    static, (full, writes) = replay(235)
    _, (short, _) = replay(174)
    _, (again, _) = replay(235)
    for arm, curve in full.items():
        assert writes[arm] > 0
        np.testing.assert_array_equal(static[:153], curve[:153])
        assert not np.array_equal(static[153:], curve[153:])
        np.testing.assert_allclose(short[arm], curve[:174], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(again[arm], curve)
