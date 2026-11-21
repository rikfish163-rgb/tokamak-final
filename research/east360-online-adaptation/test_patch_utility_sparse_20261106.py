"""Synthetic tests for sparse hard-action weights and causal single writes."""

import numpy as np
import pytest

import run_patch_utility_sparse_20261106 as sparse


def test_same_class_balances_shots_blocks_and_actions():
    hard = np.array([[0, 1, 1, -1], [0, 1, 0, 0], [0, 1, 0, 0]])
    allowed = np.ones_like(hard, dtype=bool)
    owner = np.array([0, 0, 1])
    weights = sparse.event_weights(hard, allowed, owner)
    assert weights[0, 1] == pytest.approx(1 / 24)
    assert weights[0, 2] == pytest.approx(1 / 24)
    assert weights[1, 1] == pytest.approx(1 / 12)
    assert weights[2, 1] == pytest.approx(1 / 6)
    for label in (-1, 0, 1):
        assert weights[hard == label].sum() == pytest.approx(1 / 3)
    assert not weights[:, 0].any()
    with pytest.raises(ValueError, match="class"):
        sparse.event_weights(abs(hard), allowed, owner)


def test_first_admissible_action_is_causal_and_unique():
    owner = np.array([0, 0, 0, 1, 1, 2])
    scores = np.array(
        [[0, 0.5, 40], [0, 0.6, 0.2], [0, 99, 90],
         [0, 0.6, 0.8], [0, 90, 95], [0, 20, 22]], dtype=float,
    )
    ok = np.ones_like(scores, dtype=bool)
    ok[0, 2] = False
    ok[-1, 1:] = False
    expected = [0, 1, 0, 2, 0, 0]
    assert sparse.choose_once(scores, ok, owner, 0.5).tolist() == expected
    assert sparse.choose_once(scores[:2], ok[:2], owner[:2], 0.5).tolist() == expected[:2]
    assert not sparse.choose_once(scores, ok, owner, np.inf).any()


def test_alarm_so_far_closes_writes_without_label():
    allowed = np.ones((3, 3), dtype=bool)
    data = {"allowed": allowed, "risk": np.zeros((3, 10), dtype=np.float32)}
    data["risk"][:, 8] = [0.0, 0.5, 0.8]
    assert sparse.eligible_menu(data, 0.5).tolist() == [[True] * 3, [True, False, False], [True, False, False]]
    assert allowed.all()


def test_rank_receipt_is_a_per_action_diagnostic():
    score = np.array([[0, -1, 2], [0, 1, 2], [0, 3, 2], [0, 9, 10.0]])
    hard = np.array([[0, 0, 1], [0, 1, -1], [0, -1, 1], [0, 0, 1]])
    allowed = np.ones_like(hard, dtype=bool)
    allowed[-1, 1:] = False
    rank = sparse.rank_support(score, allowed, hard, np.array([0, 0, 1, 2]), np.array([101, 102, 103]))
    assert rank["oracle_shots"] == [101, 102]
    assert rank["top_action_shots"] == [101]
    assert rank["top_action_negative_blocks"] == 2


def test_multinomial_receipt_predicts_the_fit_ranking():
    x = np.array([[[-8], [-2], [0], [3]], [[-8], [-3], [0], [2]]], dtype=float)
    hard = np.array([[0, -1, 0, 1], [0, -1, 0, 1]])
    ok = np.ones_like(hard, dtype=bool)
    model = sparse.fit_events(x, hard, ok, np.array([0, 1]))
    score = sparse.event_scores(model, x)
    assert np.all(score[:, 3] > score[:, 2])
    assert np.all(score[:, 2] > score[:, 1])
