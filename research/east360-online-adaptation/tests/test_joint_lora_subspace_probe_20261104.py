"""Synthetic tests for warm-only rank-two LoRA subspace fitting."""

import numpy as np
import pytest
import torch

import joint_lora_subspace_probe_20261104 as probe


def test_symmetric_alignment_uses_normalized_d_transpose_r():
    d = torch.tensor([[1., 2.], [0., 1.]])
    r = torch.tensor([[0., 1.], [2., 1.]])
    actual, active = probe.alignment_matrix(d, r)
    expected = (d.double().T @ r.double() + r.double().T @ d.double()) / (2 * d.double().norm() * r.double().norm())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, actual.T)
    zero, active_zero = probe.alignment_matrix(torch.zeros_like(d), r)
    assert active and not active_zero and torch.count_nonzero(zero) == 0


def test_equal_group_shot_window_weights_retain_zero_contributions():
    rows = [
        dict(group="safe", shot=1, q=torch.tensor([[3.0]])),
        dict(group="safe", shot=1, q=torch.tensor([[3.0]])),
        dict(group="safe", shot=2, q=torch.tensor([[0.0]])),
        dict(group="earlier", shot=3, q=torch.tensor([[6.0]])),
        dict(group="positive", shot=3, q=torch.tensor([[9.0]])),
    ]
    weights = probe.balanced_weights(rows)
    np.testing.assert_allclose(weights, [1/12, 1/12, 1/6, 1/3, 1/3])
    assert float(probe.weighted_alignment(rows, "q")) == pytest.approx(5.5)
    with pytest.raises(ValueError, match="group"):
        probe.balanced_weights(rows[:-1])


def test_rank_two_eigenvectors_and_a_gram_are_preserved():
    a0 = torch.tensor([[1., 2., 0.], [0., 1., 3.]], dtype=torch.float32)
    covariance = torch.diag(torch.tensor([4., 2., -3.], dtype=torch.float64))
    new, eigvals, vectors = probe.new_subspace(a0, covariance)
    torch.testing.assert_close(new @ new.T, a0 @ a0.T, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(eigvals, torch.tensor([4., 2., -3.], dtype=torch.float64))
    torch.testing.assert_close(vectors @ vectors.T, torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64)))
    for col in vectors.T:
        assert col[col.abs().argmax()] >= 0


def test_subspace_objective_describes_only_first_sgd_direction():
    d = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
    r = torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64))
    c, _ = probe.alignment_matrix(d, r)
    a0 = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
    new, _, _ = probe.new_subspace(a0, c)
    new = new.double()
    exact = torch.sum(r * (d @ new.T @ new)) / (d.norm() * r.norm())
    spectral = torch.trace(new @ c @ new.T)
    torch.testing.assert_close(exact, spectral)


def test_true_adam_updates_only_256_b_parameters():
    a = torch.nn.Parameter(torch.randn(2, 64), requires_grad=False)
    bq = torch.nn.Parameter(torch.zeros(64, 2))
    bv = torch.nn.Parameter(torch.zeros(64, 2))
    a_before = a.detach().clone()
    starts = (bq.detach().clone(), bv.detach().clone())
    grads = (torch.full_like(bq, 2.), torch.full_like(bv, -1.))
    norm, displacement = probe.b_only_step((bq, bv), starts, grads)
    assert norm > 1 and 0 < displacement < 0.5
    assert torch.equal(a, a_before) and torch.count_nonzero(bq) and torch.count_nonzero(bv)
    expected_q = torch.nn.Parameter(starts[0].clone())
    expected_v = torch.nn.Parameter(starts[1].clone())
    opt = torch.optim.Adam([expected_q, expected_v], lr=1e-4)
    expected_q.grad, expected_v.grad = (g.clone() for g in grads)
    torch.nn.utils.clip_grad_norm_([expected_q, expected_v], 1.0, error_if_nonfinite=True)
    opt.step()
    torch.testing.assert_close(bq, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(bv, expected_v, rtol=0, atol=0)


def test_zero_b_makes_changed_input_subspace_risk_neutral():
    torch.manual_seed(11)
    weight = torch.randn(64, 64)
    x = torch.randn(4, 64)
    b = torch.zeros(64, 2)
    a0, anew = torch.randn(2, 64), torch.randn(2, 64)
    baseline = x @ weight.T
    assert torch.equal(x @ (weight + b @ a0).T, baseline)
    assert torch.equal(x @ (weight + b @ anew).T, baseline)


def test_summary_weights_shots_not_windows_and_preserves_noop():
    def row(sid, group, delta, supported):
        vals = {arm: {"bce": 1.0, "prob": 0.5} for arm in probe.ARMS}
        vals["new_motion"]["bce"] += delta
        return dict(shot=sid, group=group, supported=supported, arms=vals)
    rows = [row(1, "safe", -.2, 1), row(1, "safe", -.2, 1), row(2, "safe", 0, 0)]
    group = probe.summarize(rows)["safe"]
    assert group["shots"] == 2 and group["cases"] == 3 and group["no_support"] == 1
    pair = group["new_motion_vs_none"]
    assert pair["mean_delta"] == pytest.approx(-.1)
    assert (pair["improved"], pair["worse"], pair["ties"]) == (1, 0, 1)


def test_plan_separates_warm_fit_from_frozen_holdout_evaluation():
    plan = probe.fixed_plan(list(range(1, 21)), list(range(21, 101)), {})
    assert plan["train_indices"] == [1, 20] and plan["test_indices"] == [21, 100]
    assert plan["train_shots"] == list(range(1, 21))
    assert plan["test_shots"] == list(range(21, 101))
    assert plan["train_uses_original_packed_gradient"] is True
    assert plan["test_labels_fit"] is False and plan["independent_validation"] is False
