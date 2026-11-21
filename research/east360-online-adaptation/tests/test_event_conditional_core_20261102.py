import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_conditional_feedback_20261102.context import (
    select_variation_features,
    probabilities,
)
from experiments.risk_aligned_online.event_conditional_feedback_20261102.core import (
    effective_table,
)


def test_context_uses_variation_not_mean_levels_and_mnm_is_masked():
    raw = np.arange(230, dtype=np.float32).reshape(2, 115)
    old = raw.copy()
    selected = select_variation_features(raw)
    assert not selected[:, :46].any()
    for block in (2, 3, 4):
        expected = raw[:, 23 * block : 23 * (block + 1)].copy()
        expected[:, 2] = 0  # RAW23 MNM, not a hard-coded new physical grouping.
        np.testing.assert_array_equal(
            selected[:, 23 * block : 23 * (block + 1)], expected
        )
    np.testing.assert_array_equal(raw, old)


def test_stable_soft_context_and_mean_level_changes_do_not_affect_it():
    head = dict(
        mean=np.zeros(115),
        scale=np.ones(115),
        weight=np.zeros((6, 115)),
        bias=np.arange(6) * 1000.0,
    )
    x = np.ones((1, 115), np.float32)
    a = probabilities(x, head)
    x[:, :46] *= 500
    b = probabilities(x, head)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1, 6) and np.isfinite(a).all()
    np.testing.assert_allclose(a.sum(1), 1.0)


def test_conditional_table_is_a_convex_mixture_and_keeps_all_expert_gradients():
    tables = (
        torch.arange(6 * 25 * 23, dtype=torch.double)
        .reshape(6, 25, 23)
        .requires_grad_()
    )
    q = torch.arange(1, 7, dtype=torch.double)
    q /= q.sum()
    result = effective_table(tables, q)
    expected = sum(q[i] * tables[i] for i in range(6))
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(tables.grad, q[:, None, None].expand_as(tables))
    permutation = [3, 1, 5, 0, 4, 2]
    torch.testing.assert_close(
        result, effective_table(tables[permutation], q[permutation])
    )


def test_shared_table_is_the_explicit_smaller_reference():
    table = torch.randn(25, 23, requires_grad=True)
    result = effective_table(table, torch.full((6,), 1 / 6))
    assert result is table


def test_fitted_context_matches_legacy_standard_deviation_floor_not_new_variance_floor():
    from experiments.risk_aligned_online.event_conditional_feedback_20261102.context_fit import (
        standardize_train,
    )

    raw = np.zeros((2, 115), np.float32)
    raw[:, 47] = [-1e-5, 1e-5]
    _, _, scale = standardize_train(dict(raw=raw, event_id=np.array([1, 2])))
    assert scale[47] == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "q",
    [
        torch.zeros(6),
        torch.ones(6),
        torch.tensor([1.0, 0, 0, 0, 0, -1]),
        torch.tensor([float("nan")] * 6),
    ],
)
def test_invalid_context_is_not_replaced_with_uniform(q):
    with pytest.raises(ValueError):
        effective_table(torch.zeros(6, 25, 23), q)
