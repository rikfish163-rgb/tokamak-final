import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_inverse_core_20261114 import (
    block_system,
    mature_history,
    solve_step,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_block_system_removes_channel_offsets_and_rolls_only_residual(dtype):
    f = torch.tensor([-2, -1, 0, 1, 2], dtype=dtype)
    g = torch.tensor([2, -1, -2, -1, 2], dtype=dtype)
    jacobian = torch.stack(
        (torch.stack((f, g), dim=-1), torch.stack((2 * f, -g), dim=-1)),
        dim=1,
    )
    coefficient = torch.tensor([1.5, -0.75], dtype=dtype)
    residual = jacobian @ coefficient

    gram, real, shifted = block_system(jacobian, residual)
    torch.testing.assert_close(gram, torch.diag(torch.tensor([50, 28], dtype=dtype)))
    torch.testing.assert_close(real, torch.tensor([75, -21], dtype=dtype))
    torch.testing.assert_close(shifted, torch.tensor([-41.25, 9], dtype=dtype))

    jacobian_offset = torch.tensor([[11, -3], [-6, 8]], dtype=dtype)
    residual_offset = torch.tensor([13, -9], dtype=dtype)
    moved = block_system(jacobian + jacobian_offset, residual + residual_offset)
    for actual, expected in zip(moved, (gram, real, shifted), strict=True):
        torch.testing.assert_close(actual, expected)


def test_block_system_preserves_arbitrary_leading_batch_dimensions():
    generator = torch.Generator().manual_seed(1114)
    jacobian = torch.randn(2, 3, 5, 8, 4, generator=generator)
    residual = torch.randn(2, 3, 5, 8, generator=generator)

    gram, real, shifted = block_system(jacobian, residual)
    assert gram.shape == (2, 3, 4, 4)
    assert real.shape == shifted.shape == (2, 3, 4)
    for batch in range(2):
        for member in range(3):
            expected = block_system(jacobian[batch, member], residual[batch, member])
            for actual, single in zip((gram, real, shifted), expected, strict=True):
                torch.testing.assert_close(actual[batch, member], single)


def test_known_trajectory_solves_ridge_using_only_mature_windows():
    generator = torch.Generator().manual_seed(360)
    dtype = torch.float64
    coefficient = torch.randn(60, generator=generator, dtype=dtype)
    jacobian = torch.randn(3, 5, 8, 60, generator=generator, dtype=dtype)
    offsets = torch.randn(3, 1, 8, generator=generator, dtype=dtype)
    residual = jacobian @ coefficient + offsets
    gram, real, shifted = block_system(jacobian, residual)
    gram, real, shifted = gram.sum(dim=0), real.sum(dim=0), shifted.sum(dim=0)

    estimate = solve_step(gram, real)
    shifted_estimate = solve_step(gram, shifted)
    ridge = 0.1 * gram.trace() / gram.shape[-1]
    assert ridge > 0
    torch.testing.assert_close(real, gram @ coefficient)
    torch.testing.assert_close(
        (gram + ridge * torch.eye(60, dtype=dtype)) @ estimate, real
    )
    assert not torch.allclose(estimate, shifted_estimate)

    # A separate synthetic window is assessed only after the step is calculated.
    held_jacobian = torch.randn(5, 8, 60, generator=generator, dtype=dtype)
    held_residual = held_jacobian @ coefficient
    centered_jacobian = held_jacobian - held_jacobian.mean(dim=0, keepdim=True)
    centered_residual = held_residual - held_residual.mean(dim=0, keepdim=True)
    assert (centered_jacobian @ estimate - centered_residual).norm() < centered_residual.norm()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_solve_step_batches_rank_deficiency_and_zero_system(dtype):
    gram = torch.zeros(2, 2, 3, 3, dtype=dtype)
    gram[0, 1, 0, 0] = 3
    gram[1, 0, 0, 0] = 6
    gram[1, 1] = torch.eye(3, dtype=dtype)
    rhs = torch.tensor(
        [[[4, 5, 6], [3, 0, 0]], [[6, 0, 0], [1, 2, 3]]], dtype=dtype
    )
    result = solve_step(gram, rhs, damping=0.1)

    assert result.shape == (2, 2, 3)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    expected = torch.tensor(
        [[[0, 0, 0], [3 / 3.1, 0, 0]],
         [[6 / 6.2, 0, 0], [1 / 1.1, 2 / 1.1, 3 / 1.1]]],
        dtype=dtype,
    )
    torch.testing.assert_close(result, expected)
    more_damping = solve_step(gram, rhs, damping=0.2)
    torch.testing.assert_close(
        more_damping[0, 1, 0], torch.tensor(3 / 3.2, dtype=dtype)
    )


def test_mature_history_uses_release_and_inclusive_physical_time_bounds():
    q = np.array([0, 5, 10, 31, 35, 36, 40, 45, 80])
    eligible = np.array([True, False, True, True, True, True, True, False, True])
    history = mature_history(q, eligible, horizon=30, stride=5, count=9)

    assert history.shape == (len(q), 9)
    assert np.issubdtype(history.dtype, np.integer)
    np.testing.assert_array_equal(history[2], [-1, -1, -1, -1, -1, -1, -1, -1, 2])
    np.testing.assert_array_equal(history[3], [-1, -1, -1, -1, -1, -1, -1, 0, 3])
    np.testing.assert_array_equal(history[6], [-1, -1, -1, -1, -1, -1, -1, 0, 6])
    np.testing.assert_array_equal(history[7], [-1, -1, -1, -1, -1, -1, -1, 2, -1])
    np.testing.assert_array_equal(history[8], [-1, -1, -1, -1, -1, -1, -1, 6, 8])
    for j, row in enumerate(history):
        past = row[:-1][row[:-1] >= 0]
        assert np.all(past < j)
        assert np.all(eligible[past])
        assert np.all(q[past] + 31 <= q[j])
        assert np.all(q[past] >= q[j] - 8 * 5)
        assert np.all(np.diff(past) > 0)


def test_mature_history_does_not_use_future_rows_or_eligibility():
    q = np.array([0, 5, 10, 31, 35, 40, 45, 50, 55, 60])
    eligible = np.array([True, True, False, True, True, False, True, True, False, True])
    whole = mature_history(q, eligible, horizon=30, stride=5, count=10)
    for end in range(1, len(q) + 1):
        prefix = mature_history(q[:end], eligible[:end], horizon=30, stride=5, count=10)
        np.testing.assert_array_equal(whole[:end], prefix)
        altered_q = np.concatenate((q[:end], q[end:] + 1000))
        altered_eligible = eligible.copy()
        altered_eligible[end:] = ~altered_eligible[end:]
        altered = mature_history(altered_q, altered_eligible, horizon=30, stride=5, count=10)
        np.testing.assert_array_equal(whole[:end], altered[:end])


def test_mature_history_current_slot_is_not_a_release_time():
    q = np.array([0, 30, 31])
    eligible = np.array([True, True, True])
    history = mature_history(q, eligible, horizon=30, stride=5, count=8)
    np.testing.assert_array_equal(history[:, -1], [0, 1, 2])
    assert 0 not in history[1, :-1]  # q[0] + 31 has not arrived at q[1].
    assert history[2, -2] == 0     # It is mature exactly at q[2].


def test_mature_history_keeps_the_most_recent_eligible_indices():
    q = np.arange(8)
    eligible = np.array([True, True, True, True, True, False, True, True])
    history = mature_history(q, eligible, horizon=0, stride=5, count=4)
    np.testing.assert_array_equal(history[-1], [3, 4, 6, 7])


def test_mature_history_single_slot_empty_and_duplicate_times():
    q = np.array([0, 0, 1, 2])
    eligible = np.array([True, True, False, True])
    one = mature_history(q, eligible, horizon=0, stride=1, count=1)
    np.testing.assert_array_equal(one[:, 0], [0, 1, -1, 3])
    empty = mature_history(q[:0], eligible[:0], horizon=0, stride=1, count=3)
    assert empty.shape == (0, 3)
    history = mature_history(q, eligible, horizon=0, stride=1, count=3)
    np.testing.assert_array_equal(history[1], [-1, -1, 1])
    np.testing.assert_array_equal(history[2], [0, 1, -1])
