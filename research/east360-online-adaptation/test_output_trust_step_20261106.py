"""CPU-only contract tests for output-space symmetric trust steps."""

import numpy as np
import pytest
import torch

from output_trust_step_20261106 import LIMIT, SCALES, reached_indices, select_symmetric


def test_default_grid_and_limit_are_fixed():
    assert tuple(SCALES) == (1.0, 0.5, 0.25, 0.125, 0.0625)
    assert LIMIT == 0.10


def test_inclusive_limit_and_per_action_selected_maxima():
    shifts = torch.zeros((1, 2, 5, 3), dtype=torch.float64)
    shifts[0, 0, 0] = torch.tensor([0.10, -0.08, 0.02], dtype=torch.float64)
    shifts[0, 1, 0] = torch.tensor([-0.01, 0.03, -0.09], dtype=torch.float64)
    available = torch.ones((1, 2), dtype=torch.bool)

    radius, largest_abs = select_symmetric(shifts, available)

    torch.testing.assert_close(radius, torch.tensor([[1.0, 1.0]], dtype=radius.dtype))
    torch.testing.assert_close(
        largest_abs, torch.tensor([[0.10, 0.09]], dtype=shifts.dtype)
    )
    assert radius.device == shifts.device == largest_abs.device


@pytest.mark.parametrize("sign", [0, 1])
@pytest.mark.parametrize("clock", [0, 1, 2])
def test_each_sign_and_each_reached_clock_guard_the_pair(sign, clock):
    shifts = torch.full((1, 2, 5, 3), 0.01, dtype=torch.float64)
    shifts[0, sign, 0, clock] = -0.100001
    shifts[0, 0, 1] = torch.tensor([-0.06, 0.03, 0.02], dtype=torch.float64)
    shifts[0, 1, 1] = torch.tensor([0.02, -0.095, 0.01], dtype=torch.float64)

    radius, largest_abs = select_symmetric(shifts, torch.ones((1, 2), dtype=torch.bool))

    torch.testing.assert_close(radius, torch.tensor([[0.5, 0.5]], dtype=radius.dtype))
    torch.testing.assert_close(
        largest_abs, torch.tensor([[0.06, 0.095]], dtype=shifts.dtype)
    )


def test_nonmonotonic_grid_uses_largest_evaluated_feasible_radius():
    shifts = torch.zeros((1, 4, 5, 3), dtype=torch.float64)
    shifts[0, 0, :, 0] = torch.tensor([0.2, 0.04, 0.2, 0.03, 0.2])
    shifts[0, 1, :, 1] = torch.tensor([0.2, 0.05, 0.2, 0.02, 0.2])
    shifts[0, 2, :, 0] = torch.tensor([0.07, 0.2, 0.03, 0.2, 0.01])
    shifts[0, 3, :, 2] = torch.tensor([0.08, 0.2, 0.02, 0.2, 0.01])

    radius, largest_abs = select_symmetric(shifts, torch.ones((1, 4), dtype=torch.bool))

    torch.testing.assert_close(
        radius, torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=radius.dtype)
    )
    torch.testing.assert_close(
        largest_abs,
        torch.tensor([[0.04, 0.05, 0.07, 0.08]], dtype=shifts.dtype),
    )


def test_no_feasible_step_and_unavailable_nonfinite_values_return_zero():
    shifts = torch.full((1, 4, 5, 3), 0.11, dtype=torch.float64)
    shifts[0, 2, :, :] = float("nan")
    shifts[0, 3, :, :] = float("inf")
    available = torch.tensor([[True, True, False, False]])
    original = shifts.clone()

    radius, largest_abs = select_symmetric(shifts, available)

    torch.testing.assert_close(radius, torch.zeros((1, 4), dtype=radius.dtype))
    torch.testing.assert_close(largest_abs, torch.zeros((1, 4), dtype=shifts.dtype))
    torch.testing.assert_close(shifts, original, equal_nan=True)
    assert torch.equal(available, torch.tensor([[True, True, False, False]]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_only_smallest_evaluated_radius_is_feasible(dtype):
    shifts = torch.full((1, 2, 5, 3), 0.11, dtype=dtype)
    shifts[0, 0, -1] = -0.09
    shifts[0, 1, -1] = 0.08

    radius, largest_abs = select_symmetric(shifts, torch.ones((1, 2), dtype=torch.bool))

    torch.testing.assert_close(
        radius, torch.tensor([[0.0625, 0.0625]], dtype=radius.dtype)
    )
    torch.testing.assert_close(largest_abs, torch.tensor([[0.09, 0.08]], dtype=dtype))


def test_empty_batch_preserves_output_shapes():
    radius, largest_abs = select_symmetric(
        torch.empty((0, 2, 5, 3)), torch.empty((0, 2), dtype=torch.bool)
    )
    assert radius.shape == largest_abs.shape == (0, 2)


def test_custom_descending_grid_and_zero_limit():
    shifts = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    shifts[0, 1, 0, 2] = 0.01

    radius, largest_abs = select_symmetric(
        shifts, torch.ones((1, 2), dtype=torch.bool), scales=(0.9, 0.3, 0.1), limit=0
    )

    torch.testing.assert_close(radius, torch.tensor([[0.3, 0.3]], dtype=radius.dtype))
    torch.testing.assert_close(largest_abs, torch.zeros((1, 2), dtype=shifts.dtype))


def test_randomized_pairs_match_independent_exhaustive_grid_oracle():
    generator = torch.Generator(device="cpu").manual_seed(17)
    shifts = (
        torch.rand((8, 6, 5, 3), generator=generator, dtype=torch.float64) - 0.5
    ) * 0.3
    pair_available = torch.rand((8, 3), generator=generator) > 0.3
    available = pair_available.repeat_interleave(2, dim=1)
    expected_radius = torch.zeros((8, 6), dtype=torch.float64)
    expected_max = torch.zeros((8, 6), dtype=torch.float64)

    for batch in range(8):
        for pair in range(3):
            if not pair_available[batch, pair]:
                continue
            first = pair * 2
            for grid_index, scale in enumerate(SCALES):
                if all(
                    abs(shifts[batch, action, grid_index, clock].item()) <= LIMIT
                    for action in (first, first + 1)
                    for clock in range(3)
                ):
                    for action in (first, first + 1):
                        expected_radius[batch, action] = scale
                        expected_max[batch, action] = max(
                            abs(shifts[batch, action, grid_index, clock].item())
                            for clock in range(3)
                        )
                    break

    radius, largest_abs = select_symmetric(shifts, available)

    torch.testing.assert_close(radius, expected_radius)
    torch.testing.assert_close(largest_abs, expected_max)
    assert torch.equal(radius[:, 0::2], radius[:, 1::2])


@pytest.mark.parametrize(
    "shifts, available",
    [
        (np.zeros((1, 2, 5, 3)), torch.ones((1, 2), dtype=torch.bool)),
        (torch.zeros((1, 2, 5)), torch.ones((1, 2), dtype=torch.bool)),
        (torch.zeros((1, 3, 5, 3)), torch.ones((1, 3), dtype=torch.bool)),
        (torch.zeros((1, 2, 4, 3)), torch.ones((1, 2), dtype=torch.bool)),
        (torch.zeros((1, 2, 5, 2)), torch.ones((1, 2), dtype=torch.bool)),
        (
            torch.zeros((1, 2, 5, 3), dtype=torch.int64),
            torch.ones((1, 2), dtype=torch.bool),
        ),
        (
            torch.zeros((1, 2, 5, 3), dtype=torch.complex64),
            torch.ones((1, 2), dtype=torch.bool),
        ),
        (torch.zeros((1, 2, 5, 3)), [[True, True]]),
        (torch.zeros((1, 2, 5, 3)), torch.ones((1, 2, 1), dtype=torch.bool)),
        (torch.zeros((1, 2, 5, 3)), torch.ones((2, 2), dtype=torch.bool)),
        (torch.zeros((1, 2, 5, 3)), torch.ones((1, 2))),
        (torch.zeros((1, 2, 5, 3)), torch.tensor([[True, False]])),
    ],
)
def test_rejects_malformed_or_asymmetric_inputs(shifts, available):
    with pytest.raises((TypeError, ValueError)):
        select_symmetric(shifts, available)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("sign", [0, 1])
def test_rejects_nonfinite_values_anywhere_in_an_available_action(bad_value, sign):
    shifts = torch.zeros((1, 2, 5, 3))
    shifts[0, sign, 4, 2] = bad_value
    with pytest.raises(ValueError):
        select_symmetric(shifts, torch.ones((1, 2), dtype=torch.bool))


@pytest.mark.parametrize(
    "scales",
    [
        (),
        (1.0, 0.5),
        (1.0, 0.5, 0.25, 0.25, 0.0625),
        (0.5, 1.0, 0.25, 0.125, 0.0625),
        (1.0, 0.5, 0.25, 0.125, 0.0),
        (1.0, 0.5, 0.25, 0.125, -0.1),
        (1.0, 0.5, 0.25, 0.125, float("nan")),
        (float("inf"), 0.5, 0.25, 0.125, 0.0625),
        [[1.0, 0.5, 0.25, 0.125, 0.0625]],
        torch.tensor([True, True, True, True, True]),
        torch.tensor([1 + 0j, 0.5 + 0j, 0.25 + 0j, 0.125 + 0j, 0.0625 + 0j]),
    ],
)
def test_rejects_invalid_scales(scales):
    with pytest.raises((TypeError, ValueError)):
        select_symmetric(
            torch.zeros((1, 2, 5, 3)),
            torch.ones((1, 2), dtype=torch.bool),
            scales=scales,
        )


@pytest.mark.parametrize(
    "limit", [-1, float("nan"), float("inf"), -float("inf"), True, 1j]
)
def test_rejects_invalid_limit(limit):
    with pytest.raises((TypeError, ValueError)):
        select_symmetric(
            torch.zeros((1, 2, 5, 3)), torch.ones((1, 2), dtype=torch.bool), limit=limit
        )


def test_reached_indices_are_int64_ordered_and_never_future():
    maturity = np.array([63, 64, 100], dtype=np.int32)

    indices = reached_indices(maturity)

    np.testing.assert_array_equal(indices, [[47, 55, 63], [48, 56, 64], [84, 92, 100]])
    assert indices.dtype == np.int64
    assert np.all(indices >= 0)
    assert np.all(indices <= maturity[:, None])


def test_reached_indices_accept_integer_lists_and_empty_batch():
    np.testing.assert_array_equal(reached_indices([63]), [[47, 55, 63]])
    empty = reached_indices(np.array([], dtype=np.int64))
    assert empty.shape == (0, 3)
    assert empty.dtype == np.int64


@pytest.mark.parametrize(
    "maturity",
    [
        [62],
        [-1],
        [63, 62],
        63,
        [[63]],
        [63.0],
        [float("nan")],
        [float("inf")],
        [True],
        ["63"],
        np.array([2**63], dtype=np.uint64),
    ],
)
def test_reached_indices_rejects_bad_or_unrepresentable_maturity(maturity):
    with pytest.raises((TypeError, ValueError)):
        reached_indices(maturity)
