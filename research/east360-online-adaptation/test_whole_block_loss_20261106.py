"""Synthetic CPU-only regression tests for the offline block objective."""

import numpy as np
import pytest
import torch

from whole_block_loss_20261106 import (
    PROTECT,
    RECOVER,
    SAFE,
    _first_pre_outcome_crossing,
    peak_loss,
    role_masks,
)


def _rows_equal(actual, expected):
    assert len(actual) == len(expected)
    for (slot, role, indices), (want_slot, want_role, want_indices) in zip(
        actual, expected, strict=True
    ):
        assert (slot, role) == (want_slot, want_role)
        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64
        np.testing.assert_array_equal(indices, want_indices)


def _shot(lead_ms, logits, slots, support, *, horizon=25.0, threshold=0.0):
    # With down=0, decreasing lead is an increasing endpoint clock.
    return role_masks(
        -np.asarray(lead_ms, dtype=np.float64) / 1000.0,
        np.asarray(logits, dtype=np.float64),
        np.asarray(slots, dtype=np.int64),
        np.asarray(support, dtype=bool),
        True,
        0.0,
        horizon,
        threshold,
    )


def test_role_constants_are_stable():
    assert (SAFE, PROTECT, RECOVER) == (0, 1, 2)


def test_safe_uses_every_endpoint_of_each_supported_block_including_partial():
    rows = role_masks(
        np.arange(10) / 1000.0,
        [-5, 10, 0, 1, -1, 2, 3, 4, 5, 6],
        [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2],
        [True, False, True],
        False,
        None,
        25.0,
        0.0,
    )
    _rows_equal(rows, [(0, SAFE, [2, 3, 4]), (2, SAFE, [8, 9])])


def test_safe_has_no_rows_without_writable_supported_endpoints():
    rows = role_masks([0, 1, 2], [0, 1, 2], [-1, 0, 0], [False], False, None, 25, 0)
    assert rows == []
    assert (
        role_masks([0], [0], [-1], np.array([], dtype=bool), False, None, 25, 0) == []
    )


def test_timely_first_alarm_protects_only_its_block_up_to_first_no_horizon_cap():
    # The threshold-equal first alarm has 40 ms lead, exceeding the 25 ms
    # canonical horizon. It nevertheless receives protection, not recovery.
    rows = _shot(
        [150, 100, 80, 60, 50, 40, 30, 20, 10, 5],
        [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1],
        [-1, -1, 0, 0, 1, 1, 1, 1, 2, 2],
        [True, True, True],
    )
    _rows_equal(rows, [(1, PROTECT, [4, 5])])


@pytest.mark.parametrize(
    ("slots", "support", "first"),
    [
        ([-1, -1, 0, 0, 1, 1], [True, True], 0),
        ([-1, 0, 0, 1, 1, 1], [True, False], 3),
    ],
)
def test_timely_first_in_unmodified_or_unsupported_slot_has_no_training_row(
    slots, support, first
):
    logits = [-1] * 6
    logits[first] = 0
    assert _shot([60, 50, 40, 30, 20, 10], logits, slots, support) == []


def test_protect_stops_at_first_and_does_not_relabel_earlier_blocks():
    rows = _shot(
        [100, 90, 80, 70, 60, 50, 40],
        [-1, -1, -1, -1, 0, 0, 0],
        [-1, 0, 0, 1, 1, 1, 2],
        [True, True, True],
        horizon=25,
    )
    _rows_equal(rows, [(1, PROTECT, [3, 4])])


@pytest.mark.parametrize("kind", ["missed", "late", "post_outcome_only"])
def test_missed_late_and_fake_after_down_crossings_recover_supported_blocks(kind):
    logits = [-1.0] * 10
    if kind == "late":
        logits[6] = 0.0  # exactly 10 ms is not timely
    elif kind == "post_outcome_only":
        logits[9] = 1.0  # must not be taken as a pre-outcome first alarm
    rows = _shot(
        [80, 50, 25, 20, 15, 12, 10, 0, -0.000001, -1],
        logits,
        [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2],
        [True, True, True],
        horizon=25,
    )
    _rows_equal(rows, [(0, RECOVER, [2, 3, 4]), (1, RECOVER, [5])])


def test_rounded_strict_lead_and_inclusive_horizon_tolerance_boundaries():
    leads = [
        20.000001001,  # beyond upper tolerance
        20.000001000,  # upper tolerance is inclusive
        20.000000999,
        20.0,
        10.000000001,
        10.00000000051,  # rounds up to >10
        10.00000000049,  # rounds to exactly 10
        10.0,
        9.0,
    ]
    logits = [-1] * len(leads)
    logits[6] = 0  # first lead rounds to exactly 10: RECOVER
    rows = _shot(leads, logits, [0] * len(leads), [True], horizon=20)
    _rows_equal(rows, [(0, RECOVER, [1, 2, 3, 4, 5])])


def test_rounding_just_above_ten_makes_first_alarm_timely():
    rows = _shot(
        [11.0, 10.00000000051, 10.00000000049, 10.0],
        [-1, 0, -1, -1],
        [0, 0, 0, 0],
        [True],
        horizon=25,
    )
    _rows_equal(rows, [(0, PROTECT, [0, 1])])


@pytest.mark.parametrize(
    ("raw_lead_ms", "logits", "expected"),
    [
        ([0.0, -1e-6], [-1.0, 0.0], 1),
        ([0.0, np.nextafter(-1e-6, -np.inf)], [-1.0, 0.0], None),
        ([0.0, -1e-6], [0.0, 1.0], 0),
        ([0.0, -1e-6], [-1.0, -1.0], None),
    ],
)
def test_pre_outcome_tolerance_and_crossing_equality(raw_lead_ms, logits, expected):
    # Late and missed both map to RECOVER; test the internal first-crossing
    # primitive directly so the inclusive tolerance remains observable.
    assert (
        _first_pre_outcome_crossing(np.array(raw_lead_ms), np.array(logits), 0.0)
        == expected
    )


def test_first_crossing_uses_all_endpoints_not_just_supported_endpoints():
    rows = _shot(
        [100, 80, 60, 40, 20],
        [-1, 0, -1, 1, -1],
        [-1, 0, 0, 1, 1],
        [False, True],
    )
    assert rows == []  # native first was in unsupported slot 0


def test_recover_omits_unsupported_and_empty_eligible_blocks():
    rows = _shot(
        [90, 80, 40, 30, 20, 15, 10, 5],
        [-1] * 8,
        [-1, -1, 0, 0, 1, 1, 2, 2],
        [True, False, True],
        horizon=25,
    )
    assert rows == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"times": []}, "times"),
        ({"times": [[0, 1, 2]]}, "times"),
        ({"times": [0, 0, 2]}, "times"),
        ({"times": [0, 2, 1]}, "times"),
        ({"times": [0, float("nan"), 2]}, "times"),
        ({"times": [0, "not-a-time", 2]}, "times"),
        ({"times": np.array([0, 0.001 + 1j, 0.002])}, "times"),
        ({"logits": [0, 1]}, "logits"),
        ({"logits": [0, 1, float("inf")]}, "logits"),
        ({"logits": [0, "not-a-logit", 1]}, "logits"),
        ({"logits": np.array([0, 1 + 1j, 2])}, "logits"),
        ({"slots": [-1, 0]}, "slots"),
        ({"slots": [-1.0, 0.0, 0.0]}, "slots"),
        ({"slots": [-2, 0, 0]}, "slots"),
        ({"slots": [-1, 0, 1]}, "slots"),
        ({"support": [1]}, "support"),
        ({"support": [[True]]}, "support"),
        ({"disruptive": 1}, "disruptive"),
        ({"down": None}, "down"),
        ({"down": float("nan")}, "down"),
        ({"horizon": 10.0}, "horizon"),
        ({"horizon": float("inf")}, "horizon"),
        ({"threshold_logit": float("nan")}, "threshold_logit"),
        ({"threshold_logit": [0]}, "threshold_logit"),
    ],
)
def test_role_masks_rejects_malformed_inputs(overrides, error):
    args = dict(
        times=[0.0, 0.001, 0.002],
        logits=[-1.0, -1.0, -1.0],
        slots=[-1, 0, 0],
        support=[True],
        disruptive=True,
        down=1.0,
        horizon=25.0,
        threshold_logit=0.0,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match=error):
        role_masks(**args)


def test_role_masks_rejects_overflowing_leads():
    with pytest.raises(ValueError, match="lead"):
        role_masks([-1e308, 0.0], [-1, -1], [0, 0], [True], True, 1e308, 25, 0)


def test_role_masks_matches_independent_synthetic_reference():
    rng = np.random.default_rng(86021)
    for _ in range(200):
        n = int(rng.integers(1, 36))
        times = np.sort(rng.uniform(-0.08, 0.004, size=n))
        logits = rng.choice([-1.0, 0.0, 1.0], size=n)
        slots = np.array([-1 if i < 4 else (i - 4) // 4 for i in range(n)])
        n_slots = max(0, int(slots.max()) + 1)
        support = rng.choice([False, True], size=n_slots)
        disruptive = bool(rng.integers(2))
        horizon = 25.0
        rows = role_masks(times, logits, slots, support, disruptive, 0.0, horizon, 0.0)

        first = None
        if disruptive:
            for i, (t, logit) in enumerate(zip(times, logits, strict=True)):
                if -float(t) * 1000 >= -1e-6 and logit >= 0:
                    first = i
                    break
        timely_first = first is not None and np.round(-times[first] * 1000, 9) > 10
        expected = []
        for slot in range(n_slots):
            if not support[slot]:
                continue
            if disruptive and timely_first and slots[first] != slot:
                continue
            indices = []
            for i, (t, owner) in enumerate(zip(times, slots, strict=True)):
                if owner != slot:
                    continue
                lead = np.round(-float(t) * 1000, 9)
                if not disruptive:
                    indices.append(i)
                elif timely_first and i <= first and lead > 10:
                    indices.append(i)
                elif not timely_first and 10 < lead <= horizon + 1e-6:
                    indices.append(i)
            if indices:
                role = SAFE if not disruptive else PROTECT if timely_first else RECOVER
                expected.append((slot, role, indices))
        _rows_equal(rows, expected)


def test_whole_block_and_point_loss_detect_shifted_peaks_differently():
    native = torch.tensor([[0.9, 0.1, 99.0], [0.5, 0.4, 99.0]], dtype=torch.float64)
    updated = torch.tensor(
        [[-0.4, 0.2, 999.0], [-0.6, 0.1, 999.0]], dtype=torch.float64
    )
    mask = torch.tensor([[True, True, False], [True, True, False]])
    roles = torch.tensor([SAFE, PROTECT])
    block = peak_loss(updated, native, mask, roles, 0.0)
    point = peak_loss(updated, native, mask, roles, 0.0, point=True)
    torch.testing.assert_close(block, torch.tensor([0.225, 0.0], dtype=torch.float64))
    torch.testing.assert_close(point, torch.tensor([0.0, 0.625], dtype=torch.float64))


def test_all_roles_use_same_per_row_hinge_margin_without_extra_weights():
    z = torch.tensor([[0.01], [-0.01], [-0.01], [-0.025], [0.025]], dtype=torch.float64)
    roles = torch.tensor([SAFE, PROTECT, RECOVER, SAFE, RECOVER])
    loss = peak_loss(z, z, torch.ones_like(z, dtype=torch.bool), roles, 0.0)
    torch.testing.assert_close(
        loss, torch.tensor([0.035, 0.035, 0.035, 0.0, 0.0], dtype=z.dtype)
    )
    assert loss.shape == (5,)


@pytest.mark.parametrize("point", [False, True])
def test_sparse_mask_ignores_nonfinite_padding_and_its_gradients(point):
    z = torch.tensor([[0.1, float("nan"), -0.5, float("inf")]], requires_grad=True)
    native = torch.tensor([[0.9, float("inf"), 0.1, float("nan")]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    loss = peak_loss(z, native, mask, torch.tensor([SAFE]), 0.0, point=point)
    torch.testing.assert_close(loss, torch.tensor([0.125]))
    loss.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert native.grad is None


def test_native_argmax_ties_choose_first_eligible_point():
    z = torch.tensor([[99.0, -0.1, 0.5, 0.8]], requires_grad=True)
    native = torch.tensor([[100.0, 1.0, 1.0, 0.0]])
    loss = peak_loss(
        z,
        native,
        torch.tensor([[False, True, True, True]]),
        torch.tensor([PROTECT]),
        0.0,
        point=True,
    )
    torch.testing.assert_close(loss, torch.tensor([0.125]))
    loss.sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([[0.0, -1.0, 0.0, 0.0]]))


def test_whole_block_gradient_flows_to_updated_maximum_only():
    z = torch.tensor(
        [[0.1, 0.3, -9], [-0.2, -0.1, 9], [-0.1, -0.2, 9]], requires_grad=True
    )
    native = torch.zeros_like(z)
    mask = torch.tensor([[True, True, False]] * 3)
    roles = torch.tensor([SAFE, PROTECT, RECOVER])
    peak_loss(z, native, mask, roles, 0.0).sum().backward()
    torch.testing.assert_close(
        z.grad, torch.tensor([[0, 1, 0], [0, -1, 0], [-1, 0, 0]], dtype=z.dtype)
    )


@pytest.mark.parametrize("point", [False, True])
def test_numerical_gradcheck_away_from_ties_and_hinge_boundaries(point):
    z = torch.tensor(
        [[0.7, -0.4, 4.0], [-0.5, -0.8, 5.0]], dtype=torch.float64, requires_grad=True
    )
    native = torch.tensor([[0.1, 0.5, 99.0], [0.5, 0.2, 99.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    roles = torch.tensor([SAFE, RECOVER])
    assert torch.autograd.gradcheck(
        lambda values: peak_loss(values, native, mask, roles, 0.0, point=point), (z,)
    )


@pytest.mark.parametrize("point", [False, True])
def test_peak_loss_matches_independent_rowwise_reference(point):
    rng = np.random.default_rng(371)
    updated = rng.normal(size=(32, 9))
    frozen = rng.normal(size=(32, 9))
    masks = rng.choice([False, True], size=(32, 9))
    masks[:, 0] = True
    roles = rng.integers(SAFE, RECOVER + 1, size=32)
    threshold = -0.37
    expected = []
    for z_row, native_row, mask_row, role in zip(
        updated, frozen, masks, roles, strict=True
    ):
        eligible = [i for i, allowed in enumerate(mask_row) if allowed]
        index = max(eligible, key=lambda i: native_row[i] if point else z_row[i])
        peak = z_row[index]
        expected.append(
            max(0.0, peak - (threshold - 0.025))
            if role == SAFE
            else max(0.0, threshold + 0.025 - peak)
        )
    actual = peak_loss(
        torch.from_numpy(updated),
        torch.from_numpy(frozen),
        torch.from_numpy(masks),
        torch.from_numpy(roles),
        threshold,
        point=point,
    )
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


def test_both_functions_leave_caller_inputs_unchanged():
    times = np.array([-0.04, -0.025, -0.02, -0.015])
    logits = np.array([-1.0, -1.0, -1.0, -1.0])
    slots = np.array([-1, 0, 0, 1])
    support = np.array([True, True])
    original_arrays = [value.copy() for value in (times, logits, slots, support)]
    role_masks(times, logits, slots, support, True, 0.0, 25.0, 0.0)
    for value, original in zip(
        (times, logits, slots, support), original_arrays, strict=True
    ):
        np.testing.assert_array_equal(value, original)

    z = torch.tensor([[0.2, -0.1]], requires_grad=True)
    native = torch.tensor([[-0.2, 0.1]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    role = torch.tensor([SAFE])
    original_tensors = [value.detach().clone() for value in (z, native, mask, role)]
    peak_loss(z, native, mask, role, 0.0, point=True)
    for value, original in zip((z, native, mask, role), original_tensors, strict=True):
        torch.testing.assert_close(value.detach(), original)


@pytest.mark.parametrize("point", [False, True])
def test_loss_is_invariant_to_padding_batch_order_and_common_logit_shift(point):
    z = torch.tensor([[0.1, 0.3, -0.2], [-0.4, -0.3, 0.1]], dtype=torch.float64)
    native = torch.tensor([[0.0, 0.4, 0.2], [-0.2, -0.4, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    roles = torch.tensor([SAFE, RECOVER])
    original = peak_loss(z, native, mask, roles, 0.2, point=point)
    changed_padding = z.clone()
    changed_padding[~mask] = float("nan")
    padded_z = torch.cat([changed_padding, torch.full((2, 1), float("inf"))], dim=1)
    padded_native = torch.cat([native, torch.full((2, 1), float("nan"))], dim=1)
    padded_mask = torch.cat([mask, torch.zeros((2, 1), dtype=torch.bool)], dim=1)
    padded = peak_loss(padded_z, padded_native, padded_mask, roles, 0.2, point=point)
    torch.testing.assert_close(padded, original)
    shifted = peak_loss(z + 1.25, native + 1.25, mask, roles, 1.45, point=point)
    torch.testing.assert_close(shifted, original)
    order = torch.tensor([1, 0])
    permuted = peak_loss(
        z[order], native[order], mask[order], roles[order], 0.2, point=point
    )
    torch.testing.assert_close(permuted, original[order])


def test_masks_feed_directly_into_the_batched_objective():
    rows = _shot([40, 25, 20, 15, 10], [-1] * 5, [-1, 0, 0, 1, 1], [True, True])
    _rows_equal(rows, [(0, RECOVER, [1, 2]), (1, RECOVER, [3])])
    mask = torch.zeros((len(rows), 5), dtype=torch.bool)
    for row, (_, _, indices) in enumerate(rows):
        mask[row, torch.from_numpy(indices)] = True
    roles = torch.tensor([role for _, role, _ in rows])
    native = torch.zeros((2, 5))
    updated = torch.tensor([[9.0, -0.1, -0.2, 9.0, 9.0], [9.0, 9.0, 9.0, -0.3, 9.0]])
    torch.testing.assert_close(
        peak_loss(updated, native, mask, roles, 0.0), torch.tensor([0.125, 0.325])
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("z_not_tensor", "z"),
        ("z_integer", "z"),
        ("z_vector", "z"),
        ("z_empty_batch", "z"),
        ("z_empty_length", "z"),
        ("native_shape", "native"),
        ("native_integer", "native"),
        ("native_nonfinite", "native"),
        ("z_nonfinite", "z"),
        ("mask_shape", "mask"),
        ("mask_dtype", "mask"),
        ("mask_empty_row", "mask"),
        ("role_shape", "role"),
        ("role_dtype", "role"),
        ("role_invalid", "role"),
        ("threshold_nonfinite", "threshold_logit"),
        ("point_not_boolean", "point"),
    ],
)
def test_peak_loss_rejects_malformed_inputs(change, error):
    args = dict(
        z=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        native=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        mask=torch.tensor([[True, True], [True, False]]),
        role=torch.tensor([SAFE, RECOVER]),
        threshold_logit=0.0,
        point=False,
    )
    if change == "z_not_tensor":
        args["z"] = [[0.1, 0.2], [0.3, 0.4]]
    elif change == "z_integer":
        args["z"] = torch.tensor([[1, 2], [3, 4]])
    elif change == "z_vector":
        args["z"] = torch.tensor([0.1, 0.2])
    elif change == "z_empty_batch":
        args["z"] = torch.empty((0, 2))
    elif change == "z_empty_length":
        args["z"] = torch.empty((2, 0))
    elif change == "native_shape":
        args["native"] = torch.zeros((2, 1))
    elif change == "native_integer":
        args["native"] = torch.zeros((2, 2), dtype=torch.int64)
    elif change == "native_nonfinite":
        args["native"][0, 0] = float("nan")
    elif change == "z_nonfinite":
        args["z"][0, 0] = float("inf")
    elif change == "mask_shape":
        args["mask"] = torch.tensor([[True, True]])
    elif change == "mask_dtype":
        args["mask"] = torch.tensor([[1, 1], [1, 0]])
    elif change == "mask_empty_row":
        args["mask"][1] = False
    elif change == "role_shape":
        args["role"] = torch.tensor([[SAFE], [RECOVER]])
    elif change == "role_dtype":
        args["role"] = torch.tensor([float(SAFE), float(RECOVER)])
    elif change == "role_invalid":
        args["role"] = torch.tensor([SAFE, 3])
    elif change == "threshold_nonfinite":
        args["threshold_logit"] = float("inf")
    elif change == "point_not_boolean":
        args["point"] = 1
    with pytest.raises(ValueError, match=error):
        peak_loss(**args)
