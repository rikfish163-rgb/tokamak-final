import numpy as np
import pytest

import alarm_order_views_20261106 as views


def identity(batch=1):
    return np.broadcast_to(np.arange(16, dtype=np.int64), (batch, 16)).copy()


def windows(batch=1):
    values = np.arange(batch * 128 * 23, dtype=np.float64).reshape(batch, 128, 23)
    return values, np.ones_like(values, dtype=bool)


def cycle():
    # Non-involution makes destination -> source distinguishable from its inverse.
    perm = identity()[0]
    perm[[3, 5, 8]] = [5, 8, 3]
    return perm


def test_plans_exact_four_arm_orders_and_terminal_patch():
    _, mask = windows(2)
    perm, supported = views.plans(mask)
    assert views.ARMS == ("recent_reverse", "old_reverse", "recent_inner", "old_inner")
    assert perm.shape == (2, 4, 16) and perm.dtype == np.int64
    assert supported.shape == (2, 4) and supported.dtype == bool and supported.all()
    expected = np.tile(np.arange(16), (4, 1))
    expected[0, 11:15] = [14, 13, 12, 11]
    expected[1, 7:11] = [10, 9, 8, 7]
    expected[2, 11:15] = [11, 13, 12, 14]
    expected[3, 7:11] = [7, 9, 8, 10]
    np.testing.assert_array_equal(perm, np.broadcast_to(expected, perm.shape))
    np.testing.assert_array_equal(
        np.sort(perm, axis=-1), np.broadcast_to(np.arange(16), perm.shape)
    )
    np.testing.assert_array_equal(perm[..., -1], 15)


def test_plans_require_exact_full_position_masks_and_never_partially_intervene():
    _, mask = windows(4)
    # Mismatch only the outer recent pair: recent_inner remains supported.
    mask[0, 11 * 8 + 2, 5] = False
    # Matching per-patch counts are insufficient: offsets and channels matter.
    mask[1, 12 * 8 + 1, 0] = False
    mask[1, 13 * 8 + 1, 1] = False
    # All missing is still mask-preserving, despite carrying no observed motion.
    mask[2] = False
    # The outer old pair may match with the same missing coordinates.
    mask[3, 7 * 8 + 4, 9] = mask[3, 10 * 8 + 4, 9] = False
    before = mask.copy()
    perm, supported = views.plans(mask)
    np.testing.assert_array_equal(
        supported,
        [
            [False, True, True, True],
            [False, True, False, True],
            [True, True, True, True],
            [True, True, True, True],
        ],
    )
    for batch, arm in np.argwhere(~supported):
        np.testing.assert_array_equal(perm[batch, arm], np.arange(16))
    np.testing.assert_array_equal(mask, before)
    for batch, arm in np.argwhere(supported):
        patches = mask[batch].reshape(16, 8, 23)
        np.testing.assert_array_equal(patches[perm[batch, arm]], patches)


def test_apply_uses_destination_to_source_whole_patches_and_keeps_raw_payload():
    values, mask = windows(1)
    perm = cycle()[None, :]
    # The same coordinate is absent in all three patches but has distinct payload.
    mask[0, np.array([3, 5, 8]) * 8 + 4, 7] = False
    values[0, 3 * 8 + 4, 7] = np.nan
    values[0, 5 * 8 + 4, 7] = np.inf
    values[0, 8 * 8 + 4, 7] = -12345.0
    values_before, mask_before, perm_before = values.copy(), mask.copy(), perm.copy()
    result = views.apply(values, mask, perm)
    expected = values.reshape(1, 16, 8, 23)[0, perm[0]].reshape(1, 128, 23)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[:, -8:], values[:, -8:])
    assert np.isinf(result[0, 3 * 8 + 4, 7])
    assert result[0, 5 * 8 + 4, 7] == -12345.0
    assert np.isnan(result[0, 8 * 8 + 4, 7])
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(perm, perm_before)
    inverse = np.argsort(perm, axis=-1)
    np.testing.assert_array_equal(views.apply(result, mask, inverse), values)


def test_apply_rejects_even_single_cell_mask_difference():
    values, mask = windows()
    mask[0, 3 * 8 + 2, 4] = False
    with pytest.raises(ValueError):
        views.apply(values, mask, cycle()[None, :])


def test_transport_preserves_absolute_donors_destinations_and_offsets():
    _, mask = windows(2)
    perm, _ = views.plans(mask)
    shifted = views.transport(perm, 24)
    assert shifted.shape == perm.shape and shifted.dtype == np.int64
    np.testing.assert_array_equal(
        np.sort(shifted, axis=-1), np.broadcast_to(np.arange(16), shifted.shape)
    )
    np.testing.assert_array_equal(shifted[..., -1], 15)
    for batch in range(2):
        for arm in range(4):
            moved = np.flatnonzero(perm[batch, arm] != np.arange(16))
            new_dest = moved - 3
            new_source = perm[batch, arm, moved] - 3
            np.testing.assert_array_equal(shifted[batch, arm, new_dest], new_source)
            for offset in range(8):
                # An endpoint 24 samples later uses the same absolute samples.
                np.testing.assert_array_equal(
                    1000 - 127 + 8 * moved + offset,
                    1024 - 127 + 8 * new_dest + offset,
                )
                np.testing.assert_array_equal(
                    1000 - 127 + 8 * perm[batch, arm, moved] + offset,
                    1024 - 127 + 8 * new_source + offset,
                )
    np.testing.assert_array_equal(views.transport(perm, 0), perm)


def test_transport_handles_general_cycles_and_rejects_clipping():
    perm = cycle()
    shifted = views.transport(perm, np.int64(16))
    expected = np.arange(16)
    expected[[1, 3, 6]] = [3, 6, 1]
    np.testing.assert_array_equal(shifted, expected)
    np.testing.assert_array_equal(views.transport(np.arange(16), 1600), np.arange(16))
    with pytest.raises(ValueError):
        views.transport(perm, 32)  # patch 3 would leave the window; never clip
    for advance in (-8, 1, 8.0, True, np.nan, [8]):
        with pytest.raises(ValueError):
            views.transport(perm, advance)


def test_transport_composes_and_commutes_with_inverse():
    perm = cycle()
    shifted = views.transport(perm, 24)
    np.testing.assert_array_equal(views.transport(views.transport(perm, 8), 16), shifted)
    np.testing.assert_array_equal(views.transport(np.argsort(perm), 24), np.argsort(shifted))
    # The old inner pair still fits when an unmoved outer band patch has left.
    _, support_mask = windows()
    all_plans, _ = views.plans(support_mask)
    old_inner = views.transport(all_plans[0, 3], 64)
    np.testing.assert_array_equal(old_inner[:2], [1, 0])
    with pytest.raises(ValueError):
        views.transport(all_plans[0, 1], 64)


def test_overlapping_windows_intervene_on_the_same_absolute_samples():
    stream = np.broadcast_to(np.arange(152)[:, None], (152, 23)).astype(float).copy()
    stream[:, 1] += 1000  # All channels must follow the same sample mapping.
    stream[::8, 2] = -np.arange(19)  # Distinct missing raw payloads still move.
    observed = np.ones_like(stream, dtype=bool)
    observed[::8, 2] = False
    first, later = stream[None, :128], stream[None, 24:152]
    first_mask, later_mask = observed[None, :128], observed[None, 24:152]
    permutations, supported = views.plans(first_mask)
    assert supported.all()
    for arm in range(4):
        initial = views.apply(first, first_mask, permutations[:, arm])
        later_perm = views.transport(permutations[:, arm], 24)
        subsequent = views.apply(later, later_mask, later_perm)
        np.testing.assert_array_equal(initial[:, 24:], subsequent[:, :104])
        assert initial[:, :, 0].max() <= 127  # No future sample in the earlier view.
        np.testing.assert_array_equal(initial[:, -8:], first[:, -8:])
        np.testing.assert_array_equal(subsequent[:, -8:], later[:, -8:])


def test_motion_rms_counts_only_observed_coordinates_of_moved_patches():
    z = np.zeros((3, 128, 23), dtype=np.float64)
    mask = np.zeros_like(z, dtype=bool)
    perm = np.broadcast_to(cycle(), (3, 16)).copy()
    # Identical per-offset masks in all cycled patches, with unequal differences.
    for batch in (0, 1):
        for patch in (3, 5, 8):
            mask[batch, patch * 8, 0] = True
    z[0, np.array([3, 5, 8]) * 8, 0] = [1, 4, 9]
    z[1, np.array([3, 5, 8]) * 8, 0] = [2, 2, 2]
    # Missing payload and observed unchanged patches are excluded from the RMS.
    z[~mask] = np.nan
    mask[0, 0, 1] = True
    z[0, 0, 1] = 1e100
    result = views.motion_rms(z, mask, perm)
    assert result.dtype == np.float64 and result.shape == (3,)
    np.testing.assert_allclose(result, [np.sqrt((3**2 + 5**2 + (-8)**2) / 3), 0, 0])
    np.testing.assert_array_equal(views.motion_rms(z, mask, identity(3)), np.zeros(3))


def test_motion_rms_weights_cells_not_cycles_and_avoids_square_overflow():
    z = np.zeros((1, 128, 23), dtype=np.float64)
    mask = np.zeros_like(z, dtype=bool)
    perm = identity()
    perm[0, [3, 5, 8, 9]] = [5, 3, 9, 8]
    for patch in (3, 5):
        mask[0, patch * 8, 0] = True
    for patch in (8, 9):
        mask[0, patch * 8, :3] = True
    z[0, 5 * 8, 0] = 1
    z[0, 9 * 8, :3] = 3
    np.testing.assert_allclose(views.motion_rms(z, mask, perm), [np.sqrt(7)])
    z *= 1e200
    np.testing.assert_allclose(views.motion_rms(z, mask, perm), [np.sqrt(7) * 1e200])


def test_motion_rms_uses_float64_and_rejects_nonfinite_observed_motion():
    values, mask = windows()
    values = values.astype(np.float32)
    perm = cycle()[None, :]
    result = views.motion_rms(values, mask, perm)
    moved = np.array([3, 5, 8])
    before = values.reshape(1, 16, 8, 23)[0].astype(np.float64)
    expected = np.sqrt(np.mean((before[perm[0, moved]] - before[moved]) ** 2))
    np.testing.assert_allclose(result, [expected], rtol=1e-14)
    values[0, 3 * 8, 0] = np.inf
    with pytest.raises(ValueError):
        views.motion_rms(values, mask, perm)
    mask[0, np.array([3, 5, 8]) * 8, 0] = False
    assert np.isfinite(views.motion_rms(values, mask, perm)).all()


def test_motion_rejects_nonnumeric_values_and_unrepresentable_differences():
    values, mask = windows()
    perm = cycle()[None, :]
    with pytest.raises(ValueError):
        views.motion_rms(values.astype(str), mask, perm)
    values[0, 3 * 8, 0] = 1e308
    values[0, 5 * 8, 0] = -1e308
    with pytest.raises(ValueError):
        views.motion_rms(values, mask, perm)


@pytest.mark.parametrize("which", ["values", "mask", "perm_batch", "perm_rank"])
def test_apply_and_motion_reject_incompatible_shapes(which):
    values, mask = windows(2)
    perm = identity(2)
    if which == "values":
        values = values[:, :-1]
    elif which == "mask":
        mask = mask[:, :, :-1]
    elif which == "perm_batch":
        perm = identity(1)
    else:
        perm = perm[:, None, :]
    for function in (views.apply, views.motion_rms):
        with pytest.raises(ValueError):
            function(values, mask, perm)


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.ones((1, 128, 23)),
        np.ones((128, 23), dtype=bool),
        np.ones((1, 128, 22), dtype=bool),
    ],
)
def test_plans_reject_bad_masks(bad_mask):
    with pytest.raises(ValueError):
        views.plans(bad_mask)


def test_window_consumers_reject_nonboolean_masks():
    values, mask = windows()
    for function in (views.apply, views.motion_rms):
        with pytest.raises(ValueError):
            function(values, mask.astype(np.int8), identity())


@pytest.mark.parametrize("perm", [np.arange(15), np.arange(16).reshape(2, 8), np.array(3)])
def test_transport_rejects_nonpermutation_shapes(perm):
    with pytest.raises(ValueError):
        views.transport(perm, 0)


@pytest.mark.parametrize(
    "defect", ["duplicate", "negative", "too_large", "float", "bool", "terminal"]
)
def test_all_permutation_consumers_reject_invalid_permutations(defect):
    values, mask = windows()
    perm = identity()
    if defect == "duplicate":
        perm[0, 3] = 4
    elif defect == "negative":
        perm[0, 3] = -1
    elif defect == "too_large":
        perm[0, 3] = 16
    elif defect == "float":
        perm = perm.astype(float)
    elif defect == "bool":
        perm = perm.astype(bool)
    else:
        perm[0, [14, 15]] = [15, 14]
    with pytest.raises(ValueError):
        views.transport(perm, 0)
    for function in (views.apply, views.motion_rms):
        with pytest.raises(ValueError):
            function(values, mask, perm)


def test_empty_batches_are_supported_without_special_values():
    values, mask = windows(0)
    perm, support = views.plans(mask)
    assert perm.shape == (0, 4, 16) and support.shape == (0, 4)
    assert views.transport(perm, 24).shape == (0, 4, 16)
    assert views.apply(values, mask, identity(0)).shape == (0, 128, 23)
    assert views.motion_rms(values, mask, identity(0)).shape == (0,)


def test_seeded_random_cycles_against_direct_sample_oracle():
    rng = np.random.default_rng(20261106)
    for _ in range(50):
        moved = rng.choice(np.arange(2, 15), size=int(rng.integers(2, 8)), replace=False)
        perm = identity()
        perm[0, moved] = np.roll(moved, -1)
        mask = rng.random((1, 128, 23)) > 0.25
        mask.reshape(1, 16, 8, 23)[0, moved] = rng.random((8, 23)) > 0.25
        values = rng.standard_normal(mask.shape)
        values[~mask] = np.nan

        expected = np.empty_like(values)
        for destination, source in enumerate(perm[0]):
            expected[:, destination * 8 : (destination + 1) * 8] = (
                values[:, source * 8 : (source + 1) * 8]
            )
        actual = views.apply(values, mask, perm)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(views.apply(actual, mask, np.argsort(perm)), values)
        selected = np.zeros_like(mask)
        selected.reshape(1, 16, 8, 23)[0, moved] = mask.reshape(1, 16, 8, 23)[0, moved]
        differences = actual[selected] - values[selected]
        oracle_rms = np.sqrt(np.mean(differences**2)) if differences.size else 0.0
        np.testing.assert_allclose(views.motion_rms(values, mask, perm), [oracle_rms])

        shift = int(rng.integers(0, 3))
        shifted = views.transport(perm, 8 * shift)
        expected_shifted = identity()
        expected_shifted[0, moved - shift] = perm[0, moved] - shift
        np.testing.assert_array_equal(shifted, expected_shifted)
