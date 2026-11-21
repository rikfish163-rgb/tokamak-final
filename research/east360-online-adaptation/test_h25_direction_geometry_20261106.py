"""Independent, synthetic CPU tests for the direction geometry diagnostic."""

import numpy as np
import pytest

from h25_direction_geometry_20261106 import basis, capture, matched_augments


def projector(q):
    return q @ q.T


def assert_orthonormal(q, atol=1e-10):
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=atol, rtol=0)


def test_basis_uses_relative_and_absolute_singular_value_cutoffs():
    relative = basis(np.diag([1.0, 2e-6, 5e-7, 0.0]))
    absolute = basis(np.diag([2e-10, 5e-11]))

    assert relative.shape == (4, 2)
    assert absolute.shape == (2, 1)
    assert_orthonormal(relative)
    np.testing.assert_allclose(projector(relative), np.diag([1, 1, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(projector(absolute), np.diag([1, 0]), atol=1e-12)


def test_basis_ignores_zero_rows_and_handles_collinear_and_empty_inputs():
    rows = np.array([[0, 0, 0], [1, 2, 0], [-2, -4, 0], [0, 0, 0]])
    q = basis(rows)

    assert q.shape == (3, 1)
    np.testing.assert_allclose(projector(q), np.outer([1, 2, 0], [1, 2, 0]) / 5)
    assert basis(np.empty((0, 3))).shape == (3, 0)
    assert basis(np.zeros((4, 3))).shape == (3, 0)
    assert basis(np.empty((2, 0))).shape == (0, 0)
    assert basis([[5e-11, 0, 0]]).shape == (3, 0)
    assert basis([[1e-320, 0, 0]]).shape == (3, 0)


def test_basis_rotation_and_global_scaling_preserve_projector():
    rng = np.random.default_rng(101)
    rows = rng.normal(size=(4, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    expected = rotation @ projector(basis(rows)) @ rotation.T

    for scale in (1e-3, 1.0, 1e3):
        rotated = basis(scale * rows @ rotation.T)
        assert_orthonormal(rotated)
        np.testing.assert_allclose(projector(rotated), expected, atol=1e-12)


def test_basis_accepts_large_finite_values_without_spurious_overflow():
    q = basis(np.diag([1e300, 2e294, 5e293]))
    assert q.shape == (3, 2)
    assert_orthonormal(q)


def test_capture_projects_onto_the_intersection_and_uses_full_null_denominator():
    q = np.eye(5)[:, :3]
    past = np.array([[1, 0, 0, 0, 0]])
    future = np.array([3, 4, 5, 12, 0])
    result = capture(q, past, future)

    assert set(result) == {
        "rank",
        "constrained_rank",
        "absolute_norm",
        "fraction",
        "residual_past",
        "vector",
    }
    assert result["rank"] == 3
    assert result["constrained_rank"] == 2
    assert isinstance(result["rank"], int)
    assert isinstance(result["constrained_rank"], int)
    assert all(
        isinstance(result[key], float)
        for key in ("absolute_norm", "fraction", "residual_past")
    )
    np.testing.assert_allclose(result["vector"], [0, 4, 5, 0, 0], atol=1e-12)
    assert result["absolute_norm"] == pytest.approx(np.sqrt(41))
    assert result["fraction"] == pytest.approx(np.sqrt(41 / 185))
    assert result["residual_past"] == pytest.approx(0, abs=1e-12)


def test_capture_is_not_a_sequential_projection_onto_two_different_spaces():
    q = np.eye(3)[:, :2]
    past = np.array([[1.0, 1.0, 1.0]])
    result = capture(q, past, np.array([0.0, 1.0, 2.0]))

    assert result["rank"] == 2
    assert result["constrained_rank"] == 1
    np.testing.assert_allclose(result["vector"], [-0.5, 0.5, 0], atol=1e-12)
    assert result["absolute_norm"] == pytest.approx(np.sqrt(0.5))
    assert result["fraction"] == pytest.approx(0.5)
    assert result["residual_past"] == pytest.approx(0, abs=1e-12)


def test_capture_empty_and_collinear_constraints_and_zero_basis():
    q = np.eye(3)[:, :2]
    future = np.array([1.0, 2.0, 2.0])
    no_past = capture(q, np.empty((0, 3)), future)
    collinear = capture(q, np.array([[1, 0, 0], [2, 0, 0], [0, 0, 0]]), future)
    no_q = capture(np.empty((3, 0)), np.empty((0, 3)), future)

    assert no_past["constrained_rank"] == 2
    assert no_past["fraction"] == pytest.approx(np.sqrt(5) / 3)
    assert collinear["constrained_rank"] == 1
    np.testing.assert_allclose(collinear["vector"], [0, 2, 0], atol=1e-12)
    assert no_q["rank"] == no_q["constrained_rank"] == 0
    assert no_q["absolute_norm"] == no_q["fraction"] == no_q["residual_past"] == 0
    np.testing.assert_array_equal(no_q["vector"], np.zeros(3))


def test_capture_preserves_a_tiny_component_relative_to_a_large_future():
    result = capture(np.array([[1.0], [0.0]]), np.empty((0, 2)), [1e-200, 1.0])

    np.testing.assert_array_equal(result["vector"], [1e-200, 0])
    assert result["absolute_norm"] == pytest.approx(1e-200, rel=1e-12, abs=0)
    assert result["fraction"] == pytest.approx(1e-200, rel=1e-12, abs=0)


def test_capture_treats_a_nearly_orthonormal_basis_as_its_span():
    result = capture(np.array([[1.0 + 4e-8], [0.0]]), np.empty((0, 2)), [1.0, 0.0])

    np.testing.assert_allclose(result["vector"], [1, 0], atol=1e-12, rtol=0)
    assert result["absolute_norm"] == pytest.approx(1.0, abs=1e-12)
    assert result["fraction"] == pytest.approx(1.0, abs=1e-12)


def test_capture_exposes_approximate_constraint_residual():
    past = np.diag([1.0, 5e-7, 0])
    result = capture(np.eye(3), past, [0, 1.0, 0])

    assert result["constrained_rank"] == 2
    assert result["fraction"] == pytest.approx(1)
    assert result["residual_past"] == pytest.approx(5e-7)
    np.testing.assert_allclose(result["vector"], [0, 1, 0], atol=1e-12)


def test_capture_does_not_hide_inconsistent_numerical_rank_truncation():
    q = np.array([[1.0, 0], [0, 1e-7], [0, np.sqrt(1 - 1e-14)]])
    past = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    with pytest.raises(ValueError, match="undefined capture fraction"):
        capture(q, past, [0, 1.0, 0])


def test_capture_rejects_unrepresentable_norm_and_bad_q_without_warnings():
    q = np.array([[1.0], [1.0]]) / np.sqrt(2)
    with pytest.raises(OverflowError):
        capture(q, np.empty((0, 2)), [1.5e308, 1.5e308])
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError):
            capture([[1e300], [0]], np.empty((0, 2)), [1.0, 0])


def test_capture_uses_economy_svd_for_tall_constraints(monkeypatch):
    original_svd = np.linalg.svd
    saw_constraint = []

    def checked_svd(matrix, *args, **kwargs):
        if matrix.shape == (256, 2):
            saw_constraint.append(True)
            assert kwargs.get("full_matrices") is False
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", checked_svd)
    past = np.tile([1.0, 0, 0, 0], (256, 1))
    result = capture(np.eye(4)[:, :2], past, [0, 1.0, 0, 0])

    assert saw_constraint
    assert result["constrained_rank"] == 1
    assert result["fraction"] == pytest.approx(1)


def test_capture_documents_zero_over_zero_for_future_already_in_past():
    past = np.array([[1.0, 1.0, 0], [0, 1.0, 1.0]])
    future = 3 * past[0] - 2 * past[1]
    result = capture(np.eye(3), past, future)

    assert result["fraction"] == 0.0
    assert result["absolute_norm"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(result["vector"], np.zeros(3), atol=1e-12)
    assert all(
        np.isfinite(result[key])
        for key in ("fraction", "absolute_norm", "residual_past")
    )
    zero = capture(np.eye(3), past, np.zeros(3))
    assert zero["fraction"] == zero["absolute_norm"] == zero["residual_past"] == 0
    np.testing.assert_array_equal(zero["vector"], np.zeros(3))


def test_capture_is_equivariant_to_rotation_and_future_scaling():
    rng = np.random.default_rng(202)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    q = np.eye(8)[:, :4]
    past = np.array([[1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0]])
    future = np.arange(1, 9, dtype=float)
    original = capture(q, past, future)

    for scale in (-3.0, 0.25, 1e-150):
        transformed = capture(
            rotation @ q, 7 * past @ rotation.T, scale * rotation @ future
        )
        assert transformed["rank"] == original["rank"]
        assert transformed["constrained_rank"] == original["constrained_rank"]
        np.testing.assert_allclose(
            transformed["vector"] / scale, rotation @ original["vector"], atol=1e-12
        )
        assert transformed["absolute_norm"] / abs(scale) == pytest.approx(
            original["absolute_norm"]
        )
        assert transformed["fraction"] == pytest.approx(original["fraction"])


def test_capture_matches_independent_pseudoinverse_projections():
    rng = np.random.default_rng(606)
    for dimension in (3, 8, 16):
        for _ in range(12):
            rank = int(rng.integers(0, dimension + 1))
            n_past = int(rng.integers(0, dimension + 3))
            q = np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")[0]
            past = rng.normal(size=(n_past, dimension))
            future = rng.normal(size=dimension)
            result = capture(q, past, future)

            if n_past and rank:
                a = past @ q
                null_projector = np.eye(rank) - np.linalg.pinv(a) @ a
            else:
                null_projector = np.eye(rank)
            expected = q @ null_projector @ q.T @ future
            full_null = (
                future - np.linalg.pinv(past) @ past @ future if n_past else future
            )

            assert result["rank"] == rank
            assert result["constrained_rank"] == max(0, rank - n_past)
            np.testing.assert_allclose(result["vector"], expected, atol=1e-11, rtol=0)
            assert result["absolute_norm"] == pytest.approx(
                np.linalg.norm(expected), abs=1e-11
            )
            assert result["residual_past"] == pytest.approx(
                np.linalg.norm(past @ result["vector"]), abs=1e-11
            )
            if np.linalg.norm(full_null) < 1e-11:
                assert result["fraction"] == 0
            else:
                assert result["fraction"] == pytest.approx(
                    np.linalg.norm(expected) / np.linalg.norm(full_null), abs=1e-11
                )


def test_128d_matched_augments_keep_all_old_and_dominant_extra_directions():
    rng = np.random.default_rng(303)
    rotation, _ = np.linalg.qr(rng.normal(size=(128, 128)))
    axes = rotation.T
    old_rows = axes[:4]
    true_rows = axes[[4, 4, 4, 5, 5, 6]]
    reverse_rows = axes[[7, 8]]

    old, joined_true, joined_reverse, common = matched_augments(
        old_rows, true_rows, reverse_rows
    )

    assert old.shape == (128, 4)
    assert joined_true.shape == joined_reverse.shape == (128, 6)
    assert common == 2
    for q in (old, joined_true, joined_reverse):
        assert_orthonormal(q)
    np.testing.assert_array_equal(joined_true[:, :4], old)
    np.testing.assert_array_equal(joined_reverse[:, :4], old)
    np.testing.assert_allclose(projector(old), axes[:4].T @ axes[:4], atol=1e-12)
    np.testing.assert_allclose(
        projector(joined_true), axes[:6].T @ axes[:6], atol=1e-12
    )
    expected_reverse = np.concatenate([axes[:4], axes[7:9]], axis=0)
    np.testing.assert_allclose(
        projector(joined_reverse), expected_reverse.T @ expected_reverse, atol=1e-12
    )


def test_auxiliary_row_scaling_and_sham_sign_do_not_change_the_subspace():
    old = np.eye(5)[:1]
    true = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    _, original, sham, common = matched_augments(old, true, -true)
    scaled = true * np.array([1e-300, -1e300, 3e-100])[:, None]
    _, changed, changed_sham, changed_common = matched_augments(old, scaled, -scaled)

    assert common == changed_common == 2
    np.testing.assert_allclose(projector(original), projector(sham), atol=1e-12)
    np.testing.assert_allclose(projector(changed), projector(original), atol=1e-12)
    np.testing.assert_allclose(projector(changed_sham), projector(original), atol=1e-12)


def test_auxiliary_scale_cannot_change_which_dominant_direction_is_selected():
    old = np.empty((0, 3))
    true = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    reverse = np.array([[0, 0, 1]])
    _, selected, _, common = matched_augments(old, true, reverse)
    _, scaled, _, scaled_common = matched_augments(
        old, true * [[1], [1], [1e12]], reverse
    )

    assert common == scaled_common == 1
    np.testing.assert_allclose(projector(selected), np.diag([1, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(projector(scaled), projector(selected), atol=1e-12)


def test_matched_subspaces_are_invariant_under_independent_row_scaling():
    rng = np.random.default_rng(707)
    old = rng.normal(size=(4, 16))
    true = rng.normal(size=(6, 16))
    reverse = rng.normal(size=(3, 16))
    reference = matched_augments(old, true, reverse)
    true_scale = np.array([1e-150, -1e150, 1e-50, 7.0, -1e50, 1e100])
    reverse_scale = np.array([-1e100, 1e-100, -0.1])
    scaled = matched_augments(
        old, true * true_scale[:, None], reverse * reverse_scale[:, None]
    )

    assert reference[3] == scaled[3] == 3
    for original, changed in zip(reference[:3], scaled[:3], strict=True):
        assert_orthonormal(changed)
        np.testing.assert_allclose(projector(changed), projector(original), atol=1e-11)


def test_all_apis_leave_input_arrays_unchanged():
    q = np.eye(3)[:, :2].copy()
    past = np.array([[1.0, 1.0, 0]])
    future = np.array([1.0, 2.0, 3.0])
    true = np.array([[0.0, 1.0, 1.0]])
    reverse = np.array([[0.0, 0.0, 1.0]])
    arrays = (q, past, future, true, reverse)
    before = tuple(array.copy() for array in arrays)

    basis(past)
    capture(q, past, future)
    matched_augments(past, true, reverse)

    for actual, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_matched_augments_empty_collinear_and_no_matching_extra_rank():
    old_rows = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    true_rows = np.array([[0.0, 1, 0], [0, 2, 0]])
    reverse_rows = np.array([[3.0, 0, 0], [0, 0, 0]])
    old, joined_true, joined_reverse, common = matched_augments(
        old_rows, true_rows, reverse_rows
    )

    assert common == 0
    assert old.shape == joined_true.shape == joined_reverse.shape == (3, 1)
    np.testing.assert_array_equal(joined_true, old)
    np.testing.assert_array_equal(joined_reverse, old)
    empty = np.empty((0, 3))
    empty_old, a, b, common = matched_augments(empty, empty, empty)
    assert common == 0
    assert empty_old.shape == a.shape == b.shape == (3, 0)
    all_zero = np.zeros((2, 3))
    _, a, b, common = matched_augments(empty, all_zero, all_zero)
    assert common == 0
    assert a.shape == b.shape == (3, 0)


def test_zero_dimensional_inputs_are_well_defined():
    empty_rows = np.empty((0, 0))
    old, true, reverse, common = matched_augments(empty_rows, empty_rows, empty_rows)
    result = capture(old, empty_rows, np.empty(0))

    assert common == 0
    assert old.shape == true.shape == reverse.shape == (0, 0)
    assert result["rank"] == result["constrained_rank"] == 0
    assert result["absolute_norm"] == result["fraction"] == result["residual_past"] == 0
    assert result["vector"].shape == (0,)


def test_matched_bases_feed_capture_without_rank_or_dimension_confounds():
    old, true, reverse, common = matched_augments(
        [[1.0, 0, 0, 0]], [[0, 1.0, 0, 0]], [[0, 0, 1.0, 0]]
    )
    past = [[1.0, 0, 0, 0]]
    future = [1.0, 2.0, 0, 0]
    old_result = capture(old, past, future)
    true_result = capture(true, past, future)
    reverse_result = capture(reverse, past, future)

    assert common == 1
    assert true_result["rank"] == reverse_result["rank"] == 2
    assert true_result["constrained_rank"] == reverse_result["constrained_rank"] == 1
    assert old_result["fraction"] == reverse_result["fraction"] == 0
    assert true_result["fraction"] == pytest.approx(1)
    np.testing.assert_allclose(true_result["vector"], [0, 2, 0, 0], atol=1e-12)


def test_auxiliary_nearly_in_old_span_does_not_create_roundoff_directions():
    rng = np.random.default_rng(404)
    old_rows = rng.normal(size=(7, 32))
    old_q = basis(old_rows)
    only_old = 3 * old_rows - 2 * old_rows[::-1]
    old, a, b, common = matched_augments(old_rows, only_old, only_old)

    assert common == 0
    assert a.shape == b.shape == old.shape == old_q.shape
    np.testing.assert_allclose(projector(old), projector(old_q), atol=1e-12)


def test_auxiliary_barely_above_absolute_floor_is_still_orthogonal():
    rng = np.random.default_rng(405)
    directions, _ = np.linalg.qr(rng.normal(size=(32, 2)))
    old_rows = directions[:, 0][None, :]
    auxiliary = (directions[:, 0] + 4e-10 * directions[:, 1])[None, :]

    old, a, b, common = matched_augments(old_rows, auxiliary, -auxiliary)

    assert common == 1
    assert_orthonormal(a, atol=1e-12)
    assert_orthonormal(b, atol=1e-12)
    np.testing.assert_array_equal(a[:, :1], old)
    np.testing.assert_allclose(projector(a), directions @ directions.T, atol=1e-6)
    np.testing.assert_allclose(projector(a), projector(b), atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_apis_reject_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        basis([[1.0, bad]])
    with pytest.raises(ValueError):
        capture(np.array([[1.0], [0.0]]), np.empty((0, 2)), [1.0, bad])
    with pytest.raises(ValueError):
        capture(np.array([[1.0], [bad]]), np.empty((0, 2)), [1.0, 0.0])
    with pytest.raises(ValueError):
        capture(np.eye(2), [[1.0, bad]], [1.0, 0.0])
    with pytest.raises(ValueError):
        matched_augments(np.empty((0, 2)), [[1.0, bad]], np.empty((0, 2)))
    with pytest.raises(ValueError):
        matched_augments(np.empty((0, 2)), np.empty((0, 2)), [[1.0, bad]])


@pytest.mark.parametrize(
    "bad_rows", [[], [1, 2], [[[1, 2]]], [[1 + 1j, 2]], [["not-a-number", "2"]]]
)
def test_basis_rejects_invalid_input_shape_and_values(bad_rows):
    with pytest.raises(ValueError):
        basis(bad_rows)


@pytest.mark.parametrize(
    ("q", "past", "future"),
    [
        ([1, 0], [[1, 0]], [1, 0]),
        (np.eye(2), [1, 0], [1, 0]),
        (np.eye(2), [[1, 0]], [[1, 0]]),
        (np.eye(2), [[1, 0, 0]], [1, 0]),
        (np.eye(2), [[1, 0]], [1, 0, 0]),
        ([[2], [0]], [[1, 0]], [1, 0]),
        ([[1, 1], [0, 0]], [[1, 0]], [1, 0]),
        (np.ones((2, 3)), np.empty((0, 2)), [1, 0]),
    ],
)
def test_capture_rejects_bad_shapes_and_non_orthonormal_q(q, past, future):
    with pytest.raises(ValueError):
        capture(q, past, future)


def test_matched_augments_reject_inconsistent_ambient_dimensions():
    with pytest.raises(ValueError):
        matched_augments(np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 3)))
    with pytest.raises(ValueError):
        matched_augments(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)))
