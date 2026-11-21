import numpy as np

from peak_contrast_policy_20261106 import (
    choose,
    fit_values,
    grouped_folds,
    utility,
)


def test_utility_shape_removes_uniform_shifts_but_not_raw_guard_cost():
    tau = 0.5
    native = np.full((1, 2, 32), 0.2)
    future = np.broadcast_to(native[:, :, None, None], (1, 2, 3, 2, 32)).copy()
    delta = np.zeros((1, 2, 3, 2))
    delta[..., 0] = 0.4
    future[..., 0, :] += 0.4
    future[:, 0, :, 1, 4] += 0.1
    positive = np.arange(32)[None] < 8
    raw, shape = utility(native, future, delta, positive, np.zeros_like(positive), tau)
    np.testing.assert_allclose(raw[0, 0, :, 0], 0.25)
    np.testing.assert_allclose(shape[0, 0, :, 0], 0, atol=1e-15)
    np.testing.assert_allclose(shape[0, 1, :, 0], -0.75)
    np.testing.assert_allclose(shape[0, 0, :, 1], 0.1)


def test_causal_selector_abstains_without_positive_available_evidence():
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.column_stack((x[:, 0] * 0 + 0.2, x[:, 1] * 0 - 0.2))
    fit = fit_values(x, y, np.ones_like(y, bool))
    assert choose(fit, x[:1], np.array([[True, True]])).tolist() == [0]
    assert choose(fit, x[:1], np.array([[False, True]])).tolist() == [-1]
    assert choose(fit, x[:1], np.array([[False, False]])).tolist() == [-1]


def test_ridge_ignores_unsupported_pseudorewards_and_requires_six_pairs():
    x = np.zeros((14, 2))
    y = np.ones((14, 2))
    available = np.ones_like(y, dtype=bool)
    available[10:, 1] = False
    y[~available] = 9000.0
    fit = fit_values(x, y, available)
    assert fit["trained"].tolist() == [True, False]
    assert choose(fit, x[:1], np.array([[False, True]])).tolist() == [-1]


def test_grouped_fold_keeps_both_members_together_and_covers_every_pair():
    ids = np.column_stack((np.arange(12)[::-1], np.arange(12) + 100))
    split = grouped_folds(ids)
    assert len(split) == 5
    np.testing.assert_array_equal(
        np.sort(np.concatenate([q[1] for q in split])), np.arange(12)
    )
    for train, held in split:
        assert not set(ids[train].flatten()) & set(ids[held].flatten())


def test_held_out_future_effects_and_masks_cannot_change_policy_choice():
    from run_peak_contrast_crossfit_20261106 import select_fold

    n, arms, tau = 20, 4, 0.5
    a = np.full((n, 2, 32), 0.2)
    z = np.broadcast_to(a[:, :, None, None], (n, 2, 3, arms, 32)).copy()
    z[:, 0, :, 1, 2] = 0.4
    z[:, 1, :, 1] = 0.1
    off = np.zeros(z.shape[:-1])
    desc = np.zeros((n, 2, 3, 35))
    available = np.ones((n, 3, arms), dtype=bool)
    pos, prot = np.ones((n, 32), dtype=bool), np.zeros((n, 32), dtype=bool)
    tr, held = np.arange(16), np.arange(16, 20)
    original = select_fold(z, off, a, desc, available, pos, prot, tau, tr, held)
    assert np.all(original["shape"] == 1)
    z[held], off[held], a[held] = -999, 82, 91
    pos[held], prot[held] = False, True
    changed = select_fold(z, off, a, desc, available, pos, prot, tau, tr, held)
    assert all(np.array_equal(original[k], changed[k]) for k in original)
