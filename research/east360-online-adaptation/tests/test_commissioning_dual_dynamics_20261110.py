import numpy as np

from experiments.risk_aligned_online.commissioning_dual_dynamics_20261110 import (
    fit,
    prepare,
    replay,
)


def huber(residual):
    magnitude = np.abs(residual)
    return np.where(magnitude <= 0.5, 0.5 * residual**2, 0.5 * (magnitude - 0.25))


def make_blocks(origins, targets, masks=None):
    targets = np.asarray(targets, dtype=float)
    if masks is None:
        masks = np.ones_like(targets, dtype=bool)
    return np.asarray(origins), np.ones((len(origins), 1)), targets, np.asarray(masks)


def test_prepare_windows_masks_clipping_and_targets():
    z = np.zeros((185, 3))
    mask = np.ones_like(z, dtype=bool)
    z[64:96] = [1, 1, 1]
    z[96:112] = [3, 3, 3]
    z[112:128] = [9, 7, 5]
    z[128:153] = [11, 9, 8]
    mask[112:128, 1] = False
    mask[112:119, 1] = True
    mask[112:120, 2] = False
    mask[96:112, 2] = False
    mask[96:103, 2] = True
    mask[128, 0] = False
    z[128, 0] = np.nan
    context = np.full((185, 2), 10.0)
    context_mask = np.ones_like(context, dtype=bool)
    context_mask[112:128, 1] = False
    context_mask[112:119, 1] = True
    context[120:128, 1] = np.nan

    q, x, y, m = prepare(z, mask, context, context_mask)

    np.testing.assert_array_equal(q, [127, 159])
    raw = np.array([1, 4, 0, 4, 0, 0, 2, 2, 0], dtype=float)
    np.testing.assert_allclose(x[0], raw / np.linalg.norm(raw))
    assert x.shape == (2, 9)
    assert y.shape == m.shape == (2, 25, 3)
    assert m.dtype == np.bool_
    assert not m[0, 0, 0]
    assert y[0, 0, 0] == 0
    np.testing.assert_array_equal(m[0, :, 1], False)
    np.testing.assert_array_equal(y[0, :, 1], 0)
    np.testing.assert_allclose(y[0, 1:, 0], 2)
    np.testing.assert_allclose(y[0, :, 2], 3)


def test_prepare_empty_and_exact_complete_horizon_boundary():
    for length, expected in [(0, []), (152, []), (153, [127]), (184, [127]), (185, [127, 159])]:
        z = np.zeros((length, 2))
        context = np.zeros((length, 2))
        q, x, y, m = prepare(z, np.ones_like(z, bool), context, np.ones_like(context, bool))
        np.testing.assert_array_equal(q, expected)
        assert x.shape == (len(expected), 7)
        assert y.shape == m.shape == (len(expected), 25, 2)


def test_weighted_ridge_matches_independent_solves_and_empty_cells_hold():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(9, 3))
    y = rng.normal(size=(9, 25, 2))
    m = rng.random(y.shape) > 0.35
    m[:, 3, 1] = False
    weights = np.arange(9, dtype=float) / 4
    coefficients = fit(x, y, m, weights)
    assert coefficients.shape == (3, 25, 2)
    for h in range(25):
        for c in range(2):
            w = weights * m[:, h, c]
            if w.sum() == 0:
                expected = np.zeros(3)
            else:
                expected = np.linalg.solve(
                    x.T @ (w[:, None] * x) + 0.05 * w.sum() * np.eye(3),
                    x.T @ (w * y[:, h, c]),
                )
            np.testing.assert_allclose(coefficients[:, h, c], expected, atol=1e-11)
    np.testing.assert_array_equal(fit(x[:0], y[:0], m[:0], weights[:0]), 0)


def test_fit_constant_signal_beats_hold_and_zero_weights_hold():
    x = np.ones((10, 1))
    y = np.full((10, 25, 2), 0.3)
    m = np.ones_like(y, dtype=bool)
    coefficients = fit(x, y, m, np.ones(10))
    predicted = np.einsum('bd,dhc->bhc', x, coefficients)
    assert np.mean(huber(predicted - y)) < np.mean(huber(y))
    np.testing.assert_allclose(coefficients, 0.3 / 1.05)
    np.testing.assert_array_equal(fit(x, y, m, np.zeros(10)), 0)


def test_release_153_and_first_adapted_evidence_185():
    y = np.full((2, 25, 1), 0.2)
    blocks = make_blocks([127, 159], y)
    benign = np.zeros((1, 25, 1))
    disruptive = np.full((1, 25, 1), 0.2)

    features, audit = replay(186, blocks, benign, disruptive)

    for values in features.values():
        assert values.shape == (186, 3)
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values[:153], 0)
        np.testing.assert_allclose(values[153:185, 0], 0.02, atol=1e-7)
        np.testing.assert_array_equal(values[153:185, 1], 0)
        np.testing.assert_array_equal(values[153:185, 2], 1)
    assert np.isclose(features['fixed'][185, 0], 0.02)
    assert np.isclose(features['real'][185, 0], 0.01808)
    assert np.isclose(features['reverse'][185, 0], 0.01808)
    assert np.isclose(features['real'][185, 1], 0.01808)
    assert audit['mature_blocks'] == 2
    assert audit['update_attempts'] == {'real': 2, 'reverse': 2}
    assert audit['parameter_changes'] == {'real': 2, 'reverse': 2}


def test_completed_last_block_audited_without_early_feature():
    blocks = make_blocks([127], np.full((1, 25, 1), 0.2))
    benign = np.zeros((1, 25, 1))
    disruptive = np.full((1, 25, 1), 0.2)
    features, audit = replay(153, blocks, benign, disruptive)
    for values in features.values():
        np.testing.assert_array_equal(values, 0)
    assert audit['mature_blocks'] == 1
    assert audit['update_attempts'] == {'real': 1, 'reverse': 1}
    assert audit['benign_huber_count'] == {'fixed': 25, 'real': 25, 'reverse': 25}
    assert audit['hold_huber_count'] == 25


def test_shared_gate_reverse_only_valid_targets_and_original_mask():
    y = np.zeros((2, 25, 2))
    y[0, :, 0] = np.linspace(0.02, 0.26, 25)
    y[1, :, 0] = np.linspace(0.07, 0.24, 25)
    y[1, :, 1] = 0.15
    m = np.ones_like(y, dtype=bool)
    m[0, :, :] = False
    valid = np.array([0, 2, 4, 7, 10, 13, 20, 24])
    m[0, valid, 0] = True
    m[1, 0, 0] = False
    blocks = make_blocks([127, 159], y, m)
    benign = np.zeros((1, 25, 2))
    disruptive = np.full((1, 25, 2), 0.2)

    features, audit = replay(186, blocks, benign, disruptive)

    gate = np.clip(0.5 + (huber(0.2 - y[0])[m[0]].mean() - huber(y[0])[m[0]].mean()) / 0.1, 0, 1)
    offset = np.zeros((25, 2))
    offset[valid, 0] = 0.2 * gate / 1.25 * y[0, valid, 0]
    reverse_offset = np.zeros_like(offset)
    reverse_offset[valid, 0] = 0.2 * gate / 1.25 * y[0, valid[::-1], 0]
    for arm, adjustment in [('fixed', np.zeros_like(offset)), ('real', offset), ('reverse', reverse_offset)]:
        difference = huber(adjustment - y[1]) - huber(0.2 + adjustment - y[1])
        expected = np.where(m[1], difference, 0).sum(axis=0) / m[1].sum(axis=0)
        first = features[arm][153].reshape(3, 2).T
        second = features[arm][185].reshape(3, 2).T
        np.testing.assert_allclose(second[:, 0], expected, atol=1e-7)
        np.testing.assert_allclose(first[:, 1], 0)
        np.testing.assert_allclose(first[:, 2], [8 / 25, 0])
        np.testing.assert_allclose(second[:, 2], [24 / 25, 1])
    assert not np.isclose(features['real'][185, 0], features['reverse'][185, 0], atol=1e-7)
    assert features['real'][185, 1] == features['fixed'][185, 1]
    for arm, adjustment in [('fixed', np.zeros_like(offset)), ('real', offset), ('reverse', reverse_offset)]:
        expected_sum = huber(y[0])[m[0]].sum() + huber(adjustment - y[1])[m[1]].sum()
        np.testing.assert_allclose(audit['benign_huber_sum'][arm], expected_sum)
        assert audit['benign_huber_count'][arm] == 57
    np.testing.assert_allclose(audit['hold_huber_sum'], huber(y)[m].sum())
    assert audit['hold_huber_count'] == 57


def test_persistence_uses_previous_completed_block_and_resets_on_missing_or_sign_change():
    y = np.empty((4, 25, 2))
    y[0] = 0.3
    y[1] = 0.25
    y[2] = 0.1
    y[3] = 0.3
    m = np.ones_like(y, dtype=bool)
    m[1, :, 1] = False
    blocks = make_blocks([127, 159, 191, 223], y, m)
    benign = np.zeros((1, 25, 2))
    disruptive = np.full((1, 25, 2), 0.4)
    features, _ = replay(250, blocks, benign, disruptive)
    fixed = features['fixed'].reshape(250, 3, 2).transpose(0, 2, 1)
    np.testing.assert_allclose(fixed[153, :, 0], [0.04, 0.04], atol=1e-7)
    np.testing.assert_allclose(fixed[185, :, 0], [0.02, 0], atol=1e-7)
    np.testing.assert_allclose(fixed[185, :, 1], [0.02, 0], atol=1e-7)
    np.testing.assert_allclose(fixed[217, :, 0], [-0.04, -0.04], atol=1e-7)
    np.testing.assert_allclose(fixed[217, :, 1], [0, 0], atol=1e-7)
    np.testing.assert_allclose(fixed[249, :, 1], [0, 0], atol=1e-7)


def test_no_observations_count_as_no_updates_and_missing_evidence_is_zero():
    y = np.full((1, 25, 1), 123.0)
    blocks = make_blocks([127], y, np.zeros_like(y, dtype=bool))
    coefficients = np.zeros((1, 25, 1))
    features, audit = replay(154, blocks, coefficients, coefficients)
    for values in features.values():
        np.testing.assert_array_equal(values, 0)
    assert audit['mature_blocks'] == 1
    assert audit['update_attempts'] == {'real': 0, 'reverse': 0}
    assert audit['parameter_changes'] == {'real': 0, 'reverse': 0}
    assert audit['hold_huber_count'] == 0


def test_gradient_clip_retention_missing_columns_and_norm_cap():
    origins = 127 + 32 * np.arange(20)
    y = np.full((20, 25, 1), 2.0)
    m = np.zeros_like(y, dtype=bool)
    m[:, 0, 0] = True
    m[1, 0, 0] = False
    m[1, 1, 0] = True
    m[-1, 1, 0] = True
    blocks = make_blocks(origins, y, m)
    coefficients = np.zeros((1, 25, 1))
    _, audit = replay(int(origins[-1] + 26), blocks, coefficients, coefficients)

    coefficients_now = np.zeros(25)
    expected_sum = 0.0
    changed = 0
    for index in range(len(origins)):
        positions = np.flatnonzero(m[index, :, 0])
        expected_sum += huber(coefficients_now[positions] - 2).sum()
        before = coefficients_now.copy()
        coefficients_now[positions] = np.minimum(0.5, 0.98 * coefficients_now[positions] + 0.04)
        changed += int(np.any(before != coefficients_now))
    assert coefficients_now[0] == 0.5
    np.testing.assert_allclose(coefficients_now[1], 0.0792)
    for arm in ('real', 'reverse'):
        np.testing.assert_allclose(audit['benign_huber_sum'][arm], expected_sum)
        assert audit['update_attempts'][arm] == 20
        assert audit['parameter_changes'][arm] == changed < 20


def test_last_future_point_changes_evidence_only_after_release():
    z = np.zeros((154, 1))
    mask = np.ones_like(z, dtype=bool)
    context = np.zeros((154, 2))
    context_mask = np.ones_like(context, dtype=bool)
    original = prepare(z, mask, context, context_mask)
    z[152, 0] = 1
    context[128:] = 100
    changed = prepare(z, mask, context, context_mask)
    np.testing.assert_array_equal(original[1][0], changed[1][0])
    assert original[2][0, -1, 0] != changed[2][0, -1, 0]
    benign = np.zeros((5, 25, 1))
    disruptive = np.zeros_like(benign)
    disruptive[0] = 0.2
    baseline, _ = replay(154, original, benign, disruptive)
    perturbed, _ = replay(154, changed, benign, disruptive)
    for arm in baseline:
        np.testing.assert_array_equal(baseline[arm][:153], perturbed[arm][:153])
        assert baseline[arm][153, 0] != perturbed[arm][153, 0]


def test_common_prefix_under_truncation_future_perturbation_and_shot_reset():
    rng = np.random.default_rng(71)
    z = rng.normal(size=(250, 2))
    mask = rng.random(z.shape) > 0.15
    context = rng.normal(size=(250, 2))
    context_mask = rng.random(context.shape) > 0.15
    benign = rng.normal(scale=0.05, size=(7, 25, 2))
    disruptive = rng.normal(scale=0.05, size=(7, 25, 2))
    full = prepare(z, mask, context, context_mask)
    short = prepare(z[:170], mask[:170], context[:170], context_mask[:170])
    mutated_z, mutated_mask = z.copy(), mask.copy()
    mutated_context, mutated_context_mask = context.copy(), context_mask.copy()
    mutated_z[170:] += 100
    mutated_mask[170:] = ~mutated_mask[170:]
    mutated_context[170:] -= 100
    mutated_context_mask[170:] = ~mutated_context_mask[170:]
    mutated = prepare(mutated_z, mutated_mask, mutated_context, mutated_context_mask)
    for part in range(1, 4):
        np.testing.assert_array_equal(full[part][0], short[part][0])
        np.testing.assert_array_equal(full[part][0], mutated[part][0])
    result, _ = replay(250, full, benign, disruptive)
    truncated, _ = replay(170, short, benign, disruptive)
    altered, _ = replay(250, mutated, benign, disruptive)
    repeated, _ = replay(250, full, benign, disruptive)
    for arm in result:
        np.testing.assert_array_equal(result[arm][:170], truncated[arm])
        np.testing.assert_array_equal(result[arm][:170], altered[arm][:170])
        np.testing.assert_array_equal(result[arm], repeated[arm])
