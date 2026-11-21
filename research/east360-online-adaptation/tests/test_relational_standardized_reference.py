"""CPU contracts for covariate-standardized reference fitting only."""

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.relational_attention_20261030.learning import (
    fit_reference,
    relation_target,
)
from experiments.risk_aligned_online.relational_attention_20261030.standardized_reference import (
    MIN_COVARIATE_STD,
    fit_reference_standardized,
)


U = [io23.RAW23.index(name) for name in ('Bt', 'ip')]
C = [io23.RAW23.index(name) for name in ('Mir_avg_amp', 'sxr_mean', 'xuv_ratio', 'ne_nG')]


def shot(seed, rows=256):
    rng = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, 23, dtype=torch.float64, generator=rng)
    z[:, U] = z[:, U] * torch.tensor([.001, .04]) + torch.tensor([.02, -.1])
    slopes = z.new_tensor([[120., -70., 40., 100.], [2., -3., 4., 1.]])
    z[:, C] = z[:, U] @ slopes + .3 * z[:, C] + z.new_tensor([1., -2., .5, 0.])
    return z, torch.ones_like(z, dtype=torch.bool)


def manual_fit(shots):
    """Independent NumPy weighted least-squares and loop-based patch oracle."""
    values, weights = [], []
    for z, mask in shots:
        joint = mask[:, U + C].all(-1).numpy()
        rows = z.numpy()[joint][:, U + C]
        values.append(rows)
        weights.append(np.full(len(rows), 1 / (len(shots) * len(rows))))
    observed, weight = np.concatenate(values), np.concatenate(weights)
    u, y = observed[:, :2], observed[:, 2:]
    center = np.sum(weight[:, None] * u, axis=0)
    centered = u - center
    covariance = centered.T @ (weight[:, None] * centered)
    std = np.sqrt(np.diag(covariance))
    x = np.column_stack((np.ones(len(u)), centered / std))
    gram = x.T @ (weight[:, None] * x)
    regularized = gram + np.diag([0., 1., 1.])
    beta_std = np.linalg.solve(regularized, x.T @ (weight[:, None] * y))
    slopes = beta_std[1:] / std[:, None]
    beta = np.vstack((beta_std[0] - center @ slopes, slopes))
    shot_variances, difference_counts, patch_counts = [], [], []
    for z, mask in shots:
        joint = mask[:, U + C].all(-1).numpy()
        data = z.numpy()
        means = []
        for patch in range(32):
            indices = np.arange(8 * patch, 8 * (patch + 1))
            indices = indices[joint[indices]]
            if len(indices) < 6:
                means.append(None)
            else:
                design = np.column_stack((np.ones(len(indices)), data[indices][:, U]))
                means.append((data[indices][:, C] - design @ beta).mean(0))
        differences = [means[i] - means[i - 1] for i in range(1, 32)
                       if means[i] is not None and means[i - 1] is not None]
        shot_variances.append(np.square(differences).mean(0))
        difference_counts.append(len(differences))
        patch_counts.append(sum(mean is not None for mean in means))
    return dict(beta=beta, scale=np.maximum(np.mean(shot_variances, axis=0), 1e-4),
                covariate_center=center, covariate_scale=std,
                regression_standardized_beta=beta_std,
                eigenvalues=dict(raw_covariance=np.linalg.eigvalsh(covariance),
                                 standardized_covariance=np.linalg.eigvalsh(gram[1:, 1:]),
                                 standardized_design=np.linalg.eigvalsh(gram),
                                 regularized_design=np.linalg.eigvalsh(regularized)),
                differences=difference_counts, patches=patch_counts)


def test_matches_manual_weighted_standardized_ridge_and_raw_residual_scale():
    first, second = shot(11), shot(22)
    first[1][:2, U[0]] = False  # Exactly six joint-valid rows in patch zero.
    second[1][96:, C[1]] = False
    expected = manual_fit([first, second])
    ref = fit_reference_standardized([first, second])
    for key in ('beta', 'scale', 'covariate_center', 'covariate_scale', 'regression_standardized_beta'):
        np.testing.assert_allclose(ref[key].numpy(), expected[key], rtol=2e-11, atol=2e-12)
        assert ref[key].dtype == torch.float64 and ref[key].device.type == 'cpu'
        assert torch.isfinite(ref[key]).all() and not ref[key].requires_grad
    for key, eigenvalues in expected['eigenvalues'].items():
        np.testing.assert_allclose(ref['condition_eigenvalues'][key], eigenvalues, rtol=2e-11, atol=2e-12)
    assert ref['support']['joint_valid_counts'] == [254, 96]
    assert ref['support']['valid_patch_counts'] == expected['patches'] == [32, 12]
    assert ref['support']['valid_difference_counts'] == expected['differences'] == [31, 11]
    assert ref['support']['shot_count'] == 2 and ref['support']['design_rank'] == 3
    assert MIN_COVARIATE_STD == 1e-12


@pytest.mark.parametrize('gain,offset', [([17., .004], [.2, -.03]), ([1e4, .1], [-7., .2])])
def test_positive_affine_covariate_units_preserve_predictions_residuals_scale_and_targets(gain, offset):
    shots = [shot(31), shot(32)]
    shots[1][1][160:, C[0]] = False
    held, held_mask = shot(33, 128)
    held_mask[105:108, U[0]] = False
    gain, offset = held.new_tensor(gain), held.new_tensor(offset)

    def transform(z):
        result = z.clone()
        result[:, U] = result[:, U] * gain + offset
        return result

    ref = fit_reference_standardized(shots)
    changed = fit_reference_standardized([(transform(z), m) for z, m in shots])
    design = torch.cat((torch.ones(128, 1, dtype=held.dtype), held[:, U]), 1)
    transformed_design = torch.cat((torch.ones(128, 1, dtype=held.dtype), transform(held)[:, U]), 1)
    prediction = design @ ref['beta']
    transformed_prediction = transformed_design @ changed['beta']
    torch.testing.assert_close(prediction, transformed_prediction, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(held[:, C] - prediction, held[:, C] - transformed_prediction,
                               rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(ref['scale'], changed['scale'], rtol=1e-10, atol=1e-12)
    for mode in ('real', 'reverse', 'unconditioned'):
        actual = relation_target(held[None], held_mask[None], ref, mode=mode)
        transformed = relation_target(transform(held)[None], held_mask[None], changed, mode=mode)
        for a, b in zip(actual, transformed):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def test_frozen_fit_in_standardized_coordinates_agrees_with_converted_reference():
    shots = [shot(41), shot(42)]
    ref = fit_reference_standardized(shots)
    standardized = []
    for z, mask in shots:
        converted = z.clone()
        converted[:, U] = (z[:, U] - ref['covariate_center']) / ref['covariate_scale']
        standardized.append((converted, mask))
    frozen = fit_reference(standardized)
    torch.testing.assert_close(ref['regression_standardized_beta'], frozen['beta'], rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(ref['scale'], frozen['scale'], rtol=1e-11, atol=1e-12)


def test_equal_shot_weights_are_invariant_to_repeating_valid_rows_within_one_shot():
    first, second = shot(51), shot(52)
    first[1][64:] = False
    second[0][:, U] += second[0].new_tensor([.008, .08])
    repeated = (first[0][:64].repeat(4, 1), first[1][:64].repeat(4, 1))
    sparse = fit_reference_standardized([first, second])
    dense = fit_reference_standardized([repeated, second])
    for key in ('covariate_center', 'covariate_scale', 'beta', 'regression_standardized_beta'):
        torch.testing.assert_close(sparse[key], dense[key], rtol=1e-11, atol=1e-12)
    expected_mean = (first[0][:64, U].mean(0) + second[0][:, U].mean(0)) / 2
    torch.testing.assert_close(sparse['covariate_center'], expected_mean, rtol=1e-12, atol=1e-12)
    pooled_mean = torch.cat((first[0][:64, U], second[0][:, U])).mean(0)
    assert not torch.allclose(sparse['covariate_center'], pooled_mean)


def test_uses_only_supplied_prefixes_without_reading_held_shots(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('reference fitting attempted a record or outcome read')

    monkeypatch.setattr(io23, 'read_signals', forbidden)
    monkeypatch.setattr(io23, 'read_outcome', forbidden)
    history, mask = shot(61, 300)
    ref = fit_reference_standardized([(history[:256], mask[:256])])
    history[256:] = 1e9
    held, _ = shot(62)
    held.fill_(float('nan'))
    again = fit_reference_standardized([(history[:256], mask[:256])])
    for key in ('beta', 'scale', 'covariate_center', 'covariate_scale'):
        torch.testing.assert_close(ref[key], again[key], rtol=0, atol=0)


def test_masked_values_are_ignored_and_inputs_are_unchanged_and_detached():
    z, mask = shot(71)
    mask[::13, U[1]] = False
    mask[::11, C[3]] = False
    ref = fit_reference_standardized([(z, mask)])
    dirty = z.clone()
    dirty[~mask] = float('nan')
    dirty.requires_grad_()
    before, before_mask = dirty.detach().clone(), mask.clone()
    actual = fit_reference_standardized([(dirty, mask)])
    for key in ('beta', 'scale', 'covariate_center', 'covariate_scale', 'regression_standardized_beta'):
        torch.testing.assert_close(ref[key], actual[key], rtol=0, atol=0)
        assert not actual[key].requires_grad
    torch.testing.assert_close(dirty, before, equal_nan=True, rtol=0, atol=0)
    assert torch.equal(mask, before_mask)


@pytest.mark.parametrize('kind', ['constant', 'near_zero', 'collinear'])
def test_degenerate_covariates_fail_before_ridge_can_hide_nonidentifiability(kind):
    z, mask = shot(81)
    if kind == 'constant':
        z[:, U[0]] = 0.
    elif kind == 'near_zero':
        z[:, U[0]] = torch.linspace(-1e-13, 1e-13, 256)
    else:
        z[:, U[1]] = 2 * z[:, U[0]]
    with pytest.raises(ValueError, match='standard deviation|rank'):
        fit_reference_standardized([(z, mask)])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_available_nonfinite_values_fail(value):
    z, mask = shot(91)
    z[0, C[0]] = value
    with pytest.raises(ValueError, match='finite'):
        fit_reference_standardized([(z, mask)])


def test_nonfinite_moments_from_finite_inputs_fail():
    z, mask = shot(92)
    z[:, U[0]] = torch.linspace(-1., 1., 256, dtype=z.dtype) * 1e300
    with pytest.raises(ValueError, match='finite'):
        fit_reference_standardized([(z, mask)])


@pytest.mark.parametrize('kind', ['empty', 'short', 'long', 'mask_dtype', 'few_rows', 'no_pairs'])
def test_reference_requires_exact_prefix_and_support_for_every_shot(kind):
    z, mask = shot(101)
    if kind == 'empty':
        shots = []
    elif kind == 'short':
        shots = [(z[:255], mask[:255])]
    elif kind == 'long':
        shots = [(torch.cat((z, z[:1])), torch.cat((mask, mask[:1])))]
    elif kind == 'mask_dtype':
        shots = [(z, mask.float())]
    else:
        if kind == 'few_rows':
            mask[31:, U[0]] = False
        elif kind == 'no_pairs':
            mask[:, U[0]] = False
            for patch in (0, 2, 4, 6):
                mask[patch * 8:(patch + 1) * 8, U[0]] = True
        shots = [shot(102), (z, mask)]
    with pytest.raises(ValueError):
        fit_reference_standardized(shots)


def test_exact_32_valid_rows_are_allowed_and_constant_residual_scale_has_floor():
    z, mask = shot(111)
    constants = z.new_tensor([2., -1., .5, 4.])
    z[:, C] = constants
    mask[32:] = False
    ref = fit_reference_standardized([(z, mask)])
    torch.testing.assert_close(ref['beta'][0], constants, rtol=0, atol=1e-11)
    torch.testing.assert_close(ref['beta'][1:], torch.zeros_like(ref['beta'][1:]), rtol=0, atol=1e-10)
    torch.testing.assert_close(ref['scale'], torch.full((4,), 1e-4, dtype=z.dtype), rtol=0, atol=0)
    assert ref['support']['joint_valid_counts'] == [32]
    assert ref['support']['valid_difference_counts'] == [3]
