"""Small analytic fixtures; no checkpoint loading, model fitting, or file writes."""

import math

import pytest
import torch

from experiments.risk_aligned_online.operational_transport_20261030.diagnose_fit import (
    balanced_weights,
    half_roll_indices,
    mapping_metrics,
    matrix_geometry,
    row_cosine,
    spectrum,
    summarize,
)


def test_cosine_distinguishes_opposite_orthogonal_and_undefined_zero():
    a = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., 0.]])
    b = torch.tensor([[3., 0.]]).expand_as(a)
    cosine = row_cosine(a, b)
    torch.testing.assert_close(cosine[:3], torch.tensor([1., -1., 0.], dtype=torch.float64))
    assert torch.isnan(cosine[3])
    result = summarize(cosine, torch.ones(4))
    assert (result['positive'], result['negative'], result['zero'], result['undefined']) == (1, 1, 1, 1)
    assert result['weighted_mean'] == pytest.approx(0.)
    assert summarize(torch.full((2,), float('nan')), torch.ones(2))['weighted_mean'] is None


def test_half_roll_stays_inside_shots_and_preserves_balanced_mean():
    records = [dict(shot=s, disruptive=(s == 1)) for s in (1, 2, 1, 2, 1)]
    order = half_roll_indices(records)
    assert order.tolist() == [4, 3, 0, 1, 2]
    weights = balanced_weights(records)
    torch.testing.assert_close(weights, torch.tensor([1/6, 1/4, 1/6, 1/4, 1/6], dtype=torch.float64))
    target = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 0.]])
    assert not torch.equal(target, target[order])
    torch.testing.assert_close((weights[:, None]*target).sum(0), (weights[:, None]*target[order]).sum(0))


def test_conflicting_class_labels_within_one_shot_are_rejected():
    with pytest.raises(ValueError, match='shot'):
        balanced_weights([dict(shot=1, disruptive=True), dict(shot=1, disruptive=False)])


def test_mapping_deviation_ratios_use_target_mean_and_named_denominators():
    mapped = torch.tensor([[2., 0.], [1., 1.]])
    target = torch.eye(2)
    mean = torch.tensor([1., 0.])
    result = mapping_metrics(mapped, target, mean, torch.tensor([.25, .75]))
    assert result['target_cosine']['weighted_mean'] == pytest.approx(.25+.75/math.sqrt(2))
    assert result['constant_target_cosine']['weighted_mean'] == pytest.approx(.25)
    assert result['cosine_gain_over_constant']['weighted_mean'] == pytest.approx(.75/math.sqrt(2))
    assert result['deviation_over_mapped_norm']['weighted_mean'] == pytest.approx(.25/2+.75/math.sqrt(2))
    assert result['deviation_over_constant_norm']['weighted_mean'] == pytest.approx(1.)
    assert result['deviation_energy_over_mapped_energy'] == pytest.approx(.4)


def test_gram_spectrum_has_hand_computable_effective_ranks():
    result = spectrum(torch.diag(torch.tensor([2., .5, 0.], dtype=torch.float64)), feature_dim=2)
    assert result['numerical_gram_rank'] == 2
    assert result['energy'] == pytest.approx(2.5)
    assert result['stable_rank'] == pytest.approx(1.25)
    assert result['participation_rank'] == pytest.approx(1/.68)
    assert result['entropy_rank'] == pytest.approx(math.exp(-.8*math.log(.8)-.2*math.log(.2)))
    assert result['components_90pct'] == result['components_99pct'] == 2
    zero = spectrum(torch.zeros((2, 2), dtype=torch.float64), feature_dim=2)
    assert zero['numerical_gram_rank'] == zero['participation_rank'] == zero['entropy_rank'] == 0


def test_between_and_within_shot_energy_decomposition_is_weighted():
    matrix = torch.tensor([[1., 0.], [3., 0.], [0., 2.], [0., 4.]])
    weights = torch.tensor([.125, .125, .375, .375])
    result = matrix_geometry(matrix, weights, [1, 1, 2, 2])
    assert result['uncentered']['energy'] == pytest.approx(8.75)
    assert result['centered']['energy'] == pytest.approx(3.4375)
    assert result['between_shots']['energy'] == pytest.approx(2.4375)
    assert result['within_shots']['energy'] == pytest.approx(1.)
    assert result['between_shots']['numerical_gram_rank'] == 1
    assert result['between_fraction_of_centered_energy'] == pytest.approx(2.4375/3.4375)
    assert abs(result['decomposition_error']) < 1e-12
