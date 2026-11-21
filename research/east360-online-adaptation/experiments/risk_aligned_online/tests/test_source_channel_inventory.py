import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.source_channel_inventory import record


def test_short_shot_is_explicitly_ineligible_not_zero_prediction_error():
    result = record(torch.ones(4, 23), torch.ones(4, 23, dtype=torch.bool))
    assert result['status'] == 'too_short_for_complete_four_horizon_loss'
    assert result['mse'] is None
    assert result['training_loss_value'] == 0.
    assert sum(result['count']) == 0


def test_unobserved_targets_explicitly_accounted_for():
    result = record(torch.ones(8, 23), torch.zeros(8, 23, dtype=torch.bool))
    assert result['status'] == 'no_observed_four_horizon_pairs'
    assert result['mse'] is None


def test_observed_source_persistence_loss():
    x = torch.arange(7.).unsqueeze(1).repeat(1, 23)
    result = record(x, torch.ones_like(x, dtype=torch.bool))
    assert result['status'] == 'eligible'
    assert result['mse'] == 7.5
    assert result['training_loss_value'] == 7.5
    assert result['count'] == [12] * 23


def test_bad_shapes_do_not_become_short_shot_exclusions():
    with pytest.raises(ValueError):
        record(torch.ones(3, 20), torch.ones(3, 20, dtype=torch.bool))
