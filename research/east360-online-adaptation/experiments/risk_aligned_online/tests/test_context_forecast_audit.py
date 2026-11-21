import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.forecast_audit import persistence_error


def test_constant_signal_has_zero_persistence_error():
    x = torch.full((7, 23), 3.)
    assert persistence_error(x, torch.ones_like(x, dtype=torch.bool)) == 0.


def test_linear_signal_uses_all_four_horizons():
    x = torch.arange(7.).unsqueeze(1).repeat(1, 23)
    assert persistence_error(x, torch.ones_like(x, dtype=torch.bool)) == 7.5


def test_missing_values_are_not_zero_measurements():
    x = torch.arange(7.).unsqueeze(1).repeat(1, 23)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 1:] = False
    x[:, 1:] = float('nan')
    assert persistence_error(x, mask) == 7.5


@pytest.mark.parametrize('n', [0, 4])
def test_no_complete_forecast_origin_is_rejected(n):
    x = torch.zeros(n, 23)
    with pytest.raises(ValueError, match='full four-horizon'):
        persistence_error(x, torch.ones_like(x, dtype=torch.bool))


def test_no_observations_are_rejected():
    x = torch.zeros(7, 23)
    with pytest.raises(ValueError, match='observed target'):
        persistence_error(x, torch.zeros_like(x, dtype=torch.bool))
