import pytest

from experiments.risk_aligned_online.live_history_20261023.accounting import summarize_updates


def test_zero_parameter_change_is_not_learning_even_if_loss_rounding_decreases():
    rows = [
        dict(gradient_steps=0, parameter_delta_norm=0., loss_before=1., loss_after=1.-1e-7),
        dict(gradient_steps=1, parameter_delta_norm=1e-4, loss_before=1., loss_after=.99),
        dict(gradient_steps=0, parameter_delta_norm=0., loss_before=1., loss_after=1.+1e-7),
    ]
    result = summarize_updates(rows)
    assert result['raw_forward_decreases'] == 2
    assert result['nonzero_updates'] == result['nonzero_update_decreases'] == 1
    assert result['zero_updates'] == 2
    assert result['zero_update_nonzero_loss_difference'] == 2
    assert result['zero_update_max_abs_loss_difference'] == pytest.approx(1e-7)


def test_update_accounting_requires_consistent_and_finite_receipts():
    row = dict(gradient_steps=0, parameter_delta_norm=0., loss_before=1., loss_after=1.)
    for change in ({'gradient_steps': 1}, {'loss_after': float('nan')}, {'parameter_delta_norm': -1.}):
        with pytest.raises(ValueError):
            summarize_updates([dict(row, **change)])
