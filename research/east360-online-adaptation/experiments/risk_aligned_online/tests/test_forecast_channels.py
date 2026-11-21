import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.forecast_channels import channel_errors, summarize


def fixture():
    x = torch.arange(7.).unsqueeze(1).repeat(1, 23)
    return x, torch.ones_like(x, dtype=torch.bool), torch.zeros(7, 4, 23)


def test_channel_sums_reproduce_complete_four_horizon_mse():
    x, mask, pred = fixture()
    e = channel_errors(x, mask, pred)
    assert torch.equal(e['count'], torch.full((23,), 12))
    assert torch.allclose(e['sse'], torch.full((23,), 90., dtype=torch.float64))
    assert e['mse'] == 7.5
    assert e['persistence_mse'] == 7.5
    pred[:] = torch.arange(1., 5.).view(1, 4, 1)
    assert channel_errors(x, mask, pred)['mse'] == 0.


def test_unobserved_channels_are_absent_not_perfect_predictions():
    x, mask, pred = fixture()
    mask[:, 1:] = False
    x[:, 1:] = float('nan')
    pred[:, :, 1:] = float('nan')
    e = channel_errors(x, mask, pred)
    assert e['mse'] == 7.5
    assert torch.equal(e['count'][1:], torch.zeros(22, dtype=torch.long))
    assert e['sse'][1:].sum() == 0.


def test_mask_requires_observation_at_origin_and_target():
    x, mask, pred = fixture()
    mask[0, 0] = False
    e = channel_errors(x, mask, pred)
    assert e['count'][0] == 8
    assert e['sse'][0] == 60.


@pytest.mark.parametrize('failure', ['shape', 'none', 'observed_nan'])
def test_invalid_inputs_rejected(failure):
    x, mask, pred = fixture()
    if failure == 'shape':
        pred = pred[:, :3]
    elif failure == 'none':
        mask[:] = False
    else:
        pred[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        channel_errors(x, mask, pred)


def test_summary_conserves_equal_shot_improvement_with_missing_channels():
    rows = []
    for factor in (0., .5):
        x, mask, pred = fixture()
        mask[:, 1:] = False
        pred[:] = factor * torch.arange(1., 5.).view(1, 4, 1)
        errors = channel_errors(x, mask, pred)
        rows.append({k: v.tolist() if torch.is_tensor(v) else v for k, v in errors.items()})
    result = summarize(rows)
    assert result['mean_improvement'] == 2.8125
    assert sum(c['mean_contribution_to_overall_improvement'] for c in result['channels']) == 2.8125
    assert result['channels'][1]['observed_shots'] == 0
    assert result['channels'][1]['memory_mse'] is None
    assert result['better_shots'] == 1


def test_no_net_improvement_has_no_top_five_fraction():
    x, mask, pred = fixture()
    errors = channel_errors(x, mask, pred)
    row = {k: v.tolist() if torch.is_tensor(v) else v for k, v in errors.items()}
    assert summarize([row])['top5_share_of_net_improvement'] is None
