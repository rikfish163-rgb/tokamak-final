from pathlib import Path

import pytest
import torch
from torch import nn

torch.set_num_threads(2)

from experiments.risk_aligned_online import io23  # noqa: E402
from experiments.risk_aligned_online.tcn23 import (  # noqa: E402
    TCN23,
    TCNConfig,
    fit_forecaster,
    load_checkpoint,
)


def make_model():
    torch.manual_seed(131)
    return TCN23(TCNConfig(window=128, dim=16, horizon=5),
                 torch.zeros(23), torch.ones(23))


def make_inputs(batch=2):
    torch.manual_seed(132)
    values = torch.randn(batch, 128, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    return values, available


def test_tcn_defaults_match_screening_contract_and_use_channel_layer_norm():
    cfg = TCNConfig()
    assert cfg.window == 128
    assert cfg.dim == 64
    assert cfg.horizon == 25
    assert cfg.fast_site == 'tokens'
    assert cfg.fast_rule == 'normalized'

    model = TCN23(cfg, torch.zeros(23), torch.ones(23))
    assert model.fast_origin.shape == (128,)
    assert len(model.blocks) == 5
    assert [block.dilation for block in model.blocks] == [1, 2, 4, 8, 16]
    assert model.receptive_field == 125
    assert any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert not any(isinstance(module, nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())


def test_forward_returns_scalar_risk_and_direct_multistep_forecast():
    model = make_model()
    values, available = make_inputs()
    output = model(values, available)
    assert output['logit'].shape == (2,)
    assert output['forecast'].shape == (2, 5, 23)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_causal_tcn_future_cannot_change_past_hidden_states():
    model = make_model().eval()
    values, available = make_inputs(batch=1)
    changed = values.clone()
    changed[:, 100:] += 7.

    before = model.encode(values, available)
    after = model.encode(changed, available)
    torch.testing.assert_close(before[:, :100], after[:, :100], rtol=0, atol=0)


def test_missing_numeric_values_do_not_change_risk_or_forecast():
    model = make_model().eval()
    values, available = make_inputs()
    available[:, :, 4] = False
    changed = values.clone()
    changed[:, :, 4] = float('nan')

    before = model(values, available)
    after = model(changed, available)
    for name in ('logit', 'forecast'):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_fast_changes_both_tasks_and_has_nonzero_task_gradients():
    model = make_model()
    values, available = make_inputs()
    fast = model.initial_fast(2).detach().requires_grad_(True)
    output = model(values, available, fast)
    risk_grad, = torch.autograd.grad(output['logit'].sum(), fast, retain_graph=True)
    forecast_grad, = torch.autograd.grad(output['forecast'].square().mean(), fast)

    assert torch.isfinite(risk_grad).all() and risk_grad.abs().sum() > 0
    assert torch.isfinite(forecast_grad).all() and forecast_grad.abs().sum() > 0
    changed = model(values, available, fast + .4)
    assert not torch.equal(output['logit'], changed['logit'])
    assert not torch.equal(output['forecast'], changed['forecast'])


def test_fast_path_supports_double_backward():
    model = make_model()
    values, available = make_inputs()
    fast = model.initial_fast(2).detach().requires_grad_(True)
    objective = model(values, available, fast)['forecast'].square().mean()
    first, = torch.autograd.grad(objective, fast, create_graph=True)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    second = first.square().mean()
    second.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_forecast_only_fitting_cannot_change_classifier():
    model = make_model().eval()
    values, available = make_inputs()
    truth = torch.randn(2, 5, 23)
    target_mask = torch.ones_like(truth, dtype=torch.bool)
    before = model(values, available)['logit'].detach().clone()
    optimizer = torch.optim.Adam(model.forecast_head.parameters(), lr=.01)

    loss = fit_forecaster(model, optimizer, values, available, truth, target_mask)

    assert loss >= 0 and torch.isfinite(torch.tensor(loss))
    torch.testing.assert_close(before, model(values, available)['logit'], rtol=0, atol=0)


def test_load_checkpoint_requires_tcn23_format(tmp_path):
    model = make_model()
    checkpoint = {
        'format': 'tcn23-v1',
        'config': {'format': 'tcn23-v1', 'feature_names': list(io23.RAW23)},
        'model': {key: value.detach().clone() for key, value in model.state_dict().items()},
        'model_config': model.cfg.to_dict(),
        'source_center': model.center.detach().clone(),
        'source_scale': model.scale.detach().clone(),
        'threshold': .5,
        'validation': {},
        'forecast_trained': False,
    }
    path = tmp_path / 'source.pt'
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path, 'cpu')
    assert isinstance(loaded, TCN23)
    assert loaded.cfg == model.cfg
    assert loaded.threshold == .5
    assert loaded.training is False

    checkpoint['format'] = 'simple23-v1'
    bad = tmp_path / 'bad.pt'
    torch.save(checkpoint, bad)
    with pytest.raises(ValueError, match='tcn23-v1'):
        load_checkpoint(bad, 'cpu')

    checkpoint['format'] = 'tcn23-v1'
    checkpoint['config']['feature_names'] = list(reversed(io23.RAW23))
    bad_names = tmp_path / 'bad-names.pt'
    torch.save(checkpoint, bad_names)
    with pytest.raises(ValueError, match='feature_names order'):
        load_checkpoint(bad_names, 'cpu')

    checkpoint['config']['feature_names'] = list(io23.RAW23)
    checkpoint['threshold'] = 1.01
    bad_threshold = tmp_path / 'bad-threshold.pt'
    torch.save(checkpoint, bad_threshold)
    with pytest.raises(ValueError, match='0 < threshold <= 1'):
        load_checkpoint(bad_threshold, 'cpu')


def test_training_entry_keeps_shared_helpers_and_screening_defaults():
    from experiments.risk_aligned_online import tcn_train

    parser = tcn_train.build_parser()
    args = parser.parse_args(['--output', '/tmp/tcn23-test-output'])
    assert (args.epochs, args.steps, args.batch_size) == (12, 128, 64)
    assert args.lr == 3e-4
    assert args.forecast_steps == 256

    source = Path(tcn_train.__file__).read_text()
    assert 'Simple23' not in source
    for helper in ('simple_train.batch', 'simple_train.pools',
                   'simple_train.score', 'simple_train.select_threshold',
                   'simple_train.metrics', 'simple_data'):
        assert helper in source
