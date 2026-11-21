from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from run_readout_forecast_20261107 import make_models, fit_step


def fixture():
    torch.manual_seed(41)
    base = Simple23(Config(dim=8, heads=2, layers=1, fast_site='tokens', fast_rule='normalized'),
                    np.zeros(23), np.ones(23)).eval().requires_grad_(False)
    models = make_models(base)
    e = dict(x=torch.randn(2, 128, 23), a=torch.ones(2, 128, 23, dtype=torch.bool),
             truth=torch.randn(2, 25, 23), mask=torch.ones(2, 25, 23, dtype=torch.bool),
             y=torch.ones(2))
    opts = {k: torch.optim.AdamW(models[k].forecast_head.parameters(), lr=3e-4, weight_decay=0.)
            for k in ('cls', 'last_data')}
    return models, opts, e


def test_matched_head_fit_preserves_the_exact_static_risk_path():
    models, opts, e = fixture()
    for key, value in models['cls'].forecast_head.state_dict().items():
        torch.testing.assert_close(value, models['last_data'].forecast_head.state_dict()[key], rtol=0, atol=0)
    original = deepcopy(models['original'].state_dict())
    with torch.no_grad():
        expected = models['original'](e['x'], e['a'])['logit'].clone()
    losses = fit_step(models, opts, e)
    for name in ('cls', 'last_data'):
        assert np.isfinite(losses[name]) and losses[name] > 0
        for key, value in models[name].state_dict().items():
            if not key.startswith('forecast_head.'):
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(models[name](e['x'], e['a'])['logit'], expected, rtol=0, atol=0)


def test_readout_fitting_uses_real_signals_not_the_query_outcome():
    models, opts, e = fixture()
    other, other_opts, _ = fixture()
    fit_step(models, opts, e)
    fit_step(other, other_opts, dict(e, y=1-e['y']))
    for name in ('cls', 'last_data'):
        for key, value in models[name].state_dict().items():
            torch.testing.assert_close(value, other[name].state_dict()[key], rtol=0, atol=0)
