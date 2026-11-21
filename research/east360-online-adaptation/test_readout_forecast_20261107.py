import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online.fast_meta import temporary_step
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.fixture
def token_forecast():
    from readout_forecast_20261107 import TokenForecast

    return TokenForecast


@pytest.mark.parametrize(('site', 'index'), [('cls', -1), ('last_data', -2)])
def test_only_the_requested_token_is_projected(token_forecast, site, index):
    dim, horizon = 4, 2
    head = token_forecast(dim, horizon, site)
    with torch.no_grad():
        head.projection.weight.copy_(
            1 + torch.arange(horizon * 23 * dim).reshape(horizon * 23, dim) / 10
        )
        head.projection.bias.copy_(torch.arange(horizon * 23) / 10)
    tokens = torch.arange(2 * 17 * dim, dtype=torch.float32).reshape(2, 17, dim)
    tokens.requires_grad_()

    result = head(tokens.flatten(1))

    assert result.shape == (2, horizon * 23)
    torch.testing.assert_close(
        result, F.linear(tokens[:, index], head.projection.weight, head.projection.bias),
        rtol=0, atol=0,
    )
    gradient, = torch.autograd.grad(result.sum(), tokens)
    assert torch.all(gradient[:, index] != 0)
    other = gradient.clone()
    other[:, index] = 0
    assert torch.count_nonzero(other) == 0


def test_sites_have_identical_projection_parameterization(token_forecast):
    dim, horizon = 8, 25
    cls = token_forecast(dim, horizon, 'cls')
    last_data = token_forecast(dim, horizon, 'last_data')
    last_data.projection.load_state_dict(cls.projection.state_dict())

    assert sum(p.numel() for p in cls.parameters()) == (dim + 1) * horizon * 23
    assert sum(p.numel() for p in last_data.parameters()) == (dim + 1) * horizon * 23
    for name, value in cls.projection.state_dict().items():
        assert torch.equal(value, last_data.projection.state_dict()[name])


@pytest.mark.parametrize('site', ['', 'CLS', 'last', None])
def test_invalid_site_is_rejected(token_forecast, site):
    with pytest.raises(ValueError):
        token_forecast(4, 2, site)


@pytest.mark.parametrize('shape', [(68,), (2, 17, 4), (2, 64), (2, 69)])
def test_only_flattened_seventeen_token_input_is_accepted(token_forecast, shape):
    head = token_forecast(4, 2, 'cls')
    with pytest.raises(ValueError):
        head(torch.zeros(shape))


def make_model(fast_site, dim=8, fast_rule='normalized'):
    torch.manual_seed(317)
    cfg = Config(dim=dim, heads=2, layers=1, horizon=25,
                 fast_site=fast_site, fast_rule=fast_rule)
    model = Simple23(cfg, torch.linspace(-1, 1, 23), torch.linspace(0.5, 2, 23))
    with torch.no_grad():
        model.fast_origin.copy_(torch.linspace(-0.25, 0.25, 2 * dim))
    return model.eval()


@pytest.mark.parametrize('site', ['cls', 'last_data'])
@pytest.mark.parametrize('fast_site', ['output', 'tokens'])
def test_simple23_replacement_preserves_risk_and_persistent_state(
        token_forecast, site, fast_site):
    model = make_model(fast_site)
    values = torch.randn(2, model.cfg.window, 23)
    available = torch.ones_like(values, dtype=torch.bool)
    available[:, :7, 2] = False
    initial = model.initial_fast(2)
    shifted = initial + torch.linspace(-0.1, 0.1, initial.shape[1])
    unchanged = {name: value.detach().clone() for name, value in model.state_dict().items()
                 if not name.startswith('forecast_head.')}
    with torch.no_grad():
        before = [model(values, available, fast)['logit'].clone()
                  for fast in (initial, shifted)]

    model.forecast_head = token_forecast(model.cfg.dim, model.cfg.horizon, site)

    with torch.no_grad():
        after = [model(values, available, fast) for fast in (initial, shifted)]
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual['logit'])
        assert actual['forecast'].shape == (2, 25, 23)
    current = {name: value for name, value in model.state_dict().items()
               if not name.startswith('forecast_head.')}
    assert current.keys() == unchanged.keys()
    assert all(torch.equal(current[name], value) for name, value in unchanged.items())


@pytest.mark.parametrize('site', ['cls', 'last_data'])
@pytest.mark.parametrize('fast_site', ['output', 'tokens'])
@pytest.mark.parametrize('fast_rule', ['raw', 'normalized'])
def test_real_temporary_step_receives_finite_nonzero_fast_gradient(
        token_forecast, site, fast_site, fast_rule):
    model = make_model(fast_site, dim=64, fast_rule=fast_rule)
    model.forecast_head = token_forecast(model.cfg.dim, model.cfg.horizon, site)
    model.requires_grad_(False)
    values = torch.randn(2, model.cfg.window, 23)
    truth = torch.randn(2, model.cfg.horizon, 23)
    episode = dict(x=values, a=torch.ones_like(values, dtype=torch.bool),
                   truth=truth, mask=torch.ones_like(truth, dtype=torch.bool))
    unchanged = {name: value.detach().clone() for name, value in model.state_dict().items()}
    seen_fast = []
    handle = model.register_forward_pre_hook(lambda module, args: seen_fast.append(args[2]))
    try:
        updated, signal_loss = temporary_step(model, episode, create_graph=True)
    finally:
        handle.remove()

    assert len(seen_fast) == 1
    fast = seen_fast[0]
    assert fast.shape == updated.shape == (2, 128)
    gradient, = torch.autograd.grad(signal_loss, fast)
    assert torch.isfinite(signal_loss)
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.norm(dim=1) > 0)
    assert torch.isfinite(updated).all()
    assert torch.all((updated - fast).norm(dim=1) > 0)
    assert all(torch.equal(model.state_dict()[name], value)
               for name, value in unchanged.items())
