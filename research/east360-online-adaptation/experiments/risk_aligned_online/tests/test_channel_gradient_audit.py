import pytest
import torch

from experiments.risk_aligned_online.channel_gradient_audit import (
    channel_losses, component_gradients, measure_event,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_channel_sum_matches_original_global_masked_mean_and_gradient():
    torch.manual_seed(4)
    base = Simple23(Config(), torch.zeros(23), torch.ones(23))
    prediction = torch.randn(2, 25, 23, requires_grad=True)
    truth = torch.randn_like(prediction)
    observed = torch.rand_like(prediction) > .3
    truth[0, 0, 0] = float('nan')
    pieces = channel_losses(prediction, truth, observed, base.center, base.scale)
    total = base.forecast_loss(prediction, truth, observed)
    torch.testing.assert_close(pieces.sum(), total)
    g1, = torch.autograd.grad(pieces.sum(), prediction, retain_graph=True)
    g2, = torch.autograd.grad(total, prediction)
    torch.testing.assert_close(g1, g2)
    assert torch.isfinite(g1).all()


def test_unobserved_channels_have_zero_components_and_no_gradient():
    parameter = torch.randn(5, requires_grad=True)
    prediction = parameter.sum().expand(1, 25, 23)
    truth = torch.ones_like(prediction)
    observed = torch.zeros_like(prediction, dtype=torch.bool)
    observed[:, :, 2] = True
    pieces = channel_losses(prediction, truth, observed, torch.zeros(23), torch.ones(23))
    gradients = component_gradients(pieces, parameter)
    assert gradients.shape == (23, 5)
    assert torch.isfinite(gradients).all()
    assert gradients[2].norm() > 0
    gradients[2] = 0
    assert not gradients.any()


def test_rejects_nonfinite_prediction_on_an_observed_channel():
    prediction = torch.ones(1, 25, 23, requires_grad=True)
    prediction = prediction.clone()
    prediction[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        channel_losses(prediction, torch.ones_like(prediction),
                       torch.ones_like(prediction, dtype=torch.bool), torch.zeros(23), torch.ones(23))


def test_probe_label_changes_explanation_not_actual_signal_update_or_model():
    torch.manual_seed(13)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), torch.zeros(23), torch.ones(23)).eval()
    model = PersistentFfnModel(base).eval()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    truth = torch.randn(1, 25, 23)
    mask = torch.ones_like(truth, dtype=torch.bool)
    kwargs = dict(model=model, x=x, a=a, truth=truth, observed=mask, qx=x*.8, qa=a,
                  fast=model.initial_fast(1), step=.05, radius=1.)
    zero = measure_event(**kwargs, label=0)
    one = measure_event(**kwargs, label=1)
    assert zero['actual']['probability_after'] == one['actual']['probability_after']
    assert zero['actual']['candidate_sha256'] == one['actual']['candidate_sha256']
    assert zero['actual']['risk_change'] * one['actual']['risk_change'] < 0
    assert zero['gradient_decomposition_verified'] and one['gradient_decomposition_verified']
    assert zero['offline_oracle'] and one['offline_oracle']
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, snapshot[k], rtol=0, atol=0)
