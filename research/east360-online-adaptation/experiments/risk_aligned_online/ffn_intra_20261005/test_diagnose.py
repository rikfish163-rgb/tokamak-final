"""CPU contracts for post-hoc source directions; no training or target data."""
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.ffn_intra_20261005.diagnose import diagnostic
from experiments.risk_aligned_online.ffn_intra_20261005.loss import signal_components, weighted_signal_loss


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.linspace(-1., 1., 23, dtype=torch.float64))
        self.register_buffer('scale', torch.linspace(.5, 2., 23, dtype=torch.float64))
        self.register_buffer('forecast', torch.linspace(-.2, .8, 25 * 23).double().reshape(1, 25, 23))
        self.register_buffer('jacobian', torch.linspace(-.5, .7, 25 * 23 * 2).double().reshape(25, 23, 2))
        self.fixed = nn.Parameter(torch.tensor(.2, dtype=torch.float64), requires_grad=False)
        self.eval()

    def initial_delta(self):
        return self.center.new_zeros(2, requires_grad=True)

    def forward(self, x, a, delta):
        return dict(forecast=self.forecast + torch.einsum('hcp,p->hc', self.jacobian, delta)[None],
                    logit=x[:, -1, 0] + self.fixed + 2 * delta[0] - delta[1])


def case():
    model = ToyModel()
    x = model.center.expand(1, 128, -1).clone()
    a = torch.ones_like(x, dtype=torch.bool)
    truth = model.center.expand(1, 25, -1).clone()
    observed = torch.zeros_like(truth, dtype=torch.bool)
    observed[:, :3, 0] = True
    observed[:, [0, 2], 1] = True  # Valid levels but no adjacent difference.
    qx = x.expand(2, -1, -1).clone()
    qx[1, -1, 0] = .4
    return model, x, a, truth, observed, qx, a.expand_as(qx).clone()


@pytest.mark.parametrize('label', [0, 1])
def test_exact_component_normalization_box_bound_and_real_finite_step(label):
    args = case()
    model, x, a, truth, observed, qx, qa = args
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.fixed.grad = torch.tensor(7., dtype=torch.float64)
    result = diagnostic(*args, label, .3, .02, .01)
    delta = model.initial_delta()
    parts = signal_components(model(x, a, delta)['forecast'], truth, observed, model.center, model.scale)
    # Independent decomposition: the shared LEVEL-valid denominator also applies to differences.
    terms = torch.cat((parts['level'], .25 * parts['difference']), 1).flatten() / parts['valid'].sum()
    gradients = torch.stack([torch.autograd.grad(t, delta, retain_graph=True)[0] for t in terms])
    uniform = torch.autograd.grad(weighted_signal_loss(parts).sum(), delta)[0]
    risk = F.binary_cross_entropy_with_logits(model(qx, qa, delta)['logit'].max() - .3,
                                             delta.new_tensor(float(label)))
    r = torch.autograd.grad(risk, delta)[0]
    dots = gradients @ r
    torch.testing.assert_close(torch.tensor(result['dotproducts']), dots.float())
    assert result['uniform_dot'] == pytest.approx(float(r @ uniform), abs=1e-12)
    assert result['boxmax_B'] == pytest.approx(float(torch.where(dots >= 0, 2 * dots, .25 * dots).sum()))
    assert result['boxmax_B'] >= result['uniform_dot'] - 1e-12
    updated = normalized_step(delta[None], torch.zeros_like(delta)[None], uniform[None], step=.02, radius=.01)[0]
    after = F.binary_cross_entropy_with_logits(model(qx, qa, updated)['logit'].max() - .3,
                                              delta.new_tensor(float(label)))
    assert result['risk_before'] == pytest.approx(float(risk.detach()))
    assert result['risk_after_uniform_step'] == pytest.approx(float(after.detach()))
    assert result['risk_change_uniform_step'] == pytest.approx(float((after - risk).detach()))
    assert result['delta_norm'] <= .01000001
    assert result['component_valid'][:2] == [True, True]
    assert result['component_valid'][23:25] == [True, False]
    assert result['persistence']['level_valid'][1]
    assert result['persistence']['decoder_difference'][1] is None
    assert result['decomposition_error_norm'] < 1e-12
    assert result['scope'] == 'SOURCE_POSTHOC_DIRECTION_NOT_ONLINE'
    json.dumps(result, allow_nan=False)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert model.fixed.grad == 7 and not model.training


def test_persistence_uses_last_observed_anchor_and_common_masks():
    args = list(case())
    model, x, a, truth, observed, _, _ = args
    a[:, :, 1] = False  # Mature truth alone must not invent a past anchor.
    x[0, 126, 0] = model.center[0] + model.scale[0] * torch.sinh(torch.tensor(2.))
    a[0, 127, 0], x[0, 127, 0] = False, float('nan')
    truth[0, :3, 0] = model.center[0] + model.scale[0] * torch.sinh(torch.tensor([0., 1., 3.]))
    model.forecast[0, :3, 0] = torch.tensor([0., 2., 5.])
    p = diagnostic(*args, 1, 0., .01, .1)['persistence']
    assert p['anchor_valid'][:2] == [True, False]
    assert p['level_valid'][:2] == [True, False]
    assert p['decoder_level'][0] == pytest.approx(2 / 3, abs=1e-6)
    assert p['persistence_level'][0] == pytest.approx(5 / 6, abs=1e-6)
    assert p['decoder_difference'][0] == pytest.approx(.5, abs=1e-6)
    assert p['persistence_difference'][0] == pytest.approx(1., abs=1e-6)
    assert p['decoder_level'][1] is None and p['persistence_level'][1] is None
    assert p['decoder_difference'][1] is None


def test_all_missing_and_poisoned_masked_values_are_json_safe_noops():
    args = list(case())
    args[4].zero_()
    args[3].fill_(float('nan'))
    with torch.no_grad():
        result = diagnostic(*args, 1, 0., .01, .1)
    assert not any(result['component_valid'])
    assert result['uniform_dot'] == result['boxmax_B'] == result['delta_norm'] == 0
    assert result['risk_change_uniform_step'] == 0
    assert result['persistence']['persistence_level'] == [None] * 23
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('disconnected', ['forecast', 'logit'])
def test_disconnected_readout_is_reported_as_zero_direction(disconnected):
    args = case()
    model = args[0]
    original = model.forward

    def forward(x, a, delta):
        output = original(x, a, delta)
        output[disconnected] = original(x, a, torch.zeros_like(delta))[disconnected].detach()
        if disconnected == 'logit':
            output['logit'].requires_grad_()  # A graph exists but does not reach delta.
        return output

    model.forward = forward
    result = diagnostic(*args, 1, 0., .01, .1)
    assert result['boxmax_B'] == result['uniform_dot'] == result['risk_change_uniform_step'] == 0
    assert result['uniform_gradient_norm' if disconnected == 'forecast' else 'risk_gradient_norm'] == 0


@pytest.mark.parametrize('label,boundary,step,radius', [(2, 0, .1, 1), (1, float('nan'), .1, 1),
                                                     (1, 0, 0, 1), (1, 0, .1, -1)])
def test_invalid_scalar_contract_rejected(label, boundary, step, radius):
    with pytest.raises(ValueError):
        diagnostic(*case(), label, boundary, step, radius)


def test_empty_query_and_train_mode_rejected():
    args = list(case())
    with pytest.raises(ValueError):
        diagnostic(*args[:5], args[5][:0], args[6][:0], 1, 0., .01, .1)
    args[0].train()
    with pytest.raises(ValueError):
        diagnostic(*args, 1, 0., .01, .1)


def test_real_ffn_cpu_preserves_all_registered_state():
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.ffn_intra_20261005.model import FfnModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        model = FfnModel(Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23)))
        x, truth = torch.randn(1, 128, 23), torch.randn(1, 25, 23)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = diagnostic(model, x, torch.ones_like(x, dtype=torch.bool), truth,
                        torch.ones_like(truth, dtype=torch.bool), x * .9,
                        torch.ones_like(x, dtype=torch.bool), 1, .4, .01, .1)
    assert len(result['dotproducts']) == 46 and result['risk_gradient_norm'] > 0
    assert result['decomposition_error_norm'] < 1e-5
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    json.dumps(result, allow_nan=False)
