import numpy as np
import pytest
import torch
import subprocess
import sys

from experiments.risk_aligned_online.gradient_transport import fit_transport, transport_gradient, TransportFfnModel
from experiments.risk_aligned_online.persistent_ffn import PersistentFfnModel
from experiments.risk_aligned_online.simple23 import Simple23, Config
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_ridge_mapping_can_learn_a_direction_rotation_not_only_reweighting():
    signal = torch.eye(2)
    risk = torch.tensor([[1., 1.], [0., 1.]])
    keys, coefficients = fit_transport(signal, risk, ridge=.1)
    gradient = torch.tensor([[1., 0.]])
    mapped = transport_gradient(gradient, keys, coefficients)
    assert mapped[0, 0] > 0 and mapped[0, 1] > 0
    assert torch.dot(mapped[0], gradient[0]) >= 0


def test_opposing_mapped_component_is_removed_without_identity_substitution():
    gradient = torch.tensor([[1., 0.]])
    mapped = transport_gradient(gradient, torch.eye(2), torch.tensor([[-1., 2.], [0., 0.]]))
    torch.testing.assert_close(mapped, torch.tensor([[0., 2.]]), rtol=0, atol=0)
    zero = transport_gradient(gradient, torch.eye(2), -torch.eye(2))
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)


@pytest.mark.parametrize('case', ('nonfinite', 'zero', 'shape', 'ridge'))
def test_bad_fit_cannot_silently_produce_a_default_rule(case):
    signal, risk, ridge = torch.eye(2), torch.eye(2), .1
    if case == 'nonfinite': signal[0, 0] = float('nan')
    if case == 'zero': signal[0] = 0
    if case == 'shape': risk = torch.ones(3, 2)
    if case == 'ridge': ridge = 0
    with pytest.raises(ValueError): fit_transport(signal, risk, ridge=ridge)


def models():
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'), torch.zeros(23), torch.ones(23)).eval()
    plain = PersistentFfnModel(base)
    n = plain.temporary_dim
    keys = torch.zeros(2, n); keys[0, 0] = 1.; keys[1, 1] = 1.
    return plain, TransportFfnModel(base, keys, keys.clone())


def test_transform_changes_only_update_rule_not_zero_state_risk_or_forecast():
    plain, mapped = models()
    x = torch.randn(2, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad(): before, after = plain(x, a), mapped(x, a)
    for key in before: torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert {id(p) for p in mapped.parameters()} == {id(p) for p in plain.base.parameters()} | {id(mapped.memory)}


def test_transport_online_is_causal_and_does_not_retain_fast_state():
    _, model = models()
    x = np.linspace(1, 2, 61, dtype=np.float32)[:, None].repeat(23, axis=1)
    shot = dict(device='EAST', values=x, available=np.ones_like(x, bool), times=np.arange(61)/1000.)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    first, audit = predict_shot(model, shot, intra=True)
    changed = dict(shot, values=x.copy(), is_disrupt=True, down=.061)
    changed['values'][40:] *= 10
    second, _ = predict_shot(model, changed, intra=True)
    np.testing.assert_array_equal(first[:40], second[:40])
    assert audit['fast_gradient_transform'] == 'source TRAIN ridge gradient transport'
    assert audit['attempts'][0]['maturity_index'] == 25
    assert audit['fast_discarded']
    for k, v in model.state_dict().items(): torch.testing.assert_close(v, initial[k], rtol=0, atol=0)


def test_density_runner_exposes_operator_and_rejects_missing_operator_before_cuda(tmp_path):
    cmd = [sys.executable, '-m', 'experiments.risk_aligned_online.density_online']
    help_result = subprocess.run([*cmd, '--help'], capture_output=True, text=True)
    assert '--gradient-operator' in help_result.stdout
    result = subprocess.run([*cmd, '--internal-site', 'gradient_transport', '--output', str(tmp_path/'new')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'gradient_transport requires' in result.stderr
    assert not (tmp_path/'new').exists()
