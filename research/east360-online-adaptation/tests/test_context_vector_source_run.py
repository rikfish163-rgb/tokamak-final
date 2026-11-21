import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030 import source_run as run
from experiments.risk_aligned_online.context_vector_20261030.learning import ContextTarget


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('fixed', torch.ones(64))
        self.base = torch.nn.Module()
        self.base.register_buffer('center', torch.zeros(23))
        self.base.register_buffer('scale', torch.ones(23))

    def first(self, x, available, state, mode):
        h = x[:, 0, :1] + state[:, :64] * self.fixed
        return h[:, None].expand(-1, 2, -1), None, None

    def forward(self, x, available, state, slow, mode, forecast=False):
        return {'logit': x[:, 0, 0] + state[:, :64].sum(1)}


def data():
    torch.manual_seed(131)
    return dict(x=torch.randn(12, 128, 23), available=torch.ones(12, 128, 23, dtype=torch.bool),
                context=torch.randn(12, 358), support=torch.ones(12, dtype=torch.bool),
                later_x=torch.randn(12, 128, 23), later_mask=torch.ones(12, 128, 23, dtype=torch.bool))


def test_source_step_changes_only_controller_through_actual_update():
    net = Toy().requires_grad_(False)
    c = ContextTarget()
    before = {k: v.clone() for k, v in c.state_dict().items()}
    original = net.fixed.clone()
    opt = torch.optim.Adam(c.parameters(), lr=.001)
    result = run.train_step(net, c, opt, data(), torch.tensor([0.] * 6 + [1.] * 6))
    assert result['controller_gradient_norm'] > 0
    assert any(not torch.equal(c.state_dict()[k], v) for k, v in before.items())
    assert torch.equal(net.fixed, original)
    assert result['max_state_norm'] == 0  # zero-initialized first step


def test_training_rejects_optimizer_that_updates_another_parameter():
    net, c = Toy(), ContextTarget()
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([*c.parameters(), p])
    with pytest.raises(ValueError, match='controller'):
        run.train_step(net, c, opt, data(), torch.zeros(12))


def test_group_mean_is_over_shots_not_windows_and_no_fake_population():
    records = [{'shot': 1, 'group': 'safe', 'label': 0.}] * 3 + [
        {'shot': 2, 'group': 'safe', 'label': 0.},
        {'shot': 3, 'group': 'danger', 'label': 1.},
        {'shot': 4, 'group': 'early', 'label': 0.}]
    predictions = {'baseline': np.zeros(6)}
    for arm in run.ACTIONS:
        predictions[arm + '_logit'] = np.array([1., 1., 1., -1., 1., 1.])
        predictions[arm + '_norm'] = np.ones(6) * .1
        predictions[arm + '_cos_real'] = np.ones(6)
    result = run.summarize(records, predictions)
    safe = result['groups']['real']['safe']
    expected = ((np.logaddexp(0, 1) - np.log(2)) + (np.logaddexp(0, -1) - np.log(2))) / 2
    assert safe['shots'] == 2 and safe['windows'] == 4
    assert safe['mean_delta_bce'] == pytest.approx(expected)
    assert result['source_screen_passed'] is False
    assert 'insufficient_validation_shots' in result['reasons']


def test_input_gate_denies_test_and_east_without_touching_readers(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('reader reached for forbidden identity')
    monkeypatch.setattr(run.io23, 'read_signals', unexpected)
    monkeypatch.setattr(run.io23, 'read_outcome', unexpected)
    with run.source_gate([1]):
        with pytest.raises(ValueError, match='source role'):
            run.io23.read_signals('EAST', 1)
        with pytest.raises(ValueError, match='source role'):
            run.io23.read_outcome('J-TEXT', 2)


def test_collector_integrates_population_identity_and_supported_windows(tmp_path, monkeypatch):
    shot = dict(shot=1, device='J-TEXT', values=np.ones((200, 23), np.float32),
                available=np.ones((200, 23), bool), times=np.arange(200)/1000.,
                y=np.zeros(200), valid=np.ones(200, bool), is_disrupt=False, down=None)
    monkeypatch.setattr(run.io23, 'path_for', lambda *_: tmp_path/'raw.hdf5')
    monkeypatch.setattr(run, 'sha', lambda *_: 'fixed')
    monkeypatch.setattr(run.simple_data, 'read_shot', lambda *_: shot)
    monkeypatch.setattr(run.simple_data, 'attach_outcome', lambda s, *_: s)
    tensors, records, groups, hashes = run.collect(Toy(), [1], {}, tmp_path, 'toy')
    assert len(records) == groups['safe']['windows'] == 2
    assert groups['safe']['shots'] == 1 and hashes == {'1': 'fixed'}
    assert tensors['context'].shape == (2, 358)
    assert tensors['support'].all()
