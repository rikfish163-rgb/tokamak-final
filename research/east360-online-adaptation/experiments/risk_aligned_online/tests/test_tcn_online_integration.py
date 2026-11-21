"""Small CPU integration checks for the independent TCN online contract."""

import json
import sys

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import tcn_online
from experiments.risk_aligned_online.forecast_replay import ForecastRetrainer
from experiments.risk_aligned_online.simple_online import predict_shot, run_arm
from experiments.risk_aligned_online.tcn23 import TCN23, TCNConfig


torch.set_num_threads(1)


def make_model():
    torch.manual_seed(131)
    return TCN23(TCNConfig(window=8, dim=4, horizon=25),
                 torch.zeros(23), torch.ones(23))


def _copy(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def _source_shot(shot, *, disruptive):
    rng = np.random.default_rng(1000 + shot)
    values = rng.normal(size=(60, 23)).astype(np.float32)
    times = np.arange(60, dtype=np.float64) / 1000.
    return dict(shot=shot, device='EAST', values=values,
                available=np.ones_like(values, dtype=bool), times=times,
                planning_ip=np.full(60, shot * 100_000., np.float32),
                planning_Bt=np.ones(60, np.float32), disruptive=disruptive)


class PrefixData:
    """Synthetic prefix reader that refuses outcome access before sealing."""

    def __init__(self):
        self.sources = {1: _source_shot(1, disruptive=False),
                        2: _source_shot(2, disruptive=True)}
        self.seal_dir = None
        self.events = []

    def read_shot(self, device, sid):
        assert device == 'EAST'
        self.events.append(('read', sid))
        source = self.sources[sid]
        return {key: _copy(value) for key, value in source.items()
                if key != 'disruptive'}

    def attach_outcome(self, shot, device, advanced):
        assert device == 'EAST'
        assert 'y' not in shot and 'valid' not in shot
        assert self.seal_dir is not None
        seals = list(self.seal_dir.glob(f'*_{shot["shot"]}_seal.json'))
        assert len(seals) == 1, 'outcome was read before exactly one seal'
        self.events.append(('reveal', shot['shot']))
        disruptive = bool(self.sources[shot['shot']]['disruptive'])
        down = .150 if disruptive else None
        valid = np.ones(len(shot['times']), dtype=bool)
        if disruptive:
            y = (shot['times'] >= .025).astype(np.float32)
        else:
            y = np.zeros(len(shot['times']), dtype=np.float32)
        return dict(shot, is_disrupt=disruptive, down=down,
                    valid=valid, y=y, horizon=125.)

    def labeled_shot(self, sid):
        source = self.sources[sid]
        shot = {key: _copy(value) for key, value in source.items()
                if key != 'disruptive'}
        disruptive = bool(source['disruptive'])
        shot.update(
            is_disrupt=disruptive,
            down=.150 if disruptive else None,
            valid=np.ones(len(shot['times']), dtype=bool),
            y=((shot['times'] >= .025).astype(np.float32)
               if disruptive else np.zeros(len(shot['times']), np.float32)),
            horizon=125.,
        )
        return shot

    @staticmethod
    def window(shot, end, length):
        x = np.zeros((length, 23), dtype=np.float32)
        mask = np.zeros((length, 23), dtype=bool)
        start = max(0, end - length + 1)
        count = end - start + 1
        x[-count:] = shot['values'][start:end + 1]
        mask[-count:] = shot['available'][start:end + 1]
        return x, mask


def test_tcn_predict_fast_updates_are_local_and_discarded():
    data = PrefixData()
    model = make_model().eval()
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    shot = data.read_shot('EAST', 2)

    scores, updates = predict_shot(model, shot, intra=True, device='cpu',
                                   data_api=data)

    assert scores.shape == (60,)
    assert updates == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_tcn_forecast_retrainer_plain_forecast_uses_fullscope_on_cpu():
    data = PrefixData()
    model = make_model().eval()
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    updater = ForecastRetrainer(model, supervised_steps=1, forecast_steps=1,
                                batch_size=4, maintenance='forecast')

    result = updater(model, data.labeled_shot(2), device='cpu', data_api=data)

    assert result['trainable_scope'] == 'full'
    assert result['maintenance'] == 'forecast'
    assert result['optimizer_steps'] == 1
    assert len(result['forecast_losses']) == 1
    assert result['forecast_risk_max_delta'] == 0.
    assert any(not torch.equal(value, before[name])
               for name, value in model.state_dict().items()
               if not name.startswith('forecast_head.'))
    assert any(not torch.equal(value, before[name])
               for name, value in model.state_dict().items()
               if name.startswith('forecast_head.'))


def _run_persistent_arm(tmp_path, arm):
    data = PrefixData()
    output = tmp_path / arm
    data.seal_dir = output
    model = make_model().eval()
    updater = ForecastRetrainer(model, supervised_steps=1, forecast_steps=1,
                                batch_size=4, maintenance='forecast')
    order = [dict(shot=2, proxy=2.), dict(shot=1, proxy=1.)]
    result = run_arm(model, order=order, arm=arm, threshold=.37,
                     output_dir=output, data_api=data, device='cpu',
                     startup_budget=0, persistent_updater=updater)
    return model, data, result


def test_inter_and_both_share_persistent_path_and_fixed_threshold(tmp_path):
    inter_model, inter_data, inter = _run_persistent_arm(tmp_path, 'inter')
    both_model, both_data, both = _run_persistent_arm(tmp_path, 'both')

    assert [row['shot'] for row in inter['rows']] == [2, 1]
    assert [row['shot'] for row in both['rows']] == [2, 1]
    assert inter['threshold'] == both['threshold'] == .37
    assert [row['fast_updates'] for row in inter['rows']] == [0, 0]
    assert all(row['fast_updates'] == 2 for row in both['rows'])
    assert inter_data.events == [('read', 2), ('reveal', 2),
                                 ('read', 1), ('reveal', 1)]
    assert both_data.events == inter_data.events

    for left, right in zip(inter['rows'], both['rows']):
        for field in ('sample_sha256', 'forecast_sample_sha256',
                      'classifier_sha256'):
            assert left['persistent'][field] == right['persistent'][field]
    for name, value in inter_model.state_dict().items():
        torch.testing.assert_close(value, both_model.state_dict()[name],
                                   rtol=0, atol=0)
    seal = json.loads((tmp_path / 'both' / '0001_2_seal.json').read_text())
    assert seal['threshold'] == .37


def test_tcn_source_gate_precedes_east_order_and_cuda(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        tcn_online, 'qualify_source',
        lambda path: events.append('source_gate') or
        {'qualified': False, 'reasons': ['synthetic_unqualified']},
    )
    monkeypatch.setattr(
        tcn_online.io23, 'cohort_ids',
        lambda domain: pytest.fail('EAST cohort read before source gate'),
    )
    monkeypatch.setattr(
        tcn_online.torch.cuda, 'is_available',
        lambda: pytest.fail('CUDA queried before source gate'),
    )

    monkeypatch.setattr(sys, 'argv', [
        'tcn_online.py',
        '--checkpoint', str(tmp_path / 'source.pt'),
        '--order', str(tmp_path / 'locked_order.json'),
        '--output', str(tmp_path / 'run'),
    ])
    with pytest.raises(ValueError, match='TCN source failed acceptance'):
        tcn_online.main()

    assert events == ['source_gate']


def test_tcn_locked_order_preserves_input_order_and_rejects_bad_proxy():
    order = [dict(shot=2, proxy=2.), dict(shot=1, proxy=1.)]
    assert tcn_online.validate_order(order, [1, 2]) is order
    with pytest.raises(ValueError, match='finite nonnegative'):
        tcn_online.validate_order(
            [dict(shot=2, proxy=-1.), dict(shot=1, proxy=1.)], [1, 2]
        )
