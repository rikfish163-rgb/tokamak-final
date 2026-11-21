"""Synthetic, CPU-only tests for the single-head online replay contract."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import simple_online as online
from experiments.risk_aligned_online.simple23 import Config, Simple23


torch.set_num_threads(1)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=4, horizon=25,fast_rule='raw')
        self.fast_origin = torch.nn.Parameter(torch.zeros(2))

    def initial_fast(self, batch):
        return self.fast_origin.unsqueeze(0).expand(batch, -1).clone()

    def forward(self, x, mask, fast=None):
        fast = self.initial_fast(len(x)) if fast is None else fast
        logit = fast[:, 0] + x[:, -1, 0]
        forecast = fast[:, :1, None].expand(-1, 25, 23)
        return {'logit': logit, 'forecast': forecast}

    def forecast_loss(self, prediction, truth, available):
        return ((prediction - truth.detach()).square() * available).sum() / available.sum().clamp_min(1)


class FakeData:
    def __init__(self, shots):
        self.shots = shots
        self.events = []
        self.output = None

    def read_shot(self, device, sid):
        self.events.append(('read', sid))
        return dict(self.shots[sid])

    def attach_outcome(self, shot, device, advanced):
        sid = shot['shot']
        arm_index = sum(kind == 'reveal' for kind, _ in self.events)
        arm_dir = self.output / online.ARMS[arm_index] if (self.output / 'static').exists() else self.output
        seal = arm_dir / f'0001_{sid}_seal.json'
        assert seal.exists(), 'outcome was read before durable prediction seal'
        info = json.loads(seal.read_text())
        raw = (seal.parent / info['npz']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        self.events.append(('reveal', sid))
        down = .030 if sid == 1 else None
        valid = shot['times'] <= down if down is not None else np.ones(len(shot['times']), bool)
        return dict(shot, is_disrupt=down is not None, down=down,
                    valid=valid, y=(valid & (shot['times'] >= .005)).astype(np.float32))

    @staticmethod
    def window(s, end, length):
        x = np.zeros((length, 23), np.float32)
        mask = np.zeros_like(x, bool)
        lo = max(0, end-length+1)
        count = end-lo+1
        x[-count:] = s['values'][lo:end+1]
        mask[-count:] = s['available'][lo:end+1]
        return x, mask


def shot(sid, *, n=54, late_signal=False):
    values = np.ones((n, 23), np.float32)
    if late_signal:
        values[:31, 0] = -10.
        values[31:, 0] = 10.
    else:
        values[:, 0] = 0.
    return dict(shot=sid, device='EAST', values=values,
                available=np.ones_like(values, bool), times=np.arange(n)/1000.,
                planning_ip=np.full(n, float(sid)*100_000, np.float32),
                planning_Bt=np.ones(n, np.float32))


def test_intra_update_matures_after_25_and_only_affects_next_prediction():
    data = FakeData({1: shot(1)})
    model = Toy()
    scores, updates = online.predict_shot(model, data.shots[1], intra=True,
                                          device='cpu', data_api=data)
    assert updates == 2  # endpoints 25 and 51; no incomplete tail update
    np.testing.assert_array_equal(scores[:26], np.full(26, .5))
    assert scores[26] > scores[25]
    assert scores[52] > scores[51]
    assert torch.equal(model.fast_origin, torch.zeros_like(model.fast_origin))
    again, _ = online.predict_shot(model, data.shots[1], intra=True,
                                   device='cpu', data_api=data)
    np.testing.assert_array_equal(scores, again)


def test_static_runner_seals_before_reveal_and_excludes_post_down_alarm(tmp_path):
    data = FakeData({1: shot(1, late_signal=True)})
    data.output = tmp_path / 'static'
    order = [dict(shot=1, proxy=1.)]
    result = online.run_arm(Toy(), order=order, arm='static', threshold=.9,
                            output_dir=data.output, data_api=data, limit=1)
    assert data.events == [('read', 1), ('reveal', 1)]
    assert result['rows'][0]['alarm'] is False
    assert result['rows'][0]['lead_ms'] is None
    assert result['rows'][0]['fast_updates'] == 0
    assert result['rows'][0]['persistent']['reason'] == 'static_or_intra'
    assert result['rows'][0]['phase'] == 'startup'
    assert result['metrics_full']['n_disrupt'] == 1
    assert 'metrics_prefix' not in result
    with np.load(data.output / '0001_1_sealed.npz') as sealed:
        assert len(sealed['scores']) == 54  # post-event scores remain sealed, not rescored


def test_order_reads_only_first_100_raw_planning_without_outcomes():
    data = FakeData({1: shot(1, n=110), 2: shot(2, n=110)})
    order = online.build_order('EAST', [2, 1], data_api=data)
    assert [r['shot'] for r in order] == [1, 2]
    assert order[0]['proxy'] == pytest.approx(1.)
    assert order[1]['proxy'] == pytest.approx(2.**1.2)
    assert all(kind == 'read' for kind, _ in data.events)


def test_persistent_update_scope_and_no_history():
    cfg = Config(window=4, patch=2, dim=4, heads=2, layers=1, horizon=25)
    model = Simple23(cfg, np.zeros(23), np.ones(23))
    s = shot(1, n=32)
    s['valid'] = np.ones(32, bool)
    s['y'] = np.arange(32, dtype=np.float32) % 2
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    result = online.postshot_update(model, s, [], lr=3e-4,
                                    rng=np.random.default_rng(3), device='cpu',
                                    data_api=FakeData({1: s}))
    assert result['accepted'] and result['reason'] == 'no_history'
    for name, p in model.named_parameters():
        assert p.requires_grad == flags[name]
        if name.startswith(('embedding', 'forecast_head', 'norm')) or name in ('position', 'cls'):
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
               if name.startswith('risk_head'))


def test_four_arms_share_order_and_limit_only_after_sort(tmp_path, monkeypatch):
    data = FakeData({1: shot(1, n=110), 2: shot(2, n=110)})
    order = online.build_order('EAST', [2, 1], data_api=data)
    data.events.clear()
    data.output = tmp_path / 'four'
    monkeypatch.setattr(online, 'postshot_update',
                        lambda *a, **k: dict(accepted=True, reason='no_history'))
    result = online.run_four_arms(Toy(), order=order, threshold=.4,
                                  output_dir=tmp_path / 'four', data_api=data, limit=1)
    assert set(result) == {'static', 'intra', 'inter', 'both'}
    assert all(r['rows'][0]['shot'] == 1 for r in result.values())
    assert all(r['manifest_count'] == 2 for r in result.values())
    assert all('metrics_full' not in r for r in result.values())


def test_revealed_disruptive_shot_samples_both_window_classes():
    s = shot(1, n=2000)
    s['valid'] = np.ones(2000, bool)
    s['y'] = np.zeros(2000, np.float32)
    s['y'][-8:] = 1.
    _, _, labels = online._examples([s], np.random.default_rng(131), 64,
                                     SimpleNamespace(window=4), 'cpu', FakeData({1: s}))
    assert int(labels.sum()) == 32
    assert int((labels == 0).sum()) == 32


def test_nonfinite_optimizer_candidate_rolls_back_even_without_history(monkeypatch):
    cfg = Config(window=4, patch=2, dim=4, heads=2, layers=1, horizon=25)
    model = Simple23(cfg, np.zeros(23), np.ones(23))
    s = shot(1, n=32)
    s['valid'] = np.ones(32, bool)
    s['y'] = np.arange(32, dtype=np.float32) % 2
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    original_step = torch.optim.AdamW.step

    def corrupt_step(optimizer, *args, **kwargs):
        original_step(optimizer, *args, **kwargs)
        optimizer.param_groups[0]['params'][0].data.fill_(float('nan'))

    monkeypatch.setattr(torch.optim.AdamW, 'step', corrupt_step)
    with pytest.raises(FloatingPointError, match='nonfinite persistent parameter'):
        online.postshot_update(model, s, [], lr=3e-4,
                               rng=np.random.default_rng(3), device='cpu',
                               data_api=FakeData({1: s}))
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
        assert param.requires_grad


def test_balanced_window_guard_rejects_worse_candidate(monkeypatch):
    cfg = Config(window=4, patch=2, dim=4, heads=2, layers=1, horizon=25)
    model = Simple23(cfg, np.zeros(23), np.ones(23))
    current, earlier = shot(1, n=32), shot(2, n=32)
    for s in (current, earlier):
        s['valid'] = np.ones(32, bool)
        s['y'] = np.arange(32, dtype=np.float32) % 2
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    original = online._bce
    calls = []

    def controlled_bce(candidate, examples):
        calls.append(examples[2].detach().cpu().numpy())
        if len(calls) in (1, 3):
            return torch.tensor(0. if len(calls) == 1 else 1.)
        return original(candidate, examples)

    monkeypatch.setattr(online, '_bce', controlled_bce)
    result = online.postshot_update(model, current, [earlier], lr=1e-4,
                                    rng=np.random.default_rng(3), device='cpu',
                                    data_api=FakeData({1: current, 2: earlier}))
    assert not result['accepted'] and result['reason'] == 'guard_bce_worse'
    assert result['guard_metric'] == 'balanced_window_bce_not_shot_safety'
    assert sum(calls[0]) == 32 and sum(calls[2]) == 32
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


def test_cli_rejects_untrained_forecast_before_any_arm(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'source_risk.pt'
    monkeypatch.setattr(online, 'qualify_source',
                        lambda path: {'qualified': True, 'reasons': []})
    monkeypatch.setattr(online.torch, 'load',
                        lambda *args, **kwargs: {'config': {'format': 'simple23-v1'},
                                                 'forecast_trained': False})
    monkeypatch.setattr(online, 'run_four_arms',
                        lambda *args, **kwargs: pytest.fail('arms started before forecast check'))
    with pytest.raises(ValueError, match='forecast-trained'):
        online.main(['--checkpoint', str(checkpoint), '--output', str(tmp_path / 'out'),
                     '--compute-device', 'cpu'])
