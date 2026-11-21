"""CPU contract tests for the causal commissioning runner."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning as C


class FakeModel(torch.nn.Module):
    def __init__(self, bias=2.):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
        self.cfg = SimpleNamespace(window=1, channels=1, horizon=1, fast_lr=.2,
                                   to_dict=lambda: dict(window=1, horizon=1))
        self.register_buffer('source_center', torch.zeros(1))
        self.register_buffer('source_scale', torch.ones(1))

    def forward(self, values, available):
        logit = self.bias.expand(len(values))
        return dict(logits=torch.stack((logit, logit), dim=-1))


def record(label=1.):
    return dict(device='EAST', shot=1, risk_windows=np.zeros((1, 1, 1), np.float32),
                risk_available=np.ones((1, 1, 1), bool), y=np.array([label], np.float32),
                y125=np.array([label], np.float32))


def test_strict_east_lead_rejects_exactly_30_ms():
    row = dict(shot=1, times=np.array([.07]), scores=np.array([.8]),
               is_disrupt=True, down=.1)
    exact = C.summarize_strict([row], .5)
    assert exact['predictions'][0]['timely'] is False
    assert exact['predictions'][0]['lead_ms'] == 30.
    row['times'] = np.array([.069])
    assert C.summarize_strict([row], .5)['predictions'][0]['timely'] is True


def test_single_class_rates_are_independent():
    disrupted = C.summarize_strict([dict(shot=1, times=np.array([0.]),
        scores=np.array([.9]), is_disrupt=True, down=.1)], .5)
    safe = C.summarize_strict([dict(shot=2, times=np.array([0.]),
        scores=np.array([.9]), is_disrupt=False, down=None)], .5)
    assert (disrupted['tpr'], disrupted['fpr'], disrupted['J']) == (1., None, None)
    assert (safe['tpr'], safe['fpr'], safe['J']) == (None, 1., None)


def test_historical_guard_rolls_back_new_wrong_decision(monkeypatch):
    model = FakeModel(2.)
    original = C.O.state_hash(model)

    def update(model, *args, **kwargs):
        with torch.no_grad():
            model.bias.fill_(-2.)
        return dict(steps=1), record()

    monkeypatch.setattr(C.O, 'persistent_update', update)
    audit, current, guard = C.apply_persistent_candidate(
        model, dict(shot=2), [record()], .5, np.array([.6]),
        device='cpu', lr=.001, steps=1, guarded=True)
    assert guard['accepted'] is False
    assert guard['new_wrong_previously_correct'] == 1
    assert C.O.state_hash(model) == original
    assert current['shot'] == 1
    assert guard['candidate_model_delta_norm'] > 0
    assert guard['accepted_model_delta_norm'] == 0


def test_historical_guard_rejects_bce_rise_without_flip(monkeypatch):
    model = FakeModel(2.)

    def update(model, *args, **kwargs):
        with torch.no_grad():
            model.bias.fill_(1.)
        return dict(steps=1), record()

    monkeypatch.setattr(C.O, 'persistent_update', update)
    _, _, guard = C.apply_persistent_candidate(
        model, dict(shot=2), [record()], .5, np.array([.6]),
        device='cpu', lr=.001, steps=1, guarded=True)
    assert guard['accepted'] is False
    assert guard['new_wrong_previously_correct'] == 0
    assert guard['mean_bce_increase'] > 1e-4
    assert model.bias.item() == pytest.approx(2.)


def test_first_shot_and_unguarded_reasons_are_explicit(monkeypatch):
    monkeypatch.setattr(C.O, 'persistent_update',
                        lambda *args, **kwargs: (dict(steps=1), record()))
    for guarded, reason in ((True, 'no_history'), (False, 'unguarded_ablation')):
        _, _, guard = C.apply_persistent_candidate(
            FakeModel(), dict(shot=1), [], .5, np.array([.6]),
            device='cpu', lr=.001, steps=1, guarded=guarded)
        assert guard['accepted'] is True
        assert guard['reason'] == reason
        assert guard['shot_safety_guarantee'] is False


@pytest.mark.parametrize('failure', ['raised', 'nonfinite'])
def test_candidate_failure_restores_full_model_even_without_history(monkeypatch, failure):
    model = FakeModel(2.)
    before = C.O.state_hash(model)

    def update(model, *args, **kwargs):
        with torch.no_grad():
            model.bias.fill_(float('nan') if failure == 'nonfinite' else -2.)
        if failure == 'raised':
            raise RuntimeError('injected update failure')
        return dict(steps=1), record()

    monkeypatch.setattr(C.O, 'persistent_update', update)
    with pytest.raises((RuntimeError, FloatingPointError)):
        C.apply_persistent_candidate(model, dict(shot=1), [], .5,
            np.array([.6]), device='cpu', lr=.001, steps=1)
    assert C.O.state_hash(model) == before


def test_stage_sidecar_uses_full_manifest_and_both_prediction_tracks():
    order = [dict(shot=i, proxy=float(i)) for i in (1, 2, 3, 4)]
    rows = [dict(shot=i, is_disrupt=False, alarm=False, timely=False, lead_ms=None)
            for i in (1, 2)]
    results = dict(static=dict(deployed=dict(predictions=rows),
                               same_parent_no_fast_shadow=dict(predictions=rows)))
    stages = C.stage_summaries(results, order, startup_budget=1)
    assert stages['static']['deployed']['full']['n'] == 2
    assert stages['static']['same_parent_no_fast_shadow']['full']['n'] == 2
    assert stages['static']['deployed']['high_proxy_quartile']['manifest_shots'] == [4]
    assert stages['static']['deployed']['high_proxy_quartile']['metrics']['n'] == 0


class FakeData:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.events = []

    def read_shot(self, domain, sid):
        self.events.append(('read', sid))
        return dict(shot=sid, device=domain, values=np.zeros((3, 1), np.float32),
                    available=np.ones((3, 1), bool), times=np.array([0., .001, .002]))

    def attach_outcome(self, s, domain, advanced):
        sid = s['shot']
        assert (self.output_dir/f'{sid}_sealed.npz').is_file()
        assert (self.output_dir/f'{sid}_seal.json').is_file()
        self.events.append(('reveal', sid))
        s.update(is_disrupt=False, down=None, y=np.zeros(3, np.float32),
                 y125=np.zeros(3, np.float32), valid=np.ones(3, bool))
        return s


def test_five_arm_phase_boundary_seal_before_reveal_and_source_parent(tmp_path, monkeypatch):
    calls = []

    def predict(model, s, *, intra, **kwargs):
        assert 'is_disrupt' not in s
        calls.append(('predict', s['shot'], intra))
        return dict(scores=np.full(3, .1, np.float32),
                    shadow_scores=np.full(3, .1, np.float32), fast_updates=[],
                    fast_reset=True, wall_s=0., fast_update_wall_s=0.)

    def update(model, s, replay, device, lr, steps, **kwargs):
        calls.append(('update', s['shot'], lr, tuple(r['shot'] for r in replay)))
        with torch.no_grad():
            model.bias.add_(.01)
        item = record(0.)
        item['shot'] = s['shot']
        return dict(steps=1, selected_endpoints=[]), item

    def select(model, s, **kwargs):
        item = record(0.)
        item['shot'] = s['shot']
        return item

    monkeypatch.setattr(C.O, 'predict_shot', predict)
    monkeypatch.setattr(C.O, 'persistent_update', update)
    monkeypatch.setattr(C.O, 'select_experience', select)
    order = [dict(shot=sid, proxy=float(sid)) for sid in (1, 2, 3)]
    for arm in C.ARMS:
        calls.clear()
        model = FakeModel()
        parent = C.O.state_hash(model)
        directory = tmp_path/arm
        data_api = FakeData(directory)
        result = C.run_arm(model, order=order, arm=arm, threshold=.5,
                           output_dir=directory, startup_budget=1,
                           data_api=data_api, device='cpu', guarded=False)
        assert result['parent_hash'] == parent
        assert (directory/'snapshot_1.pt').is_file()
        assert [sid for event, sid in data_api.events if event == 'reveal'] == [1, 2, 3]
        updates = [entry for entry in calls if entry[0] == 'update']
        assert len(updates) == {'static': 0, 'intra': 0, 'startup_freeze': 1,
                                'inter': 3, 'both': 3}[arm]
        if updates:
            assert updates[0][2] == .001
            assert updates[0][3] == ()
            if len(updates) > 1:
                assert updates[1][2] == .00025
                assert updates[1][3] == (1,)
        flags = [entry[2] for entry in calls if entry[0] == 'predict']
        assert flags == ([arm in ('intra', 'both')]*3)
        if arm in ('static', 'intra'):
            assert C.O.state_hash(model) == parent
