from types import SimpleNamespace

import numpy as np
import pytest

from . import run


class FakeData:
    def __init__(self, path, events):
        self.path, self.events = path, events

    def read_shot(self, device, sid):
        self.events.append(('signals', sid))
        return dict(shot=sid, device=device, values=np.zeros((60, 23), np.float32),
                    available=np.ones((60, 23), bool), times=np.arange(60)/1000.)

    def attach_outcome(self, shot, device, advanced):
        assert list(self.path.glob('*_seal.json')), 'outcome accessed before seal'
        self.events.append(('reveal', shot['shot']))
        disruptive = shot['shot'] == 1
        return dict(shot, is_disrupt=disruptive, down=.059, horizon=125.,
                    y=np.ones(60) if disruptive else np.zeros(60), valid=np.ones(60, bool))


def test_arm_seals_before_reveal_and_inter_uses_only_previous_shots(tmp_path):
    events = []
    arm_dir = tmp_path/'both'
    def predict(model, shot, **kwargs):
        assert 'y' not in shot and 'down' not in shot
        events.append(('predict', shot['shot']))
        return np.full(60, .8 if shot['shot'] == 1 else .1, np.float32), dict(
            accepted_updates=2, attempted_updates=2)
    def update(model, shot, replay, **kwargs):
        events.append(('update', shot['shot']))
        assert len(replay) == shot['shot']-1
        assert '_sealed_scores' in shot
        return dict(accepted=True, parameter_change=.01)
    result = run.run_arm(SimpleNamespace(), order=[dict(shot=1), dict(shot=2)],
        arm='both', threshold=.7, output_dir=arm_dir, device='cpu',
        data_api=FakeData(arm_dir, events), predictor=predict, updater=update)
    assert [x[0] for x in events] == ['signals','predict','reveal','update']*2
    assert result['metrics']['tp'] == 1 and result['metrics']['fp'] == 0
    assert result['accepted_intra_updates'] == 4
    assert result['accepted_inter_updates'] == 2
    assert result['evidence'] == 'DEVELOPMENT_LOCKED_STREAM'


def test_exact_30ms_is_not_timely():
    rows = [dict(shot=1, is_disrupt=True, down=.059, times=np.array([.029]),
                 scores=np.array([.8]))]
    assert run.metrics(rows, .7, lead=30.)['tp'] == 0


def test_duplicate_order_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        run.validate_order([dict(shot=1), dict(shot=1)], ids=[1, 2])


def test_roster_mismatch_rejected():
    with pytest.raises(ValueError, match='roster'):
        run.validate_order([dict(shot=1)], ids=[1, 2])


def test_static_never_calls_updater(tmp_path):
    def predict(model, shot, **kwargs):
        assert kwargs['intra'] is False
        return np.zeros(60, np.float32), dict(accepted_updates=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('static performed update')
    run.run_arm(SimpleNamespace(), order=[dict(shot=2)], arm='static', threshold=.7,
        output_dir=tmp_path/'static', device='cpu', data_api=FakeData(tmp_path/'static', []),
        predictor=predict, updater=forbidden)
