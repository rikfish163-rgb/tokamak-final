from types import SimpleNamespace

import numpy as np
import pytest

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.ended_density_replay import EndedDensityReplay
from experiments.risk_aligned_online.redesign_20261004.run import run_arm
from experiments.risk_aligned_online.simple_online import seal_predictions


DENSITY = io23.RAW23.index('ne_nG')
MODE = io23.RAW23.index('mode_number_n')


class Data:
    window = staticmethod(simple_data.window)

    def __init__(self, disruptive=False):
        self.disruptive, self.reveals = disruptive, []

    def read_shot(self, device, sid, **kwargs):
        assert device == 'EAST' and kwargs['rate_contract'] == 'nominal'
        values = np.tile(np.arange(23, dtype=np.float32), (60, 1))
        values[:, DENSITY] = sid * 10 + np.arange(60)
        values[:, MODE] = 0
        available = np.ones((60, 23), bool)
        available[:, io23.RAW23.index('MNM')] = False
        return dict(shot=sid, device=device, values=values, available=available,
                    times=np.arange(60)/1000.)

    def attach_outcome(self, shot, device, advanced):
        self.reveals.append(shot['shot'])
        return dict(shot, is_disrupt=self.disruptive,
                    down=.059 if self.disruptive else None, horizon=125.,
                    valid=np.ones(60, bool), y=np.full(60, float(self.disruptive)))


def provider(path, *, aligned=True, count=2, data=None):
    path.mkdir()
    return EndedDensityReplay(np.zeros(23, np.float32), np.ones(23, np.float32),
                             output_dir=path, order=[1, 2, 3],
                             calibration_count=count, aligned=aligned,
                             data_api=data or Data())


def finish(p, sid):
    s = p.read_shot('EAST', sid)
    seal_predictions(p.output_dir, sid, sid, s['times'], np.full(60, .1),
                     arm='static', threshold=.7)
    return p.attach_outcome(s, 'EAST', {})


def test_first_shot_has_no_target_statistics_and_switch_is_after_calibration_seal(tmp_path):
    p = provider(tmp_path/'a')
    first, second = finish(p, 1), finish(p, 2)
    assert not first['alignment_active'] and not second['alignment_active']
    np.testing.assert_array_equal(first['values'][:, DENSITY], 10+np.arange(60))
    assert p.statistics['calibration_ids'] == [1, 2]
    third = p.read_shot('EAST', 3)
    assert third['alignment_active']
    assert not np.array_equal(third['values'][:, DENSITY], 30+np.arange(60))


def test_no_outcome_or_statistics_before_real_whole_shot_seal(tmp_path):
    data = Data(); p = provider(tmp_path/'a', data=data)
    shot = p.read_shot('EAST', 1)
    with pytest.raises(ValueError, match='seal'):
        p.attach_outcome(shot, 'EAST', {})
    assert data.reveals == [] and p.ended_ids == [] and p.statistics is None


def test_modified_seal_rejected_before_label_reader(tmp_path):
    data = Data(); p = provider(tmp_path/'a', data=data)
    shot = p.read_shot('EAST', 1)
    receipt = seal_predictions(p.output_dir, 1, 1, shot['times'], np.full(60, .1),
                               arm='static', threshold=.7)
    with (p.output_dir/receipt['npz']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        p.attach_outcome(shot, 'EAST', {})
    assert not data.reveals


def test_revealed_labels_do_not_change_scale_estimate(tmp_path):
    a = provider(tmp_path/'a', data=Data(False))
    b = provider(tmp_path/'b', data=Data(True))
    for sid in (1, 2):
        finish(a, sid); finish(b, sid)
    for key in ('center', 'scale', 'usable', 'counts'):
        np.testing.assert_array_equal(a.statistics[key], b.statistics[key])


def test_input_mapping_equals_target_asinh_without_mutating_source_coordinates(tmp_path):
    p = provider(tmp_path/'a'); finish(p, 1); finish(p, 2)
    s = p.read_shot('EAST', 3)
    c = p.statistics['center'][DENSITY]
    a = max(p.statistics['scale'][DENSITY], abs(c)*.01, 1e-6)
    np.testing.assert_allclose(np.arcsinh(s['values'][:, DENSITY]),
                              np.arcsinh((s['_raw_values'][:, DENSITY]-c)/a),
                              rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(p.source_center, np.zeros(23))
    np.testing.assert_array_equal(p.source_scale, np.ones(23))
    other = np.arange(23) != DENSITY
    np.testing.assert_array_equal(s['values'][:, other], s['_raw_values'][:, other])


def test_mask_control_matches_aligned_arm_and_replay_is_reencoded_once(tmp_path):
    a = provider(tmp_path/'a'); b = provider(tmp_path/'b', aligned=False)
    old = finish(a, 1); finish(a, 2); finish(b, 1); finish(b, 2)
    mapped = a.reencode(old)
    np.testing.assert_array_equal(a.reencode(mapped)['values'], mapped['values'])
    assert not np.array_equal(old['values'][:, DENSITY], mapped['values'][:, DENSITY])
    x, y = a.read_shot('EAST', 3), b.read_shot('EAST', 3)
    np.testing.assert_array_equal(x['available'], y['available'])
    assert not x['available'][:, MODE].any()
    np.testing.assert_array_equal(y['values'], y['_raw_values'])


def test_future_signal_changes_do_not_refit_locked_ended_statistics(tmp_path):
    p = provider(tmp_path/'a'); finish(p, 1); finish(p, 2)
    stats = p.statistics['center'].copy()
    p.data_api.read_shot = lambda device, sid, **kw: dict(
        shot=sid, device=device, values=np.full((60,23), 10000., np.float32),
        available=np.ones((60,23), bool), times=np.arange(60)/1000.)
    p.read_shot('EAST', 3)
    np.testing.assert_array_equal(p.statistics['center'], stats)


def test_reads_follow_locked_order_and_one_pending_shot(tmp_path):
    p = provider(tmp_path/'a')
    with pytest.raises(ValueError, match='order'):
        p.read_shot('EAST', 2)
    p.read_shot('EAST', 1)
    with pytest.raises(ValueError, match='pending'):
        p.read_shot('EAST', 2)
    with pytest.raises(ValueError):
        p.read_shot('J-TEXT', 1)


def test_forecast_windows_receive_only_already_encoded_values(tmp_path):
    p = provider(tmp_path/'a'); finish(p, 1); finish(p, 2)
    s = p.read_shot('EAST', 3)
    prefix = {k:s[k][:10] for k in ('values', 'available')}
    x, a = p.window(prefix, 9, 128)
    np.testing.assert_array_equal(x[-10:], s['values'][:10])
    assert not a[:-10].any()


def test_constant_density_is_not_a_silent_source_fallback(tmp_path):
    data = Data(); read = data.read_shot
    def constant(*args, **kw):
        s = read(*args, **kw); s['values'][:, DENSITY] = 5.
        return s
    data.read_shot = constant
    p = provider(tmp_path/'a', data=data)
    finish(p, 1)
    with pytest.raises(ValueError, match='density'):
        finish(p, 2)


def test_density_only_mapping_preserves_observed_constant_other_channels(tmp_path):
    path=tmp_path/'observed'; path.mkdir()
    p=EndedDensityReplay(np.zeros(23),np.ones(23),output_dir=path,
        order=[1,2,3],calibration_count=2,data_api=Data(),preserve_observed_channels=True)
    finish(p,1); finish(p,2)
    assert not p.statistics['usable'][MODE]
    third=p.read_shot('EAST',3)
    assert third['available'][:,MODE].all()
    assert not third['available'][:,io23.RAW23.index('MNM')].any()
    np.testing.assert_array_equal(third['available'],third['_raw_available'])
    other=np.arange(23)!=DENSITY
    np.testing.assert_array_equal(third['values'][:,other],third['_raw_values'][:,other])


def test_preserved_observation_control_keeps_same_masks_and_raw_replay(tmp_path):
    providers=[]
    for aligned in (False,True):
        path=tmp_path/str(aligned); path.mkdir()
        p=EndedDensityReplay(np.zeros(23),np.ones(23),output_dir=path,
            order=[1,2,3],calibration_count=2,data_api=Data(),aligned=aligned,
            preserve_observed_channels=True)
        old=finish(p,1); finish(p,2)
        replay=p.reencode(old)
        np.testing.assert_array_equal(replay['available'],old['_raw_available'])
        np.testing.assert_array_equal(p.reencode(replay)['values'],replay['values'])
        providers.append(p.read_shot('EAST',3))
    np.testing.assert_array_equal(providers[0]['available'],providers[1]['available'])
    assert not np.array_equal(providers[0]['values'][:,DENSITY],providers[1]['values'][:,DENSITY])


def test_run_arm_seals_then_reveals_and_keeps_startup_cost(tmp_path):
    path=tmp_path/'stream'
    p = EndedDensityReplay(np.zeros(23), np.ones(23), output_dir=path,
                          order=[1,2,3], calibration_count=2, data_api=Data())
    seen=[]
    def predict(model, shot, **kw):
        seen.append((shot['shot'], shot['alignment_active']))
        return np.full(60,.8 if shot['shot']==1 else .1), dict(accepted_updates=0)
    result=run_arm(SimpleNamespace(), order=[dict(shot=i) for i in (1,2,3)],
                   arm='static', threshold=.7, output_dir=path, device='cpu',
                   data_api=p, predictor=predict)
    assert seen == [(1,False),(2,False),(3,True)]
    assert result['metrics']['fp']==1 and result['observed_count']==3
    assert p.data_api.reveals==[1,2,3]


def test_qa_convention_is_available_on_first_shot_without_target_fit(tmp_path):
    data=Data();read=data.read_shot
    def signals(*args,**kwargs):
        s=read(*args,**kwargs);s['values'][:,0]=2.;s['values'][:,9]=2e5
        return s
    data.read_shot=signals
    p=EndedDensityReplay(np.zeros(23),np.ones(23),output_dir=tmp_path,
        order=[1,2,3],calibration_count=2,data_api=data,qa_coefficient=.374)
    s=p.read_shot('EAST',1)
    assert p.statistics is None and data.reveals==[]
    np.testing.assert_allclose(s['values'][:,13],.374*2/2e5)
    np.testing.assert_array_equal(s['_raw_values'][:,13],13.)


def test_qa_replay_is_rebuilt_from_canonical_raw_and_not_applied_twice(tmp_path):
    p=EndedDensityReplay(np.zeros(23),np.ones(23),output_dir=tmp_path,
        order=[1,2,3],calibration_count=2,data_api=Data(),qa_coefficient=.374)
    old=finish(p,1);finish(p,2)
    new=p.reencode(old);twice=p.reencode(new)
    np.testing.assert_array_equal(new['values'],twice['values'])
    np.testing.assert_array_equal(new['_raw_values'],old['_raw_values'])
    assert not np.array_equal(new['values'][:,13],old['_raw_values'][:,13])


def test_invalid_qa_coefficient_rejected_before_target_access(tmp_path):
    data=Data()
    with pytest.raises(ValueError):
        EndedDensityReplay(np.zeros(23),np.ones(23),output_dir=tmp_path,
            order=[1,2,3],calibration_count=2,data_api=data,qa_coefficient=float('nan'))
    assert data.reveals==[]
