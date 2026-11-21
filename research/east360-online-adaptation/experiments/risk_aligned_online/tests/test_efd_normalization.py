import numpy as np
import pytest

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.efd_normalization import (
    ChannelMoments, EfdReplay, load_source_statistics, source_coordinate_map,
)
from experiments.risk_aligned_online.simple_online import seal_predictions


def moments(values, available=None):
    state = ChannelMoments()
    state.update(values, np.ones_like(values, bool) if available is None else available)
    return state.snapshot()


def arrays(offset=0.):
    return np.arange(23, dtype=np.float64)[None, :] + np.arange(60)[:, None] + offset


def test_incremental_mean_population_std_matches_full_observations():
    x = arrays()
    mask = np.ones_like(x, bool)
    mask[:7, 0] = False
    x[8, 1] = np.nan
    state = ChannelMoments()
    for lo, hi in ((0, 9), (9, 37), (37, 60)):
        state.update(x[lo:hi], mask[lo:hi])
    actual = state.snapshot()
    for c in range(23):
        v = x[:, c][mask[:, c] & np.isfinite(x[:, c])]
        assert actual['counts'][c] == len(v)
        assert actual['mean'][c] == pytest.approx(v.mean(), rel=1e-12, abs=1e-12)
        assert actual['std'][c] == pytest.approx(v.std(ddof=0), rel=1e-12)


def test_tiny_valid_variation_is_not_erased_by_unitful_epsilon():
    x = arrays() * 1e-12
    stats = moments(x)
    assert (stats['std'] > 0).all()
    np.testing.assert_allclose(stats['std'], x.std(0), rtol=1e-12)


def test_zscore_bridge_uses_both_device_means_and_stds_not_parent_robust_scale():
    source = moments(arrays(20.))
    target = moments(arrays(300.) * 4.)
    values = arrays(350.) * 4.
    mapped, receipt = source_coordinate_map(values, np.ones_like(values, bool), source, target)
    c = io23.RAW23.index('ne_nG')
    expected = source['mean'][c] + source['std'][c] * (
        values[:, c] - target['mean'][c]) / target['std'][c]
    np.testing.assert_allclose(mapped[:, c], expected, rtol=1e-6)
    assert receipt['aligned'][c]
    # The fitted parent's median/IQR and asinh still run AFTER this bridge.
    np.testing.assert_allclose((mapped[:, c]-source['mean'][c])/source['std'][c],
                              (values[:, c]-target['mean'][c])/target['std'][c], rtol=1e-6)


def test_same_distribution_bridge_is_identity_and_discrete_modes_keep_value():
    x = arrays()
    stats = moments(x)
    actual, receipt = source_coordinate_map(x, np.ones_like(x, bool), stats, stats)
    np.testing.assert_array_equal(actual, x.astype(np.float32))
    c = io23.RAW23.index('mode_number_n')
    assert not receipt['aligned'][c]
    assert receipt['status'][c] == 'discrete_source_normalization'


def test_missing_or_constant_observations_are_not_disguised_as_aligned():
    source = moments(arrays())
    x = arrays(100.)
    x[:, 3] = 7.
    mask = np.ones_like(x, bool)
    mask[:, 4] = False
    stats = moments(x, mask)
    actual, receipt = source_coordinate_map(x, mask, source, stats)
    np.testing.assert_array_equal(actual[:, 3:5], x[:, 3:5].astype(np.float32))
    assert receipt['status'][3] == 'unidentifiable_target_variance'
    assert receipt['status'][4] == 'unavailable_target'
    assert not receipt['aligned'][3:5].any()


def test_control_preservation_is_explicit_not_an_unreported_new_feature_group():
    x = arrays(100.)
    actual, receipt = source_coordinate_map(x, np.ones_like(x, bool),
        moments(arrays()), moments(x), preserve_controls=True)
    for name in ('Bt', 'ip'):
        c = io23.RAW23.index(name)
        np.testing.assert_array_equal(actual[:, c], x[:, c].astype(np.float32))
        assert receipt['status'][c] == 'physical_level_preserved_source_normalization'


class Data:
    window = staticmethod(simple_data.window)

    def __init__(self, disruptive=False):
        self.reveals = []
        self.disruptive = disruptive

    def read_shot(self, device, sid, **kwargs):
        return dict(shot=sid, device=device, values=arrays(100*sid).astype(np.float32),
                    available=np.ones((60, 23), bool), times=np.arange(60)/1000.)

    def attach_outcome(self, shot, device, advanced):
        self.reveals.append(shot['shot'])
        return dict(shot, is_disrupt=self.disruptive, down=.059 if self.disruptive else None,
                    valid=np.ones(60, bool), y=np.full(60, float(self.disruptive)), horizon=125.)


def provider(path, *, data=None, mode='fixed'):
    path.mkdir()
    return EfdReplay(np.zeros(23), np.ones(23), source_statistics=moments(arrays()),
        output_dir=path, order=[1,2,3,4], calibration_count=2, statistics_mode=mode,
        data_api=data or Data())


def finish(p, sid):
    shot = p.read_shot('EAST', sid)
    seal_predictions(p.output_dir, sid, sid, shot['times'], np.full(60, .1),
                     arm='static', threshold=.7)
    return p.attach_outcome(shot, 'EAST', {})


def test_first_shot_has_zero_target_statistics_and_unsealed_shot_cannot_fit():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        p = provider(Path(directory)/'p')
        shot = p.read_shot('EAST', 1)
        assert not shot['alignment_active'] and p.statistics is None
        with pytest.raises(ValueError, match='seal'):
            p.attach_outcome(shot, 'EAST', {})
        assert p.data_api.reveals == [] and p.statistics is None


def test_ended_moments_do_not_depend_on_revealed_class(tmp_path):
    a = provider(tmp_path/'a', data=Data(False))
    b = provider(tmp_path/'b', data=Data(True))
    for sid in (1, 2):
        finish(a, sid); finish(b, sid)
    for k in ('mean', 'std', 'counts'):
        np.testing.assert_array_equal(a.statistics[k], b.statistics[k])
    assert a.statistics['calibration_ids'] == [1, 2]
    assert a.statistics['counts'][0] == 120


def test_cumulative_statistics_update_only_after_seal_and_replay_never_double_maps(tmp_path):
    p = provider(tmp_path/'p', mode='cumulative')
    old = finish(p, 1); finish(p, 2)
    before = p.statistics['mean'].copy()
    third = p.read_shot('EAST', 3)
    np.testing.assert_array_equal(p.statistics['mean'], before)
    np.testing.assert_array_equal(third['available'], third['_raw_available'])
    seal_predictions(p.output_dir, 3, 3, third['times'], np.full(60, .1), arm='static', threshold=.7)
    p.attach_outcome(third, 'EAST', {})
    assert not np.array_equal(p.statistics['mean'], before)
    mapped = p.reencode(old)
    np.testing.assert_array_equal(p.reencode(mapped)['values'], mapped['values'])
    assert p.statistics['calibration_ids'] == [1, 2, 3]
    assert p.statistics['counts'][0] == 180


def test_fixed_statistics_do_not_refit_on_later_shot(tmp_path):
    p = provider(tmp_path/'p')
    finish(p, 1); finish(p, 2)
    before = p.statistics['mean'].copy()
    finish(p, 3)
    np.testing.assert_array_equal(p.statistics['mean'], before)
    assert p.statistics['calibration_ids'] == [1, 2]


def test_no_target_statistics_does_not_mutate_raw_or_parent_coordinates(tmp_path):
    p = provider(tmp_path/'p')
    old = finish(p, 1); finish(p, 2)
    raw = old['_raw_values'].copy()
    p.reencode(old)
    np.testing.assert_array_equal(old['_raw_values'], raw)
    np.testing.assert_array_equal(p.source_center, np.zeros(23))
    np.testing.assert_array_equal(p.source_scale, np.ones(23))


@pytest.mark.parametrize('field,bad', [('parent_sha256','wrong'),
    ('source_role','J-TEXT TEST'),('target_records_accessed',1),('train_ids',[1,2,4]),
    ('channel_order',list(reversed(io23.RAW23)))])
def test_source_moments_reject_wrong_parent_role_or_feature_order(tmp_path, field, bad):
    stats = moments(arrays())
    protocol = dict(parent_sha256='parent', source_role='J-TEXT TRAIN',
        target_records_accessed=0, train_ids=[1,2,3], channel_order=list(io23.RAW23),
        definition='population_mean_std_full_valid_points')
    protocol[field] = bad
    path = tmp_path/'moments.json'
    path.write_text(json.dumps(dict(protocol=protocol, **{k:v.tolist() for k,v in stats.items()})))
    with pytest.raises(ValueError, match='source TRAIN'):
        load_source_statistics(path, expected_parent_sha256='parent', train_ids=[1,2,3])


def test_raw_control_keeps_source_coordinates_even_when_statistics_are_ready(tmp_path):
    path = tmp_path/'control'; path.mkdir()
    p = EfdReplay(np.zeros(23),np.ones(23),source_statistics=moments(arrays()),
        output_dir=path,order=[1,2,3],calibration_count=2,data_api=Data(),aligned=False)
    finish(p,1); finish(p,2)
    third = p.read_shot('EAST',3)
    assert third['calibration_ready'] and not third['alignment_active']
    np.testing.assert_array_equal(third['values'],third['_raw_values'])
import json
