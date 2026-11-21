from contextlib import contextmanager

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.n1_alignment_20261104.calibration import (
    fit_gain, source_reference,
)


@pytest.fixture
def source_hdf(monkeypatch):
    """One genuinely in-memory HDF contains four templates for 1,353 shots."""
    original_file = h5py.File
    data = original_file('fixture.hdf5', 'w', driver='core', backing_store=False)
    for name, flag, values in (
        ('safe1', 0, [0., 2., 8., np.nan, np.inf]),
        ('safe2', 0, [4., 4.]),
        ('safe3', 0, [10.] * 20),
        ('disruptive', 1, [9000.]),
    ):
        group = data.create_group(name)
        group.create_dataset('meta/IsDisrupt', data=flag)
        group.create_dataset('meta/DownTime', data=1.5)
        group.create_dataset('data/n=1 amplitude', data=np.array(values))
    train = list(range(1, 1354))
    monkeypatch.setattr(io23, 'source_roles', lambda: {'train': train, 'val': [2001], 'test': [3001]})
    accesses = []

    class FakePath:
        def __init__(self, sid):
            self.sid = sid

        def is_file(self):
            return self.sid != 99

    def path_for(device, sid):
        assert device == 'J-TEXT' and sid in train
        return FakePath(sid)

    class View:
        def __init__(self, sid):
            self.sid = sid
            self.group = data[f'safe{sid}' if sid <= 3 else 'disruptive']

        def __getitem__(self, key):
            accesses.append((self.sid, key))
            if self.sid > 3 and key.startswith('data/'):
                raise AssertionError('disruptive source signals must not be read')
            return self.group[key]

    @contextmanager
    def open_fixture(path, mode):
        assert mode == 'r'
        yield View(path.sid)

    monkeypatch.setattr(io23, 'path_for', path_for)
    monkeypatch.setattr(h5py, 'File', open_fixture)
    try:
        yield data, train, accesses, FakePath
    finally:
        data.close()


def test_source_uses_hdf_outcomes_not_conflicting_shot_list_and_equal_shots(source_hdf, monkeypatch):
    _, train, accesses, fake_path = source_hdf
    monkeypatch.setattr(io23, 'path_for', lambda device, sid: _path(device, sid, fake_path))
    # Deliberately conflicting hypothetical list classes: shot 1 marked D and
    # shot 4 marked safe. Only HDF IsDisrupt is authoritative.
    conflicting_list = {1: 'D', 4: 'safe'}
    result = source_reference(train.copy())
    assert conflicting_list[1] == 'D' and conflicting_list[4] == 'safe'
    assert result['source_safe_ids'] == [1, 2, 3]
    assert result['per_shot_medians'] == {1: 2., 2: 4., 3: 10.}
    assert result['value'] == 4.  # not the sample-pooled median of 10
    assert result['source_train_ids'] == train
    assert result['source_data_feature'] == 'n=1 amplitude'
    assert sum(key == 'meta/IsDisrupt' for _, key in accesses) == 1353
    assert [sid for sid, key in accesses if key == 'data/n=1 amplitude'] == [1, 2, 3]


def _path(device, sid, cls):
    assert device == 'J-TEXT'
    path = cls(sid)
    path.is_file = lambda: True
    return path


def test_source_rejects_wrong_train_order_extra_id_and_missing_file(source_hdf):
    _, train, accesses, _ = source_hdf
    with pytest.raises(ValueError):
        source_reference(train[::-1])
    with pytest.raises(ValueError):
        source_reference(train + [2001])
    with pytest.raises(TypeError):
        source_reference(tuple(train))
    with pytest.raises(ValueError):
        source_reference([True] + train[1:])  # True must not alias integer shot 1
    assert accesses == []
    with pytest.raises(FileNotFoundError):
        source_reference(train)


@pytest.mark.parametrize('values', [[0., 0.], [-3., -2.], [np.nan, np.inf]])
def test_source_invalid_or_unobserved_reference_fails(source_hdf, monkeypatch, values):
    data, train, _, fake_path = source_hdf
    monkeypatch.setattr(io23, 'path_for', lambda device, sid: _path(device, sid, fake_path))
    for i in (1, 2, 3):
        key = f'safe{i}/data/n=1 amplitude'
        del data[key]
        data.create_dataset(key, data=np.asarray(values))
    with pytest.raises(ValueError):
        source_reference(train)


def test_source_invalid_hdf_outcome_fails(source_hdf, monkeypatch):
    data, train, _, fake_path = source_hdf
    monkeypatch.setattr(io23, 'path_for', lambda device, sid: _path(device, sid, fake_path))
    data['disruptive/meta/IsDisrupt'][()] = 2
    with pytest.raises(ValueError):
        source_reference(train)


def test_source_requires_a_safe_shot_and_the_named_signal(source_hdf, monkeypatch):
    data, train, _, fake_path = source_hdf
    monkeypatch.setattr(io23, 'path_for', lambda device, sid: _path(device, sid, fake_path))
    for i in (1, 2, 3):
        data[f'safe{i}/meta/IsDisrupt'][()] = 1
    with pytest.raises(ValueError, match='no HDF-safe'):
        source_reference(train)
    data['safe1/meta/IsDisrupt'][()] = 0
    del data['safe1/data/n=1 amplitude']
    with pytest.raises(KeyError):
        source_reference(train)


@pytest.fixture
def ended():
    ids = list(range(70001, 70021))
    past, truth = {}, {}
    for i, sid in enumerate(ids):
        n = 4 if i != 2 else 50
        values = np.ones((n, 23), dtype=np.float32)
        available = np.ones_like(values, dtype=bool)
        if i == 0:
            values[:, 11] = [0., 2., 6., 8.]
        elif i == 1:
            values[:, 11] = [2., 4., 2., 4.]
        elif i == 2:
            values[:, 11] = 6.
        past[sid] = {'values': values, 'available': available, 'device': 'EAST', 'shot': sid}
        truth[sid] = (i >= 3, 1.5 if i >= 3 else None)
    return {'value': 20.}, past, truth, ids


def test_fit_gain_safe_only_equal_shots_zero_is_observed_and_no_io(ended, monkeypatch):
    reference, past, truth, ids = ended
    def forbidden(*args, **kwargs):
        raise AssertionError('fit_gain must not access HDF, source, or EAST')
    for name in ('read_signals', 'read_outcome', 'path_for'):
        monkeypatch.setattr(io23, name, forbidden)
    monkeypatch.setattr(h5py, 'File', forbidden)
    before = {sid: (p['values'].copy(), p['available'].copy()) for sid, p in past.items()}
    result = fit_gain(reference, past, truth, ids)
    assert result['gain'] == 5. and result['source_reference'] == 20.
    assert result['target_reference'] == 4.
    assert result['target_safe_ids'] == ids[:3] and result['calibration_ids'] == ids
    assert result['per_shot_target_medians'] == {ids[0]: 4., ids[1]: 3., ids[2]: 6.}
    assert result['method'] == 'ratio_of_equal_shot_safe_medians'
    assert result['physical_unit_conversion_claim'] is False
    assert all(np.array_equal(past[s]['values'], before[s][0]) and
               np.array_equal(past[s]['available'], before[s][1]) for s in ids)
    past[ids[3]]['values'][:] = -1e20
    assert fit_gain(reference, past, truth, ids) == result


def test_fit_accepts_two_safe_shots_and_ignores_unavailable_nonfinite(ended):
    reference, past, truth, ids = ended
    truth[ids[2]] = (True, 1.)
    past[ids[0]]['available'][0, 0] = False
    past[ids[0]]['values'][0, 0] = np.nan
    result = fit_gain(reference, past, truth, ids)
    assert result['target_safe_ids'] == ids[:2]
    assert result['target_reference'] == 3.5


def test_fit_rejects_future_missing_duplicates_and_insufficient_safe(ended):
    reference, past, truth, ids = ended
    past[80000] = past[ids[0]]
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids)
    del past[80000]
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids[:-1])
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids[:-1] + ids[:1])
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids[:-1] + [80000])
    del truth[ids[0]]
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids)
    truth[ids[0]] = (False, None)
    truth[ids[1]] = truth[ids[2]] = (True, 1.)
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids)


@pytest.mark.parametrize('collection', ['past', 'truth'])
def test_fit_rejects_float_keys_that_alias_integer_ids(ended, collection):
    reference, past, truth, ids = ended
    target = past if collection == 'past' else truth
    entry = target.pop(ids[0])
    target[float(ids[0])] = entry
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids)


@pytest.mark.parametrize('problem', ['dtype', 'mask_dtype', 'shape', 'empty', 'nonfinite',
                                     'bad_truth', 'bad_down', 'wrong_device', 'wrong_shot'])
def test_fit_rejects_invalid_input_contract(ended, problem):
    reference, past, truth, ids = ended
    p = past[ids[3]]  # validation covers disruptive entries too
    if problem == 'dtype':
        p['values'] = p['values'].astype(np.float64)
    elif problem == 'mask_dtype':
        p['available'] = p['available'].astype(np.int64)
    elif problem == 'shape':
        p['values'] = p['values'][:, :22]
    elif problem == 'empty':
        p['values'], p['available'] = p['values'][:0], p['available'][:0]
    elif problem == 'nonfinite':
        p['values'][0, 0] = np.nan
    elif problem == 'bad_truth':
        truth[ids[3]] = (1, 1.)
    elif problem == 'bad_down':
        truth[ids[3]] = (True, np.nan)
    elif problem == 'wrong_device':
        p['device'] = 'J-TEXT'
    else:
        p['shot'] = 99999
    with pytest.raises((TypeError, ValueError)):
        fit_gain(reference, past, truth, ids)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf, True])
def test_fit_rejects_invalid_source_reference(ended, value):
    _, past, truth, ids = ended
    with pytest.raises(ValueError):
        fit_gain({'value': value}, past, truth, ids)


@pytest.mark.parametrize('kind', ['zero', 'negative', 'missing'])
def test_fit_rejects_invalid_target_reference_or_absent_safe_observations(ended, kind):
    reference, past, truth, ids = ended
    for sid in ids[:3]:
        if kind == 'missing':
            past[sid]['available'][:, 11] = False
        else:
            past[sid]['values'][:, 11] = 0. if kind == 'zero' else -1.
    with pytest.raises(ValueError):
        fit_gain(reference, past, truth, ids)
