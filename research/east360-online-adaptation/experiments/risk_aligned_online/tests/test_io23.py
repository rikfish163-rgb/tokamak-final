"""Contract tests for the standalone 23-channel HDF5 access layer."""
import subprocess
import sys
import json

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online import io23


def test_import_does_not_load_old_models():
    code = (
        "import sys; from experiments.risk_aligned_online import io23; "
        "assert not any(x.startswith(('experiments.fast_online_idea', "
        "'experiments.transfer_methods_2026')) for x in sys.modules); "
        "assert not any(x.startswith('torch') for x in sys.modules)"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize(('device', 'shot'), [('J-TEXT', 1051510), ('EAST', 54157)])
def test_real_signal_outcome_labels_match_raw_hdf_and_planning(device, shot):
    actual = io23.read_signals(device, shot)
    assert actual['v'].shape[1] == 23
    with h5py.File(io23.path_for(device, shot), 'r') as handle:
        matrix = np.stack([np.asarray(handle['data'][name], dtype=np.float32)
                           for name in io23.RAW23], axis=1)
        available = np.isfinite(matrix)
        np.testing.assert_array_equal(actual['a'], available)
        np.testing.assert_array_equal(actual['v'], np.where(available, matrix, 0.))
        rate = float(np.asarray(handle['data/ip'].attrs['SampleRate']).item())
        start = float(np.asarray(handle['data/ip'].attrs['StartTime']).item())
        np.testing.assert_array_equal(actual['times'], start + np.arange(len(matrix)) / rate)
        flag = bool(np.asarray(handle['meta/IsDisrupt']).item())
        down = float(np.asarray(handle['meta/DownTime']).item()) if flag else None
        assert io23.read_outcome(device, shot) == (flag, down)
        for field, key in [('ip', 'planning_ip'), ('Bt', 'planning_Bt')]:
            raw = np.asarray(handle['data'][field][:100], dtype=np.float32)
            np.testing.assert_array_equal(actual[key][:100], raw)
            assert len(raw) == 100

    disruptive, down = io23.read_outcome(device, shot)
    horizon = (io23.effective_horizon_ms(shot, io23.advanced_times())[0]
               if device == 'J-TEXT' else 125.)
    for h in (horizon, 125.):
        got = io23.labels_at(actual['times'], disruptive, down, h)
        if disruptive:
            ttd = (down - actual['times']) * 1000
            valid = ttd >= -1e-6
            expected = ((ttd <= h + 1e-6) & valid).astype(np.float32), valid
        else:
            expected = np.zeros(len(matrix), np.float32), np.ones(len(matrix), bool)
        for left, right in zip(got, expected, strict=True):
            np.testing.assert_array_equal(left, right)


def test_fixed_roles_and_cohort_match_canonical_manifests():
    roles = json.loads(io23.JTEXT_INDEX.read_text())['split_shot_ids']
    assert io23.source_roles() == roles
    assert {k: len(v) for k, v in roles.items()} == {'train': 1353, 'val': 160, 'test': 220}
    assert io23.cohort_ids('J-TEXT') == roles['test']
    manifest = json.loads(io23.EAST_COHORT.read_text())['test']
    assert io23.cohort_ids('EAST') == sorted(manifest['disruption_shots'] + manifest['non_disruption_shots'])
    assert len(io23.cohort_ids('EAST')) == 360
    advanced = json.loads(io23.JTEXT_ADVANCED_TIME_PATH.read_text())
    assert io23.advanced_times() == {int(s): float(v) for s, v in advanced.items()}


def test_horizon_and_label_boundaries():
    assert io23.effective_horizon_ms(7, {}) == (25., None, True)
    assert io23.effective_horizon_ms(7, {7: 10.}) == (25., 10., False)
    assert io23.effective_horizon_ms(7, {7: 40.}) == (40., 40., False)
    times = np.array([.874, .875, 1., 1.0000000005, 1.001])
    y, valid = io23.labels_at(times, True, 1., 125.)
    np.testing.assert_array_equal(y, [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(valid, [1, 1, 1, 1, 0])
    safe_y, safe_valid = io23.labels_at(times, False, None, 125.)
    assert not safe_y.any() and safe_valid.all()
    with pytest.raises(ValueError):
        io23.effective_horizon_ms(7, {7: -1.})


def test_reject_unsupported_device_and_feature():
    with pytest.raises(ValueError):
        io23.read_signals('other', 1)
    with pytest.raises(ValueError):
        io23.read_signals('EAST', 54157, 'shared17')
    with pytest.raises(ValueError):
        io23.cohort_ids('other')


def test_signal_reader_never_touches_meta(monkeypatch):
    original = h5py.File.__getitem__

    def checked(handle, key):
        assert not str(key).startswith('meta')
        return original(handle, key)

    monkeypatch.setattr(h5py.File, '__getitem__', checked)
    io23.read_signals('EAST', 54157)


def test_synthetic_missing_values_and_alignment_guard(tmp_path, monkeypatch):
    path = tmp_path / 'shot.hdf5'
    with h5py.File(path, 'w') as handle:
        data = handle.create_group('data')
        for name in io23.RAW23:
            values = np.array([1., np.nan, 0.], dtype=np.float32)
            ds = data.create_dataset(name, data=values)
            ds.attrs['SampleRate'] = 1000.
            ds.attrs['StartTime'] = .5
    monkeypatch.setattr(io23, 'path_for', lambda device, shot: path)
    result = io23.read_signals('J-TEXT', 9)
    assert result['v'].shape == (3, 23)
    assert not result['a'][1].any()
    assert result['a'][2].all()  # A physical zero remains observed.
    assert not result['v'][1].any()
    assert np.isnan(result['planning_ip'][1])  # raw planning is not cleaned.
    np.testing.assert_array_equal(result['times'], [.5, .501, .502])
    with h5py.File(path, 'r+') as handle:
        handle['data/Bt'].attrs['StartTime'] = .6
    with pytest.raises(ValueError, match='unaligned StartTime'):
        io23.read_signals('J-TEXT', 9)


def test_invalid_disruption_timestamp_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'shot.hdf5'
    with h5py.File(path, 'w') as handle:
        meta = handle.create_group('meta')
        meta.create_dataset('IsDisrupt', data=[1])
        meta.create_dataset('DownTime', data=[np.nan])
    monkeypatch.setattr(io23, 'path_for', lambda device, shot: path)
    with pytest.raises(ValueError, match='invalid disruption timestamp'):
        io23.read_outcome('EAST', 9)
    with h5py.File(path, 'r+') as handle:
        handle['meta/IsDisrupt'][0] = 0
    assert io23.read_outcome('EAST', 9) == (False, None)


def test_duplicate_east_cohort_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'cohort.json'
    path.write_text('{"test":{"disruption_shots":[1],"non_disruption_shots":[1]}}')
    monkeypatch.setattr(io23, 'EAST_COHORT', path)
    with pytest.raises(ValueError, match='distinct locked 360-shot cohort'):
        io23.cohort_ids('EAST')
