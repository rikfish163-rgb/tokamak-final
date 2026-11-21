"""CPU-only stream contracts; synthetic HDF5 lives on the mounted project."""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from . import stream_manifest as sm
from .. import io23
from ..io23 import RAW23


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory(prefix='.stream-test-', dir=Path(__file__).parent) as d:
        yield Path(d)


def shot_file(root, shot, *, disruptive=0, ip=200000., bt=2., length=150):
    path = root / f'{shot}.hdf5'
    with h5py.File(path, 'w') as f:
        for name in RAW23:
            value = ip if name == 'ip' else bt if name == 'Bt' else 0.
            ds = f.create_dataset('data/' + name, data=np.full(length, value))
            ds.attrs['SampleRate'] = 1000
            ds.attrs['StartTime'] = 3.
        f['data/CIII'][0] = -10.
        f['data/CIII'][1] = np.nan
        f['data/MNM'][:] = 42.
        f.create_dataset('meta/IsDisrupt', data=disruptive)
        f.create_dataset('meta/DownTime', data=3. + (length-1)/1000.)
    return path


def test_all_root_shots_not_balanced_or_support_filtered(root):
    for sid in [9, 2, 7]:
        shot_file(root, sid)
    rows, audit = sm.build_manifest(roots=[root])
    assert [r['shot'] for r in rows] == [2, 7, 9]
    assert audit['complete_count'] == 3
    assert all('provenance' in r and 'is_disrupt' not in r for r in rows)
    assert audit['manifest_sha256'] == hashlib.sha256(sm.manifest_bytes(rows)).hexdigest()
    json.loads(sm.manifest_bytes(rows))


def test_mask_contract_no_zero_inference(root):
    p = shot_file(root, 2)
    rows, _ = sm.build_manifest(paths=[p])
    counts = rows[0]['provenance']['available_points']
    assert len(counts) == 23
    assert counts[RAW23.index('CIII')] == 148
    assert counts[RAW23.index('MNM')] == 0
    assert counts[RAW23.index('sxr_mean')] == 150
    assert rows[0]['provenance']['sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_proxy_matches_approved_prefix_and_ignores_tail(root):
    p = shot_file(root, 9, ip=100000., bt=2.)
    shot_file(root, 2, ip=200000., bt=2.)
    with h5py.File(p, 'r+') as f:
        f['data/ip'][100:] = 1e9
    rows, _ = sm.build_manifest(roots=[root], order='prefix_ip_bt')
    assert [r['shot'] for r in rows] == [9, 2]
    assert rows[0]['proxy'] == 2.
    assert rows[0]['initial_abs_ip_raw'] == 100000.


def test_outcomes_never_enter_sorting(root):
    for sid in [8, 3]:
        shot_file(root, sid, disruptive=sid % 2)
    original = sm.lock_order
    seen = []
    def checked(signals, order):
        seen.extend(signals)
        assert all('outcome' not in r and 'is_disrupt' not in r for r in signals)
        return original(signals, order)
    with patch.object(sm, 'lock_order', checked):
        rows, _ = sm.build_manifest(roots=[root])
    with h5py.File(root / '8.hdf5', 'r+') as f:
        f['meta/IsDisrupt'][()] = 1
    changed, _ = sm.build_manifest(roots=[root])
    assert [r['shot'] for r in rows] == [r['shot'] for r in changed]
    assert seen


@pytest.mark.parametrize('damage,reason', [
    ('missing_channel', 'missing_channel'), ('time', 'unaligned_StartTime'),
    ('rate', 'sample_rate'), ('missing_outcome', 'missing_outcome'),
    ('bad_outcome', 'invalid_IsDisrupt'), ('bad_down', 'invalid_DownTime'),
    ('shape', 'channel_shape'), ('empty', 'empty_signal'),
])
def test_fail_closed_with_audit(root, damage, reason):
    p = shot_file(root, 2, disruptive=1)
    with h5py.File(p, 'r+') as f:
        if damage == 'missing_channel':
            del f['data/Bt']
        elif damage == 'time':
            f['data/Bt'].attrs['StartTime'] = 4.
        elif damage == 'rate':
            for k in RAW23:
                f['data'][k].attrs['SampleRate'] = 500
        elif damage == 'missing_outcome':
            del f['meta/IsDisrupt']
        elif damage == 'bad_outcome':
            f['meta/IsDisrupt'][()] = 2
        elif damage == 'bad_down':
            f['meta/DownTime'][()] = np.nan
        elif damage in ('shape', 'empty'):
            del f['data/ip']
            d = f.create_dataset('data/ip', data=np.zeros((2, 2)) if damage == 'shape' else [])
            d.attrs['SampleRate'], d.attrs['StartTime'] = 1000, 3.
    audit = sm.inventory(paths=[p])
    assert reason in str(audit['errors'])
    with pytest.raises(sm.ManifestError) as exc:
        sm.build_manifest(paths=[p])
    assert exc.value.audit['errors']


def test_duplicates_rejected_even_identical(root):
    p = shot_file(root, 2)
    for paths in [[p, p], [p, shot_file(root / make_dir(root, 'copy'), 2)]]:
        with pytest.raises(sm.ManifestError) as exc:
            sm.build_manifest(paths=paths)
        assert 'duplicate' in str(exc.value.audit['errors'])


def make_dir(root, name):
    (root / name).mkdir()
    return name


def test_explicit_list_missing_and_roster_mismatch(root):
    shot_file(root, 2)
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(roots=[root], shots=[2, 3])
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(roots=[root], shots=[2, 2])
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(paths=[root / '999.hdf5'])
    rows, _ = sm.build_manifest(roots=[root], shots=[2])
    assert [r['shot'] for r in rows] == [2]


def test_short_prefix_only_fails_proxy_mode(root):
    shot_file(root, 2, length=50)
    sm.build_manifest(roots=[root])
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(roots=[root], order='prefix_ip_bt')


def test_corrupt_file_and_empty_root_rejected(root):
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(roots=[root])
    p = root / '2.hdf5'
    p.touch()
    assert sm.inventory(paths=[p])['readable_count'] == 0
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(paths=[p])


def test_cli_stdout_only_and_listdict_schema(root, capsys):
    shot_file(root, 2)
    assert sm.main(['--root', str(root)]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)[0]['shot'] == 2
    assert 'manifest_sha256' in json.loads(out.err)
    assert sm.main(['--root', str(root), '--inventory', '--offline-outcomes']) == 0
    assert json.loads(capsys.readouterr().out)['offline_outcome_counts']['safe'] == 1


def test_bad_order_rejected(root):
    shot_file(root, 2)
    with pytest.raises(ValueError, match='order'):
        sm.build_manifest(roots=[root], order='label')


def test_late_down_is_audited_without_terminal_slack_or_exclusion(root):
    for sid, flag in [(2, 0), (3, 1)]:
        p = shot_file(root, sid, disruptive=flag)
        with h5py.File(p, 'r+') as f:
            f['meta/DownTime'][()] = 3.151
    audit = sm.inventory(roots=[root])
    assert audit['qc_eligible_count'] == 2
    assert not audit['errors']
    assert 'disruption_time_audit' not in audit['rows'][0]
    timing = audit['rows'][1]['disruption_time_audit']
    assert timing['unsampled_disrupt_gap_ms'] == pytest.approx(2.)
    assert timing['gap_from_exclusive_end_ms'] == pytest.approx(1.)
    assert timing['timely_evaluation_points'] > 0
    rows, audit = sm.build_manifest(roots=[root])
    assert [r['shot'] for r in rows] == [2, 3]
    assert audit['roster']['shot_ids'] == [2, 3]
    assert audit['roster']['data_root'] == str(root)
    assert isinstance(audit['roster']['selection'], str)
    assert audit['roster']['selection_protocol']['label_blind']


def test_explicit_signal_qc_exclusion_records_reason(root):
    shot_file(root, 2)
    p = shot_file(root, 3)
    with h5py.File(p, 'r+') as f:
        for k in RAW23:
            f['data'][k].attrs['SampleRate'] = 999.9
    rows, audit = sm.build_manifest(roots=[root], compatible_only=True)
    assert [r['shot'] for r in rows] == [2]
    assert audit['roster']['rejected'][0]['shot'] == 3


def test_roster_cli_and_errors(root, capsys):
    shot_file(root, 2)
    assert sm.main(['--root', str(root), '--emit', 'roster']) == 0
    output = capsys.readouterr()
    roster = json.loads(output.out)
    assert roster['shot_ids'] == [2]
    assert roster['order_sha256'] == json.loads(output.err)['manifest_sha256']
    assert sm.main(['--root', str(root / 'absent')]) == 2
    assert 'missing_root' in capsys.readouterr().err


def test_safe_timestamp_is_not_a_disruptive_timestamp(root):
    for sid, flag in [(2, 0), (3, 1)]:
        p = shot_file(root, sid, disruptive=flag)
        with h5py.File(p, 'r+') as f:
            f['meta/DownTime'][()] = 2.
    audit = sm.inventory(roots=[root])
    assert audit['qc_eligible_count'] == 1
    assert all(e['shot'] == 3 and e['kind'] == 'qc' for e in audit['errors'])
    assert 'no_preterminal_diagnostic_samples' in str(audit['errors'])
    rows, _ = sm.build_manifest(roots=[root], compatible_only=True)
    assert [r['shot'] for r in rows] == [2]


def test_missing_down_is_fatal_not_silently_discarded(root):
    p = shot_file(root, 2)
    with h5py.File(p, 'r+') as f:
        del f['meta/DownTime']
    with pytest.raises(sm.ManifestError) as exc:
        sm.build_manifest(roots=[root], compatible_only=True)
    assert any(e['kind'] == 'outcome' for e in exc.value.audit['errors'])


@pytest.mark.parametrize('down', [0., np.nan, -1., 1000.])
def test_safe_down_placeholders_match_existing_reader(root, down):
    p = shot_file(root, 2, disruptive=0)
    with h5py.File(p, 'r+') as f:
        f['meta/DownTime'][()] = down
    with patch.object(io23, 'path_for', return_value=p):
        assert io23.read_outcome('EAST', 2) == (False, None)
    rows, audit = sm.build_manifest(paths=[p])
    assert rows[0]['shot'] == 2
    assert audit['qc_eligible_count'] == 1
    assert 'disruption_time_audit' not in audit['rows'][0]


def test_exclusive_end_is_not_an_invalid_disruption(root):
    p = shot_file(root, 2, disruptive=1)
    with h5py.File(p, 'r+') as f:
        f['meta/DownTime'][()] = 3.150
    rows, audit = sm.build_manifest(paths=[p])
    assert len(rows) == 1 and not audit['errors']
    timing = audit['rows'][0]['disruption_time_audit']
    assert timing['unsampled_disrupt_gap_ms'] == pytest.approx(1.)
    assert timing['gap_from_exclusive_end_ms'] == pytest.approx(0.)
    assert timing['position'] == 'exclusive_sampling_end'


def test_large_unsampled_tail_is_kept_and_reported_not_hidden(root):
    p = shot_file(root, 2, disruptive=1)
    with h5py.File(p, 'r+') as f:
        f['meta/DownTime'][()] = 3.450
    rows, audit = sm.build_manifest(paths=[p])
    assert len(rows) == 1
    timing = audit['rows'][0]['disruption_time_audit']
    assert timing['timely_evaluation_points'] == 150
    assert timing['timely_label_points_30_125ms'] == 0
    assert timing['unsampled_disrupt_gap_ms'] == pytest.approx(301.)
    assert audit['warnings']


def test_no_timely_opportunity_is_not_a_favorable_outcome_filter(root):
    p = shot_file(root, 2, disruptive=1, length=20)
    with h5py.File(p, 'r+') as f:
        f['meta/DownTime'][()] = 3.020
    rows, audit = sm.build_manifest(paths=[p])
    assert len(rows) == 1
    assert audit['rows'][0]['disruption_time_audit']['timely_evaluation_points'] == 0


def test_disruptive_down_uses_first_value_like_io23(root):
    p = shot_file(root, 2, disruptive=1)
    with h5py.File(p, 'r+') as f:
        del f['meta/DownTime']
        f.create_dataset('meta/DownTime', data=[3.150, np.nan])
    with patch.object(io23, 'path_for', return_value=p):
        assert io23.read_outcome('EAST', 2) == (True, 3.150)
    rows, audit = sm.build_manifest(paths=[p])
    assert len(rows) == 1 and not audit['errors']


@pytest.mark.parametrize('rate', [999.9999999999998, 999.9999999999999, 1000.0000000000001])
def test_only_float_neighbors_accepted_without_rounding(root, rate):
    p = shot_file(root, 2)
    with h5py.File(p, 'r+') as f:
        for k in RAW23:
            f['data'][k].attrs['SampleRate'] = rate
    with patch.object(io23, 'path_for', return_value=p):
        raw = io23.read_signals('EAST', 2)
        assert raw['rate'] == rate
        np.testing.assert_array_equal(raw['times'], 3. + np.arange(150)/rate)
    rows, audit = sm.build_manifest(paths=[p])
    assert rows[0]['provenance']['sample_rate'] == rate
    assert audit['rate_atol_hz'] == io23.RATE_ATOL_HZ == 1e-9


def test_rate_tolerance_is_absolute_not_relative(root):
    p = shot_file(root, 2)
    with h5py.File(p, 'r+') as f:
        for k in RAW23:
            f['data'][k].attrs['SampleRate'] = 1000. + 2e-9
    with pytest.raises(sm.ManifestError):
        sm.build_manifest(paths=[p])


def test_explicit_main_gap_allowance_records_without_discard(root):
    p = shot_file(root, 2, disruptive=1)
    with h5py.File(p, 'r+') as f:
        f['meta/DownTime'][()] = 3.151
    rows, audit = sm.build_manifest(paths=[p], terminal_slack_ms=2)
    assert audit['terminal_slack_ms'] == 2.
    assert audit['roster']['selection_protocol']['terminal_slack_ms'] == 2.
    assert len(rows) == 1
    assert sm.inventory(paths=[p])['terminal_slack_ms'] is None
    with pytest.raises(ValueError):
        sm.inventory(paths=[p], terminal_slack_ms=-1.)
