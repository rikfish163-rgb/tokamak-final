import subprocess
import sys
import json

import h5py
import numpy as np
import pytest

from experiments.risk_aligned_online import simple_data as D


def test_new_training_import_has_no_legacy_model_dependency():
    code = "import sys; from experiments.risk_aligned_online import simple_train; assert not any(n.startswith(('experiments.fast_online_idea', 'experiments.transfer_methods_2026')) for n in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize(('device', 'shot'), [('J-TEXT', 1051510), ('EAST', 54157)])
def test_new_data_matches_physical_units_and_single_label_contract(device, shot):
    from experiments.risk_aligned_online import io23

    a = D.read_shot(device, shot)
    with h5py.File(io23.path_for(device, shot), 'r') as handle:
        raw = np.stack([np.asarray(handle['data'][name], np.float32)
                        for name in io23.RAW23], axis=1)
        flag = bool(np.asarray(handle['meta/IsDisrupt']).item())
        down = float(np.asarray(handle['meta/DownTime']).item()) if flag else None
    available = np.isfinite(raw)
    expected = np.where(available, raw, 0.)
    if device == 'J-TEXT':
        expected[:, io23.RAW23.index('ip')] *= 1000.
    else:
        available[:, io23.RAW23.index('MNM')] = False
    expected[~available] = 0.
    np.testing.assert_array_equal(a['values'], expected)
    np.testing.assert_array_equal(a['available'], available)
    assert not {'y', 'down', 'is_disrupt'} & a.keys()
    a = D.attach_outcome(a, device, D.advanced_times())
    assert (a['is_disrupt'], a['down']) == (flag, down)
    advanced = json.loads(io23.JTEXT_ADVANCED_TIME_PATH.read_text())
    horizon = max(25., float(advanced.get(str(shot), 25.))) if device == 'J-TEXT' else 125.
    if flag:
        lead = (down - a['times']) * 1000
        valid = lead >= -1e-6
        y = ((lead <= horizon + 1e-6) & valid).astype(np.float32)
    else:
        valid, y = np.ones(len(raw), bool), np.zeros(len(raw), np.float32)
    np.testing.assert_array_equal(a['valid'], valid)
    np.testing.assert_array_equal(a['y'], y)
    assert 'y125' not in a
    assert a['horizon'] == horizon


def test_window_cannot_read_later_values():
    s = dict(values=np.ones((20, 23), np.float32), available=np.ones((20, 23), bool))
    before = D.window(s, 7, 16)
    s['values'][8:] = 12345
    after = D.window(s, 7, 16)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert not before[1][:8].any()
