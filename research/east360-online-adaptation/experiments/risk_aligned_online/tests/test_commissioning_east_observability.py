import numpy as np

from experiments.risk_aligned_online import commissioning_east_observability_20261111 as audit
from experiments.risk_aligned_online.io23 import RAW23


def test_whole_window_is_required_and_missing_channels_are_not_votes():
    values = np.ones((185, len(RAW23)), dtype=np.float32)
    available = np.ones_like(values, dtype=bool)
    available[128:153, [RAW23.index(k) for k in audit.FAMILIES['motion']]] = False
    full = audit.signal_windows(values, available)
    np.testing.assert_array_equal(full['release'], [153])
    assert not full['supported']['motion'][0]
    assert not full['supported']['motion_magnetic'][0]
    assert full['supported']['radiation_electrical'][0]
    incomplete = audit.signal_windows(values[:153], available[:153])
    assert incomplete['release'].size == 0


def test_context_does_not_use_the_arriving_block_and_prefix_is_reproducible():
    values = np.ones((186, len(RAW23)), dtype=np.float32)
    ip = RAW23.index('ip')
    values[120:128, ip] = .9
    values[128:153, ip] = 10.
    available = np.ones_like(values, dtype=bool)
    before = audit.signal_windows(values[:154], available[:154])
    after = audit.signal_windows(values, available)
    assert before['current_trend'].tolist() == [-1]
    for key in ('release', 'current_trend'):
        np.testing.assert_array_equal(before[key], after[key][:1])
    for name, support in before['supported'].items():
        np.testing.assert_array_equal(support, after['supported'][name][:1])
