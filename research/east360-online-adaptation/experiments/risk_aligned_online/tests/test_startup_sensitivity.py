"""These interventions diagnose frozen-model sensitivity, not adaptation gains."""
import numpy as np
import pytest

from experiments.risk_aligned_online.startup_sensitivity import (
    channel_intervention, history_intervention, summarize_records,
)


def sample():
    x = np.arange(128*23, dtype=np.float32).reshape(128, 23)
    a = np.ones(x.shape, dtype=bool)
    a[:10] = False
    x[:10] = 777.
    center = np.arange(23, dtype=np.float32)/10
    return x, a, center


@pytest.mark.parametrize('method', ['missing', 'source_center'])
def test_only_selected_channel_and_observed_entries_change(method):
    x, a, c = sample()
    original_x, original_a = x.copy(), a.copy()
    y, b = channel_intervention(x, a, c, 12, method)
    other = np.arange(23) != 12
    np.testing.assert_array_equal(y[:, other], x[:, other])
    np.testing.assert_array_equal(b[:, other], a[:, other])
    np.testing.assert_array_equal(y[~a[:, 12], 12], x[~a[:, 12], 12])
    if method == 'missing':
        assert not b[:, 12].any()
        np.testing.assert_array_equal(y, x)
    else:
        np.testing.assert_array_equal(b, a)
        np.testing.assert_array_equal(y[a[:, 12], 12], c[12])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(a, original_a)


def test_missing_channel_remains_identity_not_a_synthetic_measurement():
    x, a, c = sample()
    a[:, 2] = False
    for method in ('missing', 'source_center'):
        y, b = channel_intervention(x, a, c, 2, method)
        np.testing.assert_array_equal(y, x)
        np.testing.assert_array_equal(b, a)


def test_history_truncation_keeps_original_endpoint_position_and_values():
    x, a, _ = sample()
    y, b = history_intervention(x, a, 8)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(b[-8:], a[-8:])
    assert not b[:-8].any()
    y, b = history_intervention(x, a, 128)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize('bad', ['channel', 'method', 'mask', 'nan', 'shape', 'center'])
def test_invalid_interventions_fail_explicitly(bad):
    x, a, c = sample()
    channel, method = 12, 'missing'
    if bad == 'channel': channel = True
    if bad == 'method': method = 'invented'
    if bad == 'mask': a = a.astype(float)
    if bad == 'nan': x[100, 12] = np.nan
    if bad == 'shape': x = x[:-1]
    if bad == 'center': c[0] = np.inf
    with pytest.raises(ValueError): channel_intervention(x, a, c, channel, method)


@pytest.mark.parametrize('length', [0, 129, True, 3.5])
def test_invalid_history_lengths_fail(length):
    x, a, _ = sample()
    with pytest.raises(ValueError): history_intervention(x, a, length)


def test_summary_reports_selected_endpoint_crossings_not_whole_shot_rescues():
    records = [dict(device='EAST', shot=1, group='safe_early_fp', tag='first_alarm',
                    baseline=.9, variants={'missing:ne_nG': .6, 'source_center:ne_nG': .8}),
               dict(device='EAST', shot=2, group='safe_early_fp', tag='first_alarm',
                    baseline=.8, variants={'missing:ne_nG': .7, 'source_center:ne_nG': .9})]
    rows = summarize_records(records, .75)
    row = next(r for r in rows if r['variant'] == 'missing:ne_nG')
    assert row['n_windows'] == row['n_shots'] == row['crossing_down'] == 2
    assert row['mean_delta'] == pytest.approx(-.2)
    assert row['crossing_up'] == 0
    assert not set(row) & {'tp', 'fp', 'tpr', 'fpr', 'rescued'}
