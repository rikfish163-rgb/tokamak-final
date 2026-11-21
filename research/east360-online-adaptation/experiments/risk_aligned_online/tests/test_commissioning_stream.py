"""Outcome-free commissioning order and explicit prefix quality failures."""
import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_stream as C


def signals(current, field=2., rate=1000., n=180):
    return dict(planning_ip=np.full(n, current, np.float64),
                planning_Bt=np.full(n, field, np.float64), rate=rate)


def test_order_uses_actual_first_100_ms_and_never_outcomes():
    raw = {1: signals(300_000.), 2: signals(100_000.), 3: signals(200_000.)}
    raw[2]['planning_ip'][100:] = 900_000.
    order, rejected = C.build_stream([1, 2, 3], raw.__getitem__)
    assert [r['shot'] for r in order] == [2, 3, 1]
    assert not rejected
    assert order[0]['prefix_samples'] == 100
    assert order[0]['initial_ip_A'] == 100_000.
    assert order[0]['proxy'] == 1.
    assert all('is_disrupt' not in r and 'down' not in r for r in order)


def test_nominal_noninteger_rate_does_not_mean_100_samples():
    order, _ = C.build_stream([1], lambda sid: signals(200_000., rate=1000.25))
    assert order[0]['prefix_samples'] == 101


def test_field_is_secondary_load_proxy_not_a_density_or_damage_label():
    raw = {1: signals(100_000., 3.), 2: signals(100_000., 1.)}
    order, _ = C.build_stream([1, 2], raw.__getitem__)
    assert [r['shot'] for r in order] == [2, 1]
    assert [r['proxy'] for r in order] == [1., 1.]
    assert order[0]['force_proxy_raw'] == 1.


@pytest.mark.parametrize('bad', [signals(0.), signals(100_000., n=80),
                                signals(100_000., field=float('nan'))])
def test_invalid_prefix_is_recorded_not_imputed_or_silently_skipped(bad):
    raw = {1: bad, 2: signals(100_000.)}
    order, rejected = C.build_stream([1, 2], raw.__getitem__)
    assert [r['shot'] for r in order] == [2]
    assert rejected[0]['shot'] == 1 and rejected[0]['reason']


def test_reader_io_errors_are_not_converted_to_quality_rejections():
    def broken(sid):
        raise OSError('disk error')
    with pytest.raises(OSError, match='disk error'):
        C.build_stream([1], broken)


def test_quantile_pilot_is_fixed_before_labels_and_spans_operating_range():
    order = [dict(shot=i, proxy=float(i)) for i in range(1, 12)]
    pilot = C.quantile_pilot(order, 4)
    assert [r['shot'] for r in pilot] == [1, 4, 8, 11]
    assert [r['full_stream_rank'] for r in pilot] == [1, 4, 8, 11]
    assert order[0] == dict(shot=1, proxy=1.)


@pytest.mark.parametrize('ids', [[1, 1], [0], [True], []])
def test_invalid_roster_rejected(ids):
    with pytest.raises(ValueError):
        C.build_stream(ids, lambda sid: signals(100_000.))


@pytest.mark.parametrize('count', [0, 1, 12, True])
def test_invalid_pilot_size_rejected(count):
    with pytest.raises(ValueError):
        C.quantile_pilot([dict(shot=i) for i in range(11)], count)


def test_composition_reads_outcomes_only_after_schedule_is_locked(tmp_path, monkeypatch):
    import json
    source = tmp_path / 'source_roster.json'
    source.write_text('{}')
    dest = tmp_path / 'stream'
    order, _ = C.build_stream([1, 2], lambda sid: signals(100_000. * sid))
    calls = []
    def outcome(device, sid):
        assert (dest / 'order.json').exists() and (dest / 'roster.json').exists()
        assert [r['shot'] for r in json.loads((dest / 'order.json').read_text())] == [1, 2]
        calls.append(sid)
        return sid == 2, .5 if sid == 2 else None
    monkeypatch.setattr(C.io23, 'read_outcome', outcome)
    C.publish(dest, order, dict(selection_protocol={}, selection='original QC'),
              source, [], pilot=False)
    assert calls == [1, 2]
    composition = json.loads((dest / 'composition.json').read_text())
    assert composition['full'] == dict(n=2, n_disrupt=1, n_safe=1)
    assert composition['outcome_reads_after_order_and_roster_locked']


def test_all_invalid_prefixes_cannot_publish_an_empty_experiment(tmp_path):
    order, rejected = C.build_stream([1], lambda sid: signals(0.))
    assert order == [] and len(rejected) == 1
    with pytest.raises(ValueError):
        C.quantile_pilot(order, 2)
