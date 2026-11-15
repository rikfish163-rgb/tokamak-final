import random

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030.source_episodes import (
    sample_indices,
    select_records,
)


def make_shot(n=200, *, disruptive=False, down=None, horizon=25.0, shot_id=123):
    times = np.arange(n, dtype=np.float64) / 1000.0
    if disruptive:
        if down is None:
            down = n / 1000.0
        lead = (down - times) * 1000.0
        valid = lead >= -1e-6
        y = ((lead <= horizon + 1e-6) & valid).astype(np.float32)
    else:
        valid = np.ones(n, dtype=bool)
        y = np.zeros(n, dtype=np.float32)
    return dict(
        shot=shot_id,
        device='J-TEXT',
        values=np.ones((n, 23), dtype=np.float32),
        available=np.ones((n, 23), dtype=bool),
        times=times,
        y=y,
        valid=valid,
        is_disrupt=disruptive,
        down=down,
        horizon=horizon,
    )


def test_clock_partial_last_interval_peak_tie_and_population():
    shot = make_shot(192)
    logits = np.zeros(192)
    logits[152] = 999  # maturity, not a query
    logits[153] = logits[177] = 20  # first eligible tie wins
    logits[178] = 30  # belongs to the next origin, not the first
    logits[191] = 40  # last point of the partial second interval
    records, population = select_records(shot, logits)
    assert [(r['origin'], r['maturity'], r['query']) for r in records] == [
        (127, 152, 153), (152, 177, 191),
    ]
    assert records[0] == dict(
        shot=123, origin=127, maturity=152, query=153, group='safe',
        label=0.0, lead_ms=None, disruptive=False, feedback_observed=True,
    )
    assert population == dict(
        samples=192, origins=3,
        before=dict(safe=2, danger=0, early=0),
        after=dict(safe=2, danger=0, early=0),
        excluded=dict(no_query=1, unsupported=0),
    )


def test_runner_collect_accepts_selector_records_and_population_without_io(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from experiments.risk_aligned_online.context_vector_20261030 import source_run

    shot = make_shot(154)
    signal_shape = (1, 128, 23)
    net = SimpleNamespace(base=SimpleNamespace(center=torch.zeros(23)))
    written = {}
    monkeypatch.setattr(source_run.io23, 'path_for', lambda device, sid: Path('/unused'))
    monkeypatch.setattr(source_run, 'sha', lambda path: 'fixed')
    monkeypatch.setattr(source_run.simple_data, 'read_shot', lambda device, sid: shot)
    monkeypatch.setattr(source_run.simple_data, 'attach_outcome', lambda raw, device, advanced: raw)
    monkeypatch.setattr(source_run, 'source_scores', lambda model, signal: np.zeros(154))
    monkeypatch.setattr(source_run, 'batch', lambda signal, ends, device: (
        torch.ones(signal_shape), torch.ones(signal_shape, dtype=torch.bool)))
    monkeypatch.setattr(source_run, 'observation', lambda model, signal, origin: dict(
        support=torch.tensor([True]), context=torch.zeros(1, 358),
        x=torch.ones(signal_shape), available=torch.ones(signal_shape, dtype=torch.bool),
        reference=torch.zeros(1, 64)))
    monkeypatch.setattr(source_run, '_durable_json', lambda path, value: written.update({path.name: value}))

    data, records, groups, hashes = source_run.collect(net, [123], {}, Path('/unused'), 'train')
    assert (records[0]['shot'], records[0]['group'], records[0]['label']) == (123, 'safe', 0.)
    assert (records[0]['origin'], records[0]['maturity'], records[0]['query']) == (127, 152, 153)
    assert data['context'].shape == (1, 358) and data['later_x'].shape == signal_shape
    assert groups['safe'] == dict(shots=1, windows=1) and hashes == {'123': 'fixed'}
    assert written['train_population.json']['shots'][0]['shot'] == 123
    assert written['train_population.json']['shots'][0]['before']['safe'] == 1
    assert written['train_records.json'] == records


@pytest.mark.parametrize('n,origins', [(127, 0), (128, 1), (153, 2)])
def test_short_shot_counts_nominal_origins_without_a_query(n, origins):
    records, population = select_records(make_shot(n), np.zeros(n))
    assert records == []
    assert population['origins'] == origins
    assert population['excluded'] == dict(no_query=origins, unsupported=0)


def test_last_eligible_point_and_valid_mask():
    shot = make_shot(178)
    shot['valid'][153:177] = False
    logits = np.zeros(178)
    logits[153] = 100
    records, _ = select_records(shot, logits)
    assert [(r['origin'], r['query']) for r in records] == [(127, 177)]
    shot['valid'][177] = False
    records, population = select_records(shot, logits)
    assert not records
    assert population['excluded']['no_query'] == population['origins']


@pytest.mark.parametrize(
    'offset, expected_query, expected_lead',
    [(0.0, 189, 11.0), (4e-13, 189, 11.0), (6e-13, 190, 10.000000001)],
)
def test_disruptive_strict_rounded_ten_ms_boundary_and_no_thirty_ms_cut(
    offset, expected_query, expected_lead
):
    shot = make_shot(205, disruptive=True, down=0.2 + offset)
    logits = np.zeros(205)
    logits[190], logits[189], logits[191] = 100, 99, 101
    records, _ = select_records(shot, logits)
    record = next(r for r in records if r['origin'] == 152)
    assert record['query'] == expected_query
    assert record['lead_ms'] == expected_lead
    assert record['group'] == 'danger' and record['label'] == 1.0


def test_query_label_uses_source_horizon_and_disruptive_early_is_not_safe():
    shot = make_shot(245, disruptive=True, down=.24, horizon=25)
    logits = np.zeros(245)
    logits[153], logits[228] = 10, 20
    records, _ = select_records(shot, logits)
    early = next(r for r in records if r['origin'] == 127)
    danger = next(r for r in records if r['query'] == 228)
    assert (early['group'], early['label'], early['lead_ms']) == ('early', 0.0, 87.0)
    assert (danger['group'], danger['label'], danger['lead_ms']) == ('danger', 1.0, 12.0)
    long_horizon = make_shot(190, disruptive=True, down=.22, horizon=45)
    long_logits = np.zeros(190)
    long_logits[177] = 5
    long_records, _ = select_records(long_horizon, long_logits)
    row = next(r for r in long_records if r['origin'] == 127)
    assert (row['query'], row['group'], row['label'], row['lead_ms']) == (177, 'danger', 1.0, 43.0)


def test_lead_uses_timestamps_not_index_spacing():
    shot = make_shot(180, disruptive=True, down=.3, horizon=75)
    shot['times'] = np.arange(180, dtype=float) * .0015
    lead = (shot['down'] - shot['times']) * 1000
    shot['valid'] = lead >= -1e-6
    shot['y'] = ((lead <= shot['horizon'] + 1e-6) & shot['valid']).astype(np.float32)
    logits = np.zeros(180)
    logits[153] = 2
    records, _ = select_records(shot, logits)
    row = next(r for r in records if r['origin'] == 127)
    assert (row['query'], row['lead_ms'], row['label']) == (153, 70.5, 1.0)


@pytest.mark.parametrize('case', ['mode_only', 'unmatched_channel', 'origin_nan', 'arrival_inf', 'too_late'])
def test_unsupported_feedback_is_excluded(case):
    shot = make_shot(154)
    shot['available'][:] = False
    if case == 'mode_only':
        shot['available'][127:153, 10] = True
    elif case == 'unmatched_channel':
        shot['available'][127, 0] = True
        shot['available'][128:153, 1] = True
    elif case == 'origin_nan':
        shot['available'][127:153, 0] = True
        shot['values'][127, 0] = np.nan
    elif case == 'arrival_inf':
        shot['available'][127:153, 0] = True
        shot['values'][128:153, 0] = np.inf
    else:
        shot['available'][127, 0] = True
        shot['available'][153, 0] = True
    records, population = select_records(shot, np.zeros(154))
    assert records == []
    assert population['excluded'] == dict(no_query=1, unsupported=1)


def test_a_single_finite_available_arrival_at_maturity_provides_support():
    shot = make_shot(154)
    shot['available'][:] = False
    shot['available'][127, 22] = shot['available'][152, 22] = True
    records, population = select_records(shot, np.zeros(154))
    assert len(records) == 1 and records[0]['feedback_observed'] is True
    assert population['before']['safe'] == population['after']['safe'] == 1


def test_a_single_finite_available_arrival_immediately_after_origin_provides_support():
    shot = make_shot(154)
    shot['available'][:] = False
    shot['available'][127, 0] = shot['available'][128, 0] = True
    records, _ = select_records(shot, np.zeros(154))
    assert len(records) == 1 and records[0]['feedback_observed'] is True


def test_support_agrees_with_actual_context_observation_on_cpu():
    from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
    from experiments.risk_aligned_online.context_vector_20261030.observation import observation
    from experiments.risk_aligned_online.simple23 import Config, Simple23

    torch.manual_seed(131)
    net = MatrixAttention(Simple23(Config(), torch.zeros(23), torch.ones(23)))
    shot = make_shot(154)
    shot['available'][127:] = False
    shot['available'][127, 10] = shot['available'][128, 10] = True
    signal = {key: shot[key] for key in ('values', 'available')}
    for expected in (False, True):
        records, population = select_records(shot, np.zeros(154))
        assert bool(records) is expected
        assert observation(net, signal, 127)['support'].item() is expected
        assert population['excluded']['unsupported'] == int(not expected)
        shot['available'][127, 22] = shot['available'][152, 22] = True


def test_each_group_capped_by_even_origin_spacing_not_logit_ranking():
    safe = make_shot(453)
    logits = np.arange(453, dtype=float)  # later origins have larger logits
    records, population = select_records(safe, logits, max_per_group=3)
    assert [r['origin'] for r in records] == [127, 277, 402]
    assert population['before'] == dict(safe=12, danger=0, early=0)
    assert population['after'] == dict(safe=3, danger=0, early=0)
    assert population['excluded'] == dict(no_query=2, unsupported=0)

    disruptive = make_shot(1250, disruptive=True, down=1.2, horizon=400)
    records, population = select_records(disruptive, np.arange(1250, dtype=float))
    assert population['before']['early'] > 8 and population['before']['danger'] > 8
    assert population['after'] == dict(safe=0, danger=8, early=8)
    assert [r['origin'] for r in records] == sorted(r['origin'] for r in records)
    records, population = select_records(disruptive, np.arange(1250, dtype=float), max_per_group=3)
    assert [r['origin'] for r in records if r['group'] == 'early'] == [127, 427, 727]
    assert [r['origin'] for r in records if r['group'] == 'danger'] == [752, 952, 1152]
    assert population['after'] == dict(safe=0, danger=3, early=3)


def sample_population():
    return ([dict(shot=1, group='safe')] + [dict(shot=2, group='safe')] * 8
            + [dict(shot=3, group='danger')] + [dict(shot=4, group='danger')] * 3
            + [dict(shot=5, group='early')] * 20)


def test_balanced_sampling_is_seeded_shot_uniform_then_window_uniform():
    records = sample_population()
    a = sample_indices(records, np.random.default_rng(131), per_class=6000)
    b = sample_indices(records, np.random.default_rng(131), per_class=6000)
    assert a == b and len(a) == 12000
    assert all(records[i]['group'] == 'safe' for i in a[:6000])
    assert all(records[i]['group'] == 'danger' for i in a[6000:])
    assert abs(sum(records[i]['shot'] == 1 for i in a[:6000]) / 6000 - .5) < .035
    assert abs(sum(records[i]['shot'] == 3 for i in a[6000:]) / 6000 - .5) < .035
    within_shot = [i for i in a[:6000] if records[i]['shot'] == 2]
    assert all(abs(within_shot.count(i) / len(within_shot) - 1 / 8) < .025 for i in range(1, 9))
    assert len(sample_indices(records, random.Random(1))) == 12


class LastChoice:
    def choice(self, options):
        return options[-1]


def test_sampling_is_with_replacement_and_never_uses_early():
    records = sample_population()
    assert sample_indices(records, LastChoice(), per_class=3) == [8, 8, 8, 12, 12, 12]


@pytest.mark.parametrize('bad', [0, -1, 9, 1.5, True, None])
def test_invalid_group_cap_is_rejected(bad):
    with pytest.raises(ValueError):
        select_records(make_shot(), np.zeros(200), max_per_group=bad)


@pytest.mark.parametrize('field, value', [
    ('device', 'EAST'), ('shot', True), ('is_disrupt', 1), ('down', 0.5),
    ('times', np.zeros(200)), ('times', np.full(200, np.nan)),
    ('values', np.zeros((200, 22))), ('values', np.full((200, 23), 'x')),
    ('available', np.zeros((200, 23), dtype=int)),
    ('valid', np.ones(199, dtype=bool)), ('valid', np.ones(200)),
    ('y', np.full(200, np.nan)), ('y', np.full(200, 2)),
    ('y', np.ones(200)),
])
def test_invalid_shot_schema_rejected(field, value):
    shot = make_shot()
    shot[field] = value
    with pytest.raises(ValueError):
        select_records(shot, np.zeros(200))


def test_missing_fields_invalid_terminal_and_non_cpu_inputs_rejected():
    shot = make_shot()
    del shot['available']
    with pytest.raises(ValueError):
        select_records(shot, np.zeros(200))
    shot = make_shot(disruptive=True)
    shot['down'] = np.nan
    with pytest.raises(ValueError):
        select_records(shot, np.zeros(200))
    with pytest.raises(ValueError):
        select_records(make_shot(), torch.empty(200, device='meta'))
    shot = make_shot()
    shot['values'] = torch.empty((200, 23), device='meta')
    with pytest.raises(ValueError):
        select_records(shot, np.zeros(200))


def test_cpu_tensors_are_supported_without_a_gradient_connection():
    shot = make_shot()
    shot['values'] = torch.tensor(shot['values'])
    logits = torch.zeros(200, requires_grad=True)
    records, _ = select_records(shot, logits)
    assert records and all(isinstance(r['query'], int) for r in records)


@pytest.mark.parametrize('logits', [np.zeros((200, 1)), np.zeros(199),
                                    np.full(200, np.inf), np.full(200, np.nan),
                                    np.full(200, 'x')])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError):
        select_records(make_shot(), logits)


@pytest.mark.parametrize('records, rng, per_class', [
    ([], np.random.default_rng(1), 1),
    ([dict(shot=1, group='safe'), dict(shot=2, group='early')], np.random.default_rng(1), 1),
    ([dict(shot=1, group='danger')], np.random.default_rng(1), 1),
    ([dict(shot=1, group='safe'), dict(shot=1, group='danger')], np.random.default_rng(1), 1),
    ([dict(shot=1, group='safe'), dict(shot=2, group='unknown')], np.random.default_rng(1), 1),
    ([dict(group='safe')], np.random.default_rng(1), 1),
    (sample_population(), object(), 1),
    (sample_population(), np.random.default_rng(1), 0),
    (sample_population(), np.random.default_rng(1), True),
    (sample_population(), np.random.default_rng(1), 1.5),
])
def test_invalid_sampling_inputs_rejected(records, rng, per_class):
    with pytest.raises(ValueError):
        sample_indices(records, rng, per_class=per_class)
