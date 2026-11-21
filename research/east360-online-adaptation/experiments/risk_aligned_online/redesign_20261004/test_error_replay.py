"""Tests-first CPU contract for sealed-error, class-balanced endpoint sampling."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest

from . import error_replay, inter
from .test_inter import ContextStub, shot


def record(s, *, current=False):
    positive, negative, counts = inter._pools(s)
    return dict(shot=s, positive=positive, negative=negative,
                counts=counts, current=current)


def identities(rows):
    return [(int(s['shot']), end, label, origin) for s, end, label, origin in rows]


def run_sample(records, seed=17):
    return error_replay.sample(records, np.random.default_rng(seed))


def long_shot(sid, *, safe=False, sealed=.1):
    return shot(sid, np.linspace(300. if safe else 125., 0. if safe else 31., 200),
                safe=safe, sealed=sealed)


def test_exact_sampler_api_and_reuses_baseline_counts_and_quotas():
    assert list(inspect.signature(error_replay.sample).parameters) == ['records', 'rng']
    records = [record(long_shot(1)), record(long_shot(2, safe=True)),
               record(long_shot(3), current=True)]
    selected, counts, audit = run_sample(records)
    _, expected_counts, baseline = inter._sample(records, np.random.default_rng(17))
    assert counts == expected_counts
    for key in ('strategy', 'positive', 'negative', 'current_positive',
                'history_positive', 'current_negative', 'history_negative'):
        assert audit[key] == baseline[key]
    assert len(selected) == 64 and audit['positive'] == audit['negative'] == 32
    assert audit['priority_draws'] == audit['uniform_draws'] == 32
    assert audit['priority_source'] == 'sealed_scores'
    assert audit['replay_policy'] == 'error_aware'
    assert [(g['label'], g['origin'], g['priority_draws'], g['uniform_draws'])
            for g in audit['groups']] == [(1, 'current', 8, 8),
                                         (1, 'history', 8, 8),
                                         (0, 'history', 16, 16)]


def test_priority_is_exactly_largest_absolute_label_error_per_group():
    positive = long_shot(1)
    negative = long_shot(2, safe=True)
    positive['_sealed_scores'][:] = np.linspace(.05, .95, 200)
    negative['_sealed_scores'][:] = np.linspace(.05, .95, 200)
    _, _, audit = run_sample([record(positive), record(negative, current=True)])
    pos = [r['end'] for r in audit['selected_endpoints']
           if r['role'] == 'priority' and r['label'] == 1]
    neg = [r['end'] for r in audit['selected_endpoints']
           if r['role'] == 'priority' and r['label'] == 0]
    assert pos == list(range(16))
    assert neg == list(range(199, 183, -1))


def test_uniform_half_remains_the_same_seeded_baseline_draws():
    records = [record(long_shot(1)), record(long_shot(2, safe=True), current=True)]
    selected, _, audit = run_sample(records)
    baseline, _, _ = inter._sample(records, np.random.default_rng(17))
    assert identities(selected[16:32]) == identities(baseline[16:32])
    assert identities(selected[48:64]) == identities(baseline[48:64])
    assert audit['uniform_draws'] == 32


def test_sparse_wrong_safe_peak_is_enriched_relative_to_uniform():
    positive = long_shot(1)
    safe = shot(2, np.linspace(300., 0., 10000), safe=True, sealed=.01)
    safe['_sealed_scores'][7000:7016] = np.linspace(.99, .94, 16)
    records = [record(positive), record(safe, current=True)]
    selected, _, audit = run_sample(records)
    uniform, _, _ = inter._sample(records, np.random.default_rng(17))
    selected_hard = [end for s, end, label, _ in selected
                     if not label and s['shot'] == 2 and 7000 <= end < 7016]
    baseline_hard = [end for s, end, label, _ in uniform
                     if not label and s['shot'] == 2 and 7000 <= end < 7016]
    assert len(selected_hard) >= 16 > len(baseline_hard)
    assert 7000 in selected_hard
    assert audit['positive'] == audit['negative'] == 32


def test_actual_125_labels_and_timely_strata_not_changed():
    positive = shot(1, [200., 126., 125., 80., 30.0001, 30., 0., -1.])
    positive['_sealed_scores'][:] = [.99, .99, .8, .2, .1, .001, .001, .001]
    safe = long_shot(2, safe=True)
    before = positive['y'].copy()
    selected, counts, audit = run_sample([record(safe), record(positive, current=True)])
    assert counts['current'] == dict(positive=3, negative=0,
                                    excluded_late_positive=2, excluded_early_negative=2)
    assert {end for s, end, _, _ in selected if s['shot'] == 1} <= {2, 3, 4}
    assert all(label == 1 for s, _, label, _ in selected if s['shot'] == 1)
    np.testing.assert_array_equal(positive['y'], before)
    assert audit['positive'] == audit['negative'] == 32


@pytest.mark.parametrize('safe', [False, True])
def test_absent_class_keeps_only_real_class_and_halves_its_group(safe):
    selected, counts, audit = run_sample([record(long_shot(1, safe=safe), current=True)])
    label = 0 if safe else 1
    assert len(selected) == 64 and all(row[2] == label for row in selected)
    assert audit['strategy'] == 'single_class'
    assert audit['priority_draws'] == audit['uniform_draws'] == 32
    assert counts['total']['positive' if safe else 'negative'] == 0


def test_short_record_has_no_primary_pool_or_invented_negative():
    selected, counts, audit = run_sample([record(shot(1, [30., 10., 0.]), current=True)])
    assert not selected and audit['strategy'] == 'empty'
    assert audit['priority_draws'] == audit['uniform_draws'] == 0
    assert counts['total']['positive'] == counts['total']['negative'] == 0


def test_insufficient_unique_priority_pool_falls_back_to_uniform_not_weighted_replacement():
    records = [record(shot(1, [125., 31.]), current=True),
               record(long_shot(2, safe=True))]
    selected, _, audit = run_sample(records)
    group = next(g for g in audit['groups'] if g['label'] == 1)
    assert group['requested_priority_draws'] == 16
    assert group['priority_draws'] == 2 and group['uniform_draws'] == 30
    assert group['priority_shortfall'] == 14
    assert audit['priority_draws'] == 18 and audit['uniform_draws'] == 46
    assert len(selected) == 64 and audit['positive'] == audit['negative'] == 32
    assert audit['replacement_draws'] == 64 - audit['unique_windows']


def test_seed_determinism_and_no_global_rng_use_or_record_mutation():
    records = [record(long_shot(1)), record(long_shot(2, safe=True), current=True)]
    before = deepcopy(records)
    global_state = np.random.get_state()
    a, a_counts, a_audit = run_sample(records, 41)
    b, b_counts, b_audit = run_sample(records, 41)
    assert identities(a) == identities(b) and a_counts == b_counts and a_audit == b_audit
    after_state = np.random.get_state()
    assert global_state[0] == after_state[0]
    np.testing.assert_array_equal(global_state[1], after_state[1])
    assert global_state[2:] == after_state[2:]
    for left, right in zip(records, before):
        assert left.keys() == right.keys()
        for key in ('positive', 'negative'):
            np.testing.assert_array_equal(left[key], right[key])
        for key, value in left['shot'].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, right['shot'][key])
            else:
                assert value == right['shot'][key]
    json.dumps(a_audit, allow_nan=False)


def test_sealed_errors_remain_stale_not_current_recomputed_errors():
    safe = long_shot(2, safe=True, sealed=.01)
    safe['_sealed_scores'][77] = .99
    safe['_recomputed_scores'] = np.full(200, .98)
    safe['_recomputed_scores'][77] = .001  # Pretend the model already repaired it.
    records = [record(long_shot(1)), record(safe, current=True)]
    _, _, audit = run_sample(records)
    priorities = [r for r in audit['selected_endpoints']
                  if r['label'] == 0 and r['role'] == 'priority']
    assert priorities[0]['end'] == 77 and priorities[0]['sealed_error'] > .98
    assert audit['priority_source'] == 'sealed_scores'


def test_unknown_future_and_recomputed_fields_are_never_accessed(monkeypatch):
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError('future/recomputed information accessed')

    def forbidden(*args, **kwargs):
        raise AssertionError('sampler read an outcome or diagnostic window')

    monkeypatch.setattr(inter.simple_data, 'attach_outcome', forbidden)
    monkeypatch.setattr(inter.simple_data, 'window', forbidden)
    current = long_shot(2, safe=True)
    current['_future_labels'] = Forbidden()
    current['_recomputed_scores'] = Forbidden()
    selected, _, _ = run_sample([record(long_shot(1)), record(current, current=True)])
    assert {s['shot'] for s, _, _, _ in selected} == {1, 2}


@pytest.mark.parametrize('missing', ['_sealed_scores', 'y', 'is_disrupt'])
def test_unsealed_or_unrevealed_record_rejected_before_rng_draw(missing):
    current = record(long_shot(1), current=True)
    del current['shot'][missing]
    rng = np.random.default_rng(7)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match='sealed|revealed|outcome'):
        error_replay.sample([current], rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('pool', ['positive', 'negative'])
def test_contaminated_late_positive_or_early_disruptive_negative_pool_rejected(pool):
    current = record(shot(1), current=True)
    current[pool] = np.array([4]) if pool == 'positive' else np.array([0])
    with pytest.raises(ValueError, match='pool|strat'):
        run_sample([current])


def test_incorrect_counts_source_domain_and_bad_sealed_scores_rejected():
    current = record(long_shot(1), current=True)
    current['counts']['positive'] += 1
    with pytest.raises(ValueError, match='count'):
        run_sample([current])
    current = record(long_shot(1), current=True)
    current['shot']['device'] = 'J-TEXT'
    with pytest.raises(ValueError, match='EAST'):
        run_sample([current])
    current = record(long_shot(1), current=True)
    current['shot']['_sealed_scores'][0] = np.nan
    with pytest.raises(ValueError, match='sealed'):
        run_sample([current])


def test_empty_input_and_bounded_replay_contract():
    selected, counts, audit = run_sample([])
    assert not selected and counts['total']['positive'] == 0 and not audit['groups']
    records = [record(long_shot(i, safe=True)) for i in range(32)]
    records.append(record(long_shot(100), current=True))
    selected, _, audit = run_sample(records)
    assert len(selected) == 64 and len(audit['selected_endpoints']) == 64
    records.insert(0, record(long_shot(101, safe=True)))
    with pytest.raises(ValueError, match='32|bounded'):
        run_sample(records)


def test_duplicate_shot_identity_and_multiple_current_records_rejected():
    current = record(long_shot(1), current=True)
    with pytest.raises(ValueError, match='duplicate'):
        run_sample([current, record(current['shot'])])
    with pytest.raises(ValueError, match='current'):
        run_sample([current, record(long_shot(2), current=True)])


def test_ties_have_deterministic_input_order_without_case_id_priority():
    for sid in (3, 901):
        records = [record(long_shot(sid, safe=True, sealed=.5), current=True)]
        _, _, audit = run_sample(records, 77)
        priorities = [r['end'] for r in audit['selected_endpoints'] if r['role'] == 'priority']
        assert priorities == list(range(32))


def test_runtime_hook_compatibility_in_isolated_process_without_production_file_edit(monkeypatch):
    # Only this test process is patched. The helper must call its bound baseline
    # reference, not recurse through a future dispatch/hook to itself.
    monkeypatch.setattr(inter, '_sample', error_replay.sample)
    model = ContextStub()
    current = long_shot(2)
    old = long_shot(1, safe=True)
    old['values'][:, :2] = [-2., 0.]
    result = inter.postshot_update(model, current, [old], 'cpu',
                                 np.random.default_rng(17), .5)
    assert result['accepted'] and result['sampling']['replay_policy'] == 'error_aware'
    assert result['sampling']['positive'] == result['sampling']['negative'] == 32
    assert result['scope']['parameter_names'] == ['memory']
