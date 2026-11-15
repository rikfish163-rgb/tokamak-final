import io
from pathlib import Path

import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_preq_source_probe_20261111 as probe


ARMS = ('fixed', 'real', 'shifted', 'lag')


def features(values):
    result = np.zeros((len(values), 18), np.float32)
    result[:, :2] = values
    return result


def test_weighted_ridge_matches_explicit_two_view_objective():
    fixed = features([[0, 1], [2, 0], [1, 2], [3, 1], [4, 0], [2, 3]])
    real = features([[1, 2], [1, 0], [2, 1], [2, 2], [5, 1], [3, 2]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    shots = np.array([1, 1, 2, 3, 4, 4])
    ridge = .2
    fitted = probe.fit_readout(fixed, real, labels, shots, ridge=ridge)

    # Class mass .5, then equal mass per shot within each class; two views
    # divide, rather than duplicate, each original row's mass.
    row_weight = np.array([.125, .125, .25, .25, .125, .125])
    x = np.concatenate([fixed, real]).astype(np.float64)
    design = np.column_stack((np.ones(len(x)), x))
    w = np.tile(row_weight / 2, 2)
    y = np.tile(labels, 2)
    penalty = np.diag([0.] + [ridge] * 18)
    expected = np.linalg.solve(design.T @ (w[:, None] * design) + penalty,
                               design.T @ (w * y))
    np.testing.assert_allclose(fitted['intercept'], expected[0], atol=1e-12)
    np.testing.assert_allclose(fitted['weights'], expected[1:], atol=1e-12)
    assert fitted['weight_sum'] == pytest.approx(1.)
    assert fitted['ridge'] == ridge


def test_ridge_same_views_and_within_shot_row_replication_do_not_change_weights():
    x = features([[0, 0], [1, 0], [2, 1], [3, 1]])
    y = np.array([0, 0, 1, 1])
    shots = np.array([1, 2, 3, 4])
    first = probe.fit_readout(x, x, y, shots, ridge=.1)
    indices = np.array([0, 0, 0, 1, 2, 3])
    repeated = probe.fit_readout(x[indices], x[indices], y[indices], shots[indices], ridge=.1)
    np.testing.assert_allclose(first['weights'], repeated['weights'], atol=1e-12)
    assert first['intercept'] == pytest.approx(repeated['intercept'])


@pytest.mark.parametrize('labels,ridge', [([0, 0], .1), ([0, 1], 0.), ([0, 2], .1)])
def test_ridge_rejects_missing_classes_invalid_labels_and_nonpositive_penalty(labels, ridge):
    x = features([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        probe.fit_readout(x, x, np.array(labels), np.array([1, 2]), ridge=ridge)


def pair_row(value_by_arm, *, risk, disruptive, lead):
    return dict(risk=risk, disruptive=disruptive, lead=lead,
                views={arm: features([[value_by_arm[arm], 0]])[0] for arm in ARMS})


def test_pair_comparison_keeps_exact_ids_and_counts_strict_sign_and_ties():
    records = {
        (11, 200): pair_row(dict(zip(ARMS, (1, -1, 0, 1e-12))), risk=1, disruptive=True, lead=11),
        (13, 201): pair_row(dict(zip(ARMS, (-1, 0, 1, -1e-12))), risk=1, disruptive=True, lead=20),
        (12, 199): pair_row(dict.fromkeys(ARMS, 0), risk=0, disruptive=False, lead=0),
    }
    identities = [[11, 200, 12, 199], [13, 201, 12, 199]]
    readout = dict(weights=np.r_[1., np.zeros(17)], intercept=.3)
    result = probe.compare_pairs(identities, records, readout)
    assert result['pair_ids'] == identities
    assert result['arms'] == {
        'fixed': dict(wins=1, losses=1, ties=0),
        'real': dict(wins=0, losses=1, ties=1),
        'shifted': dict(wins=1, losses=0, ties=1),
        'lag': dict(wins=1, losses=1, ties=0),
    }
    assert result['pairs'][0]['identity'] == identities[0]
    assert result['pairs'][0]['margins']['lag'] > 0


@pytest.mark.parametrize('positive', [
    dict(risk=0, disruptive=True, lead=20),
    dict(risk=1, disruptive=False, lead=20),
    dict(risk=1, disruptive=True, lead=10),
])
def test_pair_comparison_rejects_a_wrong_or_not_strictly_timely_positive(positive):
    a = pair_row(dict.fromkeys(ARMS, 1), **positive)
    b = pair_row(dict.fromkeys(ARMS, 0), risk=0, disruptive=False, lead=0)
    with pytest.raises(ValueError):
        probe.compare_pairs([[1, 200, 2, 201]], {(1, 200): a, (2, 201): b},
                            dict(weights=np.ones(18), intercept=0.))


def test_archive_requires_the_original_fifty_validation_pairs():
    pairs = [[100 + i, 200 + i, 300, 220] for i in range(50)]
    old = {'held': {'pairs': 50, 'pair_ids': pairs}}
    val = [*(100 + i for i in range(50)), 300]
    assert probe.locked_pair_ids(old, val) == pairs
    with pytest.raises(ValueError):
        probe.locked_pair_ids({'held': {'pairs': 50, 'pair_ids': pairs[:-1]}}, val)
    with pytest.raises(ValueError):
        probe.locked_pair_ids(old, val[:-1])


def test_synthetic_shot_uses_true_advanced_window_complete_grid_and_causal_views(monkeypatch):
    from experiments.risk_aligned_online import commissioning_lag_data_20261110 as lag
    from experiments.risk_aligned_online import io23, simple_data
    from experiments.risk_aligned_online.commissioning_transition_source_20261110 import decision_points

    rng = np.random.default_rng(131)
    values = rng.normal(size=(260, 23)).astype(np.float32)
    signal = dict(shot=11, device='J-TEXT', values=values,
                  available=np.ones_like(values, bool), times=np.arange(260) / 1000.)
    monkeypatch.setattr(simple_data, 'read_shot', lambda device, sid: signal)
    monkeypatch.setattr(io23, 'read_outcome', lambda device, sid: (True, .240))
    original_feedback = lag.feedback
    called_ends = []

    def feedback(z, mask, ends):
        called_ends.append(ends.copy())
        assert np.all(ends < len(z))
        return original_feedback(z, mask, ends)

    monkeypatch.setattr(lag, 'feedback', feedback)
    center, scale = np.zeros(23), np.ones(23)
    prior = np.eye(70, dtype=np.float32)[0]
    full = probe.collect_shot(11, {11: 60.}, center, scale, prior)
    np.testing.assert_array_equal(called_ends[0], np.arange(127, 260 - 25, 32) + 25)
    expected_y, valid = io23.labels_at(signal['times'], True, .240, 60.)
    expected_end = decision_points(expected_y.astype(bool), valid)
    np.testing.assert_array_equal(full['end'], expected_end)
    np.testing.assert_array_equal(full['risk'], expected_y[expected_end])
    assert 197 in expected_end  # Exact old endpoint need not lie on the 32-sample grid.
    assert full['risk'][np.flatnonzero(expected_end == 197)[0]] == 1

    # The fallback horizon is 25 ms, not the 60 ms window above.
    fallback = probe.collect_shot(11, {}, center, scale, prior)
    old_y, old_valid = io23.labels_at(signal['times'], True, .240, 25.)
    np.testing.assert_array_equal(fallback['end'], decision_points(old_y.astype(bool), old_valid))
    assert 197 not in fallback['end']

    # Future signal perturbations cannot alter any earlier published feature.
    changed = dict(signal, values=values.copy())
    changed['values'][201:] *= -7
    monkeypatch.setattr(simple_data, 'read_shot', lambda device, sid: changed)
    altered = probe.collect_shot(11, {11: 60.}, center, scale, prior)
    before = full['end'] <= 200
    for arm in ARMS:
        np.testing.assert_array_equal(full['views'][arm][before], altered['views'][arm][before])


def test_synthetic_diagnosis_fits_all_train_and_reads_only_paired_val(monkeypatch):
    calls = []

    def collect(sid, advanced, center, scale, prior):
        calls.append(sid)
        risk = np.array([0, 1]) if sid == 1 else np.array([0])
        if sid == 3:
            risk = np.array([1])
        ends = np.array([191, 207])[:len(risk)]
        x = features([[float(sid) + int(y), float(y)] for y in risk])
        return dict(shot=sid, end=ends, risk=risk, disruptive=sid in (1, 3),
                    lead=np.full(len(risk), 20.),
                    views={arm: x + (0.1 if arm == 'real' else 0.) for arm in ARMS},
                    audit={'observed_blocks': 1})

    monkeypatch.setattr(probe, 'collect_shot', collect)
    roles = {'train': [1, 2], 'val': [3, 4, 5], 'test': [99]}
    ids = [[3, 191, 4, 191]]
    result = probe.diagnose(roles, {}, np.zeros(23), np.ones(23), np.ones(70), ids,
                            expected_train_shots=[1, 2])
    assert calls == [1, 2, 3, 4]
    assert result['train_shots'] == [1, 2]
    assert result['held']['pair_ids'] == ids
    assert all(sum(v.values()) == 1 for v in result['held']['arms'].values())
    with pytest.raises(ValueError):
        probe.diagnose(roles, {}, np.zeros(23), np.ones(23), np.ones(70), ids,
                       expected_train_shots=[1])


def test_full_entrypoint_binds_sealed_sources_and_writes_only_one_json_in_memory(monkeypatch):
    import json
    from types import SimpleNamespace

    from experiments.risk_aligned_online import io23
    from experiments.risk_aligned_online.redesign_20261004 import prepare_prior

    source = Path('/virtual/lag_source')
    mean = Path('/virtual/graph_memory/source_mean.npz')
    output = Path('/virtual/diagnosis.json')
    identities = [[100 + i, 191, 300, 191] for i in range(50)]
    hashes = {name: f'hash:{source / name}' for name in ('protocol.json', 'model.npz', 'result.json')}
    archived = dict(model_sha256=hashes['model.npz'], train_shots=[1, 2],
                    held=dict(pairs=50, pair_ids=identities))
    entries = {
        source / 'protocol.json': {'parent_sha256': prepare_prior.PARENT_SHA256},
        source / 'result.json': archived,
        mean.parent / 'protocol.json': {'source_sha256': prepare_prior.PARENT_SHA256,
                                       'source_assets': hashes},
        mean.parent / 'seal.json': {'artifacts': {mean.name: f'hash:{mean}'}},
    }
    roles = {'train': [1, 2], 'val': [*range(100, 150), 300, 301], 'test': [999]}
    calls = []

    def collect(sid, advanced, center, scale, prior):
        calls.append(sid)
        risk = np.array([0, 1]) if sid == 1 else np.array([int(100 <= sid < 150)])
        end = np.array([191, 207])[:len(risk)]
        x = features([[float(y), sid / 1000.] for y in risk])
        return dict(shot=sid, end=end, risk=risk, disruptive=sid != 2 and sid != 300,
                    lead=np.full(len(end), 20.), views={arm: x for arm in ARMS},
                    audit={'observed_blocks': 1})

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Archive:
        def __enter__(self):
            return {'mean': np.ones(70, np.float32)}

        def __exit__(self, *args):
            return False

    class Captured(io.StringIO):
        def close(self):
            pass

    captured = Captured()
    monkeypatch.setattr(probe, 'sha256', lambda p: f'hash:{Path(p)}')
    monkeypatch.setattr(probe, 'read_json', lambda p: entries[Path(p)])
    monkeypatch.setattr(probe, 'collect_shot', collect)
    monkeypatch.setattr(io23, 'source_roles', lambda: roles)
    monkeypatch.setattr(io23, 'advanced_times', lambda: {})
    monkeypatch.setattr(prepare_prior, 'load_parent', lambda p: (
        SimpleNamespace(center=Tensor(np.zeros(23)), scale=Tensor(np.ones(23))), {}))
    original_load, original_exists, original_is_dir, original_open = (
        np.load, Path.exists, Path.is_dir, Path.open)
    monkeypatch.setattr(np, 'load', lambda p, *a, **kw:
                        Archive() if Path(p) == mean else original_load(p, *a, **kw))
    monkeypatch.setattr(Path, 'exists', lambda p: False if p == output else original_exists(p))
    monkeypatch.setattr(Path, 'is_dir', lambda p: True if p == output.parent else original_is_dir(p))

    def open_output(path, mode='r', *args, **kwargs):
        if path == output:
            assert mode == 'x' and kwargs.get('encoding') == 'utf-8'
            return captured
        assert mode in ('r', 'rb')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_output)
    probe.run(source, mean, output)
    payload = json.loads(captured.getvalue())
    assert payload['held']['pair_ids'] == identities
    assert payload['train_shots'] == [1, 2]
    assert len(payload['readout']['weights']) == 18
    assert payload['input_sha256']['source_mean'] == f'hash:{mean}'
    assert set(payload['code_sha256']) >= {
        'probe', 'prequential', 'lag_data', 'io23', 'transition_source',
        'transition_net', 'roughness', 'event_roughness', 'prepare_prior',
    }
    assert calls == [1, 2, *range(100, 150), 300]
    assert all(sum(arm.values()) == 50 for arm in payload['held']['arms'].values())

    entries[mean.parent / 'seal.json']['artifacts'][mean.name] = 'different-hash'
    calls.clear()
    with pytest.raises(ValueError, match='lineage'):
        probe.run(source, mean, output)
    assert calls == []
