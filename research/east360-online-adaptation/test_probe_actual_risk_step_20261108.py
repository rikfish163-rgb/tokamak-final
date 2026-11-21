"""CPU-only regression tests against the deployed rollout, without disk outputs."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import probe_actual_risk_step_20261108 as p
import run_risk_feature_intra_20261107 as native
from experiments.risk_aligned_online.simple23 import Config, Simple23
from test_joint_stream_20261107 import signal


@pytest.fixture(scope='module', autouse=True)
def cpu():
    p.d.configure_cpu()


@pytest.fixture
def inputs():
    torch.manual_seed(149)
    base = Simple23(Config(), torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    decoder = p.d.MetaOrderedTrajectory(base.center, base.scale, dim=64, scenarios=4).eval()
    raw = dict(signal(360), times=np.arange(360) / 1000., device='EAST', shot=1)
    return base, decoder, raw, dict(gain=np.ones(23), offset=np.zeros(23))


def test_full_prefix_matches_native_rollout_weights_prediction_and_last_step(inputs, monkeypatch):
    base, decoder, raw, mapping = inputs
    old = [p.d.snapshot(m) for m in (base, decoder)]
    raw_before, last = deepcopy(raw), {}
    native_update = native.update

    def capture(model, optimizer, episode, **kwargs):
        before = model.weight.detach().clone()
        stats = native_update(model, optimizer, episode, **kwargs)
        last.update(model=model, delta=model.weight.detach() - before, stats=stats)
        return stats

    monkeypatch.setattr(native, 'update', capture)
    prefix = dict(raw, **{k: raw[k][:346].copy() for k in ('values', 'available', 'times')})
    with sdpa_kernel(SDPBackend.MATH):
        curves, info = native.feature_curves(base, decoder, p.j.r.prepare_actual(prefix, 21, mapping), feedbacks=('real',))
        row, model, delta = p.probe_prefix(base, decoder, raw, 21, mapping, 345, 1)
    assert row['releases'] == info['releases'] == list(range(153, 346, 32))
    assert row['updates'] == 7 and model.weight.numel() == 8192
    assert row['after']['probability'] == pytest.approx(float(curves['real'][345]), abs=5e-6)
    torch.testing.assert_close(model.weight, last['model'].weight, atol=0, rtol=0)
    torch.testing.assert_close(delta, last['delta'], atol=0, rtol=0)
    assert row['ssl_before'] == last['stats']['signal_loss']
    for module, state in zip((base, decoder), old):
        p.d.check_state(module, state)
    for key in ('values', 'available', 'times'):
        np.testing.assert_array_equal(raw[key], raw_before[key])


def test_diagnostic_label_cannot_change_weights_prediction_or_actual_delta(inputs):
    base, decoder, raw, mapping = inputs
    a, ma, da = p.probe_prefix(base, decoder, raw, 21, mapping, 345, 0)
    b, mb, db = p.probe_prefix(base, decoder, raw, 21, mapping, 345, 1)
    p.d.check_state(ma, p.d.snapshot(mb))
    torch.testing.assert_close(da, db, atol=0, rtol=0)
    for phase in ('before', 'after'):
        assert a[phase]['probability'] == b[phase]['probability']
    assert a['actual_step_norm'] == b['actual_step_norm']
    assert a['before']['bce'] != b['before']['bce']
    assert a['actual_step_risk_gradient_dot'] != b['actual_step_risk_gradient_dot']


@pytest.mark.parametrize('q,releases', [(152, []), (153, [153]), (185, [153, 185])])
def test_release_boundaries_and_appended_future_never_reach_preprocessing(inputs, monkeypatch, q, releases):
    base, decoder, raw, mapping = inputs
    prepare, origins = p.j.r.prepare_actual, []
    make_episode = p.d.make_episode

    def checked_prepare(prefix, rank, mapping):
        assert all(len(prefix[k]) == q + 1 for k in ('values', 'available', 'times'))
        return prepare(prefix, rank, mapping)

    def checked_episode(s, origin, query):
        origins.append(origin)
        episode = make_episode(s, origin, query)
        np.testing.assert_array_equal(episode['truth'][0], s['values'][origin+1:origin+26])
        assert origin + 25 < query == q and episode['x'].shape == (1, 128, 23)
        return episode

    monkeypatch.setattr(p.j.r, 'prepare_actual', checked_prepare)
    monkeypatch.setattr(p.d, 'make_episode', checked_episode)
    prefix = dict(raw, **{k: raw[k][:q+1].copy() for k in ('values', 'available', 'times')})
    future = deepcopy(raw)
    future['values'][q+1:] = np.nan
    future['available'][q+1:, 2] = True
    a, ma, da = p.probe_prefix(base, decoder, prefix, 21, mapping, q, 1)
    b, mb, db = p.probe_prefix(base, decoder, future, 21, mapping, q, 1)
    assert a == b and a['releases'] == releases and a['updates'] == len(releases)
    assert origins == [v - 26 for v in releases] * 2
    p.d.check_state(ma, p.d.snapshot(mb))
    torch.testing.assert_close(da, db, atol=0, rtol=0)
    assert a['target_range'] == ([releases[-1]-25, releases[-1]-1] if releases else None)
    if not releases:
        assert a['status'] == 'noupdate' and a['ssl_before'] is None
        assert a['before'] == a['after'] and a['actual_step_norm'] == 0


@pytest.mark.parametrize('mismatch', [False, True])
def test_cli_retains_fixed_cases_exclusions_and_never_writes_after_parity_failure(inputs, monkeypatch, mismatch):
    base, decoder, raw, mapping = inputs
    row, model, delta = p.probe_prefix(base, decoder, raw, 21, mapping, 152, 1)
    cases = [dict(rank=i+21, shot=i+1, query_index=152, native_y_at_query=1,
                  static_status='miss', exclusion_reasons=[]) for i in range(85)]
    old = dict(status='COMPLETE_CPU_POSTHOC_LOCAL_DIRECTION', cases=cases, exclusions=[{'old': 'unchanged'}])
    monkeypatch.setattr(p.d, 'configure_cpu', lambda: None)
    monkeypatch.setattr(p.d, 'load_inputs', lambda: ({}, mapping, cases))
    monkeypatch.setattr(p.d, 'choose_and_verify', lambda case, cfg: (dict(case), raw))
    monkeypatch.setattr(p.j.r, 'read', lambda path: old)
    monkeypatch.setattr(p.j.r, 'sha', lambda path: 'unchanged')
    monkeypatch.setattr(p.j.r, 'load_parent', lambda path: (base, {'threshold': p.d.THRESHOLD}))
    monkeypatch.setattr(p.torch, 'load', lambda *args, **kwargs: decoder.state_dict())
    monkeypatch.setattr(p, 'probe_prefix', lambda *args: (dict(row), model, delta))
    expected = row['after']['probability'] + (1e-4 if mismatch else 0.)
    monkeypatch.setattr(p.np, 'load', lambda *args, **kwargs: nullcontext({'recorded_sampling_real': np.full(153, expected)}))
    monkeypatch.setattr(p, 'OUTPUT', Path('/unused-actual-step-test-output.json'))
    writes = []
    monkeypatch.setattr(p.j.r, 'write_new', lambda path, result: writes.append((path, result)))
    monkeypatch.setattr(p.sys, 'argv', ['probe_actual_risk_step_20261108.py'])
    if mismatch:
        with pytest.raises(ValueError, match='sealed real parity'):
            p.main()
        assert not writes
    else:
        p.main()
        assert len(writes) == 1 and writes[0][0] == p.OUTPUT
        result = writes[0][1]
        assert len(result['cases']) == result['summary']['all']['nomatured'] == 85
        assert result['exclusions'] == old['exclusions'] and str(Path(p.__file__)) in result['source_files']
