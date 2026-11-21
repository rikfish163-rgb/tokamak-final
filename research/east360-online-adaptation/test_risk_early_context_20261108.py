"""CPU checks for causal, shot-local early-context feature updates."""

from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.simple23 import Config, Simple23
from risk_feature_intra_20261107 import FeatureIntra, update as actual_update
from risk_meta_step_20261107 import MetaOrderedTrajectory
from run_risk_feature_intra_20261107 import feature_curves
import risk_early_context_20261108 as early


@pytest.fixture
def case():
    torch.manual_seed(71)
    base = Simple23(Config(dim=8, heads=4, layers=1, fast_site='tokens'),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    torch.manual_seed(37)
    decoder = MetaOrderedTrajectory(base.center, base.scale, dim=8, scenarios=4,
                                    motion_weight=.472).eval()
    values = np.random.default_rng(72).normal(0., .15, (180, 23)).astype(np.float32)
    values[:, [17, 22]] = abs(values[:, [17, 22]])
    available = np.ones_like(values, dtype=bool)
    available[:, 2] = False
    available[::17, 4] = False
    return base, decoder, dict(values=values, available=available)


def prefix(signal, n):
    return {key: value[:n].copy() for key, value in signal.items()}


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_same_state(module, before):
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def risk_at(model, signal, index):
    x, a = simple_data.window(signal, index, 128)
    with torch.no_grad():
        return model(torch.from_numpy(x)[None], torch.from_numpy(a)[None])['logit'].sigmoid().item()


def test_real_updates_follow_the_early_grid_and_full_arrived_h25(case, monkeypatch):
    base, decoder, signal = case
    base_before, decoder_before = snapshot(base), snapshot(decoder)
    records = []
    clipped = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def trace_clip(parameters, max_norm, *args, **kwargs):
        clipped.append((max_norm, kwargs.get('error_if_nonfinite')))
        return real_clip(parameters, max_norm, *args, **kwargs)

    def trace_update(model, optimizer, episode, *, feedback):
        assert feedback == 'real'
        assert set(episode) == {'x', 'a', 'truth', 'mask'}
        assert [name for name, p in model.named_parameters() if p.requires_grad] == [
            'risk.encoder.layers.0.linear2.weight']
        assert optimizer.param_groups[0]['params'] == [model.weight]
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[0]['weight_decay'] == 0.
        before = snapshot(model)
        saved = {k: v.detach().clone() for k, v in episode.items()}
        stats = actual_update(model, optimizer, episode, feedback=feedback)
        after = snapshot(model)
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        assert changed == ['risk.encoder.layers.0.linear2.weight']
        records.append((saved, before, after, stats))
        return stats

    monkeypatch.setattr(early, 'update', trace_update)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', trace_clip)
    curves, info = early.early_curves(base, decoder, signal)

    assert set(curves) == {'static', 'real'}
    assert info['releases'] == [57, 89, 121, 153]
    assert info['updates'] == {'real': 4}
    assert len(records) == 4
    assert clipped == [(1., True)] * 4
    assert info['weight_norm']['real'] > 0
    assert info['gradient_max']['real'] == max(record[3]['gradient_norm'] for record in records)
    assert info['gradient_max']['real'] > 0
    assert records[0][3]['prior'] == 0.
    assert records[1][3]['prior'] > 0.
    for _, _, _, stats in records:
        assert stats['objective'] == pytest.approx(stats['signal_loss'] + .01*stats['prior'], abs=1e-6)
    np.testing.assert_array_equal(curves['real'][:57], curves['static'][:57])
    assert curves['real'][57] != curves['static'][57]

    witness = FeatureIntra(base, decoder)
    weight_key = 'risk.encoder.layers.0.linear2.weight'
    last = records[-1][2]
    assert info['weight_norm']['real'] == pytest.approx(
        (last[weight_key] - last['initial_weight']).norm().item(), abs=1e-12)
    for i, (origin, release) in enumerate(zip((31, 63, 95, 127), info['releases'], strict=True)):
        episode, before, after, _ = records[i]
        assert release == origin + 26
        x, a = simple_data.window(signal, origin, 128)
        torch.testing.assert_close(episode['x'], torch.from_numpy(x)[None], rtol=0, atol=0)
        torch.testing.assert_close(episode['a'], torch.from_numpy(a)[None], rtol=0, atol=0)
        assert episode['truth'].shape == episode['mask'].shape == (1, 25, 23)
        torch.testing.assert_close(episode['truth'], torch.from_numpy(signal['values'][origin+1:release])[None],
                                   rtol=0, atol=0)
        torch.testing.assert_close(episode['mask'], torch.from_numpy(signal['available'][origin+1:release])[None],
                                   rtol=0, atol=0)
        if i:
            torch.testing.assert_close(before[weight_key], records[i-1][2][weight_key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(before[weight_key], before['initial_weight'], rtol=0, atol=0)
        with torch.no_grad():
            witness.weight.copy_(before[weight_key])
        assert curves['real'][release-1] == pytest.approx(risk_at(witness.risk, signal, release-1), abs=3e-6)
        with torch.no_grad():
            witness.weight.copy_(after[weight_key])
        assert curves['real'][release] == pytest.approx(risk_at(witness.risk, signal, release), abs=3e-6)

    assert_same_state(base, base_before)
    assert_same_state(decoder, decoder_before)


@pytest.mark.parametrize(('n', 'releases'), [
    (56, []), (57, []), (58, [57]),
    (88, [57]), (89, [57]), (90, [57, 89]),
    (120, [57, 89]), (121, [57, 89]), (122, [57, 89, 121]),
    (152, [57, 89, 121]), (153, [57, 89, 121]),
    (154, [57, 89, 121, 153]),
])
def test_truncation_excludes_only_updates_without_a_following_risk(case, n, releases):
    base, decoder, signal = case
    full, _ = early.early_curves(base, decoder, signal)
    partial, info = early.early_curves(base, decoder, prefix(signal, n))
    assert info['releases'] == releases
    assert info['updates'] == {'real': len(releases)}
    assert (info['weight_norm']['real'] == 0.) == (not releases)
    for name in ('static', 'real'):
        np.testing.assert_allclose(partial[name], full[name][:n], rtol=0, atol=3e-6)


def test_warm_is_exactly_static_and_never_builds_a_temporary_model(case, monkeypatch):
    base, decoder, signal = case
    reference, _ = feature_curves(base, decoder, signal, warm=True, feedbacks=('real',))

    def forbidden(*args, **kwargs):
        raise AssertionError('warm calls must not construct or update a temporary model')

    monkeypatch.setattr(early, 'FeatureIntra', forbidden)
    monkeypatch.setattr(early, 'update', forbidden)
    curves, info = early.early_curves(base, decoder, signal, warm=True)
    assert set(curves) == {'static', 'real'}
    np.testing.assert_array_equal(curves['static'], reference['static'])
    np.testing.assert_array_equal(curves['real'], curves['static'])
    assert info['releases'] == []
    assert info['updates'] == {'real': 0}
    assert info['weight_norm'] == {'real': 0.}
    assert info['gradient_max'] == {'real': 0.}


def test_only_signal_fields_are_read_no_outcome_or_hdf_and_shots_reset(case, monkeypatch):
    base, decoder, signal = case
    base_before, decoder_before = snapshot(base), snapshot(decoder)

    class SignalOnly:
        def __getitem__(self, key):
            assert key in ('values', 'available'), f'unexpected input field {key}'
            return signal[key]

    def forbidden(*args, **kwargs):
        raise AssertionError('prediction must not read outcome, signals, or HDF')

    monkeypatch.setattr(io23, 'read_outcome', forbidden)
    monkeypatch.setattr(io23, 'read_signals', forbidden)
    monkeypatch.setattr(h5py, 'File', forbidden)
    first, info1 = early.early_curves(base, decoder, SignalOnly())
    warm, warm_info = early.early_curves(base, decoder, SignalOnly(), warm=True)
    second, info2 = early.early_curves(base, decoder, SignalOnly())
    assert info1 == info2
    assert info1['updates']['real'] == 4 and warm_info['updates']['real'] == 0
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(warm['real'], warm['static'])
    assert_same_state(base, base_before)
    assert_same_state(decoder, decoder_before)


@pytest.mark.parametrize('cut', [32, 57, 80, 89, 121, 153])
def test_future_signal_and_mask_changes_cannot_alter_prior_risk(case, cut):
    base, decoder, signal = case
    changed = deepcopy(signal)
    changed['values'][cut:] += 3.
    changed['available'][cut:, 4::3] = False
    original, _ = early.early_curves(base, decoder, signal)
    altered, _ = early.early_curves(base, decoder, changed)
    for name in ('static', 'real'):
        np.testing.assert_allclose(original[name][:cut], altered[name][:cut], rtol=0, atol=3e-6)


@pytest.mark.parametrize('fault', ['length', 'shape', 'mask_dtype'])
def test_bad_signal_contract_is_rejected_by_the_native_views(case, fault):
    base, decoder, signal = case
    bad = deepcopy(signal)
    if fault == 'length':
        bad['available'] = bad['available'][:-1]
    elif fault == 'shape':
        bad['values'] = bad['values'][:, :-1]
    else:
        bad['available'] = bad['available'].astype(np.float32)
    with pytest.raises(ValueError, match='23-channel signals'):
        early.early_curves(base, decoder, bad)
