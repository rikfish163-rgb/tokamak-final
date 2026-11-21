"""CPU contract tests for causal partial trajectory feedback."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from risk_meta_step_20261107 import MetaOrderedTrajectory
from run_risk_feature_intra_20261107 import feature_curves
import risk_meta_step_20261107 as meta
import risk_partial_feedback_20261108 as partial


ARMS = ('static', 'full', 'partial_delayed', 'partial_early')
FIRST = dict(static=None, full=153, partial_delayed=153, partial_early=144)


@pytest.fixture(scope='module')
def parents():
    torch.manual_seed(71)
    center = torch.linspace(-.1, .1, 23)
    scale = torch.linspace(.8, 1.2, 23)
    base = Simple23(Config(dim=8, heads=4, layers=1, fast_site='tokens'),
                    center, scale).eval().requires_grad_(False)
    torch.manual_seed(37)
    decoder = MetaOrderedTrajectory(center, scale, dim=8, scenarios=4,
                                    motion_weight=.472).eval().requires_grad_(False)
    return base, decoder


def signal(length):
    values = np.random.default_rng(72).normal(0., .15, (length, 23)).astype(np.float32)
    values[:, [17, 22]] = abs(values[:, [17, 22]])
    available = np.ones((length, 23), dtype=bool)
    available[:, 2] = False
    if length > 132:
        available[132, 4] = False
    return dict(values=values, available=available)


@pytest.fixture(scope='module')
def long_curves(parents):
    return partial.partial_curves(*parents, signal(222))


def assert_parent_states_same(model, before):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_full_reference_first_affected_and_actual_arm_metadata(parents, long_curves):
    curves, info = long_curves
    reference, reference_info = feature_curves(*parents, signal(222), feedbacks=('real',))
    assert set(curves) == set(ARMS)
    np.testing.assert_array_equal(curves['static'], reference['static'])
    np.testing.assert_array_equal(curves['full'], reference['real'])
    assert info['first_affected'] == FIRST
    assert info['releases'] == dict(static=[], full=[153, 185, 217],
                                    partial_delayed=[153, 185, 217],
                                    partial_early=[144, 176, 208])
    assert info['releases']['full'] == reference_info['releases']
    assert info['updates'] == dict(static=0, full=3, partial_delayed=3, partial_early=3)
    assert info['weight_norm']['full'] == reference_info['weight_norm']['real']
    assert info['gradient_max']['full'] == reference_info['gradient_max']['real']
    for name in ARMS[1:]:
        assert np.isfinite(info['weight_norm'][name]) and info['weight_norm'][name] > 0
        assert np.isfinite(info['gradient_max'][name]) and info['gradient_max'][name] > 0
        np.testing.assert_allclose(curves[name][:FIRST[name]],
                                   curves['static'][:FIRST[name]], atol=3e-6, rtol=0)
        assert abs(curves[name][FIRST[name]] - curves['static'][FIRST[name]]) > 1e-7
    assert info['weight_norm']['static'] == info['gradient_max']['static'] == 0


def test_warm_is_static_and_declares_but_does_not_execute_releases(parents):
    curves, info = partial.partial_curves(*parents, signal(222), warm=True)
    reference, _ = feature_curves(*parents, signal(222), warm=True, feedbacks=('real',))
    for name in ARMS:
        np.testing.assert_array_equal(curves[name], reference['static'])
        assert info['releases'][name] == []
        assert info['updates'][name] == 0
        assert info['weight_norm'][name] == info['gradient_max'][name] == 0
    assert info['first_affected'] == FIRST


@pytest.mark.parametrize('length', [127, 128, 143, *range(144, 155),
                                    175, 176, 177, 185, 186, 208, 209, 217, 218])
def test_prefixes_and_per_arm_update_clocks(parents, long_curves, length):
    whole, _ = long_curves
    s = signal(222)
    prefix = {key: value[:length] for key, value in s.items()}
    curves, info = partial.partial_curves(*parents, prefix)
    for name in ARMS:
        np.testing.assert_allclose(curves[name], whole[name][:length], atol=3e-6, rtol=0)
        releases = [] if FIRST[name] is None else list(range(FIRST[name], length, 32))
        assert info['releases'][name] == releases
        assert info['updates'][name] == len(releases)


@pytest.mark.parametrize('length', [153, 185, 217])
def test_end_of_shot_extra_early_update_is_not_discarded(parents, length):
    _, info = partial.partial_curves(*parents, signal(length))
    assert info['updates']['partial_early'] == info['updates']['full'] + 1
    assert info['updates']['partial_delayed'] == info['updates']['full']
    assert info['releases']['partial_early'][-1] < length
    assert (info['releases']['partial_early'][-1] + 9) == length


def test_only_fast_weight_changes_and_each_shot_resets(parents, monkeypatch):
    base, decoder = parents
    base_before, decoder_before = deepcopy(base.state_dict()), deepcopy(decoder.state_dict())
    real_update = partial.update
    observed = []

    def checked_update(model, optimizer, episode, **kwargs):
        assert [n for n, p in model.named_parameters() if p.requires_grad] == [
            'risk.encoder.layers.0.linear2.weight']
        assert optimizer.defaults['lr'] == 1e-4 and optimizer.defaults['weight_decay'] == 0
        before = deepcopy(model.state_dict())
        result = real_update(model, optimizer, episode, **kwargs)
        for name, value in model.state_dict().items():
            if name != 'risk.encoder.layers.0.linear2.weight':
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        observed.append(result)
        return result

    monkeypatch.setattr(partial, 'update', checked_update)
    first, first_info = partial.partial_curves(base, decoder, signal(190))
    partial.partial_curves(base, decoder, signal(222))
    repeated, repeated_info = partial.partial_curves(base, decoder, signal(190))
    for name in ARMS:
        np.testing.assert_array_equal(first[name], repeated[name])
    assert first_info == repeated_info and observed
    assert_parent_states_same(base, base_before)
    assert_parent_states_same(decoder, decoder_before)


def test_every_partial_update_uses_its_own_issued_origin_window(parents, monkeypatch):
    real_update = partial.update
    episodes = []

    def capture(model, optimizer, episode, **kwargs):
        episodes.append({key: value.detach().clone() for key, value in episode.items()})
        return real_update(model, optimizer, episode, **kwargs)

    monkeypatch.setattr(partial, 'update', capture)
    s = signal(222)
    partial.partial_curves(*parents, s)
    assert len(episodes) == 6
    for arm_episodes in (episodes[:3], episodes[3:]):
        for origin, episode in zip((127, 159, 191), arm_episodes, strict=True):
            torch.testing.assert_close(episode['x'][0],
                                       torch.from_numpy(s['values'][origin - 127:origin + 1]))
            torch.testing.assert_close(episode['a'][0],
                                       torch.from_numpy(s['available'][origin - 127:origin + 1]))
            torch.testing.assert_close(episode['truth'][0, :16],
                                       torch.from_numpy(s['values'][origin + 1:origin + 17]))
            torch.testing.assert_close(episode['mask'][0, :16],
                                       torch.from_numpy(s['available'][origin + 1:origin + 17]))


def test_partial_updates_read_only_16_mature_points_and_ignore_future_nine(parents, monkeypatch):
    base, decoder = parents
    real_update = partial.update
    snapshots = []

    def capture(model, optimizer, episode, **kwargs):
        assert not optimizer.state
        with torch.no_grad():
            forecast = model(episode['x'], episode['a'])
        assert forecast['encoded'].shape == (1, 4, 25, 23)
        saved = {key: value.detach().clone() for key, value in episode.items()}
        result = real_update(model, optimizer, episode, **kwargs)
        saved['weight'] = model.weight.detach().clone()
        saved['gradient'] = model.weight.grad.detach().clone()
        saved['gradient_norm'] = result['gradient_norm']
        snapshots.append(saved)
        return result

    monkeypatch.setattr(partial, 'update', capture)
    original = signal(154)
    partial.partial_curves(base, decoder, original)
    first = snapshots[:]
    snapshots.clear()
    changed = {key: value.copy() for key, value in original.items()}
    changed['values'][144:153] += 50
    changed['available'][144:153] = ~changed['available'][144:153]
    partial.partial_curves(base, decoder, changed)
    second = snapshots[:]
    assert len(first) == len(second) == 2  # delayed and early, one each
    for before, after in zip(first, second, strict=True):
        for key in ('x', 'a', 'truth', 'mask', 'weight', 'gradient'):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        torch.testing.assert_close(before['truth'][0, :16],
                                   torch.from_numpy(original['values'][128:144]))
        torch.testing.assert_close(before['mask'][0, :16],
                                   torch.from_numpy(original['available'][128:144]))
        torch.testing.assert_close(before['truth'][0, 16:], base.center.expand(9, -1))
        assert not before['mask'][0, 16:].any()

    snapshots.clear()
    changed_observed = {key: value.copy() for key, value in original.items()}
    changed_observed['values'][135, 0] += 50
    partial.partial_curves(base, decoder, changed_observed)
    assert len(snapshots) == 2
    for before, after in zip(first, snapshots, strict=True):
        assert not torch.allclose(before['gradient'], after['gradient'], atol=1e-8, rtol=1e-5)


def test_partial_keeps_the_real_lag8_change_loss_and_gradient(parents, monkeypatch):
    base, decoder = parents
    real_energy, real_update = meta.joint_energy_no_self, partial.update
    energy_calls, gradients = [], []

    def energy(encoded, probabilities, target, available):
        result = real_energy(encoded, probabilities, target, available)
        energy_calls.append((available.detach().clone(), result.detach().clone()))
        return result

    def update(model, optimizer, episode, **kwargs):
        result = real_update(model, optimizer, episode, **kwargs)
        gradients.append(model.weight.grad.detach().clone())
        return result

    monkeypatch.setattr(meta, 'joint_energy_no_self', energy)
    monkeypatch.setattr(partial, 'update', update)
    partial.partial_curves(base, decoder, signal(145))
    assert len(energy_calls) == 2 and len(gradients) == 1
    level_mask, _ = energy_calls[0]
    lag_mask, lag_score = energy_calls[1]
    assert level_mask.shape == (1, 25, 23) and lag_mask.shape == (1, 17, 23)
    torch.testing.assert_close(lag_mask, level_mask[:, 8:] & level_mask[:, :-8])
    assert lag_mask[:, :8].any() and not lag_mask[:, 8:].any()
    assert torch.isfinite(lag_score).all() and lag_score.abs().item() > 1e-7

    no_motion = deepcopy(decoder)
    no_motion.motion_weight.zero_()
    partial.partial_curves(base, no_motion, signal(145))
    assert len(gradients) == 2
    assert not torch.allclose(gradients[0], gradients[1], atol=1e-8, rtol=1e-5)
