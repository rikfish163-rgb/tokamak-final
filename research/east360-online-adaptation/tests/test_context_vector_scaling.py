"""CPU-only contracts for train-fold context scaling and radius calibration."""

import json
import math

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030.scaling import (
    calibrate_radius,
    fit_context_scale,
    transform_context,
)
from experiments.risk_aligned_online.context_vector_20261030.teaching import context_controls


def sample_context(dtype=torch.float64):
    context = torch.zeros((4, 358), dtype=dtype)
    context[:, 0] = torch.tensor([0, 0, 0, 10], dtype=dtype)
    context[:, 1] = torch.tensor([2, 4, 6, 8], dtype=dtype)
    context[:, 2] = 7
    blocks = context[:, 128:].reshape(4, 5, 46)
    # Finite placeholders in missing cells must not be treated as observations.
    blocks[:, :, :23] = 777
    blocks[:, 0, 0] = torch.tensor([2, 4, 6, 10], dtype=dtype)
    blocks[:, 0, 23] = torch.tensor([.2, 1, 1, .2], dtype=dtype)
    blocks[0, 0, 1] = 2
    blocks[0, 0, 24] = .2
    blocks[3, :, 1] = 4
    blocks[3, :, 24] = 1
    blocks[:, 0, 3] = 0
    blocks[:, 0, 26] = 1
    blocks[:, 0, 4] = 3
    blocks[:, 0, 27] = 1
    return context


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shot_window_weighted_population_stats_and_masked_pooled_rms(dtype):
    context = sample_context(dtype)
    original = context.clone()
    stats = fit_context_scale(context, ['a', 'a', 'a', 'b'])

    assert set(stats) == {
        'hcenter', 'hscale', 'h_constant', 'change_scale',
        'change_constant', 'change_unobservable',
    }
    for name, shape in [('hcenter', (128,)), ('hscale', (128,)),
                        ('change_scale', (23,))]:
        assert stats[name].shape == shape and stats[name].dtype == dtype
        assert stats[name].device.type == 'cpu' and not stats[name].requires_grad
    assert stats['hcenter'][0].item() == pytest.approx(5)
    assert stats['hscale'][0].item() == pytest.approx(5)
    assert stats['hcenter'][1].item() == pytest.approx(6)
    assert stats['hscale'][1].item() == pytest.approx(math.sqrt(16 / 3))
    assert stats['hcenter'][2].item() == pytest.approx(7)
    assert stats['hscale'][2].item() == 1
    assert stats['h_constant'][2] and not stats['h_constant'][0]
    # Fractions .2 and 1 both mark a valid block; they are not another weight.
    assert stats['change_scale'][0].item() == pytest.approx(math.sqrt(178 / 3))
    # The denominator is the weighted number of valid blocks, not all blocks.
    assert stats['change_scale'][1].item() == pytest.approx(math.sqrt(15.25))
    assert stats['change_scale'][2].item() == 1
    assert stats['change_constant'][2] and stats['change_unobservable'][2]
    assert stats['change_scale'][3].item() == 1
    assert stats['change_constant'][3] and not stats['change_unobservable'][3]
    # A nonzero constant mean is not a near-zero RMS coordinate.
    assert stats['change_scale'][4].item() == pytest.approx(3)
    assert not stats['change_constant'][4]
    for name, size in [('h_constant', 128), ('change_constant', 23),
                       ('change_unobservable', 23)]:
        assert stats[name].shape == (size,) and stats[name].dtype == torch.bool
    torch.testing.assert_close(context, original, rtol=0, atol=0)


@pytest.mark.parametrize('amplitude,constant', [(0., True), (1e-6, True), (1.01e-6, False)])
def test_exact_small_scale_threshold(amplitude, constant):
    context = torch.zeros((2, 358), dtype=torch.float64)
    context[:, 0] = torch.tensor([-amplitude, amplitude], dtype=torch.float64)
    blocks = context[:, 128:].reshape(2, 5, 46)
    blocks[:, 0, 0] = amplitude
    blocks[:, 0, 23] = 1
    stats = fit_context_scale(context, [10, 20])
    assert bool(stats['h_constant'][0]) is constant
    assert bool(stats['change_constant'][0]) is constant
    assert not stats['change_unobservable'][0]
    assert stats['hscale'][0].item() == pytest.approx(1 if constant else amplitude)
    assert stats['change_scale'][0].item() == pytest.approx(1 if constant else amplitude)


def test_large_constant_coordinate_is_not_roundoff_amplified():
    context = torch.zeros((6, 358), dtype=torch.float64)
    context[:, 0] = 1e20
    stats = fit_context_scale(context, list(range(6)))
    assert stats['hcenter'][0].item() == 1e20
    assert stats['h_constant'][0]
    assert stats['hscale'][0].item() == 1


def test_transform_preserves_fractions_zeros_missing_and_does_not_mutate_input():
    context = sample_context()
    stats = fit_context_scale(context, [1, 1, 1, 2])
    original = context.clone()
    stats_before = {name: value.clone() for name, value in stats.items()}
    transformed = transform_context(context, stats)
    torch.testing.assert_close(transformed[:, :128],
                               (context[:, :128] - stats['hcenter']) / stats['hscale'])
    blocks = context[:, 128:].reshape(4, 5, 46)
    out = transformed[:, 128:].reshape(4, 5, 46)
    torch.testing.assert_close(out[:, :, 23:], blocks[:, :, 23:], rtol=0, atol=0)
    valid = blocks[:, :, 23:] > 0
    expected = torch.where(valid, blocks[:, :, :23] / stats['change_scale'], 0.)
    torch.testing.assert_close(out[:, :, :23], expected, rtol=0, atol=0)
    assert torch.count_nonzero(out[:, :, :23][~valid]) == 0
    torch.testing.assert_close(context, original, rtol=0, atol=0)
    for name in stats:
        torch.testing.assert_close(stats[name], stats_before[name], rtol=0, atol=0)
    assert transform_context(context[:0], stats).shape == (0, 358)


def test_reverse_control_commutes_with_fit_and_transform():
    context = sample_context()
    ids = np.array([11, 11, 11, 22])
    reverse = lambda value: context_controls(value, torch.zeros(358, dtype=value.dtype))['reverse']
    stats = fit_context_scale(context, ids)
    reverse_stats = fit_context_scale(reverse(context), ids)
    for name in stats:
        torch.testing.assert_close(stats[name], reverse_stats[name])
    torch.testing.assert_close(transform_context(reverse(context), stats),
                               reverse(transform_context(context, stats)), rtol=0, atol=0)


def test_held_transform_never_refits_or_updates_statistics():
    train = sample_context()
    stats = fit_context_scale(train, torch.tensor([1, 1, 1, 2]))
    before = {name: value.clone() for name, value in stats.items()}
    first_train_transform = transform_context(train, stats)
    held = sample_context() * 1
    held[:, :128] += 100_000
    held[:, 128:].reshape(4, 5, 46)[:, :, :23] *= 100
    transform_context(held, stats)
    torch.testing.assert_close(first_train_transform, transform_context(train, stats), rtol=0, atol=0)
    for name in stats:
        torch.testing.assert_close(stats[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize('defect', [
    'shape', 'empty', 'integer_context', 'nan_context', 'inf_context',
    'negative_availability', 'availability_over_one', 'ids_shape', 'ids_length',
    'ids_nan', 'ids_inf', 'ids_none', 'ids_boolean',
])
def test_fit_rejects_invalid_context_or_ids(defect):
    context = sample_context()
    ids = [1, 1, 1, 2]
    if defect == 'shape':
        context = context[:, :-1]
    elif defect == 'empty':
        context, ids = context[:0], []
    elif defect == 'integer_context':
        context = context.long()
    elif defect == 'nan_context':
        context[0, 130] = float('nan')
    elif defect == 'inf_context':
        context[0, 0] = float('inf')
    elif defect == 'negative_availability':
        context[0, 151] = -.1
    elif defect == 'availability_over_one':
        context[0, 151] = 1.1
    elif defect == 'ids_shape':
        ids = [[1], [1], [1], [2]]
    elif defect == 'ids_length':
        ids = [1]
    elif defect == 'ids_nan':
        ids[0] = float('nan')
    elif defect == 'ids_inf':
        ids[0] = float('inf')
    elif defect == 'ids_none':
        ids[0] = None
    elif defect == 'ids_boolean':
        ids[0] = True
    with pytest.raises((TypeError, ValueError)):
        fit_context_scale(context, ids)


@pytest.mark.parametrize('defect', [
    'missing', 'bad_shape', 'nonfinite', 'zero_scale', 'negative_scale',
    'integer_stat', 'context_nonfinite', 'context_shape', 'invalid_availability',
])
def test_transform_rejects_invalid_context_or_stats(defect):
    context = sample_context()
    stats = fit_context_scale(context, [1, 1, 1, 2])
    if defect == 'missing':
        del stats['hcenter']
    elif defect == 'bad_shape':
        stats['change_scale'] = torch.ones(22, dtype=context.dtype)
    elif defect == 'nonfinite':
        stats['hcenter'][0] = float('nan')
    elif defect == 'zero_scale':
        stats['hscale'][0] = 0
    elif defect == 'negative_scale':
        stats['change_scale'][0] = -1
    elif defect == 'integer_stat':
        stats['hcenter'] = stats['hcenter'].long()
    elif defect == 'context_nonfinite':
        context[0, 0] = float('inf')
    elif defect == 'context_shape':
        context = context[:, :-1]
    elif defect == 'invalid_availability':
        context[0, 151] = -.1
    with pytest.raises((TypeError, ValueError)):
        transform_context(context, stats)


def test_radius_uses_median_per_shot_then_median_across_shots_and_clips():
    result = calibrate_radius(
        np.array([.2, .4, 2., 4., 6., 0., -2.]),
        ['a', 'a', 'b', 'b', 'b', 'c', 'a'],
        np.array([True, True, True, True, True, False, False]),
        source_qk_norm=10.,
    )
    assert result == dict(radius=1., raw_median=pytest.approx(2.15),
                          contributing_shots=2, clipped=True, status='ok')
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize('distances,source_norm,raw,expected,clipped', [
    ([.01, .05], 100., .03, .1, True),
    ([.4, .8], 20., .6, .6, False),
    ([.4, .8], 0., .6, .1, True),
    ([.1], 1., .1, .1, False),
])
def test_radius_lower_and_source_relative_upper_bounds(distances, source_norm, raw, expected, clipped):
    result = calibrate_radius(torch.tensor(distances, dtype=torch.float64),
                              [1] * len(distances), [True] * len(distances), source_norm)
    assert result['raw_median'] == pytest.approx(raw)
    assert result['radius'] == pytest.approx(expected)
    assert result['contributing_shots'] == 1
    assert result['clipped'] is clipped and result['status'] == 'ok'


@pytest.mark.parametrize('distances,ids,eligible', [
    ([], [], []), ([0., 100.], [1, 2], [False, False]),
])
def test_radius_with_no_historical_misclassified_query(distances, ids, eligible):
    result = calibrate_radius(distances, ids, eligible, source_qk_norm=12.)
    assert result == dict(radius=.1, raw_median=None, contributing_shots=0,
                          clipped=False, status='no_historical_misclassified_query')
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize('defect', [
    'distance_shape', 'distance_length', 'distance_nan_unselected', 'distance_inf',
    'distance_none_unselected', 'distance_zero_selected', 'distance_negative_selected',
    'eligible_shape', 'eligible_length', 'eligible_nonboolean', 'ids_shape', 'ids_nan',
    'source_nan', 'source_inf', 'source_negative', 'source_boolean',
])
def test_radius_rejects_invalid_inputs(defect):
    distances, ids, eligible, source = [.2, .4], [1, 2], [True, False], 10.
    if defect == 'distance_shape':
        distances = [[.2], [.4]]
    elif defect == 'distance_length':
        distances = [.2]
    elif defect == 'distance_nan_unselected':
        distances[1] = float('nan')
    elif defect == 'distance_inf':
        distances[0] = float('inf')
    elif defect == 'distance_none_unselected':
        distances[1] = None
    elif defect == 'distance_zero_selected':
        distances[0] = 0.
    elif defect == 'distance_negative_selected':
        distances[0] = -1.
    elif defect == 'eligible_shape':
        eligible = [[True], [False]]
    elif defect == 'eligible_length':
        eligible = [True]
    elif defect == 'eligible_nonboolean':
        eligible = [1, 0]
    elif defect == 'ids_shape':
        ids = [[1], [2]]
    elif defect == 'ids_nan':
        ids[0] = float('nan')
    elif defect == 'source_nan':
        source = float('nan')
    elif defect == 'source_inf':
        source = float('inf')
    elif defect == 'source_negative':
        source = -1.
    elif defect == 'source_boolean':
        source = True
    with pytest.raises((TypeError, ValueError)):
        calibrate_radius(distances, ids, eligible, source)
