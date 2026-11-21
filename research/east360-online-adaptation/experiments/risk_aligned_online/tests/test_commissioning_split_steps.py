import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_split_steps_20261111 import replay


def _inputs(origins=(0,)):
    shape = (len(origins), 25, 10)
    return (
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=bool),
        np.asarray(origins, dtype=np.int64),
    )


def test_step_equation_uses_the_issued_snapshot_for_both_parts():
    forecast, target, observed, origins = _inputs()
    states, audit = replay(forecast, target, observed, origins, 32)
    short = np.arange(1, 9, dtype=np.float32) / 25
    tail = np.arange(9, 26, dtype=np.float32) / 25
    short_step = np.array([
        0.45 / 1.25,
        0.45 * short.mean() / (0.25 + (short**2).mean()),
    ])
    tail_step = np.array([
        0.45 / 1.25,
        0.45 * tail.mean() / (0.25 + (tail**2).mean()),
    ])

    np.testing.assert_array_equal(states['split'][:9], 0)
    np.testing.assert_allclose(states['split'][9].reshape(10, 2),
                               np.tile(short_step, (10, 1)), rtol=1e-6)
    np.testing.assert_allclose(states['split'][26].reshape(10, 2),
                               np.tile(0.92 * short_step + tail_step, (10, 1)),
                               rtol=1e-6)
    np.testing.assert_array_equal(states['delayed'][:26], 0)
    np.testing.assert_array_equal(states['split'][26:], states['delayed'][26:])
    assert audit == {
        'visible_updates': {'split': 2, 'delayed': 2, 'reversed': 2},
        'short_supported': 1,
        'tail_supported': 1,
    }


def test_paired_states_are_exactly_equal_outside_each_interval_even_at_clamp():
    origins = np.arange(0, 32 * 8, 32)
    forecast, target, observed, origins = _inputs(origins)
    target.fill(100)
    length = int(origins[-1]) + 33
    states, _ = replay(forecast, target, observed, origins, length)
    between = np.zeros(length, dtype=bool)
    for origin in origins:
        between[origin + 9:origin + 26] = True
    np.testing.assert_array_equal(states['split'][~between], states['delayed'][~between])
    assert np.any(states['split'][between] != states['delayed'][between])
    assert np.any(np.abs(states['split']) == 2)
    assert np.max(np.abs(states['split'])) <= 2
    for value in states.values():
        assert value.shape == (length, 20)
        assert value.dtype == np.float32


@pytest.mark.parametrize('seed', [1, 12, 91, 1007])
def test_paired_equality_with_varying_forecasts_targets_and_masks(seed):
    origins = np.arange(0, 26 * 5, 26)
    forecast, target, observed, origins = _inputs(origins)
    rng = np.random.default_rng(seed)
    forecast[:] = rng.normal(0, 2, forecast.shape)
    target[:] = rng.normal(0, 3, target.shape)
    observed[:] = rng.random(observed.shape) < 0.65
    length = int(origins[-1]) + 27
    states, _ = replay(forecast, target, observed, origins, length)
    between = np.zeros(length, dtype=bool)
    for origin in origins:
        between[origin + 9:origin + 26] = True
    np.testing.assert_array_equal(states['split'][~between], states['delayed'][~between])


def test_full_update_at_next_origin_is_in_its_frozen_forecast():
    forecast, target, observed, origins = _inputs((0, 26))
    initial, _ = replay(forecast, target, observed, origins, 53)
    state_at_second_origin = initial['split'][26].reshape(10, 2)
    basis = np.column_stack((np.ones(25, dtype=np.float32),
                             np.arange(1, 26, dtype=np.float32) / 25))
    target[1] = np.einsum('hc,dc->hd', basis, state_at_second_origin)

    states, _ = replay(forecast, target, observed, origins, 53)
    np.testing.assert_allclose(states['split'][35].reshape(10, 2),
                               0.92 * state_at_second_origin, rtol=1e-6)
    np.testing.assert_allclose(states['split'][52].reshape(10, 2),
                               0.92**2 * state_at_second_origin, rtol=1e-6)
    np.testing.assert_array_equal(states['split'][52], states['delayed'][52])


@pytest.mark.parametrize(
    ('length', 'short', 'tail'), [(9, 0, 0), (10, 1, 0), (26, 1, 0), (27, 1, 1)]
)
def test_updates_and_audit_only_count_when_visible(length, short, tail):
    states, audit = replay(*_inputs(), length)
    assert audit['short_supported'] == short
    assert audit['tail_supported'] == tail
    assert audit['visible_updates'] == {
        'split': short + tail,
        'delayed': 2 * tail,
        'reversed': short + tail,
    }
    if not tail:
        np.testing.assert_array_equal(states['delayed'], 0)


def test_incomplete_tail_and_future_eligibility_do_not_block_short_step():
    forecast, target, observed, origins = _inputs()
    original, audit = replay(forecast, target, observed, origins, 18)
    target[:, 8:] = -1000
    observed[:, 8:] = False
    changed, changed_audit = replay(forecast, target, observed, origins, 32)
    np.testing.assert_array_equal(original['split'], changed['split'][:18])
    np.testing.assert_array_equal(original['reversed'], changed['reversed'][:18])
    assert np.any(original['split'][9] != 0)
    np.testing.assert_array_equal(original['delayed'], 0)
    assert audit['short_supported'] == changed_audit['short_supported'] == 1
    assert audit['tail_supported'] == changed_audit['tail_supported'] == 0


def test_stage_gate_counts_families_by_half_segment_and_updates_observed_channels():
    forecast, target, observed, origins = _inputs()
    observed.fill(False)
    observed[0, :4, 0] = True  # four magnetic samples
    observed[0, :3, 3] = True  # three position samples do not qualify
    observed[0, 0, 6] = True   # one radiation sample does not qualify
    closed, audit = replay(forecast, target, observed, origins, 18)
    for value in closed.values():
        np.testing.assert_array_equal(value, 0)
    assert audit['short_supported'] == 0

    observed[0, 3, 3] = True  # position now qualifies as the second family
    opened, audit = replay(forecast, target, observed, origins, 18)
    assert audit['short_supported'] == 1
    for arm in ('split', 'reversed'):
        channels = opened[arm][9].reshape(10, 2)
        assert np.all(channels[[0, 3, 6], 0] > 0)
        np.testing.assert_array_equal(channels[9], 0)


def test_sparse_observations_use_count_normalization_and_masked_diagonal():
    forecast, target, observed, origins = _inputs()
    observed.fill(False)
    observed[0, :4, 0] = True
    observed[0, :4, 3] = True
    target[0, :4, 0] = 0.5
    states, _ = replay(forecast, target, observed, origins, 10)
    slopes = np.arange(1, 5, dtype=np.float32) / 25
    expected = np.array([
        0.45 * 0.5 / (0.25 + 4 / 8),
        0.45 * 0.5 * slopes.mean() / (0.25 + (slopes**2).sum() / 8),
    ])
    np.testing.assert_allclose(states['split'][9].reshape(10, 2)[0], expected, rtol=1e-6)


def test_tail_gate_needs_nine_samples_and_cannot_decay_unobserved_channel():
    forecast, target, observed, origins = _inputs()
    observed.fill(False)
    observed[0, :8, [0, 3]] = True
    observed[0, 8:17, 3] = True
    observed[0, 8:16, 6] = True
    closed, audit = replay(forecast, target, observed, origins, 32)
    assert audit['tail_supported'] == 0
    np.testing.assert_array_equal(closed['split'][9], closed['split'][26])

    observed[0, 16, 6] = True
    opened, audit = replay(forecast, target, observed, origins, 32)
    assert audit['tail_supported'] == 1
    np.testing.assert_array_equal(opened['split'][9].reshape(10, 2)[0],
                                  opened['split'][26].reshape(10, 2)[0])
    np.testing.assert_array_equal(opened['split'][26], opened['delayed'][26])


def test_masked_values_never_change_any_arm():
    forecast, target, observed, origins = _inputs((0, 32))
    observed[:, 1::2, 0] = False
    observed[:, ::3, 3] = False
    observed[:, :, 9] = False
    baseline, audit = replay(forecast, target, observed, origins, 65)
    target[~observed] = np.nan
    forecast[~observed] = np.nan
    changed, changed_audit = replay(forecast, target, observed, origins, 65)
    for arm in baseline:
        np.testing.assert_array_equal(baseline[arm], changed[arm])
    assert audit == changed_audit


def test_reversed_matches_per_channel_reordering_of_only_observed_slots():
    forecast, target, observed, origins = _inputs((0, 32))
    ramp = np.linspace(0.05, 0.85, 25, dtype=np.float32)
    target[:] = ramp[None, :, None]
    observed[:, 1::3, 0] = False
    observed[:, 2::4, 3] = False
    observed[:, ::2, 6] = False
    permuted = target.copy()
    for q in range(len(origins)):
        for start, stop in ((0, 8), (8, 25)):
            for channel in range(10):
                slots = np.flatnonzero(observed[q, start:stop, channel]) + start
                permuted[q, slots, channel] = target[q, slots[::-1], channel]
    np.testing.assert_array_equal(permuted[~observed], target[~observed])
    original, original_audit = replay(forecast, target, observed, origins, 65)
    reordered, reordered_audit = replay(forecast, permuted, observed, origins, 65)
    np.testing.assert_array_equal(original['reversed'], reordered['split'])
    assert original_audit == reordered_audit
    assert np.any(original['reversed'] != original['split'])


def test_each_replay_starts_from_zero_and_does_not_modify_inputs():
    forecast, target, observed, origins = _inputs((3, 35))
    before = tuple(value.copy() for value in (forecast, target, observed, origins))
    first, _ = replay(forecast, target, observed, origins, 70)
    second, _ = replay(forecast, target, observed, origins, 70)
    for arm in first:
        np.testing.assert_array_equal(first[arm], second[arm])
        np.testing.assert_array_equal(first[arm][:12], 0)
    for value, saved in zip((forecast, target, observed, origins), before):
        np.testing.assert_array_equal(value, saved)


def test_empty_origins_return_zero_states_and_no_visible_updates():
    states, audit = replay(*_inputs(()), 7)
    for value in states.values():
        np.testing.assert_array_equal(value, np.zeros((7, 20), dtype=np.float32))
    assert audit['visible_updates'] == {'split': 0, 'delayed': 0, 'reversed': 0}
    assert audit['short_supported'] == audit['tail_supported'] == 0


@pytest.mark.parametrize('origins', [[0, 25], [2, 2], [5, 3], [-1]])
def test_invalid_origin_order_or_stride_is_rejected(origins):
    with pytest.raises(ValueError):
        replay(*_inputs(origins), 64)


def test_descending_unsigned_origins_are_rejected_without_wraparound():
    forecast, target, observed, _ = _inputs((32, 0))
    with pytest.raises(ValueError):
        replay(forecast, target, observed, np.array([32, 0], dtype=np.uint64), 64)
def test_centered_slope_ignores_offsets_and_keeps_mask_and_step_count():
    y = np.tile(np.linspace(-.5, .5, 25, dtype=np.float32)[None, :, None], (2, 1, 10))
    f = np.zeros_like(y)
    m = np.ones_like(y, dtype=bool)
    m[:, [1, 8, 10, 22], [0, 2, 3, 5]] = False
    q = np.array([0, 32])
    observed, a = replay(f, y, m, q, 60, centered=True)
    shifted, b = replay(f + 2., y - 3., m, q, 60, centered=True)
    assert a == b and a['visible_updates']['split'] == 4
    for name in observed:
        np.testing.assert_allclose(observed[name], shifted[name], atol=4e-7, rtol=0.)
        np.testing.assert_array_equal(observed[name][:, ::2], 0.)
    np.testing.assert_array_equal(observed['split'][26:41], observed['delayed'][26:41])
    np.testing.assert_array_equal(observed['split'], -observed['reversed'])


def test_centered_control_flips_only_temporal_gradient_at_the_first_release():
    y = np.tile(np.linspace(0., .25, 25, dtype=np.float32)[None, :, None], (1, 1, 10))
    f = np.zeros_like(y)
    m = np.ones_like(y, dtype=bool)
    a, _ = replay(f, y, m, np.array([0]), 29, centered=True)
    assert np.linalg.norm(a['split'][9]) > 0.
    np.testing.assert_array_equal(a['split'][9], -a['reversed'][9])
    np.testing.assert_array_equal(a['split'][:9], 0.)
    np.testing.assert_array_equal(a['delayed'][:26], 0.)
