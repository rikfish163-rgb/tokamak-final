from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23, simple_data
from risk_meta_sequence_20261108 import sequence_from_note as old_sequence_from_note
from run_risk_teacher_20261107 import teacher_pools
from risk_short_context_20261108 import context_pools, sequence_from_note


GROUPS = ('0_to_0', '0_to_1', '1_to_1')


def shot(n, sid, *, down=None, advanced=None):
    times = np.arange(n, dtype=np.float64) / 1000
    values = (100 * np.arange(n)[:, None] + np.arange(23)[None, :]).astype(np.float32)
    available = np.ones((n, 23), dtype=bool)
    available[::17, 4] = False
    values[~available] = 0
    advanced_times = {} if advanced is None else {sid: advanced}
    horizon = io23.effective_horizon_ms(sid, advanced_times)[0]
    y, valid = io23.labels_at(times, down is not None, down, horizon)
    return dict(shot=sid, values=values, available=available, y=y, valid=valid)


def short_shots():
    return [shot(155, 1), shot(155, 2, down=.164),
            shot(155, 3, down=.164, advanced=60)]


def note_for(s, first):
    origins = [first + 32 * k for k in range(4)]
    q = origins[-1] + 26
    return dict(shot=s['shot'], origins=origins, query=q, label_time=q + 25,
                group=f"{int(s['y'][q])}_to_{int(s['y'][q + 25])}")


def assert_same_pools(actual, expected):
    assert list(actual) == list(expected)
    for group in expected:
        assert len(actual[group]) == len(expected[group])
        for (actual_i, actual_origins), (expected_i, expected_origins) in zip(
                actual[group], expected[group], strict=True):
            assert actual_i == expected_i
            np.testing.assert_array_equal(actual_origins, expected_origins)


def assert_same_episode(actual, expected):
    actual_es, actual_qx, actual_qa, actual_y, actual_note = actual
    expected_es, expected_qx, expected_qa, expected_y, expected_note = expected
    assert len(actual_es) == len(expected_es) == 4
    for a, b in zip(actual_es, expected_es, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype
            assert a[key].device == b[key].device
            assert torch.equal(a[key], b[key])
    for a, b in ((actual_qx, expected_qx), (actual_qa, expected_qa),
                 (actual_y, expected_y)):
        assert a.dtype == b.dtype
        assert a.device == b.device
        assert torch.equal(a, b)
    assert actual_note == expected_note


def test_155_points_create_first_7_tasks_in_three_real_native_groups():
    shots = short_shots()
    original_labels = [s['y'].copy() for s in shots]
    pools = context_pools(shots)
    assert list(pools) == list(GROUPS)
    for i, group in enumerate(GROUPS):
        assert len(pools[group]) == 1
        index, origins = pools[group][0]
        assert index == i
        np.testing.assert_array_equal(origins, [7])
        s = shots[index]
        note = note_for(s, int(origins[0]))
        assert note['group'] == group
        assert note['query'] == 129 and note['label_time'] == 154
        result = sequence_from_note(s, note, 'cpu')
        assert result[3].item() == s['y'][154]
    for s, before in zip(shots, original_labels, strict=True):
        np.testing.assert_array_equal(s['y'], before)
    assert io23.effective_horizon_ms(2, {})[0] == 25
    assert io23.effective_horizon_ms(2, {2: 15})[0] == 25
    assert io23.effective_horizon_ms(3, {3: 60})[0] == 60
    assert all(not rows for rows in teacher_pools(shots, steps=4, require_all_groups=False).values())
    with pytest.raises(ValueError):
        context_pools([shot(154, 1)])


def test_short_windows_are_native_left_padding_with_mnm_masked():
    s = short_shots()[1]
    note = note_for(s, 7)
    episodes, qx, qa, label, returned = sequence_from_note(s, note, 'cpu')
    assert returned == note
    assert len(episodes) == 4
    for o, e in zip(note['origins'], episodes, strict=True):
        assert set(e) == {'x', 'a', 'truth', 'mask'}
        x, a = simple_data.window(s, o, 128)
        padding = 127 - o
        assert padding > 0
        assert not e['a'][0, :padding].any()
        assert not e['x'][0, :padding].any()
        assert torch.equal(e['x'], torch.from_numpy(x)[None])
        a[:, 2] = False
        assert torch.equal(e['a'], torch.from_numpy(a)[None])
        assert not e['a'][..., 2].any()
        assert not e['mask'][..., 2].any()
        assert s['available'][:o + 1, 2].all()
    x, a = simple_data.window(s, note['query'], 128)
    a[:, 2] = False
    assert torch.equal(qx, torch.from_numpy(x)[None])
    assert torch.equal(qa, torch.from_numpy(a)[None])
    assert not qa[..., 2].any()
    assert label.item() == s['y'][note['label_time']]


@pytest.mark.parametrize('first', [7, 8, 31, 32, 63, 64, 95, 96, 126])
def test_short_origin_boundaries_use_exact_native_windows(first):
    s = shot(400, 4, down=.380, advanced=60)
    note = note_for(s, first)
    episodes, _, _, _, _ = sequence_from_note(s, note, 'cpu')
    for origin, episode in zip(note['origins'], episodes, strict=True):
        x, a = simple_data.window(s, origin, 128)
        a[:, 2] = False
        assert torch.equal(episode['x'], torch.from_numpy(x)[None])
        assert torch.equal(episode['a'], torch.from_numpy(a)[None])
        assert torch.equal(episode['truth'], torch.from_numpy(s['values'][origin + 1:origin + 26])[None])


def test_four_real_h25_targets_mature_before_query_and_do_not_carry_risk_labels():
    s = short_shots()[1]
    note = note_for(s, 7)
    episodes, _, _, _, _ = sequence_from_note(s, note, 'cpu')
    q = note['query']
    assert note['origins'] == [7, 39, 71, 103]
    assert q == 129 and note['label_time'] == 154
    for i, (o, e) in enumerate(zip(note['origins'], episodes, strict=True)):
        assert o + 25 < q
        if i < 3:
            assert o + 25 < note['origins'][i + 1]
        assert set(e) == {'x', 'a', 'truth', 'mask'}
        assert e['truth'].shape == e['mask'].shape == (1, 25, 23)
        assert torch.equal(e['truth'], torch.from_numpy(s['values'][o + 1:o + 26])[None])
        expected = s['available'][o + 1:o + 26].copy()
        expected[:, 2] = False
        assert torch.equal(e['mask'], torch.from_numpy(expected)[None])


def test_missing_native_group_is_an_error_and_never_synthesized():
    for shots in ([], [short_shots()[0]], short_shots()[:2]):
        with pytest.raises(ValueError):
            context_pools(shots)
    with pytest.raises(ValueError):
        context_pools(short_shots(), minimum_origin=127)


@pytest.mark.parametrize('key', ['values', 'available', 'valid', 'y'])
def test_pool_rejects_misaligned_shot_arrays(key):
    shots = short_shots()
    shots[1][key] = shots[1][key][:-1]
    with pytest.raises(ValueError):
        context_pools(shots)


@pytest.mark.parametrize('minimum', [6, 8, 126, 128, 7.0, True, None, '7'])
def test_only_two_integer_minimum_origins_are_supported(minimum):
    with pytest.raises(ValueError):
        context_pools(short_shots(), minimum_origin=minimum)


def test_full_context_pools_equal_old_order_and_membership_with_valid_holes():
    shots = [shot(600, 1), shot(600, 2, down=.550, advanced=75),
             shot(600, 3, down=.550, advanced=75)]
    shots[0]['valid'][180] = False
    shots[1]['valid'][50] = False
    shots[2]['valid'][280] = False
    old = teacher_pools(shots, steps=4)
    new = context_pools(shots, minimum_origin=127)
    assert_same_pools(new, old)
    safe_origins = dict(new['0_to_0'])[0]
    assert 127 not in safe_origins and 307 not in safe_origins
    assert 308 in safe_origins


def test_short_pool_requires_all_actual_context_through_future_label_to_be_valid():
    shots = [shot(500, 1), shot(500, 2, down=.450, advanced=50),
             shot(500, 3, down=.450, advanced=50)]
    shots[0]['valid'][0] = False
    shots[1]['valid'][29] = False
    shots[2]['valid'][270] = False
    pools = context_pools(shots)
    expected = {group: [] for group in GROUPS}
    for i, s in enumerate(shots):
        selected = {group: [] for group in GROUPS}
        for first in range(7, len(s['valid']) - 147):
            q = first + 122
            t = q + 25
            if all(s['valid'][max(0, first - 127):t + 1]):
                group = f"{int(s['y'][q])}_to_{int(s['y'][t])}"
                if group in selected:
                    selected[group].append(first)
        for group in GROUPS:
            if selected[group]:
                expected[group].append((i, np.asarray(selected[group])))
    assert_same_pools(pools, expected)
    safe_origins = dict(pools['0_to_0'])[0]
    assert 127 not in safe_origins and 128 in safe_origins


@pytest.mark.parametrize('first', [127, 230, 330, 353])
def test_full_context_episodes_are_bitwise_equal_to_old_builder(first):
    s = shot(600, 4, down=.550, advanced=75)
    note = note_for(s, first)
    assert_same_episode(sequence_from_note(s, note, 'cpu'),
                        old_sequence_from_note(s, note, 'cpu'))


@pytest.mark.parametrize('hole', [0, 7, 40, 103, 129, 154])
def test_sequence_rejects_valid_holes_in_actual_history_or_future(hole):
    s = short_shots()[1]
    note = note_for(s, 7)
    s['valid'][hole] = False
    with pytest.raises(ValueError):
        sequence_from_note(s, note, 'cpu')


@pytest.mark.parametrize('change', ['short', 'missing_origin', 'spacing', 'query', 'label_time', 'float_origin',
                                     'past_end', 'wrong_group', 'unsupported_group'])
def test_sequence_rejects_wrong_recorded_clock_or_group(change):
    s = short_shots()[1]
    note = note_for(s, 7)
    if change == 'short':
        note = note_for(s, 6)
    elif change == 'missing_origin':
        note['origins'].pop()
    elif change == 'spacing':
        note['origins'][1] += 1
    elif change == 'query':
        note['query'] += 1
    elif change == 'label_time':
        note['label_time'] -= 1
    elif change == 'float_origin':
        note['origins'][0] = 7.0
    elif change == 'past_end':
        note['origins'] = [8, 40, 72, 104]
        note['query'], note['label_time'] = 130, 155
    elif change == 'wrong_group':
        note['group'] = '1_to_1'
    elif change == 'unsupported_group':
        note['group'] = '1_to_0'
    with pytest.raises(ValueError):
        sequence_from_note(s, note, 'cpu')


@pytest.mark.parametrize('which', ['query', 'label_time'])
def test_sequence_rejects_tampering_with_recorded_native_labels(which):
    s = short_shots()[1]
    note = note_for(s, 7)
    index = note[which]
    s['y'][index] = 1 - s['y'][index]
    with pytest.raises(ValueError):
        sequence_from_note(s, note, 'cpu')


@pytest.mark.parametrize('which', ['query', 'label_time'])
def test_nonbinary_label_at_task_clock_is_not_rounded_to_a_native_group(which):
    shots = short_shots()
    s = shots[1]
    note = note_for(s, 7)
    s['y'][note[which]] = .25
    with pytest.raises(ValueError):
        context_pools(shots)
    with pytest.raises(ValueError):
        sequence_from_note(s, note, 'cpu')


def test_signal_and_availability_after_query_cannot_affect_inner_or_query():
    s = short_shots()[1]
    note = note_for(s, 7)
    before = sequence_from_note(s, note, 'cpu')
    changed = deepcopy(s)
    changed['values'][note['query'] + 1:] = -98765
    changed['available'][note['query'] + 1:] = False
    after = sequence_from_note(changed, note, 'cpu')
    assert_same_episode(after, before)


def test_changes_after_future_label_time_cannot_affect_the_sequence():
    s = shot(210, 2, down=.164)
    note = note_for(s, 7)
    before = sequence_from_note(s, note, 'cpu')
    changed = deepcopy(s)
    later = slice(note['label_time'] + 1, None)
    changed['values'][later] = -12345
    changed['available'][later] = False
    changed['valid'][later] = ~changed['valid'][later]
    changed['y'][later] = 1 - changed['y'][later]
    assert_same_episode(sequence_from_note(changed, note, 'cpu'), before)
