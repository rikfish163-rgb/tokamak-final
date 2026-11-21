import numpy as np
import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from ..ffn_intra_20261005.stream import predict_shot as original_predict
from .stream import predict_shot


def east(length=80):
    model, shot, _ = case()
    shot = {k: v[:length].copy() if isinstance(v, np.ndarray) else v for k, v in shot.items()}
    shot['device'] = 'EAST'
    return model, shot


def test_partial_and_delayed_match_at_origins_but_not_inside_cycle():
    model, shot = east()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    p, pa = predict_shot(model, shot, mode='partial')
    d, da = predict_shot(model, shot, mode='delayed')
    np.testing.assert_array_equal(p[:6], d[:6])
    np.testing.assert_array_equal(p[[26, 27, 28, 29, 30, 31, 52, 53, 54, 55, 56, 57]],
                                  d[[26, 27, 28, 29, 30, 31, 52, 53, 54, 55, 56, 57]])
    assert np.max(np.abs(p[6:26] - d[6:26])) > 1e-7
    assert pa['final_delta_sha256'] == da['final_delta_sha256']
    assert [(e['snapshot_index'], e['update_index'], e['effective_index']) for e in pa['events'][:2]] == [(5, 5, 6), (25, 25, 26)]
    assert [(e['snapshot_index'], e['update_index'], e['effective_index']) for e in da['events'][:2]] == [(5, 25, 26), (25, 25, 26)]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize('mode', ['partial', 'delayed'])
def test_future_mutation_and_outcomes_do_not_change_emitted_prefix(mode):
    model, shot = east()
    original, _ = predict_shot(model, shot, mode=mode)
    for cutoff in (5, 6, 24, 25, 26, 31, 51):
        changed = dict(shot, values=shot['values'].copy(), available=shot['available'].copy(),
                       is_disrupt=False, down=-200.)
        changed['values'][cutoff + 1:] = 999.
        changed['available'][cutoff + 1:] = False
        scores, _ = predict_shot(model, changed, mode=mode)
        np.testing.assert_array_equal(scores[:cutoff + 1], original[:cutoff + 1])


@pytest.mark.parametrize('length,partial_count,delayed_count', [(1, 0, 0), (5, 0, 0), (6, 1, 0), (25, 1, 0), (26, 2, 2), (31, 2, 2), (32, 3, 2)])
def test_short_shots_consume_only_arrived_events(length, partial_count, delayed_count):
    model, shot = east(length)
    _, p = predict_shot(model, shot, mode='partial')
    _, d = predict_shot(model, shot, mode='delayed')
    assert len(p['events']) == partial_count
    assert len(d['events']) == delayed_count
    for audit in (p, d):
        for row in audit['events']:
            assert row['snapshot_index'] <= row['update_index'] < length
            assert row['effective_index'] == (row['update_index'] + 1 if row['update_index'] + 1 < length else None)


def test_empty_feedback_no_projection_and_every_shot_resets():
    model, shot = east(52)
    shot['available'][1:26] = False
    shot['values'][1:26] = np.nan
    p, a = predict_shot(model, shot, mode='partial')
    assert a['events'][0]['reason'] == a['events'][1]['reason'] == 'no_valid_targets'
    assert a['events'][0]['delta_norm'] == a['events'][1]['delta_norm'] == 0.
    p2, a2 = predict_shot(model, shot, mode='partial')
    np.testing.assert_array_equal(p, p2)
    assert a['final_delta_sha256'] == a2['final_delta_sha256']


def test_static_matches_existing_reader_and_requires_explicit_source_access():
    model, shot = east()
    original, _ = original_predict(model, shot, intra=False)
    static, audit = predict_shot(model, shot, mode='static')
    np.testing.assert_array_equal(original, static)
    assert not audit['events']
    shot['device'] = 'J-TEXT'
    with pytest.raises(ValueError, match='source'):
        predict_shot(model, shot, mode='partial')


def test_early_truth_is_snapshotted_before_delayed_execution(monkeypatch):
    from . import stream
    model, shot = east(26)
    saved = shot['values'][1:6].copy()
    original_snapshot = stream.snapshot
    def mutate_after_snapshot(signals, origin, segment, device, dtype):
        value = original_snapshot(signals, origin, segment, device, dtype)
        if segment == 'early':
            signals['values'][origin + 1:origin + 6] = -999.
        return value
    observed = []
    original_step = stream.segment_step
    def step(*args, **kwargs):
        observed.append((args[6], args[4].detach().cpu().numpy().copy()))
        return original_step(*args, **kwargs)
    monkeypatch.setattr(stream, 'snapshot', mutate_after_snapshot)
    monkeypatch.setattr(stream, 'segment_step', step)
    predict_shot(model, shot, mode='delayed')
    np.testing.assert_array_equal(observed[0][1][0], saved)
