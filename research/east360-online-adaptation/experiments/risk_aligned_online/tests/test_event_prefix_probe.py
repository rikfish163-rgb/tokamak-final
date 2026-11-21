"""Boundaries for the offline-label / strictly-past-input research probe."""
import csv
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.event_prefix_probe import (
    Event, assign_events, read_events, causal_summary, frozen_features,
    balanced_event_weights, shuffle_event_targets,
)
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.simple_data import window


def test_half_open_intervals_and_unknown_background():
    events = [Event(11, 1, 2, .165, .180), Event(11, 2, 3, .185, .200)]
    times = .15 + np.arange(55) / 1000
    labels, ids = assign_events(times, events)
    assert labels[15] == 2 and labels[29] == 2
    assert labels[30] == -1 and labels[34] == -1
    assert labels[35] == 3 and labels[49] == 3 and labels[50] == -1
    assert (ids[:15] == -1).all()


@pytest.mark.parametrize('events', [
    [Event(11, 1, 0, .2, .1)],
    [Event(11, 1, 6, .1, .2)],
    [Event(11, 1, 0, .1, .2), Event(11, 2, 1, .15, .3)],
    [Event(11, 1, 0, .1, .2), Event(12, 2, 1, .2, .3)],
])
def test_invalid_intervals_rejected(events):
    with pytest.raises(ValueError):
        assign_events(np.arange(50) / 100, events)


def test_read_does_not_convert_unannotated_shots_to_normal(tmp_path):
    path = tmp_path / 'labels.csv'
    fields = ['shot_no', 'cohort', 'disruption_label', 'event_index', 'event_class',
              'event_start_s', 'event_end_s', 'annotation_status']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(shot_no=1, cohort='training_pool', disruption_label=1,
                             annotation_status='no_event_detected'))
        writer.writerow(dict(shot_no=2, cohort='sanity', disruption_label=0,
                             event_index=1, event_class='C', event_start_s=.2,
                             event_end_s=.3, annotation_status='event'))
    grouped, metadata = read_events(path)
    assert grouped == {2: [Event(2, 1, 2, .2, .3)]}
    assert metadata[1]['status'] == 'no_event_detected'
    assert metadata[1]['outcome'] == 1


def test_weights_give_each_event_equal_total_mass():
    ids = np.array([1, 1, 1, 2, 3, 3])
    w = balanced_event_weights(ids)
    np.testing.assert_allclose([w[ids == i].sum() for i in (1, 2, 3)], [1, 1, 1])


def test_shuffle_is_event_level_preserves_class_count_and_strata():
    event_ids = np.repeat(np.arange(12), [2, 1, 4, 3, 2, 1, 2, 3, 1, 2, 1, 4])
    event_y = np.arange(12) % 3
    y = event_y[event_ids]
    strata = (np.arange(12) // 6)[event_ids]
    result = shuffle_event_targets(y, event_ids, strata, seed=11)
    for i in range(12):
        assert len(set(result[event_ids == i])) == 1
    for s in (0, 1):
        before = [event_y[i] for i in range(12) if i // 6 == s]
        after = [result[event_ids == i][0] for i in range(12) if i // 6 == s]
        assert sorted(before) == sorted(after)
    assert np.any(result != y)


def test_causal_features_ignore_future_and_missing_values():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(220, 23)).astype('float32')
    available = np.ones_like(values, dtype=bool)
    available[60:75, 2] = False
    shot = dict(values=values.copy(), available=available.copy())
    altered = deepcopy(shot)
    altered['values'][101:] = 1e8
    altered['values'][60:75, 2] = -1e9
    x, a = window(shot, 100, 128)
    xx, aa = window(altered, 100, 128)
    first = causal_summary(torch.from_numpy(x[None]), torch.from_numpy(a[None]),
                           torch.zeros(23), torch.ones(23))
    other = causal_summary(torch.from_numpy(xx[None]), torch.from_numpy(aa[None]),
                           torch.zeros(23), torch.ones(23))
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    assert first.shape == (1, 115) and torch.isfinite(first).all()


def test_frozen_feature_hook_preserves_parent_and_has_expected_shape():
    torch.manual_seed(12)
    model = Simple23(Config(fast_site='tokens'), torch.zeros(23), torch.ones(23))
    model.eval().requires_grad_(False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(2, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    before = model(x, a, model.initial_fast(2).detach().requires_grad_(True))['logit'].detach()
    features, logits = frozen_features(model, x, a)
    assert features.shape == (2, 192) and logits.shape == (2, 1)
    assert not features.requires_grad and not logits.requires_grad
    torch.testing.assert_close(logits[:, 0], before, rtol=0, atol=0)
    assert not model.forecast_head._forward_pre_hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_classifier_targets_do_not_enter_feature_extraction():
    import inspect
    assert set(inspect.signature(frozen_features).parameters) == {'model', 'x', 'available'}
    assert set(inspect.signature(causal_summary).parameters) == {'x', 'available', 'center', 'scale'}
