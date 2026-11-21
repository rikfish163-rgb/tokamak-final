"""The learned update may consume only matured measurements."""

import numpy as np
import pytest
import torch

from same_site_arrival_20261105 import (
    action,
    arrival_features,
    fit_action,
    match_norm,
)
import same_site_arrival_20261105 as experiment


class ToyForecast:
    center = torch.zeros(23)
    scale = torch.ones(23)

    def __call__(self, values, available):
        assert values.shape == available.shape == (1, 128, 23)
        return dict(
            logit=values[:, -1, 0],
            forecast=torch.zeros(1, 25, 23),
        )


def test_arrival_features_refuse_early_reads_and_ignore_all_future_samples():
    shot = dict(
        values=np.zeros((180, 23), np.float32), available=np.ones((180, 23), bool)
    )
    shot["values"][132:136, 0] = 1
    s = ToyForecast()
    before = arrival_features(s, shot, 127, 135, 0.0)
    assert before["innovation"][0] == pytest.approx(np.arcsinh(1))
    assert before["hold"][0] == 0.0
    assert before["reversed"][0] == -before["innovation"][0]
    shot["values"][136:] = 99
    shot["available"][136:] = False
    after = arrival_features(s, shot, 127, 135, 0.0)
    assert all(np.array_equal(before[k], after[k]) for k in before)
    with pytest.raises(ValueError, match="eight"):
        arrival_features(s, shot, 127, 134, 0.0)


def test_action_learns_signed_on_site_utility_and_is_bounded():
    x = np.array([[-1.0], [1.0]])
    z = np.zeros(2)
    g = np.ones((2, 1))
    fit = fit_action(x, z, g, np.array([0, 1]), np.full(2, 0.5))
    d = action(fit, x)
    assert -0.5 < d[0, 0] < 0 < d[1, 0] < 0.5
    assert np.array_equal(action(fit, np.zeros_like(x)), np.zeros_like(d))
    assert np.array_equal(action(fit, -x), -d)


def test_norm_control_keeps_direction_and_does_not_create_missing_support():
    real = np.array([[0.0, 0.2], [0.5, 0.0]])
    placebo = np.array([[10.0, 0.0], [0.0, 0.0]])
    assert np.array_equal(match_norm(placebo, real), [[0.2, 0.0], [0.0, 0.0]])


def test_ended_east_case_exposes_the_same_query_field_as_source(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "_teaching",
        lambda _model, _signal, query, _tau: {"z": float(query)},
    )
    case = dict(origin=127, arrival=135, future=159, group="positive", label=1)
    item = experiment.east_fit_record(None, None, 65944, case, 0.0)
    assert item == dict(shot=65944, group="positive", label=1, query=159, z=159.0)
    with pytest.raises(ValueError, match="clock"):
        experiment.east_fit_record(None, None, 65944, dict(case, future=158), 0.0)
