import numpy as np
import pytest
import torch

from latent_trajectory_adapt_20261105 import (
    RADIUS,
    bounded_delta,
    causal_history,
    labelled_queries,
    routes_all,
    weights_by_class_shot,
)


def test_history_is_unchanged_by_future_and_resets_per_shot():
    h = torch.arange(10, dtype=torch.float32)[:, None].expand(-1, 64)
    features, stale = causal_history(h)
    cut_features, cut_stale = causal_history(h[:5])
    torch.testing.assert_close(cut_features, features[:5])
    torch.testing.assert_close(cut_stale, stale[:5])
    assert features[0].abs().sum() == 0
    torch.testing.assert_close(stale[4:], features[:-4])
    torch.testing.assert_close(features[7, :64], torch.full((64,), 4.0))
    torch.testing.assert_close(features[7, 64:], torch.full((64,), 5.5))


def test_teaching_danger_window_is_strictly_timely_and_safe_maxima_use_only_eligible():
    t = np.arange(1600) / 1000
    p = np.full(len(t), 0.2)
    p[[2, 400, 1310]] = 0.9
    dq, lead = labelled_queries(t, p, True, 1.0)
    assert len(dq) == 8 and np.all((lead > 30) & (lead <= 125))
    assert dq[0] == 875 and dq[-1] == 969
    sq, absence = labelled_queries(t, p, False, None)
    assert len(sq) == 8 and min(sq) >= 135 and 400 in sq and 1310 in sq
    assert np.isnan(absence).all()


def test_equal_label_and_shot_weight_ignores_window_count():
    weights = weights_by_class_shot(
        np.array([11, 11, 11, 12, 17]), np.array([False, False, False, False, True])
    )
    np.testing.assert_allclose(weights, [1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 2])
    with pytest.raises(ValueError):
        weights_by_class_shot(np.array([1, 1]), np.array([False, True]))


def test_internal_route_does_not_activate_missing_tokens_and_delta_can_have_both_signs():
    available = torch.zeros(2, 128, 23, dtype=torch.bool)
    available[0, 8:16, 3] = True
    weights = routes_all(available)
    assert weights.shape == (2, 2, 17)
    assert weights[0, 0].nonzero().ravel().tolist() == [1, 16]
    assert weights[1, 1].nonzero().ravel().tolist() == [16]
    raw = torch.randn(1, 256) * 100
    d = bounded_delta(torch.cat((raw, -raw)))
    assert torch.all(d.norm(dim=1) <= RADIUS + 1e-6)
    torch.testing.assert_close(d[0], -d[1])


def test_auditor_window_selection_matches_both_ended_shot_protocols():
    from verify_latent_trajectory_adapt_20261105 import expected_queries

    t = np.arange(4503) / 1000
    p = np.random.default_rng(27).uniform(size=len(t))
    for disrupted, down in ((False, None), (True, 1.324), (True, 0.175)):
        q, lead = labelled_queries(t, p, disrupted, down)
        reference_q, reference_lead = expected_queries(t, p, disrupted, down, 135)
        np.testing.assert_array_equal(reference_q, q)
        np.testing.assert_array_equal(reference_lead, lead)
