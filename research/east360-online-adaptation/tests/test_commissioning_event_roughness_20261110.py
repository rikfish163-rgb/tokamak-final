import numpy as np

from experiments.risk_aligned_online.commissioning_event_roughness_20261110 import (
    directional_features, event_endpoints, prefix_context,
)


def test_directional_features_distinguish_persistence_from_turning():
    z = np.zeros((70, 2), np.float32)
    z[0:2] = [3., -4.]
    z[32:34] = [[1., -8.], [-3., 2.]]
    out = directional_features(z)
    np.testing.assert_array_equal(out[:, :2], z)
    np.testing.assert_array_equal(out[:32, 2:], 0.)
    np.testing.assert_array_equal(out[32], [1., -8., 1., -4.])
    np.testing.assert_array_equal(out[33], [-3., 2., 0., 0.])
    np.testing.assert_array_equal(out[:61], directional_features(z[:61]))


def test_event_endpoints_keep_only_observed_labelled_early_samples():
    t = .15 + np.arange(300) / 1000.
    event = np.full(len(t), -1)
    event[120:230] = 3
    event[235:270] = 4
    active = np.ones_like(event, bool)
    active[172:185] = False
    selected = event_endpoints(t, event, active, disruptive=True, down=t[250])
    assert selected == [(3, 171)]
    # Exact 30ms is out, even if it is the only available annotation endpoint.
    active[:] = False
    active[220] = True
    assert event_endpoints(t, event, active, disruptive=True, down=t[250]) == []


def test_safe_candidates_require_annotation_never_treat_unknown_as_normal():
    t = .15 + np.arange(300) / 1000.
    event = np.full(len(t), -1)
    event[129:152] = 3
    active = np.ones_like(event, bool)
    active[136] = False
    assert event_endpoints(t, event, active, disruptive=False, down=None) == [(3, 144)]


def test_context_is_causal_and_requires_available_current_history():
    t = np.arange(100, dtype=float)
    z = np.stack((t / 16., np.full_like(t, 2.)), axis=1)
    mask = np.ones_like(z, bool)
    mask[82:, 0] = False
    ctx, ok = prefix_context(z, mask)
    np.testing.assert_allclose(ctx[75], [75 / 16., 2., 2.])
    assert ok[75] and not ok[85] and not ok[0]
    a, b = prefix_context(z[:80], mask[:80])
    np.testing.assert_array_equal(ctx[:80], a)
    np.testing.assert_array_equal(ok[:80], b)
