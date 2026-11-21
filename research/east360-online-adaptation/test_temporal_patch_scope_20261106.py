import numpy as np
import pytest

from run_temporal_patch_scope_20261106 import (
    energy_factor, first_spliced, temporal_weights, summarize,
)


def test_release_weight_is_on_observed_overlap_and_expires():
    r = 128
    ends = np.array([r - 8, r, r + 1, r + 4, r + 8, r + 24, r + 32, r + 60])
    fade = temporal_weights(ends, r, "fade")
    assert not fade[:, :-4].any()
    np.testing.assert_array_equal(fade[:, -4:], [
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, .875], [1, 1, 1, .5],
        [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(temporal_weights(ends, r, "recent")[:, -4:], 1.)
    np.testing.assert_array_equal(temporal_weights(ends, r, "full"), 1.)
    with pytest.raises(ValueError):
        temporal_weights(ends, r, "not-a-scope")


def test_energy_is_matched_per_state_and_endpoint_in_actual_patch_space():
    rng = np.random.default_rng(37)
    f = rng.normal(size=(3, 16, 8))
    f[1] = 0.0
    p, _ = np.linalg.qr(rng.normal(size=(64, 8)))
    u = rng.normal(size=(2, 3, 64))
    u[1, 2] = 0.0
    weights = temporal_weights(np.array([128, 133, 149]), 128, "fade")
    ratio = energy_factor(f, u, weights)
    assert ratio.shape == (2, 3, 3)
    assert not ratio[..., 1].any() and not ratio[1, 2].any()
    assert np.all((ratio >= 0.) & (ratio <= 1.))
    for i, j in np.ndindex(u.shape[:-1]):
        v = (f @ u[i, j].reshape(8, 8)) @ p.T
        np.testing.assert_allclose(
            np.linalg.norm(v * ratio[i, j, :, None, None], axis=(1, 2)),
            np.linalg.norm(v * weights[:, :, None], axis=(1, 2)),
            atol=1e-12,
        )


def test_first_alarm_preserves_unwritten_past_and_future_and_strict_lead():
    t = np.arange(8) / 1000.
    z = np.array([-1., -1., -1., -1., -1., .3, -1., -1.])
    idx = np.array([2, 3])
    hit = np.array([[.25, .7], [-.5, -.4], [-.5, .3]])
    k, lead = first_spliced(t, z, idx, hit, down=.012, tau=.25)
    np.testing.assert_array_equal(k, [2, 5, 3])
    np.testing.assert_array_equal(lead, [10., 7., 9.])
    assert not (lead > 10.).any()
    z[1] = .3
    np.testing.assert_array_equal(first_spliced(t, z, idx, hit, down=None, tau=.25)[0], 1)
    z[:] = 0.
    np.testing.assert_array_equal(first_spliced(t, z, idx, hit, down=.001, tau=.25)[0], -1)


def test_gate_counts_physical_shots_and_excludes_full_energy_and_scalar():
    modes = ("full", "recent", "fade", "recent_energy", "fade_energy")

    def one(shot, rescues=(), scalar=()):
        return dict(shot=shot, exact={name: dict(
            danger=int(name in rescues), paired=int(name in rescues),
            neural_only=int(name in rescues), scalar_any=int(name in scalar), safe_reject=0,
        ) for name in modes})

    records = {"native": [one(1, ("recent",)), one(2, ("full", "recent")),
                           one(3, ("recent", "recent_energy")), one(4, ("recent",))],
               "no_MNM": [one(1, ("recent",)), one(5, ("recent",), ("fade",))],
               "low": [one(6, ("recent",))]}
    summary, gate = summarize(records, {4})
    assert summary["native"]["recent"]["neural_only"] == [1, 2, 3, 4]
    assert gate["recent"]["original_strict"] == [1, 2, 3, 4]
    assert gate["recent"]["original_local_only"] == [1]
    assert gate["recent"]["promote"]
    assert gate["recent"]["all_strict"] == [1, 2, 3, 4, 6]
    assert not gate["fade"]["promote"]
