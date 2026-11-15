import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_radiation_reference_20261110 import (
    RADIATION, mixed_features,
)
from experiments.risk_aligned_online.commissioning_event_roughness_20261110 import CUE


def test_hold_and_only_are_complementary_interventions_on_two_profiles():
    n = len(CUE)
    z = np.arange(40 * n, dtype=np.float32).reshape(-1, 2*n)
    y = z + (np.arange(2*n, dtype=np.float32) + 1.)
    modes = mixed_features(z, y)
    hold, only = modes['hold_profiles'], modes['only_profiles']
    radiation = list(RADIATION) + [n + i for i in RADIATION]
    other = np.setdiff1d(np.arange(2*n), radiation)
    np.testing.assert_array_equal(hold[:, radiation], z[:, radiation])
    np.testing.assert_array_equal(hold[:, other], y[:, other])
    np.testing.assert_array_equal(only[:, radiation], y[:, radiation])
    np.testing.assert_array_equal(only[:, other], z[:, other])
    np.testing.assert_array_equal((hold-z)+(only-z), y-z)


def test_anchor_persistent_profile_cue_without_rewriting_recent_cue():
    n = len(CUE)
    z = np.zeros((155, 2*n), np.float32)
    y = z.copy()
    y[153:] = 1.
    out = mixed_features(z, y)['hold_profile_persistence']
    keep = [n+i for i in RADIATION]
    np.testing.assert_array_equal(out[:153], z[:153])
    np.testing.assert_array_equal(out[:, keep], z[:, keep])
    rest = np.setdiff1d(np.arange(2*n), keep)
    np.testing.assert_array_equal(out[:, rest], y[:, rest])
    assert y[-1].sum() == 2*n
    with pytest.raises(ValueError, match='shape'):
        mixed_features(z, y[:, :-1])
