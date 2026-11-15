import numpy as np
import pytest

from experiments.risk_aligned_online.commissioning_event_roughness_20261110 import CUE
from experiments.risk_aligned_online.commissioning_shared_variation_20261110 import PAIR, pair_view


def test_pair_view_replaces_only_two_profile_contrasts_and_their_persistent_terms():
    base = np.full((48, 2 * len(CUE)), 7.0, dtype=np.float32)
    pair = np.full((48, 2), (2.0, -3.0), dtype=np.float32)
    pair[33, 1] = 4.0
    view = pair_view(base, pair)
    current = list(PAIR)
    persistent = [len(CUE) + j for j in PAIR]
    other = np.setdiff1d(np.arange(base.shape[1]), current + persistent)
    np.testing.assert_array_equal(view[:, other], base[:, other])
    np.testing.assert_array_equal(view[:, current], pair)
    np.testing.assert_array_equal(view[:32, persistent], 0.0)
    np.testing.assert_array_equal(view[32, persistent], [2.0, -3.0])
    np.testing.assert_array_equal(view[33, persistent], [2.0, 0.0])
    np.testing.assert_array_equal(base, 7.0)


def test_pair_view_does_not_accept_misaligned_rows():
    base = np.zeros((30, 2 * len(CUE)), np.float32)
    with pytest.raises(ValueError, match='shape'):
        pair_view(base, np.zeros((29, 2), np.float32))
