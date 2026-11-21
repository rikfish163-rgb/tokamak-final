import numpy as np
from scipy.spatial import cKDTree

from experiments.risk_aligned_online.commissioning_revealed_support_20261114 import matching


def test_window_matching_requires_prior_reveal_same_validity_and_all_distance_axes():
    shot = dict(rank=81, coords=np.array([[1., 2., 3.], [2., 4., 5.]]), mask_state=np.array([1, 0]))
    support = np.array([True, False])
    yes = cKDTree([[1.5, 2.75, 3.]])
    too_far = cKDTree([[1., 2., 4.1]])
    bank = [(1,81,{1:yes}), (2,82,{1:yes}), (3,81,{0:yes}), (4,81,{1:too_far})]
    assert matching(shot,bank,support)==[1]
