import numpy as np

import run_patch_utility_crossing_last_write_20261106 as a


def test_wait_excludes_crossing_sample_and_rechecks_both_signs():
    z = np.zeros((3,32))
    z[2,8] = .11
    z[1,9] = .1
    assert a.legal_delays(z,np.zeros(33),0,11,1,.9,.1).tolist() == [8,10]


def test_wait_stops_when_static_prefix_has_crossed():
    z = np.zeros((3,32))
    native = np.zeros(33)
    native[9] = .9
    assert a.legal_delays(z,native,0,13,1,.9,.1).tolist() == [8]
