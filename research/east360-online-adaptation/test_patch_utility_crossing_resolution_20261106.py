import numpy as np

import run_patch_utility_crossing_resolution_20261106 as r


def test_exact_segment_average_can_erase_a_single_frame_alarm():
    z = np.full((1,2,24),-.05)
    z[0,1,6] = .03
    averaged = r.proxy(z[:,0],r.exact_mean(z))
    assert averaged[0,1,6] < 0 < z[0,1,6]
    assert np.array_equal(averaged[:,0],z[:,0])


def test_reconstructed_static_does_not_depend_on_candidate_prediction():
    static = np.arange(24,dtype=float)[None]
    shifts = np.array([[[1.,2.,3.],[.1,.2,.3]]])
    z = r.proxy(static,shifts)
    assert np.array_equal(z[:,0],static)
    assert np.allclose(z[0,1]-static[0],np.repeat([.1,.2,.3],8))
