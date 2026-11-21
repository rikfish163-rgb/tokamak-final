import numpy as np

from experiments.risk_aligned_online import commissioning_radial_response_20261111 as radial


def test_outward_and_return_differ_even_at_equal_endpoint_and_step_size():
    outward = radial.excursion(np.array([[1., 0.], [2., 0.], [3., 0.]]), np.zeros(2), 1.)
    inward = radial.excursion(np.array([[5., 0.], [4., 0.], [3., 0.]]), np.zeros(2), 1.)
    assert (outward[:3] > 0.).all()
    assert (inward[:3] < 0.).all()
    np.testing.assert_array_equal(outward[3:], inward[3:])


def test_only_completed_measurements_create_a_state_and_future_is_inert():
    x = np.zeros((250, 4))
    x[:, 2] = np.arange(len(x))*.01
    m = np.ones_like(x, bool)
    full = radial.replay(x, m)
    np.testing.assert_array_equal(full['release'], [153, 185, 217, 249])
    assert not full['state'][:153].any()
    assert full['state'][153, 0] > 0.
    for end in (153, 154, 218):
        part = radial.replay(x[:end], m[:end])
        np.testing.assert_array_equal(part['state'], full['state'][:end])


def test_no_reference_or_bad_context_means_no_current_adaptation():
    x = np.zeros((154, 4))
    x[:, 2] = np.arange(len(x))*.01
    m = np.ones_like(x, bool)
    m[49:57, 2] = False
    result = radial.replay(x, m)
    assert not result['eligible'][0]
    assert not result['state'].any()
    m[:] = True
    x[121:153, 0] = 3.
    assert not radial.replay(x, m)['eligible'][0]
