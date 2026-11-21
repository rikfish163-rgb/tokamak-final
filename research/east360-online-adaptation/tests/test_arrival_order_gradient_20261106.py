import numpy as np

from audit_arrival_order_gradient_20261106 import geometry


def test_shared_and_antisymmetric_components_separate():
    a = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, -2.0], [1.0, -1.0]])
    g = geometry(a, b)
    np.testing.assert_allclose(g[:, 0], [1, -1, 0])
    np.testing.assert_allclose(g[:, 1], [0, 1, 0.5])
