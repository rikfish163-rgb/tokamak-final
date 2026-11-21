import numpy as np
import pytest

from experiments.risk_aligned_online.state_capacity_20261031.audit import geometry


def test_missing_subspace_direction_has_zero_local_response():
    a = np.array([[2., 0.], [0., 1.], [0., 0.]])
    r = geometry(a, np.array([0., 0., 1.]))
    assert r['rank'] == 2 and r['radius'] == 2
    assert r['gradient_subspace_fraction'] == 0
    assert r['local_context_to_free_ratio'] == 0


def test_anisotropy_and_direction_alignment_are_separate():
    a = np.array([[2., 0.], [0., 1.], [0., 0.]])
    strong = geometry(a, np.array([1., 0., 0.]))
    weak = geometry(a, np.array([0., 1., 0.]))
    assert strong['gradient_subspace_fraction'] == weak['gradient_subspace_fraction'] == 1
    assert strong['local_context_to_free_ratio'] == 1
    assert weak['local_context_to_free_ratio'] == .5


def test_degenerate_geometry_is_explicitly_not_identifiable():
    assert geometry(np.zeros((3, 2)), np.ones(3))['local_context_to_free_ratio'] is None
    assert geometry(np.eye(3), np.zeros(3))['gradient_subspace_fraction'] is None


def test_nonfinite_geometry_is_rejected():
    with pytest.raises(ValueError):
        geometry(np.array([[np.nan]]), np.ones(1))
