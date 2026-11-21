import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import commissioning_surplus_20261111 as run


def test_source_noise_scales_keep_fixed_amplitude_and_ignore_unobserved():
    a = np.array([[0., 0.], [.1, .01], [.2, .02], [.3, 123.]], np.float32)
    good = np.array([[True, True], [True, True], [True, True], [True, False]])
    base = run.noise_scale([(a, good)])
    np.testing.assert_allclose(base, [.225, .025], atol=1e-7)
    np.testing.assert_array_equal(base, run.noise_scale([(np.where(good,a,np.nan), good)]))
    good[:,1] = False
    with pytest.raises(ValueError, match='observed'):
        run.noise_scale([(a, good)])


def test_drive_does_not_change_representation_before_full_feedback():
    q = run.FIRST
    start = q+run.HORIZON+1
    key = np.ones((start+10,4),np.float32)/2.
    phi = np.ones((1,run.PHI_DIM),np.float32)
    delta,n = run.drive(key,np.array([q]),phi,np.array([True]))
    assert n == 1
    np.testing.assert_array_equal(delta[:start],0)
    np.testing.assert_allclose(delta[start:],run.UPDATE_STEP/np.sqrt(run.PHI_DIM),rtol=1e-6)
    pre = torch.full((len(key),2),.4)
    mapper = torch.full((run.PHI_DIM,2),.3)
    h = run.project(pre,torch.from_numpy(delta),mapper)
    torch.testing.assert_close(h[:start],pre[:start].tanh(),atol=0,rtol=0)
    assert torch.all(h[start:] > pre[start:].tanh())


def test_bounded_map_is_identity_on_small_weights_and_caps_large_weights():
    small = torch.full((run.PHI_DIM,24),.01)
    torch.testing.assert_close(run.bounded_map(small),small,atol=0,rtol=0)
    assert float(run.bounded_map(small*1000).norm()) == pytest.approx(run.MAPPER_RADIUS)
