import pytest
import torch
from experiments.risk_aligned_online.anchored_fast_window_20261023.reverse_probe import match_write_norm


def test_matched_write_preserves_norm_and_direction():
    a=torch.tensor([[[3.,4.],[0.,0.]],[[0.,1.],[0.,0.]]]);b=a*torch.tensor([2.,3.])[:,None,None]
    c=match_write_norm(a,b)
    torch.testing.assert_close(c,b)
    torch.testing.assert_close(match_write_norm(a,torch.zeros_like(b)),torch.zeros_like(a))


def test_zero_write_cannot_be_given_a_nonzero_direction():
    z=torch.zeros(1,2,2)
    torch.testing.assert_close(match_write_norm(z,z),z,atol=0,rtol=0)
    with pytest.raises(ValueError): match_write_norm(z,torch.ones_like(z))
