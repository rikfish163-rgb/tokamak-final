import pytest
import torch

from run_rls_conditional835_20261106 import delayed_direction


def test_exact_norm_and_causal_delayed_nonzero_source():
    d = torch.tensor([[0.,0.],[2.,0.],[0.,0.],[0.,3.],[0.,4.],[0.,5.],[0.,0.]])
    c, source = delayed_direction(d, lag=2)
    torch.testing.assert_close(c.tanh().norm(dim=1),d.tanh().norm(dim=1),atol=2e-6,rtol=0.)
    ref=torch.tensor([[0.,0.],[2.,0.],[0.,0.],[3.,0.],[4.,0.],[0.,5.],[0.,0.]])
    torch.testing.assert_close(c.tanh(),ref.tanh(),atol=2e-6,rtol=0.)
    assert source.tolist() == [-1,-1,-1,1,1,3,-1]


def test_truncated_trace_cannot_change_earlier_directions():
    generator = torch.Generator().manual_seed(114)
    x = torch.randn(31,5,generator=generator)
    x[:3] = 0.
    x[12:17] = 0.
    full, ix = delayed_direction(x,lag=7)
    for n in (2,7,10,15,19,31):
        old, previous = delayed_direction(x[:n],lag=7)
        torch.testing.assert_close(old,full[:n],rtol=0.,atol=0.)
        torch.testing.assert_close(previous,ix[:n],rtol=0.,atol=0.)


def test_no_nonzero_history_is_an_explicit_shared_prefix():
    x = torch.zeros(4,128)
    x[-1,2] = 7
    old, source = delayed_direction(x,lag=128)
    torch.testing.assert_close(old,x)
    assert (source == -1).all()
    with pytest.raises(ValueError):
        delayed_direction(x,lag=0)


def test_equal_raw_norm_is_not_a_match_for_saturating_token_coefficients():
    x=torch.tensor([[3.,3.],[4.242640687,0.]])
    old, source=delayed_direction(x,lag=1)
    assert source.tolist()==[-1,0]
    assert old[1,0] == old[1,1]
    assert old[1].norm() < x[1].norm()
    torch.testing.assert_close(old.tanh().norm(dim=1),x.tanh().norm(dim=1),atol=2e-6,rtol=0.)


def test_unreachable_effective_energy_is_an_explicit_shared_action():
    x=torch.tensor([[2.,0.],[3.,3.]])
    old, source=delayed_direction(x,lag=1)
    assert source.tolist()==[-1,-1]
    torch.testing.assert_close(old,x)
