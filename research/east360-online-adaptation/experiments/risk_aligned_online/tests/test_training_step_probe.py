import copy
import pytest
import torch

from ..training_step_probe_20261023.learning import probe_steps, subtract


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def setup():
    torch.manual_seed(151)
    m=torch.nn.Linear(2,1).double()
    opt=torch.optim.Adam(m.parameters(),lr=.03)
    x=torch.tensor([[1.,2.]],dtype=torch.float64)
    m(x).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    return m,opt,x


def test_probe_matches_independent_adam_step_and_restores_all_state():
    m,opt,x=setup();pars=list(m.parameters())
    grad=list(torch.autograd.grad(m(x).square().sum(),pars))
    state=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    expected=copy.deepcopy(m);other=torch.optim.Adam(expected.parameters(),lr=.03)
    other.load_state_dict(copy.deepcopy(optimizer))
    for p,g in zip(expected.parameters(),grad):p.grad=g.clone()
    torch.nn.utils.clip_grad_norm_(list(expected.parameters()),1.);other.step()
    r=probe_steps(m,opt,pars,{'full':grad,'zero':[torch.zeros_like(g) for g in grad]},lambda:m(x).detach())
    torch.testing.assert_close(r['full'],expected(x).detach(),rtol=0,atol=0)
    # Zero CURRENT gradient does not erase Adam's accumulated momentum.
    assert not torch.equal(r['zero'],m(x).detach())
    equal(m.state_dict(),state);equal(opt.state_dict(),optimizer)
    assert all(p.grad is None for p in pars)


def test_probe_restores_state_even_when_reader_raises():
    m,opt,x=setup();pars=list(m.parameters());m(x).sum().backward()
    grad=[p.grad.clone() for p in pars];state=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    def fail():raise RuntimeError('diagnostic reader failure')
    with pytest.raises(RuntimeError,match='reader failure'):probe_steps(m,opt,pars,{'full':grad},fail)
    equal(m.state_dict(),state);equal(opt.state_dict(),optimizer)
    for p,g in zip(pars,grad):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)


def test_gradient_subtraction_checks_lengths_shapes_and_finiteness():
    a=[torch.tensor([1.,3.]),torch.tensor([2.])];b=[torch.tensor([.2,.3]),torch.tensor([.5])]
    for x,y,z in zip(subtract(a,b),a,b):torch.testing.assert_close(x,y-z)
    with pytest.raises(ValueError):subtract(a,b[:1])
    with pytest.raises(ValueError):subtract(a,[torch.ones(3),b[1]])
    with pytest.raises(FloatingPointError):subtract(a,[torch.tensor([float('nan'),0.]),b[1]])
