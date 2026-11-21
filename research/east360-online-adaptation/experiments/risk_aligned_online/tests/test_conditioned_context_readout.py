import torch
import pytest

from experiments.risk_aligned_online.context_memory_20261028.conditioned import ConditionedReadout
from experiments.risk_aligned_online.context_memory_20261028.model import ContextAdapter
from experiments.risk_aligned_online.tests.test_source_task_memory import setup


@pytest.mark.parametrize('kind', ['additive', 'state_conditioned'])
def test_zero_feedback_is_exact_parent_and_only_readout_gets_gradients(kind):
    old, e = setup()
    model = ConditionedReadout(old.base, kind=kind).eval()
    baseline = ContextAdapter(old.base).eval()
    delta = torch.randn(4, 92)
    actual = model.risk(e['h'], e['valid'], delta)
    torch.testing.assert_close(actual, baseline.risk(e['h'], e['valid'], torch.zeros_like(delta)), atol=0, rtol=0)
    actual.sum().backward()
    nonzero = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
    assert nonzero == ['memory.output.weight']
    with torch.no_grad():
        model.memory.output.weight.normal_()
    torch.testing.assert_close(model.risk(e['h'], e['valid'], torch.zeros_like(delta)),
                               baseline.risk(e['h'], e['valid'], torch.zeros_like(delta)), atol=0, rtol=0)


def test_same_feedback_conditions_shift_on_current_internal_features():
    old, _ = setup()
    a = ConditionedReadout(old.base, kind='additive')
    c = ConditionedReadout(old.base, kind='state_conditioned')
    with torch.no_grad():
        a.memory.output.weight.normal_(std=.2)
        c.memory.output.weight.copy_(a.memory.output.weight)
    last = torch.randn(1, old.base.cfg.dim).repeat(2, 1)
    last[1] *= -1
    d = torch.randn(1, 92).repeat(2, 1)
    sa, sc = a.internal_shift(last, d), c.internal_shift(last, d)
    torch.testing.assert_close(sa[0], sa[1])
    torch.testing.assert_close(sc[0], -sc[1])
    assert (sa.norm(dim=-1) <= .1*old.base.cfg.dim**.5+1e-6).all()
    assert (sc.norm(dim=-1) <= .1*old.base.cfg.dim**.5+1e-6).all()
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in c.parameters())
    assert torch.equal(c.internal_shift(torch.zeros_like(last), d), torch.zeros_like(last))


def test_old_readout_is_unchanged_by_new_hook():
    old, e = setup()
    model = ContextAdapter(old.base).eval()
    with torch.no_grad():
        model.memory.output.weight.normal_(std=.01)
    d = torch.randn(4, 92)
    torch.testing.assert_close(model.internal_shift(e['h'][:, -2], d), model.memory.output(d), atol=0, rtol=0)
