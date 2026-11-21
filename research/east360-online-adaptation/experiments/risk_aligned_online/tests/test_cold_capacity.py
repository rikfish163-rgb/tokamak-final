import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cold_capacity_20261029.probe import choose_ends, optimize, logits
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.latent_completion_20261028.model import tokens, readout


def test_selection_obeys_strict_lead_and_post_update_clock():
    times = np.arange(300) / 1000
    p = np.zeros(300); p[50] = .99; p[270] = .98; p[160] = .9; p[250] = .8
    rows = choose_ends(times, p, True, .300)
    assert rows == [dict(end=160, reasons=['timely_peak']), dict(end=250, reasons=['positive_window_peak'])]
    p[250] = .95
    assert choose_ends(times, p, True, .300) == [dict(end=250, reasons=['timely_peak', 'positive_window_peak'])]
    assert choose_ends(times, p, False, None) == [dict(end=270, reasons=['safe_peak'])]
    assert choose_ends(times[:128], p[:128], False, None) == []


def test_normalized_optimizer_is_bounded_independent_and_saves_best():
    weights = torch.tensor([[2., -1.], [1., 3.]])
    signs = torch.tensor([1., -1.])
    f = lambda s: (weights*s).sum(1)
    result = optimize(f, torch.zeros(2, 2), signs, radius=.4, steps=8, step=.1)
    assert result['logits'].shape == (9, 2)
    assert torch.all(result['norms'] <= .400001)
    torch.testing.assert_close(result['state'], .4 * signs[:, None]*weights/weights.norm(dim=1, keepdim=True))
    for i in range(2):
        one = optimize(lambda s: (weights[i:i+1]*s).sum(1), torch.zeros(1, 2), signs[i:i+1], radius=.4, steps=8, step=.1)
        torch.testing.assert_close(result['state'][i:i+1], one['state'])
    assert not result['state'].requires_grad


def test_best_includes_unmodified_state_and_zero_gradient():
    result = optimize(lambda s: -s.square().sum(1), torch.zeros(1, 2), torch.ones(1), radius=1., steps=3, step=.1)
    assert result['best_step'].tolist() == [0]
    assert torch.equal(result['state'], torch.zeros(1, 2))
    assert torch.equal(result['gradient_norm'], torch.zeros(1))


def test_bad_direction_or_nonfinite_objective_rejected():
    with pytest.raises(ValueError):
        optimize(lambda s: s.sum(1), torch.zeros(1, 2), torch.zeros(1), radius=1., steps=1, step=.1)
    with pytest.raises(FloatingPointError):
        optimize(lambda s: s.sum(1)*float('nan'), torch.zeros(1, 2), torch.ones(1), radius=1., steps=1, step=.1)


def test_model_probe_matches_current_readout_and_keeps_parameters_frozen():
    torch.set_num_threads(1); torch.manual_seed(42); torch.backends.mha.set_fastpath_enabled(False)
    base = Simple23(Config(fast_site='tokens'), np.zeros(23), np.ones(23)).eval().requires_grad_(False)
    saved = {k:v.clone() for k,v in base.state_dict().items()}
    x = torch.randn(2, 128, 23); a = torch.ones_like(x, dtype=torch.bool)
    slow = torch.randn(64, 64)*.01
    z = logits(base, x, a, torch.zeros(2,128), slow)
    h, valid = tokens(base, x, a, torch.zeros(2,128))
    torch.testing.assert_close(z, readout(base,h,valid,slow), rtol=0, atol=0)
    result = optimize(lambda s: logits(base,x,a,s,slow), torch.zeros(2,128), torch.tensor([1.,-1.]), radius=1., steps=4, step=.1)
    for i in range(2):
        one = optimize(lambda s: logits(base,x[i:i+1],a[i:i+1],s,slow), torch.zeros(1,128), torch.tensor([1. if i==0 else -1.]), radius=1., steps=4, step=.1)
        torch.testing.assert_close(result['state'][i:i+1], one['state'], atol=2e-6, rtol=1e-4)
    assert all(torch.equal(v,saved[k]) for k,v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())


def test_late_signal_site_zero_equivalence_and_no_cls_shortcut():
    from experiments.risk_aligned_online.cold_capacity_20261029.site_probe import adapt_signals, late_logits
    torch.set_num_threads(1);torch.manual_seed(1);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    x=torch.randn(2,128,23);a=torch.ones_like(x,dtype=torch.bool);s=torch.randn(2,128)*.1
    h,v=tokens(base,x,a,torch.zeros_like(s))
    changed=adapt_signals(h,s)
    assert torch.equal(changed[:,-1],h[:,-1])
    assert not torch.equal(changed[:,:-1],h[:,:-1])
    torch.testing.assert_close(late_logits(base,x,a,torch.zeros_like(s),None),logits(base,x,a,torch.zeros_like(s),None),rtol=0,atol=0)
    both=optimize(lambda u:late_logits(base,x,a,u,None),torch.zeros_like(s),torch.tensor([1.,-1.]),radius=1,steps=3,step=.1)
    for i in range(2):
        one=optimize(lambda u:late_logits(base,x[i:i+1],a[i:i+1],u,None),torch.zeros(1,128),torch.tensor([1. if i==0 else -1.]),radius=1,steps=3,step=.1)
        torch.testing.assert_close(both['state'][i:i+1],one['state'],rtol=1e-4,atol=2e-6)
    assert all(p.grad is None for p in base.parameters())
