import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.lifecycle_stream_20261025.protocol import (
    order_stream, AccessLedger, sampled_endpoints,
)
from experiments.risk_aligned_online.lifecycle_stream_20261025.model import LifecycleAdapter
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_order_uses_prefix_signals_only_and_never_drops_roster_members():
    def read(sid):
        return dict(planning_ip=np.full(100, sid*1e5), planning_Bt=np.ones(100), rate=1000.)
    order = order_stream([3, 1, 2], reader=read)
    assert [r['shot'] for r in order] == [1, 2, 3]
    assert order[0]['proxy'] == 1.
    def bad(sid):
        r=read(sid); r['planning_ip'][0]=np.nan; return r
    with pytest.raises(ValueError):order_stream([1], reader=bad)


def test_outcomes_require_current_seal_and_cold_start_is_inside_stream():
    ledger=AccessLedger([30, 10, 20], warmup=2)
    with pytest.raises(ValueError):ledger.reveal(30)
    ledger.seal(30, 'a'*64); ledger.reveal(30)
    assert not ledger.ready and ledger.history == [30]
    with pytest.raises(ValueError):ledger.seal(20, 'b'*64)
    ledger.seal(10, 'b'*64); ledger.reveal(10)
    assert ledger.ready and ledger.history == [30, 10]
    with pytest.raises(ValueError):ledger.reveal(10)


def test_sampled_training_windows_include_timely_positive_and_hard_safe():
    times=np.arange(500)/1000
    score=np.zeros(500);score[211]=1
    y=(times>=.375).astype(float); valid=np.ones(500,bool)
    ix=sampled_endpoints(times,y,valid,score,down=.5)
    assert 211 in ix and np.any((times[ix]>=.375)&(times[ix]<.47))
    assert len(ix)<=64 and np.all(ix>=127)


def test_zero_adapter_matches_qualified_architecture_and_fast_signal_is_causal():
    torch.manual_seed(131); torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    cfg=Config(fast_site='tokens',fast_rule='normalized')
    base=Simple23(cfg,np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    model=LifecycleAdapter(base).eval()
    x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    h,v=model.tokens(x,mask)
    with torch.no_grad():
        z=model.risk(h,v,torch.zeros(2,92)); expected=base(x,mask)['logit']
    torch.testing.assert_close(z,expected,atol=0,rtol=0)
    signal=dict(values=np.random.default_rng(1).normal(size=(160,23)).astype('float32'),
                available=np.ones((160,23),bool))
    d,_=model.memory_delta(signal)
    later=dict(values=signal['values'].copy(),available=signal['available'].copy())
    later['values'][80:]+=10
    dd,_=model.memory_delta(later)
    torch.testing.assert_close(d[:80],dd[:80],atol=0,rtol=0)
    d2,_=model.memory_delta(signal)
    torch.testing.assert_close(d,d2,atol=0,rtol=0)
    assert not d[:40].any() and d[40:].abs().max()>0


def test_internal_adapter_backward_updates_only_adaptation_weights():
    torch.manual_seed(131);torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized'),
                  np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    model=LifecycleAdapter(base).eval()
    x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool)
    h,v=model.tokens(x,mask);original=h.detach().clone()
    delta=torch.randn(2,92)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(
        model.risk(h,v,delta),torch.tensor([0.,1.]))
    loss.backward()
    torch.testing.assert_close(h,original,rtol=0,atol=0)
    for parameter in (model.slow.weight,model.memory.output.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    assert all(p.grad is None for p in base.parameters())
