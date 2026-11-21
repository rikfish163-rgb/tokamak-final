import pytest
import torch

from experiments.risk_aligned_online.context_memory_20261028.scaled import ScaledContextMemory, weights_from_moments
from experiments.risk_aligned_online.context_memory_20261028.model import ContextMemory
from experiments.risk_aligned_online.tests.test_context_memory import observed


def test_weights_have_unit_horizon_mean_and_bounded_channel_ratio():
    moments = torch.ones(4, 23, dtype=torch.float64)
    moments[:, 0] = 1e6
    moments[:, 1] = 0
    weights = weights_from_moments(moments)
    assert torch.allclose(weights.mean(1), torch.ones(4))
    assert weights.max() < 2
    assert float((weights.max(1).values/weights.min(1).values).max()) == pytest.approx(100.)
    assert torch.equal(weights[:, 1], weights[:, 2])
    for bad in (torch.zeros(4, 23), torch.ones(3, 23), moments*float('nan'), -moments):
        with pytest.raises(ValueError):
            weights_from_moments(bad)


def test_scaled_write_matches_same_weighted_objective_gradient():
    weights = weights_from_moments(torch.arange(1., 24.).repeat(4, 1).square())
    m = ScaledContextMemory(64, weights)
    torch.manual_seed(4)
    c = (torch.randn(8)*.02).requires_grad_()
    basis = torch.randn(3, 23, 8)*.1
    target = torch.randn(3, 23)*.2
    mask = torch.rand(3, 23) > .3
    w = weights[:3]*mask
    loss = .5*((basis@c-target).square()*w).sum()
    g, = torch.autograd.grad(loss, c)
    candidate = c-m.eta*g/(1+(basis.square()*w[..., None]).sum())
    expected = candidate/candidate.norm().clamp_min(1.)
    actual = m.write(c, basis, target, mask)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert .5*((basis@actual-target).square()*w).sum() <= loss


def test_unit_weights_reproduce_legacy_context_without_reinitialization_change():
    torch.manual_seed(131); old = ContextMemory(64)
    torch.manual_seed(131); new = ScaledContextMemory(64, torch.ones(4, 23))
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    x,a = observed(20)
    left = old.trace(x,a,mode='memory',return_before=True)
    right = new.trace(x,a,mode='memory',return_before=True)
    for k in ('forecast','context','delta','before_delta','forecast_loss'):
        assert torch.allclose(left[k],right[k],atol=1e-7,rtol=1e-6)


def test_scaled_forecast_loss_and_prefix_causality():
    m = ScaledContextMemory(64, weights_from_moments(torch.arange(1.,24.).repeat(4,1)))
    x,a = observed(20)
    full=m.trace(x,a,mode='memory',return_before=True)
    short=m.trace(x[:10],a[:10],mode='memory',return_before=True)
    assert torch.allclose(full['delta'][:10],short['delta'],atol=1e-7)
    target=torch.stack([x[h:16+h]-x[:16] for h in range(1,5)],1)
    expected=((full['forecast'][:16]-target).square()*m.forecast_weights).sum()/(16*m.forecast_weights.sum())
    assert torch.allclose(full['forecast_loss'],expected)
    assert torch.equal(full['unweighted_forecast_loss'],super(ScaledContextMemory,m).trace(x,a,mode='memory')['forecast_loss'])
    constant=m.trace(torch.ones_like(x),a,mode='memory')
    assert not constant['context'].count_nonzero()
    assert torch.equal(full['delta'],m.trace(x,a,mode='memory')['delta'])


def test_scaled_memory_rejects_nonpositive_or_badly_normalized_weights():
    for w in (torch.zeros(4,23),torch.ones(3,23),torch.ones(4,23)*2,torch.full((4,23),float('nan'))):
        with pytest.raises(ValueError):
            ScaledContextMemory(64,w)


def test_weighted_persistence_uses_same_denominator_and_legacy_default():
    from experiments.risk_aligned_online.context_memory_20261028.forecast_audit import persistence_error
    x,a=observed(20)
    w=weights_from_moments(torch.arange(1.,24.).repeat(4,1).square())
    a[:,2]=False
    target=torch.stack([x[h:16+h]-x[:16] for h in range(1,5)],1)
    valid=torch.stack([a[h:16+h]&a[:16] for h in range(1,5)],1)
    expected=(target.square()*w*valid).sum()/(w*valid).sum()
    assert persistence_error(x,a,weights=w)==pytest.approx(float(expected),abs=1e-7)
    assert persistence_error(x,a,weights=torch.ones(4,23))==persistence_error(x,a)


def test_scaled_checkpoint_requires_explicit_profile_and_weights_provenance():
    from experiments.risk_aligned_online.context_memory_20261028.checkpoint import validate_metadata, SCALED_PROFILE
    from experiments.risk_aligned_online.redesign_20261004.prepare_prior import PARENT_SHA256
    roles=dict(train=list(range(1353)),val=list(range(2000,2160)),test=list(range(3000,3220)))
    record=dict(profile=SCALED_PROFILE,kind='context8',context_dim=8,epoch=2,feedback='progressive',readout='full',
        source_sha256=PARENT_SHA256,train_ids=roles['train'],val_ids=roles['val'],objective='detached_scaled_forecast',
        forecast_weight=1.,forecast_weights_sha256='a'*64,memory=ScaledContextMemory(64,torch.ones(4,23)).state_dict())
    validate_metadata(record,roles)
    for bad in (dict(record,forecast_weights_sha256=''),dict(record,profile='context8_forecast_v2'),
                dict(record,forecast_weight=0.)):
        with pytest.raises(ValueError):
            validate_metadata(bad,roles)
    assert 'forecast_weights' not in dict(ScaledContextMemory(64,torch.ones(4,23)).named_parameters())
