import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.supervised_density_gain_20261023.learning import (
    apply_gain, export_map,
)


def test_gain_matches_frozen_numpy_mapping_and_changes_only_observed_density():
    from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform
    x=np.random.default_rng(131).normal(size=(2,128,23)).astype(np.float32)
    m=np.ones_like(x,dtype=bool);m[:,3,12]=False;x[:,7,12]=0
    g=0.00036155007389467784
    out=apply_gain(torch.tensor(x),torch.tensor(m),torch.tensor(np.log(g),dtype=torch.float64))
    mapping=dict(gain=[g if c==12 else 1. for c in range(23)],offset=[0.]*23)
    for i in range(2):
        expected=transform(dict(values=x[i],available=m[i]),mapping)['values']
        np.testing.assert_array_equal(out.numpy()[i],expected)
    np.testing.assert_array_equal(out.numpy()[...,np.arange(23)!=12],x[...,np.arange(23)!=12])
    assert np.all(out.numpy()[:,7,12]==0)
    np.testing.assert_array_equal(out.numpy()[:,3,12],x[:,3,12])


def test_gradient_updates_one_scalar_not_frozen_risk_parameters():
    from experiments.risk_aligned_online.simple23 import Config,Simple23
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    torch.set_num_threads(1);torch.manual_seed(7)
    model=PairedFeedback(Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen');before={k:v.clone() for k,v in model.state_dict().items()}
    x=torch.randn(2,128,23);m=torch.ones_like(x,dtype=torch.bool)
    logg=torch.tensor(-1.,dtype=torch.float64,requires_grad=True)
    h=model(apply_gain(x,m,logg),m,mode='risk_state')['risk_state']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model.base.risk_head(h).squeeze(-1),torch.tensor([0.,1.]))
    loss.backward()
    assert logg.grad is not None and torch.isfinite(logg.grad) and logg.grad.abs()>0
    assert all(p.grad is None for p in model.parameters())
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_invalid_gain_and_shapes_are_errors():
    x=torch.ones(2,128,23);m=torch.ones_like(x,dtype=torch.bool)
    for g in (torch.tensor([0.]),torch.tensor(float('nan')),torch.tensor(float('inf'))):
        with pytest.raises(ValueError):apply_gain(x,m,g)
    with pytest.raises(ValueError):apply_gain(x,m.float(),torch.tensor(0.))
    with pytest.raises(FloatingPointError):apply_gain(x,m,torch.tensor(1000.,dtype=torch.float64))


def test_export_keeps_statistics_and_other_channels_and_does_not_mutate_parent():
    old=dict(gain=[.001 if c==12 else 1. for c in range(23)],offset=[0.]*23,
             channels=[dict(name=str(c),gain=.001 if c==12 else 1.,offset=0.) for c in range(23)],
             policy='density_gain',calibration_ids=[1,2],source_statistics=dict(example=3))
    out=export_map(old,.002)
    assert old['gain'][12]==.001 and out['gain'][12]==.002 and out['channels'][12]['gain']==.002
    assert out['source_statistics']==old['source_statistics'] and out['calibration_ids']==[1,2]
    assert out['fitted_input_parameters']==1 and not out['physical_calibration_certified']
