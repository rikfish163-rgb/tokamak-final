import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.decoder_intra_20261005.source import (
    source_origins, weighted_quantile, activity_thresholds, sampling_pools,
    sample_indices, normalized_level_loss,
)


def shot(disrupt=True):
    return dict(device='J-TEXT',shot=1,is_disrupt=disrupt,down=.109,
                times=np.arange(130)/1000,valid=np.ones(130,dtype=bool))


def test_offline_source_cutoff_uses_next_prediction_strict_30ms():
    # o=52 first affects t=.078, lead31ms; o=78 is too late.
    assert source_origins(shot()).tolist()==[0,26,52]
    s=shot();s['down']=.108
    assert source_origins(s).tolist()==[0,26]  # equality at 30ms excluded
    assert source_origins(shot(False)).tolist()==[0,26,52,78]
    s=shot();s['device']='EAST'
    with pytest.raises(ValueError,match='source'):
        source_origins(s)


def test_weighted_quantile_and_activity_are_shot_balanced():
    assert weighted_quantile([1,2,9],[.25,.25,.5],.75)==9
    rows=[dict(shot=1,label=0,activity=1.) for _ in range(100)]
    rows += [dict(shot=2,label=0,activity=9.),dict(shot=3,label=1,activity=5.)]
    assert activity_thresholds(rows)=={0:9.,1:5.}
    with pytest.raises(ValueError):
        weighted_quantile([1],[0],.75)


def test_sampling_never_uses_absent_stratum_or_validation_labels():
    rows=[dict(shot=10*c+2*h+j,label=c,activity=float(h))
          for c in (0,1) for h in (0,1) for j in range(2)]
    pools=sampling_pools(rows,{0:.5,1:.5})
    idx=sample_indices(pools,np.random.default_rng(1),4000)
    groups=[(rows[i]['label'],int(rows[i]['activity']>.5)) for i in idx]
    assert all(850<groups.count(g)<1150 for g in pools)
    with pytest.raises(ValueError,match='empty'):
        sampling_pools(rows,{0:1.,1:1.})


def test_level_loss_is_masked_channel_balanced_and_no_target_grad():
    p=torch.zeros(1,2,23,requires_grad=True)
    t=torch.full_like(p,float('nan'));t[0,:,0]=2.;t[0,0,1]=4.
    t.requires_grad_();m=torch.isfinite(t)
    loss=normalized_level_loss(p,t,m)
    assert float(loss.detach())==pytest.approx((1.5+3.5)/2)
    loss.sum().backward()
    assert torch.isfinite(p.grad).all() and t.grad is None
    assert normalized_level_loss(p,t,torch.zeros_like(m)).item()==0.


def test_observed_nonfinite_forecast_fails():
    p=torch.zeros(1,25,23);p[0,0,0]=float('inf')
    with pytest.raises(FloatingPointError):
        normalized_level_loss(p,torch.zeros_like(p),torch.ones_like(p,dtype=torch.bool))
