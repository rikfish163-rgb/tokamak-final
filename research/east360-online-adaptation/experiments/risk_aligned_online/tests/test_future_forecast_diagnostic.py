import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.future_forecast_20261023.diagnostic import (
    issue, forecast_error, replay, summarize,
)
from experiments.risk_aligned_online.tests.test_forecast_token_scope import models
from experiments.risk_aligned_online.forecast_token_scope_20261023.learning import temporary, ARMS


def signal(n):
    return dict(values=np.random.default_rng(9).normal(size=(n,23)).astype('float32'),
                available=np.ones((n,23),bool))


def test_issue_takes_only_observed_prefix_and_does_not_modify_parameters():
    m=models()['full'];fast=temporary(m,'all');before={k:v.clone() for k,v in fast.items()}
    s=signal(256);p=issue(m,s,fast,device='cpu')
    assert p.shape==(25,23) and np.isfinite(p).all()
    for k,v in fast.items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with pytest.raises(ValueError):issue(m,dict(s,y=1),fast,device='cpu')
    with pytest.raises(ValueError):issue(m,signal(127),fast,device='cpu')


def test_forecast_loss_masks_missing_targets_and_never_calls_empty_zero_success():
    p=np.ones((25,23));truth=np.zeros_like(p);mask=np.ones_like(p,dtype=bool)
    result=forecast_error(p,truth,mask)
    assert result['mean']==pytest.approx(.5) and result['observed']==575
    mask[:,0]=False;truth[:,0]=np.nan
    result=forecast_error(p,truth,mask)
    assert result['mean']==pytest.approx(.5) and result['per_channel'][0] is None
    empty=forecast_error(p,truth,np.zeros_like(mask))
    assert empty['mean'] is None and empty['observed']==0
    p[0,1]=np.nan
    with pytest.raises(ValueError):forecast_error(p,truth,mask)


def test_future_mutation_cannot_change_issued_forecasts_or_first_update():
    m=models();s=signal(410)
    p,updates=replay(m,s,device='cpu')
    s['values'][256:]=1234
    q,other=replay(m,s,device='cpu')
    assert p['origins'].tolist()==[255,383]
    for k in p:
        if k not in ('origins','counts'):
            np.testing.assert_array_equal(p[k][0],q[k][0])
    for arm in ARMS:
        assert updates[arm][0]==other[arm][0]
        assert len(updates[arm])==2
    r,_=replay(m,signal(400),device='cpu')
    assert r['origins'].tolist()==[255]  # 17-point tail is not a 25-point target


def test_summary_equal_weights_shots_not_numbers_of_windows():
    rows=[dict(shot=1,arm='a',before=3.,after=2.,static=4.,persistence=1.),
          dict(shot=1,arm='a',before=3.,after=2.,static=4.,persistence=1.),
          dict(shot=2,arm='a',before=1.,after=3.,static=2.,persistence=1.)]
    result=summarize(rows)['a']
    assert result['n_shots']==2 and result['n_windows']==3
    assert result['mean_delta_step']==pytest.approx(.5)
    assert result['mean_delta_static']==pytest.approx(-.5)
    assert result['improved_shots_step']==1
