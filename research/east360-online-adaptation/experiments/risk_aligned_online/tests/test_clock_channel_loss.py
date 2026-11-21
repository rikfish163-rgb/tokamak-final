import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.clock_channel_loss_20261024.learning import (
    GROUPS, future_groups, group_means, bounded_step, predict,
)


def test_future_group_boundaries_no_late_or_post_disruption_targets():
    times=np.arange(500)/1000
    groups=future_groups(times,384,True,.45)
    assert set(groups)=={'danger'}
    assert groups['danger'].tolist()==list(range(384,420))
    groups=future_groups(times,256,True,.45)
    assert groups['early'].tolist()==list(range(256,325))
    assert groups['danger'].tolist()==list(range(325,384))
    assert future_groups(times,384,False,None)['safe'].tolist()==list(range(384,500))


def test_group_statistics_equal_shots_not_equal_endpoints():
    rows=[]
    for group in GROUPS:
        for sid,count,value in ((1,1,2.),(2,9,6.)):
            rows.extend(dict(group=group,shot=sid,a=[value]*23,b=[1.]*23) for _ in range(count))
    a,b=group_means(rows)
    np.testing.assert_array_equal(a,np.full((3,23),4.))
    np.testing.assert_array_equal(b,np.ones((3,23)))
    with pytest.raises(ValueError):group_means([r for r in rows if r['group']!='danger'])


def test_bounded_step_keeps_cancellation_does_not_amplify_to_full_radius():
    parts=torch.zeros(23,2);parts[0,0]=3.;parts[1,0]=-3.
    w=np.zeros(23);w[0]=w[1]=.5
    step=bounded_step(parts,w)
    assert not step.any()
    w[0]=.75;w[1]=.25
    torch.testing.assert_close(bounded_step(parts,w),torch.tensor([.005,0.]))
    w[0]=1.;w[1]=0.
    assert float(bounded_step(parts,w).norm())==pytest.approx(.01)
    with pytest.raises(ValueError):bounded_step(parts,w*2)
    assert not bounded_step(torch.zeros_like(parts),w).any()


def test_complete_stream_reset_no_labels_no_unarrived_data_access():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    model=small_model();before={k:v.clone() for k,v in model.state_dict().items()}
    values=np.random.default_rng(11).normal(size=(400,23)).astype('float32')
    signal=dict(values=values,available=np.ones_like(values,bool))
    w=np.zeros(23);w[5]=1.
    policy=dict(block=w.tolist(),point=(np.ones(23)/23).tolist(),permuted=w[::-1].copy().tolist())
    curves,updates=predict(model,signal,policy,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm][:256],curves['off'][:256])
    for arm,rs in updates.items():
        assert [r['first_affected_prediction'] for r in rs]==[256,384]
        assert all(r['actual_step_norm']<=.01001 for r in rs)
    for arm in ('uniform','permuted','hold'):
        np.testing.assert_allclose([r['actual_step_norm'] for r in updates[arm]],
                                   [r['actual_step_norm'] for r in updates['block']],atol=2e-6,rtol=1e-3)
    changed={**signal,'values':values.copy()};changed['values'][330:]=123.
    after,_=predict(model,changed,policy,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm][:330],after[arm][:330])
    again,_=predict(model,signal,policy,device='cpu')
    for arm in curves:np.testing.assert_array_equal(curves[arm],again[arm])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    with pytest.raises(ValueError):predict(model,dict(signal,label=1),policy,device='cpu')
