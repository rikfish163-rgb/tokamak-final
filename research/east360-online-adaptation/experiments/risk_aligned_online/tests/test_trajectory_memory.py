import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.trajectory_memory_20261024.learning import (
    describe, distances, select_weights, combine_step, observed_window,
)


def test_order_changes_distance_but_constant_control_does_not():
    q = np.tile(np.linspace(-1, 1, 128)[:, None], (1, 23))
    m = np.ones_like(q, bool)
    d = describe(q, m, np.zeros(23), np.ones(23))
    bank = {k:np.stack([v, v[::-1] if k in ('path','valid') else v]) for k,v in d.items()}
    real = distances(d, bank, 'trajectory')
    reverse = distances(d, bank, 'reverse')
    level = distances(d, bank, 'level')
    assert real[0] == 0 and real[1] > 0
    np.testing.assert_allclose(real, reverse[::-1])
    np.testing.assert_array_equal(level, np.zeros(2))


def test_missing_values_are_not_reconstruction_targets_or_distance_evidence():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(128,23)); mask = np.ones_like(x,bool); mask[:,2] = False
    a = describe(x, mask, np.zeros(23), np.ones(23)); x[:,2] = np.nan
    b = describe(x, mask, np.zeros(23), np.ones(23))
    for k in a: np.testing.assert_array_equal(a[k], b[k])
    bank = {k:v[None] for k,v in a.items()}
    assert distances(a,bank,'trajectory')[0] == 0
    empty = describe(x, np.zeros_like(mask), np.zeros(23), np.ones(23))
    assert np.isinf(distances(empty,bank,'trajectory')[0])


def test_selection_excludes_whole_shot_and_uses_one_window_per_shot():
    shots = np.array([1,1,2,3,4]); labels = np.array([1,1,1,0,0])
    indices, weights = select_weights(np.array([0.,.1,.2,.3,.4]),shots,labels,1.,exclude=1)
    assert indices.tolist() == [2,3,4]
    assert np.isclose(weights.sum(),1.) and (weights>0).all()
    _, uniform = select_weights(np.zeros(5),shots,labels,1.)
    assert np.allclose(uniform,[.25,.25,.25,.25])
    indices, weights = select_weights(np.full(5,np.inf),shots,labels,1.)
    assert not len(indices) and not len(weights)
    with pytest.raises(ValueError):select_weights(np.zeros(5),shots,labels,0.)


def test_weighted_gradient_equals_bce_gradient_and_does_not_edit_parent():
    torch.manual_seed(9)
    w = torch.randn(3,requires_grad=True); x = torch.randn(4,3); y = torch.tensor([0.,1.,0.,1.])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(x@w,y,reduction='none')
    gradients = torch.stack([torch.autograd.grad(z,w,retain_graph=True)[0] for z in loss])
    weights = torch.tensor([.1,.2,.3,.4]); parent = w.detach().clone()
    actual = torch.autograd.grad((weights*loss).sum(),w)[0]
    out, receipt = combine_step({'w':w},gradients,weights,learning_rate=.01)
    expected = w-.01*actual/actual.norm().clamp_min(1.)
    torch.testing.assert_close(out['w'],expected)
    torch.testing.assert_close(w,parent,rtol=0,atol=0)
    assert 0<receipt['step_norm']<=.010001
    zero, info = combine_step({'w':w},torch.zeros_like(gradients),weights,learning_rate=.01)
    torch.testing.assert_close(zero['w'],w,rtol=0,atol=0)
    assert info['step_norm']==0


def test_current_query_has_no_outcome_and_future_cannot_change_window():
    x=np.arange(400*23,dtype=np.float32).reshape(400,23)
    signal=dict(values=x,available=np.ones_like(x,bool))
    a=observed_window(signal,256);x[256:]=-10000
    b=observed_window(signal,256)
    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
    with pytest.raises(ValueError):observed_window(dict(signal,y=1),256)
    with pytest.raises(ValueError):observed_window(signal,127)


def test_step_length_controls_match_and_zero_direction_is_not_replaced():
    parent={'w':torch.ones(3)}; g=torch.tensor([[1.,0.,0.],[0.,2.,0.]])
    a,info=combine_step(parent,g,torch.tensor([.5,.5]),learning_rate=.01)
    b,other=combine_step(parent,g,torch.tensor([1.,0.]),learning_rate=.01,length=info['step_norm'])
    assert abs(other['step_norm']-info['step_norm'])<1e-6
    assert not torch.equal(a['w'],b['w'])
    zero,r=combine_step(parent,g*0,torch.tensor([1.,0.]),learning_rate=.01,length=.01)
    assert r['step_norm']==0 and torch.equal(zero['w'],parent['w'])


def test_smoke_with_one_outcome_class_cannot_qualify_as_full_stream():
    from experiments.risk_aligned_online.trajectory_memory_20261024.run import eligible
    rows={a:dict(tpr=None,fpr=0.,J=None,n_safe=1,n_disrupt=0)
          for a in ('off','trajectory','level','reverse','uniform')}
    assert not eligible(rows)
    rows={a:dict(tpr=.7,fpr=.2,J=.5,n_safe=180,n_disrupt=180) for a in rows}
    rows['trajectory'].update(tpr=.8,J=.6)
    assert eligible(rows)
    rows['uniform'].update(tpr=.8,J=.6)
    assert not eligible(rows)


def test_neural_stream_reset_and_no_future_influence():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.trajectory_memory_20261024.learning import make_bank,predict
    model=small_model();rng=np.random.default_rng(18);history=[]
    before={k:v.clone() for k,v in model.state_dict().items()}
    for sid,disruptive in enumerate([False,False,True,True]):
        x=rng.normal(size=(300,23)).astype('float32');times=np.arange(300)/1000
        history.append(dict(shot=sid,values=x,available=np.ones_like(x,bool),times=times,
                            valid=np.ones(300,bool),is_disrupt=disruptive,down=.3 if disruptive else None))
    bank=make_bank(model,history,device='cpu')
    x=rng.normal(size=(300,23)).astype('float32');signal=dict(values=x,available=np.ones_like(x,bool))
    a,ra=predict(model,signal,bank,device='cpu')
    b,rb=predict(model,signal,bank,device='cpu')
    assert ra==rb
    for arm in a:np.testing.assert_array_equal(a[arm],b[arm])
    x[256:]=1e4;c,rc=predict(model,signal,bank,device='cpu')
    for arm in a:
        np.testing.assert_array_equal(a[arm][:256],c[arm][:256])
        np.testing.assert_array_equal(a[arm][:128],a['off'][:128])
    assert ra==rc and any(not np.array_equal(a[arm],a['off']) for arm in a if arm!='off')
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_posthoc_timing_respects_strict_lead_and_serializable_counts():
    import json
    from experiments.risk_aligned_online.trajectory_memory_20261024.diagnose import timely_update
    times=np.arange(1001)/1000
    assert timely_update(times,876,1.) is True  # observed lead125,next124
    assert timely_update(times,875,1.) is False # observed lead126
    assert timely_update(times,970,1.) is False # first affected prediction lead30
    flags=[timely_update(times,i,1.) for i in (876,875,970)]
    assert json.dumps(dict(count=sum(flags),flags=flags))
