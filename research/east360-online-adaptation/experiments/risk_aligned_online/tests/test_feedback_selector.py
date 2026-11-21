import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.feedback_selector_20261023.policy import (
    make_context, fit_policy, choose_action, ACTIONS, N_FEATURES,
)


def test_context_ignores_unobserved_values_and_detaches():
    p=torch.ones(4,25,23,requires_grad=True);t=torch.zeros_like(p);m=torch.ones_like(p,dtype=torch.bool)
    level=torch.ones(128,23);a=torch.ones_like(level,dtype=torch.bool)
    m[:,:,0]=False;a[:,0]=False
    x=make_context(p,t,m,level,a,.3)
    t=t.detach().clone();t[:,:,0]=float('nan');level[:,0]=float('nan')
    y=make_context(p,t,m,level,a,.3)
    assert x.shape==(N_FEATURES,) and np.isfinite(x).all()
    np.testing.assert_array_equal(x,y)


def rows():
    result=[]
    for sid,label in [(1,False),(2,False),(3,True),(4,True)]:
        x=np.zeros(N_FEATURES);x[0]=sid
        for trajectory in ('static','recent','lag_bias'):
            result.append(dict(shot=sid,is_disrupt=label,trajectory=trajectory,
                               context=x.tolist(),utility=[sid*.01,-sid*.01]))
    return result


def test_ridge_equal_shot_weight_not_long_shot_weight():
    r=rows();p=fit_policy(r,allowed_ids=[1,2,3,4])
    repeated=r+[r[0].copy() for _ in range(5)]
    q=fit_policy(repeated,allowed_ids=[1,2,3,4])
    np.testing.assert_allclose(p['mean'],q['mean'],atol=1e-14)
    np.testing.assert_allclose(p['coefficients'],q['coefficients'],atol=1e-12)


def test_future_training_ids_rejected():
    with pytest.raises(ValueError):fit_policy(rows(),allowed_ids=[1,2,3])


def test_hold_and_fixed_mean_context_are_explicit():
    p=fit_policy(rows(),allowed_ids=[1,2,3,4])
    p['coefficients']=np.zeros((N_FEATURES+1,2)).tolist()
    assert choose_action(p,np.ones(N_FEATURES))[0]=='hold'
    p['coefficients'][-1]=[-.1,.2]
    assert choose_action(p,np.zeros(N_FEATURES))[0]=='recent'
    assert choose_action(p,np.ones(N_FEATURES),mean_context=True)[0]=='recent'
    assert ACTIONS==('hold','recent','lag_bias')


def test_missing_outcome_class_cannot_train_a_two_class_policy():
    with pytest.raises(ValueError):fit_policy(rows()[:2],allowed_ids=[1,2])


def test_hierarchical_weights_and_unpenalized_intercept():
    r=[]
    for item in rows()[::3]:
        for trajectory,n in [('static',1),('recent',2),('lag_bias',3)]:
            for _ in range(n):
                r.append(dict(item,trajectory=trajectory,utility=[-.1,.2],context=[0.]*N_FEATURES))
    p=fit_policy(r,allowed_ids=[1,2,3,4])
    for sid in range(1,5):
        for trajectory in ('static','recent','lag_bias'):
            assert sum(w for w,a in zip(p['weights'],r) if a['shot']==sid and a['trajectory']==trajectory)==pytest.approx(1/12)
    np.testing.assert_allclose(p['coefficients'][-1],[-.1,.2],atol=1e-12)
    assert np.all(np.asarray(p['scale'])==.0001)


def test_old_new_availability_cannot_alias_zero_residuals():
    p=torch.zeros(4,25,23);m=torch.ones_like(p,dtype=torch.bool)
    level=torch.zeros(128,23);available=torch.ones_like(level,dtype=torch.bool)
    old_missing=m.clone();old_missing[:2]=False
    new_missing=m.clone();new_missing[2:]=False
    x=make_context(p,p,old_missing,level,available,0.)
    y=make_context(p,p,new_missing,level,available,0.)
    assert not np.array_equal(x,y)


def test_consistently_missing_training_trajectory_rejected():
    with pytest.raises(ValueError):
        fit_policy([r for r in rows() if r['trajectory']!='lag_bias'],allowed_ids=[1,2,3,4])


def test_context_is_causal_and_label_fields_are_rejected():
    from experiments.risk_aligned_online.tests.test_lag_bias_feedback import fixture
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    from experiments.risk_aligned_online.feedback_selector_20261023.execution import context_at
    model,s=fixture();state=initial_encoder(model)
    x=context_at(model,s,256,state,device='cpu')
    changed={k:v.copy() for k,v in s.items()};changed['values'][256:]=999.;changed['available'][256:]=False
    truncated={k:v[:256].copy() for k,v in s.items()}
    np.testing.assert_array_equal(x,context_at(model,changed,256,state,device='cpu'))
    np.testing.assert_array_equal(x,context_at(model,truncated,256,state,device='cpu'))
    with pytest.raises(ValueError):context_at(model,dict(s,down=1),256,state,device='cpu')


def test_bce_of_peak_and_strict_timely_eligibility():
    from experiments.risk_aligned_online.feedback_selector_20261023.execution import eligible_query,peak_loss
    out=dict(times=np.array([.969,.970,.971]),valid=np.ones(3,bool),is_disrupt=True,down=1.)
    np.testing.assert_array_equal(eligible_query(out,0,3),[True,False,False])
    z=np.array([-2.,2.]);mask=np.ones(2,bool)
    assert peak_loss(z,mask,True)==pytest.approx(np.logaddexp(0.,-2))
    assert peak_loss(z,mask,False)==pytest.approx(np.logaddexp(0.,2))
    with pytest.raises(ValueError):peak_loss(z,np.zeros(2,bool),True)


def test_forced_actions_match_archived_algorithm_and_reset():
    from experiments.risk_aligned_online.tests.test_lag_bias_feedback import fixture
    from experiments.risk_aligned_online.lag_bias_feedback_20261022.run import predict_arms
    from experiments.risk_aligned_online.feedback_selector_20261023.execution import predict_policy
    model,s=fixture();old,_=predict_arms(model,s,device='cpu')
    p=fit_policy(rows(),allowed_ids=[1,2,3,4]);p['coefficients']=np.zeros((N_FEATURES+1,2)).tolist()
    for intercept,arm in [([0.,0.],'static'),([-.1,.1],'recent'),([.1,-.1],'lag_bias')]:
        p['coefficients'][-1]=intercept
        actual,records=predict_policy(model,s,p,device='cpu')
        np.testing.assert_array_equal(actual['contextual'],old[arm])
        np.testing.assert_array_equal(actual['mean_context'],old[arm])
        assert all(r['first_affected_prediction']>r['last_observation'] for r in records['contextual'])


def test_counterfactual_order_preserves_state_and_candidate_peak():
    from experiments.risk_aligned_online.tests.test_lag_bias_feedback import fixture
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    from experiments.risk_aligned_online.feedback_selector_20261023.execution import counterfactual
    model,s=fixture();state=initial_encoder(model);before={k:v.clone() for k,v in state.items()}
    mask=np.ones(128,bool)
    a,states=counterfactual(model,s,256,state,mask,False,device='cpu')
    b,_=counterfactual(model,s,256,state,mask,False,device='cpu',candidate_order=('lag_bias','recent'))
    np.testing.assert_array_equal(a['utility'],b['utility'])
    assert states['hold'] is state and a['losses']['hold']>=0
    for k,v in before.items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)


def test_prefix_gate_rejects_incomplete_or_modified_seals(tmp_path):
    from experiments.risk_aligned_online.feedback_selector_20261023.run import validate_prefix_seals
    from experiments.risk_aligned_online.simple_train import sha
    (tmp_path/'shots').mkdir();seals=[]
    for sid in range(20):
        p=tmp_path/'shots'/f'{sid}.npz';np.savez(p,x=np.ones(2))
        seals.append(dict(shot=sid,prediction_sha256=sha(p)))
    validate_prefix_seals(tmp_path,seals,list(range(30)))
    with pytest.raises(ValueError):validate_prefix_seals(tmp_path,seals[:-1],list(range(30)))
    np.savez(tmp_path/'shots'/'3.npz',x=np.zeros(2))
    with pytest.raises(ValueError):validate_prefix_seals(tmp_path,seals,list(range(30)))
