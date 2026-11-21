import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.arrived_teacher_20261023.learning import (
    pairs, step, sample_queries, summarize,
)
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def signal(n=400):
    return dict(values=np.random.default_rng(87).normal(size=(n,23)).astype('float32'),
                available=np.ones((n,23),bool))


def test_pairs_are_matured_and_require_full_past_context():
    assert pairs(153) == [(127,152)]
    assert pairs(228) == [(202,227),(177,202),(152,177),(127,152)]
    with pytest.raises(ValueError): pairs(152)


def test_real_neural_change_no_parent_change_and_future_suffix_independence():
    model=small_model(); original={k:v.clone() for k,v in model.state_dict().items()}
    x=signal(); a,receipt,gradient=step(model,x,228,'advance',device='cpu')
    assert receipt['gradient_steps']==1 and 0<receipt['parameter_delta_norm']<=.00101
    assert receipt['first_affected_prediction']==228
    assert receipt['last_observation']==227 and receipt['target_ends']==[227,202,177,152]
    assert set(gradient)==set(model.encoder_names())
    x['values'][228:]=1e6
    b,other,_=step(model,x,228,'advance',device='cpu')
    assert other==receipt
    for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    with pytest.raises(ValueError): step(model,dict(x,y=1),228,'advance',device='cpu')


def test_same_time_teacher_has_zero_gradient_and_missing_is_not_zero_truth():
    model=small_model(); x=signal()
    fast,receipt,gradient=step(model,x,228,'same',device='cpu')
    assert receipt['loss_before']==0 and receipt['parameter_delta_norm']==0
    for k,v in initial_encoder(model).items():torch.testing.assert_close(fast[k],v,rtol=0,atol=0)
    x['available'][:]=False
    _,receipt,_=step(model,x,228,'advance',device='cpu')
    assert receipt['gradient_steps']==0 and receipt['reason']=='no_observed_pairs'


def test_sampling_uses_strict_timely_positives_and_separate_early_negatives():
    s=dict(valid=np.ones(500,bool),times=np.arange(500)/1000.,down=.5,
           y=np.r_[np.zeros(375),np.ones(125)],is_disrupt=True)
    q=sample_queries(s)
    assert len(q)==32
    assert sum(row['group']=='disruptive_timely' for row in q)==16
    assert all(30<500-r['query']<=125 for r in q if r['group']=='disruptive_timely')
    assert all(r['query']>=177 for r in q)


def test_gate_requires_three_of_four_disruptive_shots_not_only_pooled_gain():
    rows=[]
    for sid in range(8):
        group='disruptive_timely' if sid<4 else 'safe'
        for arm in ('advance','reverse','forecast'):
            rows.append(dict(shot=sid,group=group,arm=arm,bce_before=1.,
                             bce_after=.9 if sid<3 or sid>=4 else 1.01,cosine=.1,
                             gradient_dot=.02,actual_delta=-.1))
    r=summarize(rows)
    assert r['advance']['passes_screen'] and r['advance']['improved_disruptive_shots']==3
    for row in rows:
        if row['shot']==2:row['bce_after']=1.01
    assert not summarize(rows)['advance']['passes_screen']


def test_query_label_changes_diagnostics_not_predictions_or_updates():
    from experiments.risk_aligned_online.arrived_teacher_20261023.run import probe
    model=small_model(); x=signal()
    row=dict(query=252,count=228,group='safe',shot=1,index=0)
    a=probe(model,x,row,0,device='cpu');b=probe(model,x,row,1,device='cpu')
    for r,s in zip(a,b):
        for key in ('logit_before','logit_after','update','aux_gradient_norm'):
            assert r[key]==s[key]
        assert r['gradient_dot']*s['gradient_dot']<=0


def test_forecast_control_is_existing_operator_not_new_loss():
    from experiments.risk_aligned_online.deepcoral_prefix_20261022.learning import adapt_arrived
    model=small_model(); x=signal()
    actual,receipt,_=step(model,x,228,'forecast',device='cpu')
    expected,old=adapt_arrived(model,x,228,initial_encoder(model),device='cpu')
    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert receipt['loss_before']==old['loss_before']
