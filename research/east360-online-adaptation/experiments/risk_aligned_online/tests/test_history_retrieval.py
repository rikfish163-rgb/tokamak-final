import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.history_retrieval_20261023.learning import (
    normalize, select_memory, step_all, sample_queries,
)
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_balanced_unique_shot_neighbours_exclude_the_entire_query_shot():
    memory=dict(keys=normalize(np.array([[1.,0.],[1.,.01],[.8,.2],[.6,.4],
                                        [1.,0.],[.7,.3],[.5,.5]])),
                labels=np.array([1,1,1,1,0,0,0]),shots=np.array([1,1,2,3,4,5,6]))
    indices,dist=select_memory(memory,np.array([1.,0.]),'nearest',excluded_shot=1)
    assert indices==[4,5,2,3]
    assert len(dist)==4 and not any(memory['shots'][i]==1 for i in indices)
    scarce={k:v[[0,1,2,4,5,6]] for k,v in memory.items()}
    with pytest.raises(ValueError):select_memory(scarce,np.array([1.,0.]),'nearest',excluded_shot=2)


def test_centroid_selection_does_not_use_current_query_and_rejects_bad_geometry():
    memory=dict(keys=normalize(np.array([[1.,0.],[.8,.2],[.1,.9],[0.,1.]])),
                labels=np.array([0,0,1,1]),shots=np.arange(4))
    a=select_memory(memory,np.array([1.,0.]),'centroid')
    b=select_memory(memory,np.array([0.,1.]),'centroid')
    assert a==b
    with pytest.raises(ValueError):normalize(np.zeros((1,2)))
    with pytest.raises(ValueError):normalize(np.array([[np.nan,1.]]))


def test_signal_causality_actual_neural_update_and_matched_lengths():
    model=small_model();parent={k:v.clone() for k,v in model.state_dict().items()}
    rng=np.random.default_rng(13)
    x=torch.tensor(rng.normal(size=(8,128,23)),dtype=torch.float32)
    mask=torch.ones_like(x,dtype=torch.bool)
    h=model.predict_fast(x,mask,initial_encoder(model),mode='risk_state')['risk_state'].detach().numpy()
    memory=dict(keys=normalize(h),labels=np.array([0]*4+[1]*4),shots=np.arange(8),x=x,mask=mask)
    signal=dict(values=rng.normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    out,receipts,gradients=step_all(model,signal,256,memory,device='cpu')
    assert 0<receipts['nearest']['parameter_delta_norm']<=.00101
    assert all(abs(r['parameter_delta_norm']-receipts['nearest']['parameter_delta_norm'])<2e-7 for r in receipts.values())
    signal['values'][256:]=1e6
    again,other,_=step_all(model,signal,256,memory,device='cpu')
    assert other==receipts
    for a in out:
        assert receipts[a]['last_observation']==255
        for key in out[a]:torch.testing.assert_close(out[a][key],again[a][key],rtol=0,atol=0)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,parent[key],rtol=0,atol=0)
    with pytest.raises(ValueError):step_all(model,dict(signal,y=1),256,memory,device='cpu')


def test_query_sampling_preserves_deadline_and_minimum_observation_boundary():
    shot=dict(valid=np.ones(700,bool),times=np.arange(700)/1000,down=.7,
              y=np.r_[np.zeros(575),np.ones(125)],is_disrupt=True)
    rows=sample_queries(shot)
    assert len(rows)==32 and all(r['count']>=256 and r['query']==r['count']+24 for r in rows)
    assert all(30<700-r['query']<=125 for r in rows if r['group']=='disruptive_timely')


def test_query_truth_is_diagnostic_only_and_cannot_change_retrieval_or_update():
    from experiments.risk_aligned_online.history_retrieval_20261023.run import probe
    model=small_model();rng=np.random.default_rng(71)
    x=torch.tensor(rng.normal(size=(8,128,23)),dtype=torch.float32);mask=torch.ones_like(x,dtype=torch.bool)
    keys=model.predict_fast(x,mask,initial_encoder(model),mode='risk_state')['risk_state'].detach().numpy()
    memory=dict(x=x,mask=mask,keys=normalize(keys),labels=np.array([0]*4+[1]*4),shots=np.arange(8))
    signal=dict(values=rng.normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    row=dict(query=280,count=256,shot=99,group='safe',index=0)
    zero=probe(model,signal,row,0,memory,device='cpu');one=probe(model,signal,row,1,memory,device='cpu')
    for a,b in zip(zero,one):
        assert a['update']==b['update'] and a['logit_after']==b['logit_after']
        assert a['gradient_dot']*b['gradient_dot']<=0


def test_screen_must_outperform_controls_not_only_static():
    from experiments.risk_aligned_online.history_retrieval_20261023.run import summarize_rows
    rows=[]
    for sid in range(20):
        for arm in ('nearest','initial','centroid'):
            rows.append(dict(shot=sid,group='disruptive_timely' if sid<4 else 'safe',
                arm=arm,before=1.,after=.9 if arm=='nearest' else .95))
    summary,gate=summarize_rows(rows)
    assert gate and summary['nearest']['passes_screen']
    for row in rows:
        if row['arm']=='centroid':row['after']=.9
    assert not summarize_rows(rows)[1]
