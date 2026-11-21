import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.trajectory_rank_20261023.learning import (
    pairwise_loss, trajectory_vectors, step_all,
)
from experiments.risk_aligned_online.history_retrieval_20261023.learning import normalize
from experiments.risk_aligned_online.tests.test_task_geometry import small_model
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_pairwise_loss_is_shift_invariant_and_has_balanced_output_derivatives():
    z=torch.tensor([3.,2.,3.,3.2],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0.,0.,1.,1.],dtype=z.dtype)
    loss=pairwise_loss(z,y)
    torch.testing.assert_close(loss,pairwise_loss(z+100.,y))
    grad,=torch.autograd.grad(loss,z)
    assert (grad[:2]>0).all() and (grad[2:]<0).all()
    assert abs(float(grad.sum()))<1e-14
    manual=torch.stack([torch.nn.functional.softplus(z[n]-z[p]) for n in (0,1) for p in (2,3)]).mean()
    torch.testing.assert_close(loss,manual)
    with pytest.raises(ValueError):pairwise_loss(z,torch.zeros(4))


def test_trajectory_key_uses_order_but_pooled_key_does_not():
    states=np.array([[1.,0.],[.6,.8],[0.,1.]])
    a=trajectory_vectors(states);b=trajectory_vectors(states[::-1])
    np.testing.assert_array_equal(a['pooled'],b['pooled'])
    assert not np.array_equal(a['trajectory'],b['trajectory'])
    assert a['trajectory'].shape==(6,) and a['pooled'].shape==(2,)
    for v in a.values():assert np.linalg.norm(v)==pytest.approx(1.)
    # Stable signals have a valid constant key, not an arbitrary fallback.
    flat=trajectory_vectors(np.ones((3,2)));assert np.isfinite(flat['trajectory']).all()
    with pytest.raises(ValueError):trajectory_vectors(np.ones((2,2)))


def test_matched_neural_steps_future_independence_and_parent_unchanged():
    model=small_model();origin={k:v.clone() for k,v in model.state_dict().items()}
    rng=np.random.default_rng(17)
    x=torch.tensor(rng.normal(size=(8,128,23)),dtype=torch.float32);mask=torch.ones_like(x,dtype=torch.bool)
    h=model.predict_fast(x,mask,initial_encoder(model),mode='risk_state')['risk_state'].detach().numpy()
    keys=normalize(h);tr=np.concatenate([keys,keys,keys],axis=1)/np.sqrt(3)
    memory=dict(x=x,mask=mask,keys=keys,trajectory_keys=tr,pooled_keys=keys,
                labels=np.array([0]*4+[1]*4),shots=np.arange(8))
    signal=dict(values=rng.normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    states,receipts,_=step_all(model,signal,256,memory,device='cpu')
    assert len(states)==6 and 0<receipts['state_bce']['parameter_delta_norm']<.00101
    norms=[r['parameter_delta_norm'] for r in receipts.values()]
    assert max(norms)-min(norms)<2e-7
    signal['values'][256:]=1e8
    again,others,_=step_all(model,signal,256,memory,device='cpu')
    assert receipts==others
    for a in states:
        assert receipts[a]['retrieval_ends']==[205,230,255]
        for k in states[a]:torch.testing.assert_close(states[a][k],again[a][k],rtol=0,atol=0)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,origin[k],rtol=0,atol=0)
    with pytest.raises(ValueError):step_all(model,dict(signal,label=1),256,memory,device='cpu')


def test_primary_gate_cannot_be_replaced_by_a_better_secondary_arm():
    from experiments.risk_aligned_online.trajectory_rank_20261023.run import summary_and_gate
    from experiments.risk_aligned_online.trajectory_rank_20261023.learning import ARMS
    rows=[]
    for sid in range(20):
        for arm in ARMS:
            rows.append(dict(shot=sid,group='disruptive_timely' if sid<4 else 'safe',arm=arm,
                             before=1.,after=.8 if arm=='trajectory_rank' else .9))
    assert summary_and_gate(rows)[1]
    for r in rows:
        if r['arm']=='state_rank':r['after']=.7
    assert not summary_and_gate(rows)[1]


def test_posthoc_query_truth_does_not_enter_trajectory_retrieval_or_gradient():
    from experiments.risk_aligned_online.trajectory_rank_20261023.run import probe
    model=small_model();rng=np.random.default_rng(47)
    x=torch.tensor(rng.normal(size=(8,128,23)),dtype=torch.float32);mask=torch.ones_like(x,dtype=torch.bool)
    keys=normalize(model.predict_fast(x,mask,initial_encoder(model),mode='risk_state')['risk_state'].detach().numpy())
    memory=dict(x=x,mask=mask,keys=keys,trajectory_keys=np.tile(keys,(1,3))/np.sqrt(3),
                pooled_keys=keys,labels=np.array([0]*4+[1]*4),shots=np.arange(8))
    signal=dict(values=rng.normal(size=(400,23)).astype('float32'),available=np.ones((400,23),bool))
    row=dict(query=280,count=256,shot=99,group='safe',index=0)
    a=probe(model,signal,row,0,memory,device='cpu');b=probe(model,signal,row,1,memory,device='cpu')
    for ra,rb in zip(a,b):
        assert ra['update']==rb['update'] and ra['logit_after']==rb['logit_after']
        assert ra['gradient_dot']*rb['gradient_dot']<=0
