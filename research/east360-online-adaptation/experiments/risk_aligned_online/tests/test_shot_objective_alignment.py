import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.shot_objective_alignment_20261023.learning import (
    candidate_ends,bag_loss,backward_bag,logits_at,
)


def test_three_arm_factorization_changes_negative_then_positive_aggregation():
    from experiments.risk_aligned_online.shot_objective_alignment_20261023.learning import aggregation_for
    assert [aggregation_for('mean',d) for d in (False,True)]==['mean','mean']
    assert [aggregation_for('safe_peak',d) for d in (False,True)]==['peak','mean']
    assert [aggregation_for('peak',d) for d in (False,True)]==['peak','peak']
    with pytest.raises(ValueError):aggregation_for('unknown',False)


def test_candidates_keep_early_safe_points_and_strict_30_inclusive_125():
    times=np.array([0.,.875,.969,.970,.999,1.,1.001])
    shot=dict(device='EAST',times=times,is_disrupt=True,down=1.,horizon=125.,
              valid=times<=1.,y=((times>=.875)&(times<=1.)).astype(float))
    assert candidate_ends(shot).tolist()==[1,2]
    safe=dict(shot,is_disrupt=False,down=None,valid=np.ones(7,bool),y=np.zeros(7))
    assert candidate_ends(safe).tolist()==list(range(7))
    with pytest.raises(ValueError):candidate_ends(dict(shot,horizon=25.))
    with pytest.raises(ValueError):candidate_ends(dict(shot,times=np.array([1.])))


def test_mean_and_peak_are_distinct_and_positive_peak_is_not_hardest_positive():
    z=torch.tensor([-2.,1.,1.],requires_grad=True)
    loss=bag_loss(z,True,.5,'peak');loss.backward()
    assert z.grad[0]==0 and z.grad[1]<0 and z.grad[2]==0
    assert loss.item()==pytest.approx(torch.nn.functional.softplus(torch.tensor(-.5)).item())
    assert bag_loss(z,False,.5,'peak').item()==pytest.approx(torch.nn.functional.softplus(torch.tensor(.5)).item())
    assert bag_loss(z,False,.5,'mean').item()==pytest.approx(torch.nn.functional.softplus(z.detach()-.5).mean().item())
    with pytest.raises(ValueError):bag_loss(z,True,.5,'unknown')
    with pytest.raises(ValueError):bag_loss(torch.empty(0),True,.5,'mean')


@pytest.mark.parametrize('mode',['mean','peak'])
def test_chunked_backward_matches_full_tensor_and_updates_only_allowed_encoder(mode):
    from experiments.risk_aligned_online.simple23 import Config,Simple23
    from experiments.risk_aligned_online.paired_feedback_20261022.model import PairedFeedback
    torch.set_num_threads(1);torch.manual_seed(11)
    model=PairedFeedback(Simple23(Config(dim=8,heads=2,fast_site='tokens'),np.zeros(23),np.ones(23)),np.ones(23))
    model.set_stage('frozen');allowed=set(model.encoder_names())
    for k,p in model.named_parameters():p.requires_grad_(k in allowed)
    signal=dict(values=np.random.default_rng(7).normal(size=(140,23)).astype(np.float32),available=np.ones((140,23),bool))
    ends=np.arange(127,140);params=[p for p in model.parameters() if p.requires_grad]
    z=logits_at(model,signal,ends,device='cpu')
    full=.25*bag_loss(z,False,.9,mode);full.backward()
    expected=[p.grad.clone() for p in params];model.zero_grad(set_to_none=True)
    receipt=backward_bag(model,signal,ends,False,.9,mode,weight=.25,device='cpu',batch_size=4)
    assert receipt['loss']==pytest.approx(float(full.detach())/.25,abs=2e-6)
    for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=3e-5,atol=3e-6)
    assert all(p.grad is None for k,p in model.named_parameters() if k not in allowed)
    assert receipt['candidates']==len(ends)
    if mode=='peak':assert receipt['peak_end']==int(ends[int(z.detach().argmax())])
