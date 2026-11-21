import numpy as np
import pytest
import torch

from ..east_joint_source_replay_20261023.learning import replay_loss, validate_source
from ..east_joint_feedback_20261023.learning import enable
from .test_credit_horizon import fixture


def test_source_replay_rejects_wrong_domain_holdouts_and_duplicate_ids():
    shot=dict(device='J-TEXT',shot=11,y=np.array([0,1]),valid=np.array([True,True]))
    validate_source([shot],[11])
    for shots,ids in (([dict(shot,device='EAST')],[11]),([shot],[12]),([shot,shot],[11,11])):
        with pytest.raises(ValueError):validate_source(shots,ids)


def test_source_replay_only_trains_allowed_encoder():
    m,_=fixture();allowed=enable(m)
    x=torch.randn(4,128,23);mask=torch.ones_like(x,dtype=torch.bool);y=torch.tensor([0.,0.,1.,1.])
    result=replay_loss(m,x,mask,y)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(m(x,mask)['logit'].flatten(),y)
    torch.testing.assert_close(result,expected)
    result.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for k,p in m.named_parameters() if k in m.encoder_names())
    assert all(p.grad is None for k,p in m.named_parameters() if k not in allowed)
    assert all(p.grad is None for p in m.base.forecast_head.parameters())
    with pytest.raises(ValueError):replay_loss(m,x,mask,torch.full((4,),.5))
