import numpy as np
import torch
import pytest

from experiments.risk_aligned_online.head_operator_20261022.run import token_update, token_predictions
from experiments.risk_aligned_online.simple23 import Config,Simple23


def test_token_update_internal_bounded_and_causal():
    torch.set_num_threads(2);torch.manual_seed(13)
    model=Simple23(Config(fast_site='tokens',fast_rule='normalized'),np.zeros(23),np.ones(23)).eval()
    model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    signal=dict(values=np.random.default_rng(7).normal(size=(400,23)).astype(np.float32),
                available=np.ones((400,23),bool))
    initial=model.initial_fast(1).detach()
    fast,receipt=token_update(model,signal,256,initial,initial,device='cpu')
    assert fast.shape==(1,128) and 0<receipt['distance_from_origin']<=1.000001
    assert receipt['first_affected_prediction']==256
    signal['values'][256:]=1000
    other,_=token_update(model,signal,256,initial,initial,device='cpu')
    torch.testing.assert_close(fast,other,rtol=0,atol=0)
    for k,v in before.items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    with pytest.raises(ValueError): token_predictions({'x':model},dict(signal,y=1),device='cpu')


def test_token_stream_resets_each_shot_and_initial_predictions_match():
    torch.set_num_threads(2);torch.manual_seed(2)
    model=Simple23(Config(fast_site='tokens',fast_rule='normalized'),np.zeros(23),np.ones(23)).eval()
    model.requires_grad_(False)
    signal=dict(values=np.random.default_rng(8).normal(size=(385,23)).astype(np.float32),
                available=np.ones((385,23),bool))
    a,ra=token_predictions({'x':model},signal,device='cpu')
    b,rb=token_predictions({'x':model},signal,device='cpu')
    np.testing.assert_array_equal(a['x'],b['x'])
    signal['values'][256:]=999
    c,_=token_predictions({'x':model},signal,device='cpu')
    np.testing.assert_array_equal(a['x'][:256],c['x'][:256])
    assert [r['first_affected_prediction'] for r in ra['x']]==[256,384]
