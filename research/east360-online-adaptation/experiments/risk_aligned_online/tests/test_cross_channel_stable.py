import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.cross_channel_prefix import CrossChannelModel
from experiments.risk_aligned_online.cross_channel_stable import (
    FourViewModel,embedding_change,limited_step,predict_shot,
)
from experiments.risk_aligned_online.tests.test_cross_channel_prefix import make_model,signals


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model():
    original=make_model();m=FourViewModel(original.base,site='embedding')
    m.decoder.load_state_dict(original.decoder.state_dict())
    return m.eval().requires_grad_(False)


def test_batched_four_views_match_original_losses_and_gradients():
    m=model();x=torch.randn(1,128,23);a=torch.rand_like(x)>.15
    a[:,:120]=False
    f=m.initial_fast(1).requires_grad_()
    losses,counts=m.four_losses(x,a,f)
    assert losses.shape==(4,) and counts.shape==(4,)
    for phase in range(4):
        expected=CrossChannelModel.reconstruction_loss(m,x,a,f,phase=phase)
        torch.testing.assert_close(losses[phase],expected,rtol=2e-5,atol=2e-6)
        ga,=torch.autograd.grad(losses[phase],f,retain_graph=True)
        gb,=torch.autograd.grad(expected,f)
        torch.testing.assert_close(ga,gb,rtol=1e-4,atol=2e-6)


def test_measured_embedding_change_matches_original_layer_not_risk_output():
    m=model();x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool);a[:,:64]=False
    f=torch.randn(1,m.temporary_dim)*.002
    outputs=[];hook=m.base.embedding.register_forward_hook(lambda mod,inp,out:outputs.append(out.detach()))
    m(x,a);m(x,a,f);hook.remove()
    expected=(outputs[1][:,8:]-outputs[0][:,8:]).double().square().mean().sqrt()
    torch.testing.assert_close(embedding_change(m,x,a,f),expected,rtol=1e-6,atol=1e-8)
    changed=x.clone();changed[~a]=1e8
    torch.testing.assert_close(embedding_change(m,changed,a,f),embedding_change(m,x,a,f),rtol=0,atol=0)


def test_functional_limit_respects_current_input_and_parameter_radius():
    m=model();x=torch.ones(1,128,23)*1e6;a=torch.ones_like(x,dtype=torch.bool)
    current=m.initial_fast(1);g=torch.ones_like(current)
    candidate,info=limited_step(m,x,a,current,g,cap=.001)
    assert info['limited'] and info['scale']<1
    assert embedding_change(m,x,a,candidate-current)<=.001001
    assert candidate.norm()<=1
    with pytest.raises(ValueError):limited_step(m,x,a,current,g,cap=0.)


@pytest.mark.parametrize('feedback',['rotate','joint'])
def test_stream_is_causal_and_resets_per_shot(feedback):
    m=model();s=signals(105);initial={k:v.clone() for k,v in m.state_dict().items()}
    p,au=predict_shot(m,s,intra=True,feedback=feedback,cap=.01)
    s2=dict(s,values=s['values'].copy());s2['values'][60:]+=99
    p2,_=predict_shot(m,s2,intra=True,feedback=feedback,cap=.01)
    again,_=predict_shot(m,s,intra=True,feedback=feedback,cap=.01)
    np.testing.assert_array_equal(p[:60],p2[:60]);np.testing.assert_array_equal(p,again)
    assert au['fast_discarded'] and len(au['attempts'])==4
    assert au['fast_initial']['l2']==0 and au['accepted_updates']>0
    assert all(r['observed_through']==r['update_index'] for r in au['attempts'])
    for k,v in m.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='outcome'):
        predict_shot(m,dict(s,is_disrupt=False),intra=True,feedback=feedback,cap=.01)
