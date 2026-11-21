import torch
import numpy as np

from experiments.risk_aligned_online.completion_gradient_20261028.analysis import loss_parts, selected_ends


def test_short_records_use_every_available_complete_window_without_duplication():
    assert selected_ends(127).tolist()==[]
    assert selected_ends(128).tolist()==[127]
    assert selected_ends(384).tolist()==[127,255,383]
    assert np.array_equal(selected_ends(1024),[127,383,639,1023])


def test_mean_dynamic_cls_decomposition_is_exact_with_missing_tokens():
    torch.manual_seed(21)
    p=torch.randn(4,17,64,requires_grad=True)
    t=torch.randn_like(p,requires_grad=True)
    v=torch.rand(4,17)>.3;v[:,-1]=True;v[0,:-1]=False
    parts=loss_parts(p,t,v)
    expected=((p-t).square()*v[...,None]).sum((1,2))/(v.sum(1)*64)
    torch.testing.assert_close(sum(parts.values()),expected)
    g=torch.autograd.grad(expected.sum(),(p,t),retain_graph=True)
    summed=torch.autograd.grad(sum(parts.values()).sum(),(p,t))
    for a,b in zip(g,summed):torch.testing.assert_close(a,b)


def test_shared_offset_has_no_dynamic_penalty_and_cls_is_separate():
    p=torch.ones(2,17,64);t=torch.zeros_like(p);v=torch.ones(2,17,dtype=torch.bool)
    parts=loss_parts(p,t,v)
    assert torch.equal(parts['dynamic'],torch.zeros(2))
    torch.testing.assert_close(parts['mean'],torch.full((2,),16/17))
    torch.testing.assert_close(parts['cls'],torch.full((2,),1/17))


def test_target_and_prediction_branch_gradients_add_exactly():
    state=torch.randn(128,requires_grad=True)
    p=(state[:64]*2).expand(2,17,64)
    t=state[64:].expand_as(p);v=torch.ones(2,17,dtype=torch.bool)
    full=sum(loss_parts(p,t,v).values()).mean()
    pred=sum(loss_parts(p,t.detach(),v).values()).mean()
    target=sum(loss_parts(p.detach(),t,v).values()).mean()
    gs=[torch.autograd.grad(x,state,retain_graph=True)[0] for x in (full,pred,target)]
    torch.testing.assert_close(gs[0],gs[1]+gs[2])


def test_batched_window_gradients_do_not_mix_examples():
    from experiments.risk_aligned_online.completion_gradient_20261028.analysis import components
    from experiments.risk_aligned_online.simple23 import Config, Simple23
    from experiments.risk_aligned_online.latent_completion_20261028.model import Imputer
    torch.set_num_threads(1);torch.manual_seed(82);torch.backends.mha.set_fastpath_enabled(False)
    base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23)).eval().requires_grad_(False)
    net=Imputer().eval().requires_grad_(False)
    x=torch.randn(2,128,23);a=torch.rand_like(x)>.1
    both=components(base,net,x,a,None)
    for i in range(2):
        one=components(base,net,x[i:i+1],a[i:i+1],None)[0]
        for field in ('loss','norm','cosine_to_full','fraction_along_full','normalized_step_logit_derivative'):
            for k in one[field]:
                assert np.isclose(both[i][field][k],one[field][k],rtol=2e-4,atol=1e-7)
