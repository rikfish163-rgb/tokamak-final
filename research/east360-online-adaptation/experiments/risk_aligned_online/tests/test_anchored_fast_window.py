import torch
import pytest
from experiments.risk_aligned_online.anchored_fast_window_20261023.model import centered_step, AnchoredLayer


def test_normalized_step_common_scale_invariant_and_descends():
    torch.manual_seed(1); k=torch.randn(3,7,4); v=torch.randn_like(k)
    mask=torch.ones(3,7,dtype=torch.bool); w=torch.eye(4); rate=torch.tensor(.25)
    d,info=centered_step(k,v,mask,w,rate,rule='normalized',reverse=False)
    for scale in (.001,10.):
        dd,_=centered_step(k*scale,v*scale,mask,w,rate,rule='normalized',reverse=False)
        torch.testing.assert_close(d,dd,atol=1e-6,rtol=1e-5)
    assert torch.all(info['loss_after']<=info['loss_before'])


def test_constant_nonzero_measurements_exact_zero_with_missing_and_cls():
    k=torch.full((1,7,4),.1234567); v=torch.full_like(k,1.234567)
    k[:,0]=100.;v[:,0]=-100.;k[:,-1]=90.;v[:,-1]=80.
    mask=torch.tensor([[False,True,True,True,True,True,False]])
    d,info=centered_step(k,v,mask,torch.eye(4),torch.tensor(.25),rule='normalized',reverse=False)
    assert torch.count_nonzero(d)==0 and not info['updated'].any()
    assert torch.isfinite(info['effective_eta']).all()


def test_step_matches_autograd():
    torch.manual_seed(2); k=torch.randn(2,6,4,dtype=torch.float64); v=torch.randn_like(k)
    mask=torch.ones(2,6,dtype=torch.bool); w=torch.eye(4,dtype=torch.float64,requires_grad=True)
    kc=k-k.mean(1,keepdim=True);vc=v-v.mean(1,keepdim=True)
    for rule in ('fixed','normalized'):
        d,info=centered_step(k,v,mask,w,torch.tensor(.25),rule=rule,reverse=False)
        loss=(kc@w-vc).square().sum((1,2))/12
        grads=torch.stack([torch.autograd.grad(loss[i],w,retain_graph=True)[0] for i in range(2)])
        torch.testing.assert_close(d,-info['effective_eta'][:,None,None]*grads,atol=1e-12,rtol=1e-12)


def test_off_exact_after_parameters_changed_and_on_initial_gradient_live():
    torch.manual_seed(4); layer=AnchoredLayer(8,rank=4,rule='normalized')
    h=torch.randn(2,6,8); mask=torch.ones(2,6,dtype=torch.bool)
    on,_=layer(h,mask,mode='on'); torch.testing.assert_close(on,h,atol=0,rtol=0)
    on.square().sum().backward(); assert layer.output.weight.grad.norm()>0
    with torch.no_grad():
        for p in layer.parameters(): p.add_(torch.randn_like(p)*.2)
    off,_=layer(h,mask,mode='off'); torch.testing.assert_close(off,h,atol=0,rtol=0)
    on,_=layer(h,mask,mode='on'); assert not torch.equal(on,h)


def test_no_cross_window_state_and_reverse_changes_relation():
    torch.manual_seed(6); layer=AnchoredLayer(8,rank=4,rule='normalized')
    with torch.no_grad(): layer.output.weight.normal_()
    h=torch.randn(2,6,8);mask=torch.ones(2,6,dtype=torch.bool)
    before={k:v.clone() for k,v in layer.state_dict().items()}
    a,_=layer(h,mask,mode='on');b,_=layer(h,mask,mode='reverse');c,_=layer(h,mask,mode='on')
    torch.testing.assert_close(a,c,atol=0,rtol=0);assert not torch.allclose(a,b)
    assert all(torch.equal(v,before[k]) for k,v in layer.state_dict().items())


def test_empty_or_single_observation_and_invalid_rule():
    k=torch.randn(2,5,4);v=torch.randn_like(k);mask=torch.zeros(2,5,dtype=torch.bool);mask[1,3]=True
    d,info=centered_step(k,v,mask,torch.eye(4),torch.tensor(.25),rule='normalized',reverse=False)
    assert torch.count_nonzero(d)==0 and not info['updated'].any()
    with pytest.raises(ValueError): centered_step(k,v,mask,torch.eye(4),torch.tensor(.25),rule='other',reverse=False)


def test_ranking_never_uses_exactly_30ms_peak_as_timely():
    import numpy as np
    from experiments.risk_aligned_online.anchored_fast_window_20261023.run import ranking
    rows=[dict(is_disrupt=True,down=.100,times=np.array([.069,.070,.071]),scores=np.array([.1,.99,.99])),
          dict(is_disrupt=False,down=None,times=np.array([.069,.070,.071]),scores=np.array([.2,.2,.2]))]
    result=ranking(rows)
    assert result['timely_peak_auc']==0.
    assert result['oracle_tp_at_fpr_cap']=={'0.1':0,'0.15':0,'0.2':0}
