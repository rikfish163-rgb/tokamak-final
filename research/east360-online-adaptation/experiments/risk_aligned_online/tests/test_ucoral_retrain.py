import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.ucoral_retrain_20261022.alignment import fit_statistics, fit_alignment, coordinates, validate_prefix
from experiments.risk_aligned_online.ucoral_retrain_20261022.model import risk_from_coordinates, CoordinateClassifier
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.ucoral_retrain_20261022.run import phase_metrics


def signal(seed, sid, device, n=100):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,23)); x[:,1]=.8*x[:,0]+.2*x[:,1]
    return dict(shot=sid,device=device,values=x,available=np.ones_like(x,dtype=bool))


def test_direction_matches_regularized_target_covariance_and_not_inverse():
    s=signal(1,10,'J-TEXT');t=signal(2,20,'EAST');t['values'][:,1]*=2
    t['values'][:,3]=t['values'][:,0]-.7*t['values'][:,2]
    fit=fit_alignment([s],[t]);a=np.asarray(fit['matrix']);cs=np.asarray(fit['source_cov']);ct=np.asarray(fit['target_cov'])
    np.testing.assert_allclose(a.T@(cs+np.eye(23))@a,ct+np.eye(23),atol=1e-10)
    assert np.max(abs(a@(cs+np.eye(23))@a.T-(ct+np.eye(23))))>1e-4
    zs,mask=coordinates(s['values'],s['available'],fit,'source',True)
    z0,_=coordinates(s['values'],s['available'],fit,'source',False)
    np.testing.assert_allclose(zs,z0@a,atol=1e-6)
    zt,_=coordinates(t['values'],t['available'],fit,'target',True)
    zt0,_=coordinates(t['values'],t['available'],fit,'target',False)
    np.testing.assert_array_equal(zt,zt0)


def test_missing_constant_and_padding_handling_is_explicit():
    s=signal(1,10,'J-TEXT');t=signal(2,20,'EAST')
    s['values'][:,0]=5;t['available'][:,2]=False
    s['values'][0,4]=np.nan;s['available'][1,4]=False
    s['values'][1,4]=1e30
    fit=fit_alignment([s],[t]);assert not fit['active'][0] and not fit['active'][2]
    expected=s['values'][2:,4]
    assert fit['source']['mean'][4]==pytest.approx(expected.mean())
    x=np.zeros((3,23));mask=np.zeros_like(x,dtype=bool)
    z,m=coordinates(x,mask,fit,'source',True)
    assert not z.any() and not m.any()
    with pytest.raises(ValueError):fit_statistics([dict(s,y=np.ones(100))])


def test_prefix_rejects_future_outcomes_reordering_and_incomplete_seals():
    order=list(range(10,32));prefix=[signal(i,i,'EAST') for i in order[:20]]
    seals={str(i):'a'*64 for i in order[:20]}
    validate_prefix(prefix,order,seals)
    with pytest.raises(ValueError):validate_prefix(prefix+[signal(30,30,'EAST')],order,seals)
    with pytest.raises(ValueError):validate_prefix(prefix[::-1],order,seals)
    with pytest.raises(ValueError):validate_prefix(prefix,order,{})
    with pytest.raises(ValueError):validate_prefix([dict(prefix[0],down=1)]+prefix[1:],order,seals)


def test_coordinate_forward_exactly_preserves_existing_encoder_and_mask():
    torch.set_num_threads(2);torch.manual_seed(5)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized',dim=16,heads=2),np.arange(23),np.ones(23)*2).eval()
    x=torch.randn(2,128,23);mask=torch.rand_like(x)>.2;mask[:,:5]=False
    clean=torch.where(mask,x,base.center);level=torch.asinh((clean-base.center)/base.scale)
    expected=base(x,mask)['logit'];actual=risk_from_coordinates(base,level,mask)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.sum().backward();assert base.embedding.weight.grad.norm()>0
    assert all(p.grad is None for p in base.forecast_head.parameters())


def test_tensor_transform_matches_numpy_and_keeps_mixed_missing_coordinates():
    torch.manual_seed(4);s=signal(1,10,'J-TEXT');t=signal(2,20,'EAST')
    fit=fit_alignment([s],[t]);base=Simple23(Config(fast_site='tokens'),np.zeros(23),np.ones(23))
    model=CoordinateClassifier(base,fit,coral=True)
    x=torch.randn(2,128,23);mask=torch.ones_like(x,dtype=torch.bool);mask[0,5,1]=False;mask[:,:3]=False
    level,actualmask=model.transform(x,mask,'source')
    z,m=coordinates(x.numpy(),mask.numpy(),fit,'source',True)
    np.testing.assert_allclose(level.detach().numpy(),z,atol=2e-6);np.testing.assert_array_equal(actualmask,m)
    assert abs(float(level[0,5,1]))>1e-5
    assert torch.isfinite(model(x,mask)['logit']).all()
    assert not base.fast_origin.requires_grad
    assert all(not p.requires_grad for p in base.forecast_head.parameters())


def test_warmup_predictions_keep_original_threshold_not_new_selected_threshold():
    rows=[dict(shot=i,is_disrupt=False,down=1.,times=np.array([.1]),scores=np.array([.6])) for i in range(21)]
    result=phase_metrics(rows,.5,.7)
    assert result['fp']==20 and result['n_safe']==21
    assert result['predictions'][0]['threshold']==.5 and result['predictions'][-1]['threshold']==.7
