import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.forecast_profile_20261025.learning import decompose,shuffle_targets,feedback,predict


def test_exact_loss_and_gradient_decomposition_with_missing_values():
    torch.manual_seed(7)
    p=torch.randn(4,25,23,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(p);mask=torch.rand_like(p)>.3;mask[:,:,2]=False
    y[~mask]=float('nan')
    losses=decompose(p,y,mask)
    torch.testing.assert_close(losses['full'],losses['level']+losses['shape'],atol=1e-14,rtol=1e-12)
    gs={k:torch.autograd.grad(v,p,retain_graph=True)[0] for k,v in losses.items()}
    torch.testing.assert_close(gs['full'],gs['level']+gs['shape'],atol=1e-14,rtol=1e-12)
    assert all(torch.isfinite(v).all() and not v[~mask].any() for v in gs.values())
    empty=decompose(p,y,torch.zeros_like(mask))
    assert all(float(v.detach())==0 for v in empty.values())


def test_shape_ignores_common_offset_but_retains_between_segment_change():
    p=torch.zeros(4,25,23,dtype=torch.float64,requires_grad=True);m=torch.ones_like(p,dtype=torch.bool)
    y=torch.ones_like(p)*5
    assert decompose(p,y,m)['shape']==0
    y[0]+=2
    a=decompose(p,y,m);b=decompose(p,y+9,m)
    assert a['shape']>0
    torch.testing.assert_close(a['shape'],b['shape'],rtol=0,atol=1e-14)
    ga=torch.autograd.grad(a['shape'],p)[0];gb=torch.autograd.grad(b['shape'],p)[0]
    torch.testing.assert_close(ga,gb,rtol=0,atol=1e-14)


def test_shuffle_preserves_observed_values_and_missing_positions():
    y=torch.arange(4*25*23,dtype=torch.float64).reshape(4,25,23)
    m=torch.ones_like(y,dtype=torch.bool);m[0,::2]=False;m[:,:,2]=False
    shuffled=shuffle_targets(y,m)
    torch.testing.assert_close(shuffled,shuffle_targets(y,m),rtol=0,atol=0)
    for c in range(23):
        torch.testing.assert_close(y[:,:,c][m[:,:,c]].sort().values,shuffled[:,:,c][m[:,:,c]].sort().values)
    assert not torch.equal(shuffled[m],y[m])
    assert not shuffled[~m].any()


def test_arrived_only_and_end_to_end_parent_reset():
    from experiments.risk_aligned_online.tests.test_task_geometry import small_model
    from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
    model=small_model();original={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(8).normal(size=(400,23)).astype('float32')
    s=dict(values=x,available=np.ones_like(x,bool));s['available'][:,2]=False
    fast=initial_encoder(model)
    gradients,r=feedback(model,s,256,fast,device='cpu')
    altered={**s,'values':x.copy()};altered['values'][256:]=321
    g2,r2=feedback(model,altered,256,fast,device='cpu')
    assert r==r2
    for k in gradients:torch.testing.assert_close(gradients[k],g2[k],rtol=0,atol=0)
    with pytest.raises(ValueError):feedback(model,dict(s,down=3),256,fast,device='cpu')
    curves,updates=predict(model,s,device='cpu')
    for k in curves:np.testing.assert_array_equal(curves[k][:256],curves['off'][:256])
    for rows in updates.values():
        assert [r['first_affected_prediction'] for r in rows]==[256,384]
        assert all(r['actual_step_norm']<=.001001 for r in rows)
    altered['values'][:330]=x[:330]
    later,_=predict(model,altered,device='cpu')
    again,_=predict(model,s,device='cpu')
    for k in curves:
        np.testing.assert_array_equal(curves[k][:330],later[k][:330])
        np.testing.assert_array_equal(curves[k],again[k])
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
