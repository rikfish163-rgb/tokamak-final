import pytest
import torch

from experiments.risk_aligned_online.trajectory_separation_20261023.analysis import (
    separation, parameter_group, channel_separation,
)


def test_positive_and_negative_curvature_and_clipped_identity():
    a,b=torch.tensor([2.,1.]),torch.tensor([1.,1.])
    positive=separation(a,b,a,b,learning_rate=.1,clip=100.)
    negative=separation(a,b,-a,-b,learning_rate=.1,clip=100.)
    assert positive['distance_ratio']==pytest.approx(.9)
    assert negative['distance_ratio']==pytest.approx(1.1)
    assert positive['secant_curvature']>0 and negative['secant_curvature']<0
    clipped=separation(a,b,torch.tensor([20.,0.]),torch.tensor([0.,5.]),learning_rate=.1,clip=1.)
    assert clipped['scale_early']==pytest.approx(.05)
    assert clipped['scale_late']==pytest.approx(.2)
    assert clipped['distance_after_squared']==pytest.approx(.82)
    assert clipped['identity_error']<1e-12


def test_channel_differences_sum_to_actual_direction_difference():
    d=torch.tensor([1.,2.])
    early=torch.tensor([[3.,4.],[1.,0.],[0.,0.]])
    late=torch.tensor([[1.,0.],[0.,2.],[0.,0.]])
    result=channel_separation(d,early,late,.2,.5,.001)
    expected=float(-2*.001*torch.dot(d,early.sum(0)*.2-late.sum(0)*.5))
    assert sum(r['linear_distance_squared_change'] for r in result)==pytest.approx(expected)
    assert result[-1]['early_gradient_norm']==0
    with pytest.raises(ValueError): channel_separation(d,early,late[:1],1.,1.,.001)


def test_groups_cover_exact_encoder_without_head():
    names=['base.embedding.weight','base.embedding.bias','base.cls','base.position',
           'base.encoder.layers.0.self_attn.in_proj_weight',
           'base.encoder.layers.0.self_attn.out_proj.bias',
           'base.encoder.layers.0.linear1.weight','base.encoder.layers.0.norm2.weight']
    assert [parameter_group(n) for n in names]==[
        'embedding','embedding','position_cls','position_cls','attention','attention','ffn','normalization']
    with pytest.raises(ValueError):parameter_group('base.risk_head.weight')
