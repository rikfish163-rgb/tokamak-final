import pytest
import torch

from ..ffn_intra_20261005.test_meta import case
from .feedback import segment_components, segment_step, event_clock


def tensors():
    prediction=torch.zeros(1,25,23,requires_grad=True)
    return prediction,torch.zeros_like(prediction),torch.ones_like(prediction,dtype=torch.bool)


def test_level_and_difference_partition_does_not_duplicate_boundary():
    p,y,m=tensors();c=torch.zeros(23);s=torch.ones(23)
    early=segment_components(p,y[:,:5],m[:,:5],c,s,'early')
    late=segment_components(p,y[:,4:],m[:,4:],c,s,'late')
    assert early['level_count']==5*23 and late['level_count']==20*23
    assert early['difference_count']==4*23 and late['difference_count']==20*23
    # h5 enters the second difference but NOT the second level term.
    p.data[:,4]=1.
    late=segment_components(p,y[:,4:],m[:,4:],c,s,'late')
    assert late['level'].sum()==0
    assert late['difference'].sum()>0


def test_missing_h5_removes_both_adjacent_pairs_without_bridge():
    p,y,m=tensors();m[:,4]=False
    args=(torch.zeros(23),torch.ones(23))
    early=segment_components(p,y[:,:5],m[:,:5],*args,'early')
    late=segment_components(p,y[:,4:],m[:,4:],*args,'late')
    assert early['difference_count']==3*23
    assert late['difference_count']==19*23
    assert late['level_count']==20*23


def test_no_future_block_allowed_in_early_interface():
    p,y,m=tensors()
    with pytest.raises(ValueError,match='arrived'):
        segment_components(p,y,m,torch.zeros(23),torch.ones(23),'early')


@pytest.mark.parametrize('segment,length', [('early',5),('late',21)])
def test_missing_targets_leave_actual_delta_bit_exact(segment,length):
    model,shot,_=case()
    delta=torch.randn_like(model.initial_delta()).requires_grad_()
    x=torch.tensor(shot['values'][:128])[None];a=torch.ones_like(x,dtype=torch.bool)
    truth=torch.full((1,length,23),float('nan'));observed=torch.zeros_like(truth,dtype=torch.bool)
    result,row=segment_step(model,delta,x,a,truth,observed,segment)
    assert result is delta and torch.equal(result,delta)
    assert row['reason']=='no_valid_targets'


def test_clock_records_arrival_separately_from_activation():
    assert event_clock(0,'partial')==[(5,5,6,'early'),(25,25,26,'late')]
    assert event_clock(26,'partial')==[(31,31,32,'early'),(51,51,52,'late')]
    assert event_clock(0,'delayed')==[(5,25,26,'early'),(25,25,26,'late')]


def test_two_step_operator_has_real_ffn_and_head_gradient():
    model,shot,_=case();model.base.forecast_head.requires_grad_(True)
    x=torch.tensor(shot['values'][:128])[None];a=torch.ones_like(x,dtype=torch.bool)
    truth=torch.tensor(shot['values'][128:153])[None]
    observed=torch.ones_like(truth,dtype=torch.bool)
    start=model.initial_delta()
    early,row=segment_step(model,start,x,a,truth[:,:5],observed[:,:5],'early',create_graph=True)
    final,_=segment_step(model,early,x,a,truth[:,4:],observed[:,4:],'late',create_graph=True)
    assert row['state_changed'] and 0<early.norm()<=.020001 and final.norm()<=.100001
    risk=model(x,a,final)['logit'].sum()
    gradient,=torch.autograd.grad(risk,model.base.forecast_head.weight)
    assert torch.isfinite(gradient).all() and gradient.norm()>1e-8


def test_same_feedback_operators_match_at_cycle_end_not_midcycle():
    model,shot,_=case()
    x=torch.tensor(shot['values'][:128])[None];a=torch.ones_like(x,dtype=torch.bool)
    truth=torch.tensor(shot['values'][128:153])[None];observed=torch.ones_like(truth,dtype=torch.bool)
    initial=model.initial_delta()
    early,_=segment_step(model,initial,x,a,truth[:,:5],observed[:,:5],'early')
    # Emitting intermediate risk does not mutate the stored state or buffers.
    risk_partial=model(x,a,early)['logit'].detach()
    risk_delayed=model(x,a,initial)['logit'].detach()
    assert not torch.equal(risk_partial,risk_delayed)
    partial_end,_=segment_step(model,early,x,a,truth[:,4:],observed[:,4:],'late')
    delayed_first,_=segment_step(model,model.initial_delta(),x,a,truth[:,:5],observed[:,:5],'early')
    delayed_end,_=segment_step(model,delayed_first,x,a,truth[:,4:],observed[:,4:],'late')
    torch.testing.assert_close(partial_end,delayed_end,rtol=0,atol=0)
