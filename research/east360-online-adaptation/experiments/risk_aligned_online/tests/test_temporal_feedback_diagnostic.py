import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.temporal_feedback_diagnostic_20261023.run import timing_record,loss_components
from experiments.risk_aligned_online.lag_bias_feedback_20261022.learning import feedback_loss,split_residuals


@pytest.mark.parametrize('lead,expected',[(0,'late_0_10'),(10,'late_0_10'),(11,'late_10_25'),
    (25,'late_10_25'),(26,'late_25_30'),(30,'late_25_30'),(31,'timely_30_125'),
    (125,'timely_30_125'),(126,'timely_125_3000'),(3000,'timely_125_3000'),(3001,'early_over3000')])
def test_first_alarm_strict_lead_partitions(lead,expected):
    times=np.array([0.,(4000-lead)/1000.,4.]);p=np.array([.1,.8,.9])
    r=timing_record(times,p,np.ones(3,bool),True,4.,.8)
    assert r['category']==expected and r['lead_ms']==lead
    assert r['alarm_index']==1


def test_missing_alarm_and_masked_postterminal_never_count():
    r=timing_record(np.array([0.,1.]),np.array([.1,.9]),np.array([True,False]),True,.5,.8)
    assert r['category']=='never' and r['alarm_index'] is None
    r=timing_record(np.array([0.,1.]),np.array([.1,.9]),np.ones(2,bool),False,None,.8)
    assert r['category']=='safe_fp'
    r=timing_record(np.array([0.,1.]),np.array([.1,.1]),np.ones(2,bool),False,None,.8)
    assert r['category']=='safe_tn'


@pytest.mark.parametrize('correct',[False,True])
def test_channel_terms_exactly_sum_to_current_feedback(correct):
    torch.manual_seed(10)
    prediction=torch.randn(4,25,23,requires_grad=True);truth=torch.randn_like(prediction)
    mask=torch.ones_like(prediction,dtype=torch.bool);mask[:2,:,0]=False;mask[2:,:3,5]=False
    bias,valid=split_residuals(prediction,truth,mask)
    parts=loss_components(prediction,truth,mask,bias,valid,correct=correct)
    expected=feedback_loss(prediction,truth,mask,bias,valid,correct=correct)
    torch.testing.assert_close(parts.sum(),expected)
    gradient,=torch.autograd.grad(parts.sum(),prediction,retain_graph=True)
    target,=torch.autograd.grad(expected,prediction)
    torch.testing.assert_close(gradient,target)
    assert not gradient[:2].any() and not gradient[:,:,0].any()
