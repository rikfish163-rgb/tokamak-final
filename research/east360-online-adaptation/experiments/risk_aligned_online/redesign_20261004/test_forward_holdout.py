"""Chronological signal-only acceptance; later H25 points never fit the step."""
import numpy as np
import pytest
import torch

from .test_intra import MockContext, PrefixData, OpaqueOutcomeShot, shot
from .intra import predict_shot


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def run(s=None,model=None,**kw):
    return predict_shot(model or MockContext(),s if s is not None else shot(27),
                        True,objective='forward_holdout',data_api=PrefixData(),**kw)


def opposing():
    s=shot(27);s['values'][13:26]=-1.
    return s


def test_accepts_only_after_first12_fit_and_last13_validation_improve():
    scores,audit=run();r=audit['attempts'][0]
    assert r['accepted'] and r['effective_index']==26
    assert r['training_targets']==12*23 and r['holdout_targets']==13*23
    assert r['holdout_before']>r['holdout_after'] and r['holdout_gain']>0
    assert r['loss_before']>r['loss_after']
    assert r['predictions_sealed_through_index']==25
    assert scores[26]>scores[25] and audit['forecast_fit_steps']==[1,12]
    assert audit['forecast_holdout_steps']==[13,25]


def test_rejects_same_gradient_if_later_arrived_truth_worsens():
    good,ga=run();bad,ba=run(opposing())
    a,b=ga['attempts'][0],ba['attempts'][0]
    np.testing.assert_array_equal(a['gradient'],b['gradient'])
    assert b['loss_after']<b['loss_before']
    assert b['holdout_after']>b['holdout_before']
    assert not b['accepted'] and b['reason']=='holdout_not_improved'
    assert b['effective_index'] is None and b['fast_change_norm']==0
    np.testing.assert_array_equal(bad,np.full(27,bad[0],np.float32))
    np.testing.assert_array_equal(good[:26],bad[:26])


def test_fit_gradient_never_sees_last13_truth_or_validation_mask():
    left,right=shot(27),shot(27)
    right['values'][13:26]=100.
    right['available'][13:26,1:]=False
    _,a=run(left);_,b=run(right)
    np.testing.assert_array_equal(a['attempts'][0]['gradient'],b['attempts'][0]['gradient'])
    assert b['attempts'][0]['holdout_targets']==13


@pytest.mark.parametrize('part,reason',[(slice(1,13),'no_training_targets'),
                                     (slice(13,26),'no_holdout_targets')])
def test_missing_partition_is_explicitly_no_update(part,reason):
    s=shot(27);s['available'][part]=False
    _,audit=run(s);r=audit['attempts'][0]
    assert not r['accepted'] and r['reason']==reason
    assert r['grad_norm'] is None and r['fast_before']==r['fast_after']


def test_cannot_use_the_holdout_before_it_arrives():
    s=shot(14);_,a=run(s)
    assert a['attempts']==[] and a['accepted_updates']==0
    assert a['pending'][0]['maturity_index']==25


def test_static_output_is_bitwise_old_level_control():
    s=shot(160);model=MockContext()
    a,aa=predict_shot(model,s,False,objective='forward_holdout',data_api=PrefixData())
    b,bb=predict_shot(model,s,False,objective='level',data_api=PrefixData())
    np.testing.assert_array_equal(a,b)
    assert aa['accepted_updates']==bb['accepted_updates']==0


def test_resets_temporary_state_and_never_reads_outcomes():
    model=MockContext();before={k:v.clone() for k,v in model.state_dict().items()}
    s=OpaqueOutcomeShot(shot(160),y=object(),is_disrupt=object(),down=object(),horizon=object())
    a,aa=run(s,model);b,bb=run(s,model)
    np.testing.assert_array_equal(a,b)
    assert aa['fast_discarded'] and aa['accepted_updates']==bb['accepted_updates']
    for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_keeps_autograd_attention_path_under_callers_no_grad():
    model=MockContext(numerical_trap=True)
    with torch.no_grad():_,audit=run(model=model)
    assert audit['accepted_updates']==1
    assert all(c['grad'] and c['fast_grad'] for c in model.calls)


def test_rejected_holdout_does_not_change_default_level_algorithm():
    s=shot(27);a,aa=predict_shot(MockContext(),s,True,objective='level',data_api=PrefixData())
    b,bb=predict_shot(MockContext(),s,True,data_api=PrefixData())
    np.testing.assert_array_equal(a,b)
    assert aa['attempts']==bb['attempts']
