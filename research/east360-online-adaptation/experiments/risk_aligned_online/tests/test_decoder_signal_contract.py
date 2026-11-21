"""Input value is not the same decision as auxiliary-gradient supervision."""
import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.continual_decoder import RollingDecoder
from experiments.risk_aligned_online.ended_context_rule import causal_peak_logit
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel, maintenance_loss, train_pair
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.simple23 import Config, Simple23

MNN = io23.RAW23.index('mode_number_n')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(contract):
    torch.manual_seed(131)
    base = Simple23(Config(fast_site='tokens', fast_rule='normalized'),
                    torch.zeros(23), torch.ones(23))
    return DecoderFfnModel(base, signal_contract=contract).eval()


def shot(label=False):
    rng = np.random.default_rng(4)
    return dict(device='EAST', shot=12 if label else 13,
        values=rng.normal(size=(230,23)).astype(np.float32), available=np.ones((230,23),bool),
        times=np.arange(230)/1000.,valid=np.ones(230,bool), is_disrupt=label,
        down=.229 if label else None,horizon=125.,_sealed_scores=np.linspace(.1,.9,230))


def test_original_contract_is_bitexact_not_a_redefined_default():
    m = model('all23')
    prediction=torch.randn(1,25,23,requires_grad=True)
    truth=torch.randn_like(prediction);mask=torch.rand_like(prediction)>.2
    actual=m.forecast_loss(prediction,truth,mask)
    expected=m.base.forecast_loss(prediction,truth,mask)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_mnn_exclusion_changes_only_loss_not_input_or_zero_fast_risk():
    full, excluded = model('all23'),model('without_mnn')
    x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    before_x=x.clone();before_a=a.clone()
    left,right=full(x,a),excluded(x,a)
    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
    torch.testing.assert_close(x,before_x,rtol=0,atol=0)
    torch.testing.assert_close(a,before_a,rtol=0,atol=0)
    assert excluded.signal_contract=='without_mnn'


def test_excluded_output_and_truth_do_not_change_loss_or_gradient():
    m=model('without_mnn')
    pred=torch.randn(1,25,23,requires_grad=True)
    truth=torch.randn_like(pred);mask=torch.ones_like(pred,dtype=torch.bool)
    saved=mask.clone();modified=pred.detach().clone();changed_truth=truth.clone()
    modified[:,:,MNN]+=1000.;changed_truth[:,:,MNN]-=1000.
    loss=m.forecast_loss(pred,truth,mask)
    other=m.forecast_loss(modified,changed_truth,mask)
    torch.testing.assert_close(loss,other,rtol=0,atol=0)
    grad,=torch.autograd.grad(loss,pred)
    assert not grad[:,:,MNN].any() and grad[:,:,:MNN].abs().sum()>0
    torch.testing.assert_close(mask,saved,rtol=0,atol=0)


def test_all_other_observed_targets_keep_original_masked_smooth_l1():
    m=model('without_mnn')
    pred=torch.randn(1,25,23,requires_grad=True);truth=torch.randn_like(pred)
    mask=torch.rand_like(pred)>.4;mask[:,:,MNN]=False
    actual=m.forecast_loss(pred,truth,mask)
    expected=m.base.forecast_loss(pred,truth,mask)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_only_mnn_observed_is_zero_information_not_a_fabricated_target():
    m=model('without_mnn');pred=torch.randn(1,25,23,requires_grad=True)
    mask=torch.zeros_like(pred,dtype=torch.bool);mask[:,:,MNN]=True
    loss=m.forecast_loss(pred,torch.randn_like(pred),mask)
    grad,=torch.autograd.grad(loss,pred)
    assert loss==0 and not grad.any()


def test_reference_uses_the_same_signal_contract_as_deployed_meta_learner():
    m=model('without_mnn');learner=RollingDecoder(m,mode='early_fixed')
    assert learner.reference.signal_contract==m.signal_contract


def test_live_intra_and_meta_replay_use_the_same_changed_loss():
    m=model('without_mnn');s=shot()
    scores,audit=predict_shot(m,s,intra=True)
    for q in (25,26,125,126,180):
        z,_=causal_peak_logit(m,s,q)
        assert abs(float(z.detach().sigmoid())-float(scores[q]))<2e-6
    assert audit['fast_discarded'] and not m.memory.any()


def test_ended_maintenance_and_risk_meta_gradient_retain_the_changed_contract():
    m=model('without_mnn');s=shot()
    # Maintenance still reads MNN as an INPUT; only targets are excluded.
    # Hence do not assert input interventions leave predictions invariant.
    loss=maintenance_loss(m,s)
    assert torch.isfinite(loss)
    result=train_pair(m,[(s,180),(shot(True),180)],
        torch.optim.Adam(m.decoder.parameters(),lr=.0003),boundary=.7,mode='risk_after_update')
    assert result['risk_gradient_norm']>0 and result['decoder_delta_norm']>0
    assert not m.memory.any() and m.memory.grad is None


def test_unknown_contract_raises_instead_of_reverting_to_all23():
    with pytest.raises(ValueError,match='signal contract'):model('unknown')
