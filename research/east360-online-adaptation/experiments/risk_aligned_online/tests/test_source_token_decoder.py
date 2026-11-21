"""Test the parent's actual token path, with offsets from its learned origin."""
import numpy as np
import pytest
import torch
from experiments.risk_aligned_online.simple23 import Config,Simple23
from experiments.risk_aligned_online.ended_decoder import DecoderFfnModel,SourceTokenDecoderModel,train_pair
from experiments.risk_aligned_online.continual_decoder import RollingDecoder
from experiments.risk_aligned_online.ended_context_rule import causal_peak_logit
from experiments.risk_aligned_online.redesign_20261004.intra import predict_shot
from experiments.risk_aligned_online.operational_memory_experiment import validate_temporary_site


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model():
    torch.manual_seed(131)
    base=Simple23(Config(fast_site='tokens',fast_rule='normalized'),torch.zeros(23),torch.ones(23))
    with torch.no_grad():base.fast_origin.fill_(.03)
    return SourceTokenDecoderModel(base).eval()


def shot(label=False):
    rng=np.random.default_rng(4)
    return dict(device='EAST',shot=12 if label else 13,values=rng.normal(size=(230,23)).astype(np.float32),
        available=np.ones((230,23),bool),times=np.arange(230)/1000.,valid=np.ones(230,bool),
        is_disrupt=label,down=.229 if label else None,horizon=125.,_sealed_scores=np.linspace(.1,.9,230))


def test_zero_offset_is_parent_at_its_learned_nonzero_origin():
    m=model();x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    assert m.temporary_dim==128 and not m.initial_fast(1).any()
    left,right=m(x,a),m.base(x,a,m.base.initial_fast(1).detach().requires_grad_(True))
    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def test_offset_changes_the_internal_token_path_not_a_score_correction():
    m=model();x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    delta=torch.randn(1,128,requires_grad=True)*.1
    actual=m(x,a,delta);direct=m.base(x,a,m.base.initial_fast(1)+delta)
    for key in actual:torch.testing.assert_close(actual[key],direct[key],rtol=0,atol=0)
    g,=torch.autograd.grad(actual['logit'].sum(),delta)
    assert g.abs().sum()>0


def test_learned_decoder_can_change_forecast_but_not_current_zero_offset_risk():
    m=model();x=torch.randn(1,128,23);a=torch.ones_like(x,dtype=torch.bool)
    before=m(x,a)
    with torch.no_grad():m.decoder.bias.add_(.1)
    after=m(x,a)
    torch.testing.assert_close(before['logit'],after['logit'],rtol=0,atol=0)
    assert not torch.equal(before['forecast'],after['forecast'])


def test_reference_uses_same_token_geometry_but_original_source_forecast_head():
    m=model()
    with torch.no_grad():m.decoder.bias.add_(.1)
    learner=RollingDecoder(m,mode='early_fixed')
    assert isinstance(learner.reference,SourceTokenDecoderModel)
    torch.testing.assert_close(learner.reference.decoder.bias,m.base.forecast_head.bias,rtol=0,atol=0)


def test_causal_meta_replay_matches_live_token_updates():
    m=model();s=shot();scores,audit=predict_shot(m,s,intra=True)
    for q in (0,25,26,125,126,180):
        z,_=causal_peak_logit(m,s,q)
        assert abs(float(z.detach().sigmoid())-float(scores[q]))<2e-6
    assert audit['fast_discarded'] and not hasattr(m,'memory')


def test_ended_risk_gradient_changes_only_auxiliary_head():
    m=model();original={k:v.clone() for k,v in m.base.state_dict().items()}
    result=train_pair(m,[(shot(False),180),(shot(True),180)],
        torch.optim.Adam(m.decoder.parameters(),lr=.0003),boundary=.7,mode='risk_after_update')
    assert result['risk_gradient_norm']>0 and result['decoder_delta_norm']>0
    for key,v in m.base.state_dict().items():torch.testing.assert_close(v,original[key],rtol=0,atol=0)
    m.assert_persistent_inactive()


def test_current_outcome_never_changes_same_shot_predictions():
    m=model();a,_=predict_shot(m,shot(False),intra=True);b,_=predict_shot(m,shot(True),intra=True)
    np.testing.assert_array_equal(a,b)


def test_cuda_no_grad_and_gradient_prediction_have_one_exact_attention_dispatch():
    assert torch.cuda.is_available(),'this repository task explicitly uses local rl_env CUDA'
    m=model().to('cuda').requires_grad_(False)
    reference=DecoderFfnModel(m.base).to('cuda').eval().requires_grad_(False)
    x=torch.randn(3,128,23,device='cuda');a=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():left=m(x,a)['logit'];old=reference(x,a)['logit']
    delta=m.initial_fast(3).requires_grad_(True);right=m(x,a,delta)['logit']
    torch.testing.assert_close(left,right,rtol=0,atol=0)
    torch.testing.assert_close(left,old,rtol=0,atol=0)


@pytest.mark.parametrize('arm',['mapped_static','intra_control'])
def test_token_pilot_has_no_unimplemented_persistent_route(arm):
    validate_temporary_site('source_tokens',arm)


@pytest.mark.parametrize('site,arm',[('source_tokens','inter_only'),('source_tokens','both'),('unknown','intra_control')])
def test_unsupported_persistent_or_unknown_site_is_not_silently_substituted(site,arm):
    with pytest.raises(ValueError):validate_temporary_site(site,arm)
