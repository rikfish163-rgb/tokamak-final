import pytest
import torch

from experiments.risk_aligned_online.expected_dynamics_20261028.forecast import DirectForecast, observed_features
from experiments.risk_aligned_online.expected_dynamics_20261028.anchored import AnchoredForecast


def initialized(*, anchored, omit_mnm):
    torch.manual_seed(131)
    m=AnchoredForecast(anchored=anchored,omit_mnm=omit_mnm)
    with torch.no_grad():
        m.net[-1].weight.normal_(std=.03);m.net[-1].bias.normal_()
    return m


def test_regular_arm_is_original_direct_predictor():
    m=initialized(anchored=False,omit_mnm=False)
    old=DirectForecast();old.load_state_dict(m.state_dict())
    x=torch.randn(13,92)
    torch.testing.assert_close(m(x),old(x),atol=0,rtol=0)


def test_constant_observed_signal_has_exact_zero_predicted_change():
    m=initialized(anchored=True,omit_mnm=False)
    values=torch.randn(1,23).expand(12,-1)
    features=observed_features(values,torch.ones_like(values,dtype=torch.bool))
    assert m(features).count_nonzero()==0
    assert m(features*17).count_nonzero()==0


def test_mnm_input_and_availability_have_no_effect_in_missing_channel_arm():
    m=initialized(anchored=True,omit_mnm=True)
    x=torch.randn(12,92);other=x.clone();other[:,[2,25,48,71]]=999.
    torch.testing.assert_close(m(x),m(other),atol=0,rtol=0)
    before=x.clone();m(x);assert torch.equal(before,x)


def test_prefix_predictions_and_gradient_survive_anchor_constraint():
    m=initialized(anchored=True,omit_mnm=True)
    x=torch.randn(16,23);mask=torch.ones_like(x,dtype=torch.bool)
    features=observed_features(x,mask)
    torch.testing.assert_close(m(features)[:8],m(observed_features(x[:8],mask[:8])))
    loss=m(features).square().mean();loss.backward()
    assert m.net[0].weight.grad.norm()>0 and m.net[-1].weight.grad.norm()>0
    torch.testing.assert_close(m.net[-1].bias.grad,torch.zeros_like(m.net[-1].bias.grad),atol=1e-7,rtol=0)
    assert m.net[0].weight.grad[:,[2,25,48,71]].count_nonzero()==0
    with pytest.raises(ValueError):AnchoredForecast(anchored='yes',omit_mnm=False)
