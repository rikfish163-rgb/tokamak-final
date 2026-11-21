import numpy as np
import pytest
import torch
from types import SimpleNamespace

from experiments.risk_aligned_online.startup_intervention_replay import predict_intervention
from experiments.risk_aligned_online import simple_data, io23


class ToyModel:
    center = torch.zeros(23)
    cfg = SimpleNamespace(window=128)

    def __call__(self, x, a):
        return {'logit': torch.where(a, x, 0.).sum((1, 2)) / 2000.}


def shot(n=150):
    return dict(shot=1, device='EAST', values=np.ones((n,23),np.float32),
                available=np.ones((n,23),bool), times=np.arange(n)/1000.)


def test_replay_uses_only_observed_past_and_does_not_mutate_inputs():
    s=shot(); old=s['values'].copy()
    p,a=predict_intervention(ToyModel(),s,False,device='cpu',mode='mask_ne_nG')
    s['values'][80:]=1000.
    changed,_=predict_intervention(ToyModel(),s,False,device='cpu',mode='mask_ne_nG')
    np.testing.assert_array_equal(p[:80],changed[:80])
    np.testing.assert_array_equal(s['values'][:80],old[:80])
    expected=1/(1+np.exp(-22*np.minimum(np.arange(150)+1,128)/2000))
    np.testing.assert_allclose(p,expected,atol=1e-7)
    assert a['accepted_updates']==0 and a['model_parameter_change_norm']==0


def test_history_control_never_copies_future_or_earlier_record_data():
    p,_=predict_intervention(ToyModel(),shot(),False,device='cpu',mode='history32')
    expected=1/(1+np.exp(-23*np.minimum(np.arange(150)+1,32)/2000))
    np.testing.assert_allclose(p,expected,atol=1e-7)


@pytest.mark.parametrize('mode,intra',[('invented',False),('static',True)])
def test_no_update_mode_must_not_pretend_to_be_neural_adaptation(mode,intra):
    with pytest.raises(ValueError):
        predict_intervention(ToyModel(),shot(),intra,device='cpu',mode=mode)
