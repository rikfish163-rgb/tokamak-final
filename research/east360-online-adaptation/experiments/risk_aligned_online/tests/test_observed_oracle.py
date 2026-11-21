import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.tests.test_contextual_feedback import setup
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder
from experiments.risk_aligned_online.observed_oracle_20261023.learning import step, predict


def test_supervised_direction_changes_only_temporary_encoder():
    m,_,s=setup(); saved={k:v.clone() for k,v in m.state_dict().items()}
    prefix={k:s[k][:256] for k in ('values','available')}
    for label in (0.,1.):
        state=initial_encoder(m)
        new,r=step(m,prefix,np.full(256,label),np.ones(256,bool),state,steps=1,device='cpu')
        assert r['loss_after'] < r['loss_before']
        assert r['parameter_delta_norm'] > 0
        assert r['supervised_ends']==list(range(143,256,16))
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items())
        assert any(not torch.equal(state[k],new[k]) for k in state)


def test_oracle_rejects_unmatched_prefix_or_label_contract():
    m,_,s=setup();prefix={k:s[k][:256] for k in ('values','available')};state=initial_encoder(m)
    with pytest.raises(ValueError):step(m,prefix,np.zeros(257),np.ones(256,bool),state,steps=1,device='cpu')
    with pytest.raises(ValueError):step(m,prefix,np.full(256,.5),np.ones(256,bool),state,steps=1,device='cpu')
    with pytest.raises(ValueError):step(m,dict(prefix,down=1.),np.zeros(256),np.ones(256,bool),state,steps=1,device='cpu')


def test_no_update_before_clock_and_no_future_signal_dependence():
    m,_,s=setup();signal={k:s[k][:385].copy() for k in ('values','available')}
    y=np.zeros(385);valid=np.ones(385,bool)
    a,ra=predict(m,signal,y,valid,device='cpu');b,rb=predict(m,signal,y,valid,device='cpu')
    changed={k:v.copy() for k,v in signal.items()};changed['values'][384:]=100
    c,rc=predict(m,changed,y,valid,device='cpu')
    for arm in a:
        np.testing.assert_array_equal(a[arm],b[arm]);np.testing.assert_array_equal(a[arm][:384],c[arm][:384])
    np.testing.assert_array_equal(a['oracle1'][:256],a['oracle4'][:256])
    assert ra==rb==rc
    assert [r['first_affected_prediction'] for r in ra['oracle1']]==[256,384]


def test_unobserved_supervision_is_not_replaced_with_zero_labels():
    m,_,s=setup();prefix={k:s[k][:256] for k in ('values','available')};state=initial_encoder(m)
    new,r=step(m,prefix,np.zeros(256),np.zeros(256,bool),state,steps=4,device='cpu')
    assert r['gradient_steps']==0 and r['reason']=='no_valid_supervision'
    for k in state:assert torch.equal(state[k],new[k])
