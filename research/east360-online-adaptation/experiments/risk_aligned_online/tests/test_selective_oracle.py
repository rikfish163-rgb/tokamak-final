import numpy as np
import pytest

from experiments.risk_aligned_online.observed_oracle_20261023.selective import teaching_mask
from experiments.risk_aligned_online.observed_oracle_20261023.learning import step
from experiments.risk_aligned_online.tests.test_contextual_feedback import setup
from experiments.risk_aligned_online.paired_feedback_20261022.model import initial_encoder


def test_mask_does_not_fabricate_positive_labels():
    y=np.array([0,0,1,1,0]);valid=np.array([1,1,1,1,0],bool);copy=y.copy()
    np.testing.assert_array_equal(teaching_mask(y,valid,True),[0,0,1,1,0])
    np.testing.assert_array_equal(y,copy)
    np.testing.assert_array_equal(teaching_mask(np.zeros(5),valid,False),valid)
    with pytest.raises(ValueError):teaching_mask(y,valid,False)


def test_early_disruptive_prefix_is_identity_without_oracle_positive():
    m,_,s=setup();signal={k:s[k][:256] for k in ('values','available')}
    valid=teaching_mask(np.zeros(256),np.ones(256,bool),True);state=initial_encoder(m)
    new,receipt=step(m,signal,np.zeros(256),valid,state,steps=4,device='cpu')
    assert new is state and receipt['gradient_steps']==0


def test_replay_receipt_of_reused_safe_shot_comes_from_verified_parent():
    from experiments.risk_aligned_online.observed_oracle_20261023.verify_selective import expected_updates
    parent=dict(shot=96375,updates={'oracle1':[1],'oracle4':[4]})
    safe=dict(shot=96375,reused_safe_parent=True)
    assert expected_updates(safe,parent)==parent['updates']
    disruptive=dict(shot=64581,reused_safe_parent=False,updates={'oracle1':[],'oracle4':[]})
    assert expected_updates(disruptive,dict(shot=64581,updates={} ))==disruptive['updates']
    with pytest.raises(ValueError):expected_updates(safe,dict(parent,shot=1))
