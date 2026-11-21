"""Matched head learning may persist; neither comparison arm learns FFN memory."""
from copy import deepcopy

import pytest

from experiments.risk_aligned_online.verify_operational_intra_pair import verify_prediction_audit


def receipt(index=31,intra=True):
    return dict(intra=intra,fast_discarded=True,model_parameter_change_norm=0.,model_buffer_change_norm=0.,
        persistent=dict(head=dict(actual_steps=0,effective_from_index=index+1),
                        memory=dict(actual_steps=0,accepted=False)))


@pytest.mark.parametrize('intra',[True,False])
def test_same_postseal_feedback_without_persistent_ffn(intra):
    verify_prediction_audit(receipt(intra=intra),index=31,intra=intra)


@pytest.mark.parametrize('bad',['missing_feedback','wrong_intra','source_change','memory','late_head','early_effect'])
def test_missing_auxiliary_training_or_wrong_arm_is_detected(bad):
    row=deepcopy(receipt())
    if bad=='missing_feedback':row['persistent']=None
    elif bad=='wrong_intra':row['intra']=False
    elif bad=='source_change':row['model_parameter_change_norm']=.1
    elif bad=='memory':row['persistent']['memory']['actual_steps']=1
    elif bad=='late_head':row['persistent']['head']['actual_steps']=4
    else:row['persistent']['head']['effective_from_index']=31
    with pytest.raises(AssertionError):verify_prediction_audit(row,index=31,intra=True)
