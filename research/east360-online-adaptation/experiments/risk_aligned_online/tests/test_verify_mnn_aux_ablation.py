from copy import deepcopy
import numpy as np
import pytest
from experiments.risk_aligned_online.verify_mnn_aux_ablation import verify_protocol_pair, sealed_decisions


def protocols():
    common=dict(signal_contract='without_mnn',input_feature_count=23,
        auxiliary_excluded_features=['mode_number_n'],mode_number_n_is_assumed_categorical=False,
        persistent_enabled=False,alarm_tuning=False,auxiliary_postseal_feedback_in_all_arms=True,
        auxiliary_head_frozen_after30=True,parent_sha256='source',threshold=.7,
        order_sha256='order',input_mapping='same',seed=131,first30_steps=4,bootstrap_steps=40,
        intra_step=.05,intra_radius=1.,stride=100,horizon=25)
    return {a:dict(deepcopy(common),arm=a,intra=a=='intra_control')
            for a in ('mapped_static','intra_control')}


def test_declared_same_input_loss_ablation_passes():
    verify_protocol_pair(protocols())


@pytest.mark.parametrize('field,value',[('input_feature_count',22),('signal_contract','all23'),
    ('mode_number_n_is_assumed_categorical',True),('persistent_enabled',True),('alarm_tuning',True),
    ('threshold',.8),('intra_step',.1),('input_mapping','changed')])
def test_changed_input_alarm_or_budget_is_not_the_declared_ablation(field,value):
    p=protocols();p['intra_control'][field]=value
    with pytest.raises(ValueError):verify_protocol_pair(p)


def test_independent_decisions_preserve_label_without_duplicate_keyword_and_exclude_exact30ms():
    rows=[dict(shot=1,is_disrupt=True,down=.1,times=np.array([.068,.070]),
               valid=np.ones(2,bool),scores=np.array([.1,.9]))]
    result=sealed_decisions(rows,.7)
    assert result[0]['is_disrupt'] and result[0]['alarm']
    assert not result[0]['timely'] and result[0]['lead_ms']==30.
