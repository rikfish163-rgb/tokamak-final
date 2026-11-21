from copy import deepcopy
import numpy as np
import pytest
from experiments.risk_aligned_online.verify_mnn_aux_ablation import verify_efd_protocol_pair,curve_error


def protocols():
    p=dict(signal_contract='all23',input_feature_count=23,auxiliary_excluded_features=[],
        normalization='efd_density_zscore',normalization_channels=['ne_nG'],source_moments_file_sha256='moments',
        mode_number_n_is_assumed_categorical=False,persistent_enabled=False,alarm_tuning=False,
        auxiliary_postseal_feedback_in_all_arms=True,auxiliary_head_frozen_after30=True,
        parent_sha256='source',threshold=.7,order_sha256='order',input_mapping='same',seed=131,
        first30_steps=4,bootstrap_steps=40,intra_step=.05,intra_radius=1.,stride=100,horizon=25)
    return {a:dict(deepcopy(p),arm=a,intra=a=='intra_control') for a in ('mapped_static','intra_control')}


def test_same_input_statistics_and_full_signal_loss_are_required():
    verify_efd_protocol_pair(protocols())


@pytest.mark.parametrize('key,value',[('signal_contract','without_mnn'),('normalization','density_robust'),
    ('normalization_channels',['ne_nG','ip']),('source_moments_file_sha256',None),
    ('source_moments_file_sha256','other'),('threshold',.8),('intra_step',.1),('persistent_enabled',True)])
def test_confounded_moments_alarm_loss_or_memory_is_rejected(key,value):
    p=protocols();p['intra_control'][key]=value
    with pytest.raises(ValueError):verify_efd_protocol_pair(p)


def test_curve_comparison_does_not_silently_zip_different_stream_lengths():
    a=[dict(times=np.array([0.]),scores=np.array([.1]))]
    with pytest.raises(ValueError):curve_error(a,[])
    assert curve_error(a,a)==0.
    b=[dict(times=np.array([0.]),scores=np.array([.2]))]
    assert curve_error(a,b)==pytest.approx(.1)
