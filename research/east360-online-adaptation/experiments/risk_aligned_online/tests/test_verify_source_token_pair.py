from copy import deepcopy
import pytest
from experiments.risk_aligned_online.verify_mnn_aux_ablation import verify_token_protocol_pair


def protocols():
    p=dict(signal_contract='all23',input_feature_count=23,auxiliary_excluded_features=[],
        normalization='density_robust',normalization_channels=['ne_nG'],source_moments_file_sha256=None,
        temporary_site='source_tokens',temporary_count=128,temporary_is_offset_from_parent_origin=True,
        persistent_site='not_present',persistent_enabled=False,alarm_tuning=False,
        auxiliary_postseal_feedback_in_all_arms=True,auxiliary_head_frozen_after30=True,
        parent_sha256='source',threshold=.7,order_sha256='order',input_mapping='same',seed=131,
        first30_steps=4,bootstrap_steps=40,intra_step=.05,intra_radius=1.,stride=100,horizon=25)
    return {a:dict(deepcopy(p),arm=a,intra=a=='intra_control') for a in ('mapped_static','intra_control')}


def test_token_site_is_a_separate_same_input_and_loss_pair():
    verify_token_protocol_pair(protocols())


@pytest.mark.parametrize('key,value',[('temporary_site','final_ffn'),('temporary_count',16576),
    ('temporary_is_offset_from_parent_origin',False),('normalization','efd_density_zscore'),
    ('source_moments_file_sha256','other'),('signal_contract','without_mnn'),
    ('persistent_enabled',True),('threshold',.8),('intra_step',.1)])
def test_wrong_origin_geometry_input_or_memory_is_rejected(key,value):
    p=protocols();p['intra_control'][key]=value
    with pytest.raises(ValueError):verify_token_protocol_pair(p)
