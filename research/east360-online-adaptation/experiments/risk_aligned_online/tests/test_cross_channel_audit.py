import json
import pytest

from experiments.risk_aligned_online.cross_channel_audit import update_summary


def test_selected_loss_decrease_does_not_imply_other_views_improve():
    row=dict(accepted=True,views_before=[1.,1.,1.,1.],views_after=[.9,1.2,1.1,1.],
             scale=.5,actual_encoding_rms=.01,proposed_encoding_rms=.02,limited=True)
    shot=dict(attempts=[row],fast_final=dict(l2=.99))
    result=update_summary([shot],cap=.01)
    assert result['accepted']==1 and result['accepted_some_view_worsens']==1
    assert result['accepted_mean_view_loss_increases']==1 and result['limited_attempts']==1
    assert result['final_radius_ge095']==1
    json.dumps(result,allow_nan=False)
    with pytest.raises(AssertionError):update_summary([shot],cap=.001)


def test_static_empty_updates_are_valid_and_not_summarized_as_success():
    result=update_summary([dict(attempts=[],fast_final=dict(l2=0.))],cap=None)
    assert result['accepted']==0 and result['attempted']==0
    assert result['encoding_step_quantiles'] is None
