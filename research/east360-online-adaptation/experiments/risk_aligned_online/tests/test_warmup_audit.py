import pytest
from ..warmup_audit import first_effect, prefix_bound


@pytest.mark.parametrize('window,effect',[(1,6),(6,6),(26,26),(128,130),(768,780)])
def test_existing_clock_after_complete_window(window,effect):
    assert first_effect(window)==effect


def test_already_emitted_alarm_cannot_be_undone_but_effect_point_can_change():
    result=prefix_bound([None,0,5,6,20],6)
    assert result['irrevocable_fp']==2 and result['fpr_lower_bound']==.4


def test_invalid_window_rejected():
    with pytest.raises(ValueError): first_effect(0)
