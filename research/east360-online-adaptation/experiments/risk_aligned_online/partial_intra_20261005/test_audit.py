import pytest
from .audit import compare_rows


def test_comparison_uses_shot_identity_not_order():
    old = [dict(shot=1, label=0, loss=.2, first_alarm=None), dict(shot=2, label=1, loss=.3, first_alarm=10)]
    new = [dict(shot=2, label=1, loss=.1, first_alarm=9), dict(shot=1, label=0, loss=.21, first_alarm=None)]
    out = compare_rows(old, new)
    assert out['classes']['0']['mean_loss_change'] == pytest.approx(.01)
    assert out['classes']['1']['mean_loss_change'] == pytest.approx(-.2)
    assert out['changed_first_alarms'] == 1


def test_comparison_rejects_roster_or_label_mismatch():
    row = dict(shot=1, label=0, loss=.2, first_alarm=None)
    with pytest.raises(ValueError): compare_rows([row, row], [row])
    with pytest.raises(ValueError): compare_rows([row], [dict(row, label=1)])
