import numpy as np

from experiments.risk_aligned_online.simple_train import metrics, select_threshold, validation_rank


def row(shot, disrupted, score, time=.08):
    return dict(shot=shot, is_disrupt=disrupted, down=.1 if disrupted else None,
                times=np.array([time]), scores=np.array([score]))


def test_strict_ten_ms_boundary_and_safe_false_alarm():
    result = metrics([row(1, True, .9, .09), row(2, True, .9, .089),
                      row(3, False, .8)], .5)
    assert result['tp'] == 1 and result['fp'] == 1
    assert result['predictions'][0]['timely'] is False


def test_selects_source_validation_cap_without_test_labels():
    rows = [row(1, True, .9), row(2, True, .8), row(3, False, .7), row(4, False, .2)]
    selected = select_threshold(rows, .2)
    assert selected['tpr'] == 1 and selected['fpr'] == 0
    assert selected['threshold'] == .8


def test_epoch_rank_prioritizes_sensitivity_when_not_qualified():
    assert validation_rank(dict(tpr=.85, fpr=.1)) < validation_rank(dict(tpr=.7, fpr=0))
    assert validation_rank(dict(tpr=.9, fpr=.1)) < validation_rank(dict(tpr=.85, fpr=0))
