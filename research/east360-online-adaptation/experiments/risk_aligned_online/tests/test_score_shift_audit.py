import numpy as np
import pytest
import json

from experiments.risk_aligned_online.score_shift_audit import summarize_pair, read_seals, sha


def test_disruptive_groups_exclude_post_event_and_keep_strict_lead():
    row = summarize_pair(np.array([800., 900., 969., 970., 999., 1000., 1001.]),
                         np.zeros(7), np.ones(7) * .8,
                         disruptive=True, down=1000., threshold=.7)
    assert row['groups']['early_disruptive']['points'] == 1
    assert row['groups']['precursor']['points'] == 5
    assert row['groups']['actionable_precursor']['points'] == 2
    assert row['groups']['precursor']['mean_probability_delta'] == pytest.approx(.8)
    assert row['baseline_timely'] is False
    assert row['candidate_timely'] is True


def test_exact_30ms_is_not_timely():
    row = summarize_pair(np.array([969., 970., 1000.]), np.zeros(3),
                         np.array([0., .9, .9]), disruptive=True,
                         down=1000., threshold=.7)
    assert row['candidate_alarm'] is True
    assert row['candidate_timely'] is False


def test_safe_shot_does_not_need_event_time():
    row = summarize_pair(np.array([0., 1.]), np.array([.1, .2]),
                         np.array([.2, .8]), disruptive=False,
                         down=None, threshold=.7)
    assert row['candidate_alarm'] is True
    assert row['baseline_alarm'] is False
    assert row['groups']['safe_shot']['mean_probability_delta'] == pytest.approx(.35)
    assert 'precursor' not in row['groups']


@pytest.mark.parametrize('times,scores', [([1., 0.], [0., .2]),
                                          ([0., 1.], [0., float('nan')]),
                                          ([0., 1.], [0., 1.1])])
def test_bad_arrays_rejected(times, scores):
    with pytest.raises(ValueError):
        summarize_pair(np.array(times), np.array(scores), np.zeros(2),
                       disruptive=False, down=None, threshold=.7)


def test_seal_tampering_and_identity_rejected(tmp_path):
    target = tmp_path / 'shot.npz'
    np.savez(target, shot=1, times=np.array([0., .001]), scores=np.array([.1, .2]))
    receipt = tmp_path / '0001_seal.json'
    payload = dict(shot=1, npz=target.name, threshold=.7, sha256=sha(target))
    receipt.write_text(json.dumps(payload))
    assert read_seals(tmp_path)[1]['scores'].tolist() == [.1, .2]
    np.savez(target, shot=1, times=np.array([0., .001]), scores=np.array([.1, .9]))
    with pytest.raises(ValueError, match='hash mismatch'):
        read_seals(tmp_path)
    payload.update(shot=2, sha256=sha(target))
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='identity mismatch'):
        read_seals(tmp_path)


def test_missing_intervals_are_not_averaged_as_zero():
    result = summarize_pair(np.array([1000.]), np.array([.2]), np.array([.3]),
                            disruptive=True, down=1000., threshold=.7)
    assert set(result['groups']) == {'precursor'}
