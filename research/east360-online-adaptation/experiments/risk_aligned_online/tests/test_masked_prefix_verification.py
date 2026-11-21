from copy import deepcopy

import numpy as np
import pytest

from experiments.risk_aligned_online.verify_masked_prefix import verify_audit, early_alarm_floor


def receipt():
    rows = [dict(update_index=t, feedback_end_index=t, predictions_sealed_through_index=t,
                 effective_index=t+1, phase=i, hidden_targets=736, accepted=True,
                 fast_before={'l2': i*.05}, fast_after={'l2': (i+1)*.05},
                 loss_before=1., loss_after=.9, reason='same_observed_loss_decreased')
            for i, t in enumerate((152, 252))]
    return dict(intra=True, lag=0, score_count=300, attempts=rows, attempted_updates=2,
                accepted_updates=2, fast_initial={'l2': 0.}, fast_discarded=True,
                outcome_access=False, persistent=None, auxiliary_decoder_updated=False,
                model_parameter_change_norm=0., model_buffer_change_norm=0.)


def test_matched_clock_receipt_is_valid():
    verify_audit(receipt(), mode='signal_fresh')
    r = receipt(); r['lag'] = 25
    for row in r['attempts']: row['feedback_end_index'] -= 25
    verify_audit(r, mode='risk_stale')


@pytest.mark.parametrize('mutation', ['same_time', 'future_input', 'wrong_phase', 'no_improvement',
    'nonzero_start', 'label_access', 'decoder_change', 'registered_change', 'too_large',
    'missing_tick', 'persistent', 'incorrect_count'])
def test_inconsistent_update_evidence_is_rejected(mutation):
    r = deepcopy(receipt())
    if mutation == 'same_time': r['attempts'][0]['effective_index'] = 152
    elif mutation == 'future_input': r['attempts'][0]['feedback_end_index'] = 153
    elif mutation == 'wrong_phase': r['attempts'][0]['phase'] = 1
    elif mutation == 'no_improvement': r['attempts'][0]['loss_after'] = 1.
    elif mutation == 'nonzero_start': r['fast_initial']['l2'] = .1
    elif mutation == 'label_access': r['outcome_access'] = True
    elif mutation == 'decoder_change': r['auxiliary_decoder_updated'] = True
    elif mutation == 'registered_change': r['model_parameter_change_norm'] = .1
    elif mutation == 'too_large': r['attempts'][0]['fast_after']['l2'] = 2.
    elif mutation == 'missing_tick': r['attempts'].pop()
    elif mutation == 'persistent': r['persistent'] = {'accepted': True}
    elif mutation == 'incorrect_count': r['accepted_updates'] = 0
    with pytest.raises(AssertionError): verify_audit(r, mode='signal_fresh')


def test_first_alarm_before_update_cannot_be_removed_by_later_good_predictions():
    def row(sid, scores, disruptive=False):
        return dict(shot=sid, scores=np.array(scores), valid=np.ones(len(scores), bool),
                    times=np.arange(len(scores))/1000., is_disrupt=disruptive)
    rows = [row(1, [.1, .8, .1]), row(2, [.1, .1, .8]), row(3, [.8, .1, .1], True)]
    result = early_alarm_floor(rows, .7, last_unadapted_index=1)
    assert result['safe_count'] == 2 and result['locked_false_alarms'] == 1
    assert result['fpr_lower_bound'] == .5
    assert result['impossible_at_fpr_20_under_this_timing']
    rows[0]['valid'][1] = False
    assert early_alarm_floor(rows, .7, last_unadapted_index=1)['locked_false_alarms'] == 0
