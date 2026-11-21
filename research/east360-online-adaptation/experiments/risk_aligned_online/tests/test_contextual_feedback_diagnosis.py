from experiments.risk_aligned_online.contextual_feedback_20261023.diagnose import acceptance


def test_no_disruptive_shots_cannot_establish_a_J_gain():
    arms = ('static', 'uniform', 'fixed', 'online', 'hold', 'mean')
    metrics = {a+'/calibrated': dict(tpr=None, fpr=0., tp=0) for a in arms}
    conditions = acceptance(metrics)
    assert conditions['better_J_than_uniform'] is None
    assert not all(v is True for v in conditions.values())


def test_equal_scores_do_not_pass_strict_gain_acceptance():
    arms = ('static', 'uniform', 'fixed', 'online', 'hold', 'mean')
    metrics = {a+'/calibrated': dict(tpr=.8, fpr=.1, tp=144) for a in arms}
    conditions = acceptance(metrics)
    assert conditions['fpr_at_most_point2'] is True
    assert conditions['better_J_than_hold'] is False
