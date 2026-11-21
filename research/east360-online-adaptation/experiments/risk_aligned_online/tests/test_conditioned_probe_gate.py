from experiments.risk_aligned_online.context_memory_20261028.conditioned_run import gate


def summary():
    return {part: {'off': {'bag': 1., 'local': 1.},
                   'memory': {'bag': .8, 'local': .8},
                   'reverse': {'bag': .9, 'local': .9}}
            for part in ('all', 'D', 'S')}


def test_gate_rejects_safe_degradation_even_when_overall_loss_improves():
    candidate, control = summary(), summary()
    control['all']['memory'] = dict(bag=.85, local=.85)
    assert gate(candidate, control)['eligible_for_next_experiment']
    candidate['S']['memory']['bag'] = 1.01
    assert not gate(candidate, control)['eligible_for_next_experiment']


def test_gate_requires_true_feedback_to_beat_wrong_feedback_and_additive():
    candidate, control = summary(), summary()
    assert not gate(candidate, control)['eligible_for_next_experiment']
    control['all']['memory'] = dict(bag=.85, local=.85)
    candidate['all']['reverse']['local'] = .7
    assert not gate(candidate, control)['eligible_for_next_experiment']
