import copy
import pytest

from experiments.risk_aligned_online.source_task_memory_20261026.audit import verify_training


def test_training_audit_requires_coverage_and_actual_rule_gradients():
    rows = []
    for epoch in (1, 2):
        arms = dict(fixed=dict(gradients={'key.weight': None, 'prior': None, 'rate': None,
                                        'output.weight': .1}),
                    learned=dict(gradients={'key.weight': .1, 'prior': .2, 'rate': .01,
                                          'output.weight': .1}))
        rows.append(dict(epoch=epoch, ids=[1, 2], shots_seen=2, arms=arms))
    assert verify_training(rows, [1, 2])['complete_epochs'] == 2
    bad = copy.deepcopy(rows); bad[-1]['ids'] = [1, 1]
    with pytest.raises(ValueError, match='coverage'):
        verify_training(bad, [1, 2])
    bad = copy.deepcopy(rows); bad[0]['arms']['fixed']['gradients']['rate'] = .1
    with pytest.raises(ValueError, match='fixed'):
        verify_training(bad, [1, 2])
    bad = copy.deepcopy(rows)
    for row in bad:
        row['arms']['learned']['gradients']['key.weight'] = 0.
    with pytest.raises(ValueError, match='gradient'):
        verify_training(bad, [1, 2])
