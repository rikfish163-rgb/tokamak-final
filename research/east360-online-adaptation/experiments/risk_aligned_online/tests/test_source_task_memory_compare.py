import copy

import pytest

from experiments.risk_aligned_online.source_task_memory_20261026.compare import validate_pair


def protocols():
    common = dict(threshold=.7, readout='full', source_sha256='base',
                  code_sha256={'run.py': 'abc'}, target_loss='disruptive_positive_only')
    fixed = dict(common, source_memory=dict(kind='fixed', epoch=2,
        source_protocol_sha256='training', path='fixed.pt', sha256='fixed-weights'))
    learned = dict(common, source_memory=dict(kind='learned', epoch=2,
        source_protocol_sha256='training', path='learned.pt', sha256='learned-weights'))
    return fixed, learned


def test_source_task_pair_rejects_other_experiment_changes():
    fixed, learned = protocols()
    validate_pair(fixed, learned)
    for field, value in [('threshold', .8), ('readout', 'shape'),
                         ('code_sha256', {'run.py': 'changed'})]:
        changed = dict(learned, **{field: value})
        with pytest.raises(ValueError, match=field):
            validate_pair(fixed, changed)


def test_source_task_pair_requires_same_training_and_final_epoch():
    fixed, learned = protocols()
    for field, value in [('kind', 'fixed'), ('epoch', 1),
                         ('source_protocol_sha256', 'other-training')]:
        changed = copy.deepcopy(learned)
        changed['source_memory'][field] = value
        with pytest.raises(ValueError, match='source memory'):
            validate_pair(fixed, changed)
