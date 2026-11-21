from copy import deepcopy

import pytest

from experiments.risk_aligned_online.progressive_feedback_20261026.compare import validate_pair


def test_pair_accepts_only_declared_feedback_protocol_changes():
    old = dict(source_sha256='source', threshold=.7, warmup=20,
               source_memory={'kind': 'learned', 'sha256': 'memory'},
               code_sha256={'experiments/risk_aligned_online/lifecycle_stream_20261025/model.py': 'old'})
    new = deepcopy(old)
    new.update(feedback='progressive', feedback_plan_sha256='a'*64,
               source_memory_training_feedback='delayed', source_deployment_feedback_match=False)
    new['code_sha256']['experiments/risk_aligned_online/lifecycle_stream_20261025/model.py'] = 'new'
    validate_pair(old, new)
    for key, value in [('threshold', .8), ('warmup', 10), ('source_sha256', 'other')]:
        bad = deepcopy(new); bad[key] = value
        with pytest.raises(ValueError): validate_pair(old, bad)
    bad = deepcopy(new); bad['source_memory']['sha256'] = 'different'
    with pytest.raises(ValueError): validate_pair(old, bad)
    bad = deepcopy(new); bad['code_sha256']['unrelated.py'] = 'unreviewed'
    with pytest.raises(ValueError): validate_pair(old, bad)
    bad = deepcopy(new); bad['source_deployment_feedback_match'] = True
    with pytest.raises(ValueError): validate_pair(old, bad)
