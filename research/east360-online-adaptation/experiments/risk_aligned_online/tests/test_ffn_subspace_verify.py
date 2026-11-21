from copy import deepcopy
import pytest

from experiments.risk_aligned_online.ffn_subspace_verify import validate_audit


def audit():
    return dict(scope='OFFLINE_SOURCE_DIAGNOSTIC_NOT_JTEXT_ONLINE', fast_parameter_count=8,
        score_count=60, fast_discarded=True, model_parameter_change_norm=0.,
        model_buffer_change_norm=0., attempted_updates=1, accepted_updates=1,
        attempts=[dict(origin_index=0, maturity_index=25, predictions_sealed_through_index=25,
            effective_index=26, accepted=True, fast_at_origin=[0.] * 8)])


def test_verifier_checks_matured_signal_only_timing_and_per_shot_reset():
    assert validate_audit(audit()) == dict(attempted=1, accepted=1)
    for change in [dict(scope='EAST_CAUSAL_REPLAY'), dict(fast_discarded=False),
                   dict(model_parameter_change_norm=.01)]:
        with pytest.raises(ValueError): validate_audit(dict(audit(), **change))
    for field, value in [('effective_index', 25), ('predictions_sealed_through_index', 24),
                         ('maturity_index', 24), ('fast_at_origin', [1.] * 8)]:
        bad = deepcopy(audit()); bad['attempts'][0][field] = value
        with pytest.raises(ValueError): validate_audit(bad)
