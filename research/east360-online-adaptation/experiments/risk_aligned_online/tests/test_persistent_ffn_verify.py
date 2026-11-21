import hashlib
import json

import numpy as np
import pytest

from experiments.risk_aligned_online.persistent_ffn_verify import check_timing, counts, site_contract


def audit():
    digest = hashlib.sha256(np.zeros(16576, np.float32).tobytes()).hexdigest()
    receipt = dict(count=16576, dtype='torch.float32', l2=0., sha256=digest)
    return dict(scope='EAST_CAUSAL_REPLAY', fast_parameter_count=16576,
                fast_discarded=True, model_parameter_change_norm=0.,
                model_buffer_change_norm=0., score_count=61,
                attempted_updates=1, accepted_updates=0,
                attempts=[dict(origin_index=0, maturity_index=25,
                    predictions_sealed_through_index=25, effective_index=None,
                    accepted=False, fast_at_origin=receipt, fast_before=receipt,
                    fast_after=receipt)])


def test_receipt_enforces_initial_zero_arrival_seal_reset_and_state_scope():
    a = audit()
    assert check_timing(a) == dict(attempted=1, accepted=0)
    a['attempts'][0]['maturity_index'] = 24
    with pytest.raises(ValueError, match='maturity'):
        check_timing(a)
    a = audit(); a['attempts'][0]['fast_at_origin']['sha256'] = 'a'*64
    with pytest.raises(ValueError, match='zero'):
        check_timing(a)
    a = audit(); a['model_parameter_change_norm'] = .1
    with pytest.raises(ValueError, match='scope'):
        check_timing(a)


def test_numpy_outcome_recount_is_finite_json_not_numpy_scalars():
    m = counts([dict(is_disrupt=np.bool_(True), timely=np.bool_(True),
                     alarm=True, lead_ms=np.float64(3001.)),
                dict(is_disrupt=np.bool_(False), timely=False,
                     alarm=True, lead_ms=None)])
    assert json.loads(json.dumps(m, allow_nan=False))['tp'] == 1


def test_routed_receipt_requires_its_explicit_dimension_and_exact_reset():
    a = audit()
    a['fast_parameter_count'] = 512
    digest = hashlib.sha256(np.zeros(512, np.float32).tobytes()).hexdigest()
    for key in ('fast_at_origin', 'fast_before', 'fast_after'):
        a['attempts'][0][key] = dict(count=512, dtype='torch.float32', l2=0., sha256=digest)
    assert check_timing(a, dimension=512) == dict(attempted=1, accepted=0)
    with pytest.raises(ValueError, match='scope'): check_timing(a)
    a['attempts'][0]['fast_after']['count'] = 511
    with pytest.raises(ValueError, match='bound'): check_timing(a, dimension=512)


def test_internal_weight_site_has_explicit_shape_and_artifact_contract():
    assert site_contract('routed_ffn_weight') == (65536,'final_routed_ffn_weight.pt',(8,64,128))
    assert site_contract('routed_ffn') == (512,'final_routed_ffn.pt',(8,64))
    assert site_contract('persistent_ffn') == (16576,'final_persistent_ffn.pt',(16576,))
    assert site_contract('gradient_transport') == (16576,'final_gradient_transport.pt',(16576,))
    with pytest.raises(ValueError,match='site'): site_contract('unknown')
