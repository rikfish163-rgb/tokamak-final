import inspect

import numpy as np
import pytest

from experiments.risk_aligned_online.lifecycle_stream_20261025 import learning, run


def shot():
    return dict(shot=7, times=np.arange(200)/1000, y=np.zeros(200),
                valid=np.ones(200, bool), is_disrupt=False, down=None)


def test_default_route_keeps_original_endpoint_sequence(monkeypatch):
    s = shot()
    scores = np.linspace(0, 1, 200)
    monkeypatch.setattr(learning, 'cache_example', lambda model, signal, ends, labels, **kw: ends)
    expected = learning.sampled_endpoints(s['times'], s['y'], s['valid'], scores, down=None)
    assert np.array_equal(learning.target_example(None, s, scores, {}, None), expected)
    assert np.array_equal(learning.target_example(None, s, scores, {}, None,
                                                  endpoint_policy='full_window'), expected)


def test_unknown_route_is_rejected_before_cache(monkeypatch):
    monkeypatch.setattr(learning, 'cache_example', lambda *a, **kw: pytest.fail('cached invalid policy'))
    with pytest.raises(ValueError, match='endpoint'):
        learning.target_example(None, shot(), np.ones(200), {}, None, endpoint_policy='unknown')


def test_prefix_route_retains_old_indices_and_adds_only_short_history(monkeypatch):
    s = shot()
    scores = np.linspace(0, 1, 200)
    monkeypatch.setattr(learning, 'cache_example', lambda model, signal, ends, labels, **kw: ends)
    old = learning.target_example(None, s, scores, {}, None)
    new = learning.target_example(None, s, scores, {}, None, endpoint_policy='prefix16')
    assert set(old) < set(new)
    added = np.setdiff1d(new, old)
    assert len(added) == 16 and added[0] == 0 and added[-1] == 126


def test_runner_exposes_explicit_opt_in_and_propagates_both_training_sites():
    signature = inspect.signature(run.main)
    assert signature.parameters['endpoint_policy'].default == 'full_window'
    source = inspect.getsource(run.main)
    assert source.count('endpoint_policy=endpoint_policy') >= 3  # protocol, cold, online
    assert 'short_history_20261031' in source
    assert 'cold_training_endpoints' in source and 'training_endpoints' in source


def test_evidence_digest_accepts_script_filename_strings():
    from pathlib import Path
    from SHORT_HISTORY_PRECHECK_20261031 import digest
    from SHORT_HISTORY_VERIFY_20261031 import sha
    for function in (digest, sha):
        assert function(__file__) == function(Path(__file__))
