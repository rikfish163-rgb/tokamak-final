import numpy as np
import pytest

from run_paired_diagnostic_update_20261106 import commit_trace, train_counts


def test_train_coverage_uses_distinct_shots_per_role():
    q = [{"shot": 1, "counts": [2, 0, 1]}, {"shot": 2, "counts": [0, 99, 0]}]
    counts = train_counts(q)
    assert counts["shots"] == [1, 1, 1]
    assert counts["windows"] == [2, 99, 1]
    assert not counts["passes"]


def test_committed_trace_preserves_native_early_and_sets_only_explicit_rows():
    z = np.linspace(0, 1, 90).astype(np.float32)
    use = np.array([64, 67], dtype=np.int64)
    later = np.ones((2, 3, 4), np.float32) * -1
    score = np.ones((2, 3), np.float32) * -2
    out = commit_trace(z, use, later, score)
    assert out.shape == (90, 3, 6)
    assert np.array_equal(out[:, 0, 0], z)
    assert (out[use, :, 1:5] == -1).all() and (out[use, :, 5] == -2).all()
    assert np.array_equal(out[:64, 2, 3], z[:64])
    with pytest.raises(ValueError, match="causal"):
        commit_trace(z, np.array([63, 67]), later, score)
