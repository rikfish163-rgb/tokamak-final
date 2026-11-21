import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.full_path_meta_20261102.neural_replay import (
    reference_state,
)


@pytest.mark.parametrize(
    "query,index",
    [(0, None), (159, None), (160, 0), (287, 0), (288, 1), (415, 1), (416, 2)],
)
def test_borrowed_reference_state_uses_last_update_strictly_before_query(query, index):
    after = np.arange(3 * 4 * 64, dtype=np.float32).reshape(3, 4, 64)
    state = reference_state(
        "truncated_metric", query, {"innovation_metric_after": after}, "cpu"
    )
    expected = np.zeros((4, 64), np.float32) if index is None else after[index]
    np.testing.assert_array_equal(state.numpy(), expected)


def test_cold_never_uses_nonzero_archived_adaptation_state():
    after = np.ones((3, 4, 64), np.float32)
    assert torch.equal(
        reference_state("cold", 416, {"innovation_metric_after": after}, "cpu"),
        torch.zeros(4, 64),
    )


def test_missing_or_invalid_reference_trace_does_not_become_zero_state():
    with pytest.raises(KeyError):
        reference_state("truncated_metric", 416, {}, "cpu")
    with pytest.raises(ValueError):
        reference_state(
            "truncated_metric",
            416,
            {"innovation_metric_after": np.zeros((2, 4, 64), np.float32)},
            "cpu",
        )
    with pytest.raises(ValueError):
        reference_state("unknown", 416, {}, "cpu")
