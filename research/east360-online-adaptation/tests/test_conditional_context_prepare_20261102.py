from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.conditional_channel_policy_20261102.prepare import (
    source_context_features,
)


def test_context_uses_only_prefix_and_does_not_require_outcome_or_event_truth():
    rng = np.random.default_rng(131)
    signal = dict(
        values=rng.normal(size=(220, 23)).astype(np.float32),
        available=np.ones((220, 23), bool),
    )
    fitted = dict(
        mean=[0.0] * 115,
        scale=[1.0] * 115,
        weight=rng.normal(size=(6, 115)).tolist(),
        bias=[0.0] * 6,
    )
    heads = dict(true=fitted, shuffled=deepcopy(fitted))
    before = source_context_features(
        signal, 150, torch.zeros(23), torch.ones(23), heads
    )
    signal["values"][151:] = 1e9
    after = source_context_features(signal, 150, torch.zeros(23), torch.ones(23), heads)
    assert before == after
    assert sum(before["true"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        source_context_features(
            dict(signal, outcome=1), 150, torch.zeros(23), torch.ones(23), heads
        )
