import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.simple23 import Config, Simple23
from joint_lora_neutral_stream_20261104 import (
    compare_shot,
    load_risk_neutral,
    require_unmoved_risk,
)


@pytest.fixture
def parent():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    config = Config(fast_site="tokens", fast_rule="normalized")
    base = (
        Simple23(config, torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)
    )
    w0 = torch.zeros(168, 70)
    adapted = JointLoRA(base, w0).eval().requires_grad_(False)
    with torch.no_grad():
        adapted.residual_decoder.weight.normal_(0, 0.05)
    return base, adapted


def test_warm_cannot_open_before_twenty_reveals(parent, tmp_path):
    base, _ = parent
    with pytest.raises(ValueError, match="first 20"):
        load_risk_neutral(base, tmp_path, list(range(20)), list(range(19)), 0.7)


def test_reject_nonzero_lora_even_if_state_hash_matches(parent):
    base, adapted = parent
    pristine = JointLoRA(base, adapted.W0)
    require_unmoved_risk(pristine, adapted)
    with torch.no_grad():
        # B tensors are zero before any shot-specific adaptation.
        b = next(p for p in adapted.neural_parameters() if not p.count_nonzero())
        b[0, 0] = 0.1
    with pytest.raises(ValueError, match="non-decoder"):
        require_unmoved_risk(pristine, adapted)


def test_same_parent_prefix_and_only_true_temporary_feedback(parent):
    base, adapted = parent
    rng = np.random.default_rng(17)
    signal = dict(
        values=rng.normal(size=(174, 23)).astype(np.float32),
        available=np.ones((174, 23), bool),
    )
    scores, updates, forecasts, gap = compare_shot(base, adapted, signal, device="cpu")
    assert gap < 3e-6
    assert np.array_equal(scores["intra"][:135], scores["static"][:135])
    assert not np.array_equal(scores["intra"][135:], scores["static"][135:])
    assert not np.array_equal(scores["intra"][135:], scores["hold"][135:])
    assert forecasts["intra__released_at"].tolist() == [135, 167]
    assert not any(row["updated"] for row in updates["static"])
    assert all(row["updated"] for row in updates["intra"])
    next_scores, *_ = compare_shot(base, adapted, signal, device="cpu")
    for key in scores:
        np.testing.assert_array_equal(scores[key], next_scores[key])
