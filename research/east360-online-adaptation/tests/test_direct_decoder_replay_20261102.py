import pytest
import torch

from experiments.risk_aligned_online.direct_decoder_meta_20261102 import replay


def test_hook_comparison_matches_encoding_batch_not_unrelated_cached_batch(monkeypatch):
    monkeypatch.setattr(replay, "encode", lambda *a: dict(h=torch.tensor([1.5])))
    monkeypatch.setattr(replay, "forward", lambda *a: dict(logit=a[1]["h"]))
    monkeypatch.setattr(replay, "direct", lambda *a: dict(logit=torch.tensor([1.5])))
    value = replay.matching_batch_hook(
        None, {}, 4, None, None, torch.tensor([1.500003])
    )
    assert value["same_batch_ffn_error"] == 0.0
    assert value["encoding_batch_difference"] > 2e-6
    monkeypatch.setattr(
        replay, "direct", lambda *a: dict(logit=torch.tensor([1.500001]))
    )
    with pytest.raises(AssertionError):
        replay.matching_batch_hook(None, {}, 4, None, None, torch.tensor([1.5]))
