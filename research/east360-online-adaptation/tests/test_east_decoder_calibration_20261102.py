import pytest
import torch

from experiments.risk_aligned_online.east_decoder_calibration_20261102.core import (
    LowRankDecoder,
    training_eligible,
)


def test_zero_rank_change_preserves_prior_and_only_delta_trains():
    prior = torch.nn.Linear(1088, 575)
    basis = torch.eye(1088)[:4]
    head = LowRankDecoder(prior, basis)
    x = torch.randn(2, 1088)
    assert torch.equal(head(x), prior(x))
    assert sum(p.numel() for p in head.parameters() if p.requires_grad) == 2300
    head(x).sum().backward()
    assert head.delta.grad is not None
    assert all(p.grad is None for p in head.prior.parameters())
    with torch.no_grad():
        head.delta.fill_(0.1)
    torch.testing.assert_close(head(x) - prior(x), ((x @ basis.T) / 2) @ head.delta.T)
    assert prior.weight.data_ptr() != head.prior.weight.data_ptr()


def test_fit_uses_east_positive_timely_window_not_disruptive_early_negatives():
    assert training_eligible(dict(disruptive=False, actual_lead_ms=None, label=0))
    assert training_eligible(dict(disruptive=True, actual_lead_ms=35.0, label=1))
    assert not training_eligible(dict(disruptive=True, actual_lead_ms=30.0, label=1))
    assert not training_eligible(dict(disruptive=True, actual_lead_ms=150.0, label=0))
    with pytest.raises(ValueError):
        training_eligible(dict(disruptive=True, actual_lead_ms=35.0, label=0))
