import numpy as np
import torch

from experiments.risk_aligned_online.qk_forecast_stream_20261029.prequential import (
    channel_losses,
)
from experiments.risk_aligned_online.tests.test_qk_matched_head import fixture


def test_channel_recount_matches_masked_training_loss_and_ignores_missing():
    net, _ = fixture()
    truth = torch.ones(1, 25, 23)
    pred = torch.zeros_like(truth)
    mask = torch.ones_like(truth, dtype=torch.bool)
    mask[:, :, 10] = False
    mask[:, :2, 0] = False
    truth[:, :2, 0] = float("nan")
    sums, counts = channel_losses(net.base, pred, truth, mask)
    actual = sum(sums) / sum(counts)
    expected = float(net.base.forecast_loss(pred, truth, mask))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert counts[10] == 0 and counts[0] == 23
    blank = torch.zeros_like(mask)
    sums, counts = channel_losses(net.base, pred, truth, blank)
    assert not any(sums) and not any(counts)
