import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.commissioning_latent_rate_20261110 import (
    initialise_readout, score_warning, supervised_rows,
)


def test_warning_loss_rewards_earlier_equal_peak_only_for_deadline_objective():
    scores = torch.tensor([[3., -2., -2., -3., -3.], [-2., -2., 3., -3., -3.]])
    early, late = (r.unsqueeze(0) for r in scores)
    groups = ([slice(0, 3)], [slice(3, 5)])
    old = [score_warning(z, *groups, 0., deadline=False) for z in (early, late)]
    new = [score_warning(z, *groups, 0., deadline=True) for z in (early, late)]
    torch.testing.assert_close(*old, rtol=0, atol=0)
    assert new[0] < new[1]


def test_late_points_are_not_timely_and_safe_penalty_applies_to_the_whole_shot():
    scores = torch.tensor([[-2., -1., 20., -3., -3.], [-2., -1., -20., -3., -3.]])
    groups = ([slice(0, 2)], [slice(3, 5)])
    z, reference = scores[:1].clone().requires_grad_(), scores[1:]
    loss = score_warning(z, *groups, 0., deadline=True)
    torch.testing.assert_close(loss, score_warning(reference, *groups, 0., deadline=True))
    loss.backward()
    assert z.grad[0, 2] == 0
    assert z.grad[0, 0] < 0
    bad = reference.clone()
    bad[0, 4] = 2
    assert score_warning(bad, *groups, 0., deadline=True) > loss


def test_supervised_rows_keep_125ms_hdf_label_and_strict_30ms_deadline():
    times = np.array([.874, .875, .969, .970, .999, 1., 1.001])
    data = np.zeros((2, len(times), 5), np.float32)
    x, y, timely, safe = supervised_rows(data, times, True, 1.)
    assert x.shape == (2, 6, 5)
    assert y.tolist() == [False, True, True, True, True, True]
    assert timely.tolist() == [False, True, True, False, False, False]
    assert not safe.any()
    _, y, timely, safe = supervised_rows(data, times, False, None)
    assert not y.any() and not timely.any() and safe.all()


def test_new_risk_readout_initializes_to_original_source_logit_with_any_drift():
    rng = torch.Generator().manual_seed(39)
    source = torch.nn.Linear(7, 1)
    x = torch.randn(50, 7 + 23 + 7, generator=rng)
    model = initialise_readout(source, x.shape[-1])
    torch.testing.assert_close(model(x), source(x[:, :7]), rtol=1e-6, atol=3e-7)
    model(x).square().mean().backward()
    assert source.weight.grad is None
    assert model[0].weight.grad[:, 7:].abs().max() > 0


def test_empty_timely_group_does_not_silently_fit_the_objective():
    with pytest.raises(ValueError, match="timely"):
        score_warning(torch.zeros(2, 5), [], [slice(3, 5)], 0., deadline=True)
