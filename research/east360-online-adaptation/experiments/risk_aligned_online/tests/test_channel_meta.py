from copy import deepcopy

import torch
import numpy as np

from experiments.risk_aligned_online.channel_loss import weighted_loss
from experiments.risk_aligned_online.channel_meta import Weighted23
from experiments.risk_aligned_online.simple_online import predict_shot
from experiments.risk_aligned_online.sequence_fast import sequence, unroll
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture


def test_equal_weights_reproduce_original_loss():
    losses = torch.rand(2, 25, 23)
    mask = torch.rand(2, 25, 23) > .2
    actual = weighted_loss(losses, mask, torch.zeros(23))
    expected = losses.masked_fill(~mask, 0.).sum((1, 2))/mask.sum((1, 2))
    torch.testing.assert_close(actual, expected)
    assert weighted_loss(losses, torch.zeros_like(mask), torch.zeros(23)).sum() == 0


def test_outer_risk_trains_channel_weights_without_modifying_base():
    model, shot = fixture()
    before = deepcopy(model.state_dict())
    model.requires_grad_(False)
    logits = torch.zeros(23, requires_grad=True)
    result = unroll(model, sequence(shot, 0, 3, model.cfg, 'cpu'),
                    create_graph=True, channel_logits=logits)
    torch.nn.functional.softplus(result['logits'][-1]).mean().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all()
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())


def test_nonuniform_training_matches_causal_online_update():
    parent, shot = fixture()
    weights = torch.linspace(-2., 2., 23)
    model = Weighted23(dict(model_config=parent.cfg.to_dict(), source_center=parent.center,
                            source_scale=parent.scale, model=parent.state_dict()), weights).eval()
    seq = sequence(shot, 0, 3, model.cfg, 'cpu')
    predicted = unroll(model, seq, create_graph=False, channel_logits=weights)['logits']
    scores, _ = predict_shot(model, shot, intra=True, device='cpu')
    np.testing.assert_allclose(predicted.detach().sigmoid().numpy()[:, 0], scores[[26, 52, 78]], atol=1e-6)
    plain, _ = predict_shot(parent, shot, intra=True, device='cpu', channel_logits=weights)
    np.testing.assert_allclose(plain, scores, atol=1e-6)
