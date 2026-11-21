import numpy as np
import torch
from experiments.risk_aligned_online.state_loss import StateLoss
from experiments.risk_aligned_online.sequence_fast import sequence, unroll
from experiments.risk_aligned_online.simple_online import predict_shot
from experiments.risk_aligned_online.tests.test_sequence_fast import fixture


def test_initial_rule_equals_uniform_and_ignores_unavailable():
    model, _ = fixture()
    rule = StateLoss()
    p = torch.randn(2, 25, 23, requires_grad=True)
    truth = torch.randn_like(p)
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[:, :, 2] = False
    a = rule(model, p, truth, mask)
    expected = model.forecast_loss(p, truth, mask)
    torch.testing.assert_close(a.mean(), expected)
    truth[:, :, 2] = float('nan')
    torch.testing.assert_close(a, rule(model, p, truth, mask))


def test_state_rule_meta_gradient_and_online_equivalence():
    model, shot = fixture()
    model.requires_grad_(False)
    rule = StateLoss()
    seq = sequence(shot, 0, 3, model.cfg, 'cpu')
    result = unroll(model, seq, create_graph=True, loss_rule=rule)
    result['logits'][-1].sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in rule.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.parameters())
    with torch.no_grad():
        rule.net[-1].weight.normal_(std=.1)
    result = unroll(model, seq, create_graph=False, loss_rule=rule)
    scores, _ = predict_shot(model, shot, intra=True, device='cpu', loss_rule=rule)
    np.testing.assert_allclose(result['logits'].detach().sigmoid().numpy()[:, 0],
                               scores[[26, 52, 78]], atol=1e-6)
    for e in seq:
        e['y'] = 1-e['y']
    repeat = unroll(model, seq, create_graph=False, loss_rule=rule)
    torch.testing.assert_close(result['logits'], repeat['logits'], rtol=0, atol=0)
