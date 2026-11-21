import numpy as np
import torch

from experiments.risk_aligned_online.event_adapt_20261022.model import Config, EventForecastModel
from experiments.risk_aligned_online.event_adapt_20261022.gradients import assign_task_gradients


def model():
    torch.manual_seed(15)
    torch.set_num_threads(2)
    return EventForecastModel(Config(dim=8, rank=2, horizon=4, dilations=(1,)),
                              np.zeros(23), np.ones(23), np.ones(23), np.ones(23, bool))


def test_forecast_head_cannot_be_optimized_as_a_risk_code():
    m = model(); w = m.forecast_head[-1].weight
    risk = 100*w.sum()+m.risk_head[-1].weight.sum()
    forecast = w.square().sum()
    event = m.event_head.weight.sum()
    assign_task_gradients(m, [risk, forecast, event], [1., .2, .1], normalize_shared=False)
    torch.testing.assert_close(w.grad, .4*w)
    torch.testing.assert_close(m.risk_head[-1].weight.grad, torch.ones_like(m.risk_head[-1].weight))


def test_shared_balancing_removes_loss_scale_but_not_direction():
    m = model(); w = m.embedding.weight
    a = w[0].sum()*300
    b = w[1].sum()
    c = m.event_head.weight.sum()*0
    info = assign_task_gradients(m, [a, b, c], [1., .2, .1], normalize_shared=True)
    torch.testing.assert_close(w.grad[1].norm()/w.grad[0].norm(), torch.tensor(.2))
    assert w.grad[0].min() > 0 and w.grad[1].min() > 0
    assert info['risk_shared_norm'] > 200*info['forecast_shared_norm']


def test_zero_or_unused_task_does_not_generate_nan():
    m = model(); w = m.embedding.weight
    losses = [w.sum(), m.forecast_head[-1].weight.sum()*0, m.event_head.weight.sum()*0]
    assign_task_gradients(m, losses, [1., .2, .1], normalize_shared=True)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
