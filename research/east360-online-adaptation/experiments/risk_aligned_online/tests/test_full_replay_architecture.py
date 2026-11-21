"""Full-model replay must not require Transformer-specific layer names."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from experiments.risk_aligned_online.replay_retraining import ReplayRetrainer
from experiments.risk_aligned_online.tests.test_ended_adaptation import example


class NonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=8)
        self.encoder = nn.Sequential(nn.Linear(23, 8), nn.Tanh())
        self.risk_head = nn.Linear(8, 1)
        self.forecast_head = nn.Linear(8, 23)
        self.fast_origin = nn.Parameter(torch.zeros(16))

    def forward(self, x, mask):
        h = self.encoder(x.masked_fill(~mask, 0.).mean(1))
        return {'logit': self.risk_head(h).squeeze(-1)}


def test_full_replay_works_without_transformer_layers_and_keeps_forecast_frozen():
    torch.manual_seed(31)
    model = NonTransformer()
    before = deepcopy(model.state_dict())
    learner = ReplayRetrainer(model, mode='continue', scope='full', steps=2, batch_size=8)
    result = learner(model, example())
    assert result['optimizer_steps'] == 2
    assert result['trainable_scope'] == 'full'
    assert not torch.equal(model.encoder[0].weight, before['encoder.0.weight'])
    assert torch.equal(model.forecast_head.weight, before['forecast_head.weight'])
    assert torch.equal(model.forecast_head.bias, before['forecast_head.bias'])
