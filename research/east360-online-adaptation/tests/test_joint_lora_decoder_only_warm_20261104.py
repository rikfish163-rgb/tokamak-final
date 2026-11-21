import torch
from torch import nn

from joint_lora_decoder_only_warm_20261104 import fit_decoder


class HeadOnlyFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_decoder = nn.Linear(192, 168)
        self.risk = nn.Linear(192, 1)
        self.lora = nn.Parameter(torch.ones(512))
        nn.init.zeros_(self.residual_decoder.weight)
        nn.init.zeros_(self.residual_decoder.bias)


def test_fit_moves_only_decoder_and_leaves_inactive_targets_out():
    model = HeadOnlyFixture()
    h = torch.ones(3, 192)
    prior = torch.zeros(3, 168)
    target = torch.ones_like(prior)
    observed = torch.zeros_like(target, dtype=torch.bool)
    observed[:, :10] = True
    target[~observed] = float("nan")
    risk_before = model.risk(h).detach().clone()
    lora_before = model.lora.detach().clone()
    history = fit_decoder(
        model,
        h,
        prior,
        target,
        observed,
        [{"step": i, "indices": [0, 1, 2]} for i in range(1, 5)],
    )
    assert len(history) == 4
    assert history[-1]["mse"] < history[0]["mse"]
    assert torch.equal(model.risk(h), risk_before)
    assert torch.equal(model.lora, lora_before)
    assert not any(p.requires_grad for p in model.parameters())
    assert model.residual_decoder.weight[:10].count_nonzero()
    assert model.residual_decoder.weight[10:].count_nonzero() == 0
