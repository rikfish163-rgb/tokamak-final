"""State-conditioned, positive diagnostic-loss weights from matured signals."""
import torch
from torch import nn
from torch.nn import functional as F


class StateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(69, 32), nn.Tanh(), nn.Linear(32, 23))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, model, prediction, truth, available):
        mask = available & torch.isfinite(truth)
        truth = torch.where(mask, truth.detach(), model.center)
        target = torch.asinh((truth-model.center)/model.scale)
        residual = (prediction-target).masked_fill(~mask, 0.)
        count = mask.sum(1).clamp_min(1)
        # Context is observed data, not an extra differentiable escape route
        # through fast parameters. Outer gradients still reach the rule net.
        context = torch.cat(((residual.sum(1)/count).tanh(),
                             (residual.abs().sum(1)/count).log1p(),
                             mask.float().mean(1)), -1).detach()
        weights = 23*self.net(context).softmax(-1)[:, None, :]
        losses = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~mask, 0.)
        return (losses*weights).sum((1, 2))/(mask*weights).sum((1, 2)).clamp_min(1e-12)
