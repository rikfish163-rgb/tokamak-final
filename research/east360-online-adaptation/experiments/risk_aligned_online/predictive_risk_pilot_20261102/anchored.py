"""Causal initial condition for a direct, non-autoregressive diagnostic path."""

import torch


class AnchoredHead(torch.nn.Module):
    """Keep the predicted temporal differences, start at the measured state.

    This is a one-sample initial-condition approximation, not a positivity
    projection or a learned physical equation. The original decoder is not
    changed. In particular this wrapper must not receive a later query's
    measurement when computing the gradient of an earlier matured forecast.
    """

    def __init__(self, head, last):
        super().__init__()
        if last.shape != (23,) or not torch.isfinite(last).all():
            raise ValueError("one finite observed23 state in source asinh coordinates")
        self.head = head
        self.register_buffer("last", last.detach().clone())

    def forward(self, x):
        predicted = self.head(x).reshape(len(x), 25, 23)
        result = self.last[None, None] + predicted - predicted[:, :1]
        return result.reshape(len(x), 575)


class PersistenceHead(torch.nn.Module):
    """Explicit no-dynamics baseline; no decoder training or adaptation."""

    def __init__(self, last):
        super().__init__()
        if last.shape != (23,) or not torch.isfinite(last).all():
            raise ValueError("one finite observed23 state required")
        self.register_buffer("last", last.detach().clone())

    def forward(self, x):
        return self.last[None, None].expand(len(x), 25, 23).reshape(len(x), 575)
