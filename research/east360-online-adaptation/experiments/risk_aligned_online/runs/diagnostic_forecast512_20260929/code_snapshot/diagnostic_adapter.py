"""Diagnostic-coordinate adaptation between frozen Transformer blocks.

Each of six diagnostics has an endpoint-change coefficient and an in-window
curvature coefficient. This local polynomial is NOT a plasma equation. A
right-inverse decoder makes a fast coordinate change equal the decoded change
of the internal CLS representation. There is no added risk-score head.
"""
import torch
from torch import nn
from torch.nn import functional as F


def bounded_step(current, gradient, *, step=.1, radius=2.):
    if current.shape != gradient.shape or current.ndim != 2 or step <= 0 or radius <= 0:
        raise ValueError('matching batched state/gradient and positive bounds required')
    norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    updated = current-step*gradient*(1./norm).clamp(max=1.)
    return updated*(radius/updated.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)


class DiagnosticAdapter(nn.Module):
    def __init__(self, dim=64, horizon=25):
        super().__init__()
        if dim < 12 or horizon < 2:
            raise ValueError('12 physical-proxy coordinates and >=2 forecast points required')
        self.dim, self.horizon = dim, horizon
        self.basis = nn.Parameter(torch.randn(dim, 12)*.02)
        self.gain_raw = nn.Parameter(torch.zeros(12))
        self.bias = nn.Parameter(torch.zeros(12))
        time = torch.arange(1, horizon+1).float()/horizon
        self.register_buffer('time_basis', torch.stack((time, time*(1-time)), -1))

    def coordinate_map(self):
        q, _ = torch.linalg.qr(self.basis, mode='reduced')
        gain = F.softplus(self.gain_raw)+.1
        return q, gain

    def decode(self, internal):
        q, gain = self.coordinate_map()
        return (internal@q)*gain+self.bias

    def initial_fast(self, batch):
        return self.bias.new_zeros(batch, 12)

    def forward(self, parent, values, available, fast=None):
        if parent.cfg.dim != self.dim or parent.cfg.horizon != self.horizon or parent.cfg.layers < 2:
            raise ValueError('matching two-or-more-block temporal parent required')
        if parent.cfg.fast_site != 'tokens':
            raise ValueError('tested double-backward math-attention parent required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if fast.shape != (len(values), 12):
            raise ValueError('one 12-coordinate temporary state per episode required')
        # PyTorch disables its fused encoder fast path when hooks are present.
        # Its rounding differs from the unhooked frozen parent. For a zero
        # non-differentiable state, return the actual parent's original logit
        # (not an approximation) while separately decoding its hidden state.
        # A differentiable zero state still needs the hooked path so the
        # gradient of a possible internal move remains available.
        fresh = not fast.requires_grad and not bool(fast.detach().count_nonzero())
        original = parent(values, available)['logit'] if fresh else None
        q, gain = self.coordinate_map()
        captured = {}

        def insert(module, args, output):
            before = output[:, -1]
            change = (fast/gain)@q.T
            # Preserve all signal tokens; move the internal CLS state before
            # it attends to those tokens in the following Transformer block.
            after = before+change
            captured.update(internal_before=before, internal_after=after,
                            coefficients=(before@q)*gain+self.bias+fast)
            return torch.cat((output[:, :-1], after[:, None]), 1)

        handle = parent.encoder.layers[0].register_forward_hook(insert)
        try:
            out = parent(values, available)
        finally:
            handle.remove()
        coefficients = captured['coefficients'].reshape(len(values), 6, 2)
        trajectory = torch.einsum('bck,tk->btc', coefficients, self.time_basis)
        return dict(logit=original if fresh else out['logit'], trajectory=trajectory, **captured)
