"""Source mechanism test: does an explicitly observed change disambiguate keys?

These are changes per 8-sample patch, not physical derivatives. No phase label,
outcome, future signal, or predicted signal enters a key. Both arms have the
same nominal parameter count; only the trajectory arm uses the extra columns.
"""
import torch
from torch import nn

from ..context_memory_20261028.model import ContextAdapter
from ..context_memory_20261028.scaled import ScaledContextMemory


class TrajectoryMemory(ScaledContextMemory):
    def __init__(self, dim, weights, *, trajectory):
        if not isinstance(trajectory, bool):
            raise ValueError('explicit boolean trajectory arm required')
        super().__init__(dim, weights)
        old = self.key
        self.key = nn.Linear(92, 8)
        with torch.no_grad():
            self.key.weight.zero_()
            self.key.weight[:, :46].copy_(old.weight)
            self.key.bias.copy_(old.bias)
        self.trajectory = trajectory

    def key_features(self, clean, complete):
        level = torch.cat((clean, complete.to(clean)), -1)
        changes = torch.zeros_like(clean)
        pairs = torch.zeros_like(complete)
        if self.trajectory and len(clean) > 1:
            pairs[1:] = complete[1:] & complete[:-1]
            changes[1:] = torch.where(pairs[1:], clean[1:] - clean[:-1], 0.)
        return torch.cat((level, changes, pairs.to(clean)), -1)


class TrajectoryAdapter(ContextAdapter):
    def __init__(self, base, weights, *, trajectory):
        super().__init__(base)
        self.memory = TrajectoryMemory(base.cfg.dim, weights, trajectory=trajectory)
        self.slow.requires_grad_(False)
        self.memory.requires_grad_(True)
