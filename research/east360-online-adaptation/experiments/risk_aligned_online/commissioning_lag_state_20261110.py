"""Delayed test-time learning of cross-diagnostic temporal contrast."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .commissioning_transition_net_20261110 import HORIZON, RANK, STRIDE, WIDTH, move


MODES = ('fixed', 'real', 'shifted')


class LagState(nn.Module):
    def __init__(self, inputs, edges):
        super().__init__()
        self.stem = nn.Linear(inputs, WIDTH)
        self.key = nn.Linear(inputs, RANK)
        self.decoder = nn.Linear(WIDTH, edges)
        self.risk = nn.Linear(WIDTH, 1)

    def initial(self, batch):
        return self.stem.weight.new_zeros((batch, WIDTH, RANK))

    def design(self, x):
        return self.stem(x), self.key(x).tanh() / np.sqrt(RANK)

    @staticmethod
    def project(pre, key, state):
        return (pre + (state @ key[..., None]).squeeze(-1)).tanh()

    def update(self, pre, key, state, target, eligible):
        h = self.project(pre, key, state)
        margin = (self.decoder(h)*target).sum(-1)
        direction = ((target @ self.decoder.weight) * (1.-h.square())
                     * -torch.sigmoid(-margin)[..., None])
        gradient = direction[..., None] * key[..., None, :]
        return move(state, gradient, eligible), F.softplus(-margin) * eligible


@torch.no_grad()
def replay(net, x, origins, aligned, shifted, eligible):
    """Use a complete q+1..q+25 block no earlier than q+26."""
    x = torch.from_numpy(x)
    pre, key = net.design(x)
    fixed = pre.tanh()
    h = {k: fixed.numpy().copy() for k in MODES}
    state = net.initial(2)
    target = torch.from_numpy(np.stack((aligned, shifted), axis=1))
    present = torch.ones(2, dtype=torch.bool)
    correct = torch.zeros(3, dtype=torch.int64)
    loss = torch.zeros(3, dtype=torch.float64)
    changes = torch.zeros(2, dtype=torch.int64)
    n = 0
    for j, q in enumerate(origins):
        if q + HORIZON >= len(x):
            break
        if eligible[j]:
            a, k = pre[q].expand(2, -1), key[q].expand(2, -1)
            pred = torch.cat((fixed[q:q+1], net.project(a, k, state)))
            score = net.decoder(pred) @ target[j, 0]
            correct += (score > 0)
            loss += F.softplus(-score)
            nxt, _ = net.update(a, k, state, target[j], present)
            changes += (nxt != state).any((-1,-2))
            state = nxt
            n += 1
        first = int(q+HORIZON+1)
        stop = min(first+STRIDE, len(x))
        for k, a in zip(MODES[1:], state, strict=True):
            h[k][first:stop] = net.project(pre[first:stop], key[first:stop], a).numpy()
    return h, dict(observed_blocks=n, updates=dict(zip(MODES[1:], changes.tolist(), strict=True)),
                   true_order_loss_sum=dict(zip(MODES, loss.tolist(), strict=True)),
                   true_order_wins=dict(zip(MODES, correct.tolist(), strict=True)))
