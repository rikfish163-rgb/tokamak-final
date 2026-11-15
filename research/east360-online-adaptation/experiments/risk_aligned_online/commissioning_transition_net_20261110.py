"""Risk-taught neural state with masked, delayed direct transition feedback."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 24
RANK = 4
HORIZON = 25
STRIDE = 32
FIRST = 127
RETAIN = .85
STEP = .30
STABILIZER = .02
RADIUS = 1.5
MODES = ('fixed', 'real', 'reverse')


def move(state, gradient, has_feedback):
    norm = gradient.square().sum((-1, -2), keepdim=True).clamp_min(1e-16).sqrt()
    proposal = RETAIN * state - STEP * gradient / (norm + STABILIZER)
    length = proposal.square().sum((-1, -2), keepdim=True).clamp_min(1e-16).sqrt()
    projected = proposal / (length / RADIUS).clamp_min(1.)
    return torch.where(has_feedback[..., None, None], projected, state)


class TransitionNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.stem = nn.Linear(2 + 11*channels, WIDTH)
        self.context = nn.Linear(2 + 11*channels, RANK)
        self.transition = nn.Linear(WIDTH, 5*channels*3)
        self.risk = nn.Linear(WIDTH, 1)

    def initial(self, batch):
        return self.stem.weight.new_zeros((batch, WIDTH, RANK))

    def design(self, x):
        return self.stem(x), self.context(x).tanh() / np.sqrt(RANK)

    @staticmethod
    def project(pre, key, state):
        return (pre + (state @ key[..., None]).squeeze(-1)).tanh()

    def logits(self, hidden):
        return self.transition(hidden).unflatten(-1, (5, self.channels, 3))

    def update(self, pre, key, state, y, mask):
        hidden = self.project(pre, key, state)
        logits = self.logits(hidden)
        count = mask.sum((-1, -2), keepdim=True).clamp_min(1)
        logp = logits.log_softmax(-1)
        loss = -(logp.gather(-1, y[..., None]).squeeze(-1) * mask / count).sum((-1, -2))
        g = (logp.exp() - F.one_hot(y, 3)) * (mask / count)[..., None]
        direction = (g.flatten(-3) @ self.transition.weight) * (1. - hidden.square())
        return move(state, direction[..., None] * key[..., None, :], mask.any((-1, -2))), loss


@torch.no_grad()
def replay(net, x, origins, y, mask):
    """Reset each shot and write new representations starting at q+26.

    In a block with incomplete horizon the target does not enter the update.
    Only the training target is reversed; both arms' auxiliary loss is scored
    against the same chronological observations.
    """
    from .commissioning_transition_data_20261110 import reverse_valid

    def linear(layer):
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()

    ws, bs = linear(net.stem)
    wk, bk = linear(net.context)
    wd, bd = linear(net.transition)
    pre, key = x @ ws.T + bs, np.tanh(x @ wk.T + bk) / np.float32(np.sqrt(RANK))
    fixed = np.tanh(pre)
    views = {k: fixed.copy() for k in MODES}
    actual = reverse_valid(y, mask)
    state = np.zeros((2, WIDTH, RANK), dtype=np.float32)
    changed = np.zeros(2, int)
    loss = np.zeros(3, np.float64)
    seen = observed_blocks = different_targets = different_blocks = 0
    classes = np.eye(3, dtype=np.float32)
    for i, q in enumerate(origins):
        if q + HORIZON >= len(x):
            break
        m = mask[i]
        count = int(m.sum())
        first = int(q + HORIZON + 1)
        end = min(first + STRIDE, len(x))
        if count:
            observed_blocks += 1
            seen += count
            differences = int(np.count_nonzero(m & (y[i] != actual[i])))
            different_targets += differences
            different_blocks += differences > 0
            feature = np.vstack((fixed[q], np.tanh(pre[q] + state @ key[q])))
            s = (feature @ wd.T + bd).reshape(3, 5, net.channels, 3)
            shift = s - s.max(axis=-1, keepdims=True)
            exp = np.exp(shift)
            denom = exp.sum(axis=-1, keepdims=True)
            picked = np.take_along_axis(shift, y[i, ..., None][None], axis=-1)[..., 0]
            loss += (np.log(denom[..., 0]) - picked)[:, m].sum(axis=1, dtype=np.float64)
            g = exp[1:] / denom[1:] - classes[np.stack((y[i], actual[i]))]
            g *= m[None, ..., None] / np.float32(count)
            dh = (g.reshape(2, -1) @ wd) * (1. - feature[1:]**2)
            grad = dh[:, :, None] * key[q][None, None, :]
            norm = np.sqrt(np.maximum((grad**2).sum((1, 2), keepdims=True), 1e-16))
            nxt = RETAIN * state - STEP * grad / (norm + STABILIZER)
            norms = np.sqrt((nxt**2).sum((1, 2), keepdims=True))
            nxt /= np.maximum(1., norms / RADIUS)
            changed += np.any(nxt != state, axis=(1, 2))
            state = nxt
        for j, mode in enumerate(MODES[1:]):
            views[mode][first:end] = np.tanh(pre[first:end] + key[first:end] @ state[j].T)
    return views, dict(observed_blocks=observed_blocks, transition_labels=seen,
                       reversed_labels_different=different_targets, reversed_blocks_different=different_blocks,
                       nonzero_steps=dict(zip(MODES[1:], changed.tolist(), strict=True)),
                       nll_sum=dict(zip(MODES, loss.tolist(), strict=True)))
