"""Post-reveal learning of a signal-only intra-shot loss, with full rollback."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .sequence_fast import unroll
from .sequence_train import pools, sample
from . import simple_data


def fit_weights(model, episodes, logits, optimizer):
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        result = unroll(model, episodes, create_graph=True, channel_logits=logits)
        loss = F.binary_cross_entropy_with_logits(result['logits'][-1], episodes[-1]['y'])
        loss.backward()
        if not torch.isfinite(loss) or logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError('nonfinite ended channel gradient')
        norm = float(logits.grad.norm())
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite channel weights')
        return dict(loss=float(loss.detach()), gradient_norm=norm)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


class EndedChannelLearner:
    supports_intra = True

    def __init__(self, base, channel_logits, *, steps=4, depth=4, batch_size=8):
        if min(steps, depth, batch_size) < 1:
            raise ValueError('positive ended learning budget required')
        self.base = base
        self.logits = channel_logits
        self.steps, self.depth, self.batch_size = steps, depth, batch_size

    def __call__(self, model, shot, *, device='cpu', data_api=simple_data):
        state = deepcopy(model.state_dict())
        logits_before = self.logits.detach().clone()
        supervised = self.base.supervised
        history, seen, index = deepcopy(supervised.history), set(supervised.seen), supervised.round
        try:
            result = self.base(model, shot, device=device, data_api=data_api)
            after_base = deepcopy(model.state_dict())
            ended = list(supervised.history)
            pool = pools(ended, model.cfg, self.depth, require_both=False)
            rng = np.random.default_rng(2000131+index)
            candidate = torch.nn.Parameter(self.logits.detach().clone())
            optimizer = torch.optim.Adam([candidate], lr=.01)
            losses, manifest = [], []
            for _ in range(self.steps):
                episodes, ids = sample(ended, pool, model.cfg, rng, self.depth, self.batch_size, device)
                losses.append(fit_weights(model, episodes, candidate, optimizer))
                manifest.extend(ids)
            if not all(torch.equal(v, after_base[k]) for k,v in model.state_dict().items()):
                raise AssertionError('loss learner modified persistent network')
            with torch.no_grad():
                self.logits.copy_(candidate.detach())
            result['channel_meta'] = dict(steps=self.steps, depth=self.depth, losses=losses,
                sample_sha256=hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
                revealed_history_ids=[int(s['shot']) for s in ended],
                weights=(23*self.logits.softmax(0)).detach().cpu().tolist())
            return result
        except Exception:
            model.load_state_dict(state)
            with torch.no_grad():
                self.logits.copy_(logits_before)
            supervised.history, supervised.seen, supervised.round = history, seen, index
            raise
