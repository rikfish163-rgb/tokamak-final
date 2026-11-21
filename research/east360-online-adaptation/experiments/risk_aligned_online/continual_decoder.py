"""Natural ended-shot feedback learns the next shot's signal-update direction.

The qualified source classifier and alarm never change. Only an independent
copy of its H25 head is learned. Same-shot FFN deltas remain disposable. Recent
ended-task replay follows the online meta-learning idea, not FTML guarantees.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch

from .ended_context_rule import causal_peak_logit, peak_index
from .ended_decoder import DecoderFfnModel, train_pair
from .redesign_20261004.intra import predict_shot

MODES = ('early_fixed', 'late_continual', 'early_continual')


def head_digest(model):
    digest = hashlib.sha256()
    for name, value in model.decoder.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class RollingDecoder:
    def __init__(self, model, *, mode, online_steps=4, bootstrap_steps=40):
        if mode not in MODES or min(online_steps, bootstrap_steps) < 1:
            raise ValueError('declared schedule and positive learning budgets required')
        self.mode, self.online_steps, self.bootstrap_steps = mode, online_steps, bootstrap_steps
        self.initial_head = {k:v.detach().clone() for k,v in model.decoder.state_dict().items()}
        model.requires_grad_(False)
        # Only TRAINING replay uses this original-head reference. Its fixed
        # peaks are independent of future labels and of the deployed learner.
        self.reference = model.make_reference().requires_grad_(False).eval()
        self.optimizer = None
        self.optimizer_steps = 0
        self.last_index = 0
        self.rng = np.random.default_rng(131)
        self.cache = {}

    def _prepare_shot(self, shot, *, phase, device):
        key = (phase, shot['shot'])
        if key not in self.cache:
            scores, _ = predict_shot(self.reference, shot, intra=True, device=device)
            query = peak_index(shot, scores)
            logit, _ = causal_peak_logit(self.reference, shot, query)
            if abs(float(logit.detach().sigmoid())-float(scores[query])) > 2e-6:
                raise AssertionError('reference training peak differs from causal replay')
            self.cache[key] = query
        return shot, self.cache[key]

    def _train_pair(self, *args, **kwargs):
        # Default remains the archived single-peak objective. Descendants may
        # change only the ended-task objective while retaining this schedule.
        return train_pair(*args, **kwargs)

    def __call__(self, model, shot, replay, *, provider, threshold, **kwargs):
        history = [*replay, shot]
        index = len(provider.ended_ids)
        ids = [s['shot'] for s in history]
        if (provider.pending is not None or not index or index <= self.last_index
                or ids != provider.ended_ids[-len(ids):]
                or any(s.get('device') != 'EAST' or '_sealed_scores' not in s
                       or s.get('horizon') != 125. for s in history)):
            raise ValueError('new current shot and exact sealed ended history suffix required')
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('finite fixed source threshold required')
        if (index < 30 and self.mode == 'late_continual') or (
                index > 30 and self.mode == 'early_fixed'):
            self.last_index = index
            return dict(accepted=False, reason='outside_declared_head_learning_schedule',
                        actual_steps=0, effective_from_index=index+1)
        history = history if index == 30 else history[-32:]
        if index == 30 and [s['shot'] for s in history] != provider.ended_ids:
            raise ValueError('coordinate transition requires the exact first30 ended shots')
        labels = {int(s['is_disrupt']) for s in history}
        if labels != {0,1}:
            if index == 30:
                raise ValueError('bootstrap cannot estimate a balanced task from one class')
            # Explicit protocol: do NOT train a one-class risk direction. All
            # shots still receive their ordinary signal-only temporary updates.
            self.last_index = index
            return dict(accepted=False, reason='waiting_for_both_ended_classes',
                        actual_steps=0, history_ids=[s['shot'] for s in history],
                        effective_from_index=index+1)
        saved_head = {k:v.detach().clone() for k,v in model.decoder.state_dict().items()}
        old_optimizer = self.optimizer
        saved_optimizer = None if old_optimizer is None else deepcopy(old_optimizer.state_dict())
        saved_steps, saved_rng = self.optimizer_steps, deepcopy(self.rng.bit_generator.state)
        saved_cache = self.cache.copy()
        head_before = head_digest(model)
        reset = index == 30
        records = []
        try:
            if reset:
                # The raw-to-density-coordinate input contract changes once at
                # this boundary. Do not reinterpret a raw-coordinate decoder
                # or its Adam moments as if trained in the new coordinates.
                model.decoder.load_state_dict(self.initial_head)
                self.optimizer = None
                self.optimizer_steps = 0
                self.rng = np.random.default_rng(131)
                self.cache = {}
            encoded = [provider.reencode(s) for s in history]
            episodes = [self._prepare_shot(s, phase=int(index>=30), device=model.center.device)
                        for s in encoded]
            pools = [[e for e in episodes if int(e[0]['is_disrupt']) == label] for label in (0,1)]
            model.decoder.requires_grad_(True)
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(model.decoder.parameters(), lr=.0003)
            budget = self.bootstrap_steps if reset else self.online_steps
            boundary = math.log(threshold/(1-threshold))
            for iteration in range(budget):
                if reset:
                    pair = [pool[int(self.rng.integers(len(pool)))] for pool in pools]
                else:
                    current = episodes[-1]
                    opposite = pools[1-int(current[0]['is_disrupt'])]
                    pair = [current, opposite[int(self.rng.integers(len(opposite)))]]
                    pair.sort(key=lambda e:int(e[0]['is_disrupt']))
                record = self._train_pair(model, pair, self.optimizer, boundary=boundary,
                                          mode='risk_after_update')
                self.optimizer_steps += 1
                records.append(dict(iteration=iteration+1, **record))
        except Exception:
            # Numerical/training failure aborts this run after restoring the
            # previous learned head and optimizer. No silent static substitute.
            model.decoder.load_state_dict(saved_head)
            self.optimizer = old_optimizer
            if old_optimizer is not None:
                old_optimizer.load_state_dict(saved_optimizer)
            self.optimizer_steps = saved_steps
            self.rng.bit_generator.state = saved_rng
            self.cache = saved_cache
            raise
        finally:
            model.decoder.requires_grad_(False)
        self.last_index = index
        # Peaks for discarded older tasks are not a hidden unbounded replay.
        kept = {s['shot'] for s in history}
        self.cache = {k:v for k,v in self.cache.items() if k[1] in kept}
        adam_steps = {name:float(self.optimizer.state[p]['step'])
                      for name,p in model.decoder.named_parameters() if p in self.optimizer.state}
        return dict(accepted=True, reason='ended_auxiliary_meta_update',
            actual_steps=len(records), coordinate_transition_reset=reset,
            optimizer_steps_after=self.optimizer_steps, actual_adam_steps=adam_steps,
            history_ids=[s['shot'] for s in history],
            effective_from_index=index+1, head_before_sha256=head_before,
            head_after_sha256=head_digest(model),
            fixed_peaks=[dict(shot=s['shot'],query=q,label=int(s['is_disrupt'])) for s,q in episodes],
            records=records, source_changed=False, persistent_ffn_changed=False)
