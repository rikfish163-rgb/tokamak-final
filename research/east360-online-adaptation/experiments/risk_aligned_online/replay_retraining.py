"""Paired source-restart/continuation diagnostic, not a paper reproduction.

Both arms use identical ended-shot windows and fresh AdamW each round, so only
the parameter initialization differs. Fixed source scales/threshold and original
inputs are retained. No guard rejects candidates in this diagnostic; no safety
claim is made. Forecast parameters are not refitted: intra use is prohibited.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .simple23 import Config, Simple23
from .simple_online import run_arm, _durable_json, _hash
from .source_gate import qualify_source


def actionable_weights(shot):
    """Ended-shot importance weights; keep original labels unchanged.

    Transfer the expected positive loss mass from the late interval into
    positive windows before the strict 30ms deadline. An entirely unobserved
    actionable interval contributes no positive loss, not fabricated positives.
    """
    valid = np.asarray(shot['valid'],bool)
    positive = valid & (np.asarray(shot['y']) > .5)
    weights = valid.astype(np.float32)
    if not shot['is_disrupt']:
        return weights
    lead = np.round((shot['down']-np.asarray(shot['times']))*1000.,9)
    actionable = positive & (lead > 30.)
    weights[positive] = 0.
    if actionable.any():
        weights[actionable] = positive.sum()/actionable.sum()
    return weights


class ReplayRetrainer:
    def __init__(self, model, *, mode, steps=16, capacity=200, batch_size=64, lr=1e-4, seed=131,
                 objective='bce', scope='last_block'):
        if mode not in ('source_restart', 'continue') or min(steps,capacity,batch_size) < 1:
            raise ValueError('invalid retraining configuration')
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('positive finite learning rate required')
        if objective not in ('bce','actionable','safe_negative'):
            raise ValueError('unknown training objective')
        if scope not in ('last_block','full'):
            raise ValueError('unknown trainable scope')
        self.scope = scope
        self.objective = objective
        self.source = deepcopy(model.state_dict())
        self.mode, self.steps, self.capacity = mode, steps, capacity
        self.batch_size, self.lr, self.seed = batch_size, lr, seed
        self.history = deque(maxlen=capacity)
        self.seen = set()
        self.round = 0

    def __call__(self, model, shot, *, device='cpu', data_api=simple_data):
        sid = int(shot['shot'])
        if sid in self.seen:
            raise ValueError('duplicate ended shot')
        if shot['device'] != 'EAST' or not {'valid','y','is_disrupt'}.issubset(shot):
            raise ValueError('revealed EAST outcome required')
        shots = [*self.history, shot][-self.capacity:]
        pools = []
        for s in shots:
            ends = np.flatnonzero(s['valid'])
            if not len(ends):
                raise ValueError('ended shot has no valid endpoints')
            labels = np.asarray(s['y'])
            pools.append((ends[labels[ends] <= .5], ends[labels[ends] > .5]))
        weights = [actionable_weights(s) for s in shots] if self.objective == 'actionable' else None
        if self.objective == 'safe_negative':
            # Shen et al., NF64 066036, section 3.1 excludes early windows
            # of disruptive shots from supervised training. Preserve the old
            # sample manifest and mask their loss, without changing labels.
            weights = [np.asarray(s['valid'],np.float32) *
                       ((np.asarray(s['y']) > .5) if s['is_disrupt'] else 1.) for s in shots]
        before = deepcopy(model.state_dict())
        flags = [(p,p.requires_grad) for p in model.parameters()]
        if self.scope == 'full':
            selected = [p for name,p in model.named_parameters() if not name.startswith('forecast_head.')]
        else:
            # The last-block experiment is Transformer-specific. Full-model
            # replay only needs named parameters and also supports a causal TCN.
            selected = [*model.encoder.layers[-1].parameters(), *model.risk_head.parameters(), model.fast_origin]
        chosen = {id(p) for p in selected}
        rng = np.random.default_rng(self.seed + self.round)
        digest = hashlib.sha256()
        losses, positive, excluded = [], 0, 0
        effective, optimizer_steps = 0, 0
        try:
            if self.mode == 'source_restart':
                model.load_state_dict(self.source)
            for p,_ in flags:
                p.requires_grad_(id(p) in chosen)
            model.eval()
            optimizer = torch.optim.AdamW(selected, lr=self.lr, weight_decay=0.)
            for _ in range(self.steps):
                examples = []
                batch_weights = []
                for index in range(self.batch_size):
                    # Guarantee current-shot exposure, then sample history uniformly
                    # by shot, not duration. Both arms consume identical RNG draws.
                    group = len(shots)-1 if index == 0 else int(rng.integers(len(shots)))
                    neg,pos = pools[group]
                    if len(neg) and len(pos):
                        ends = pos if rng.integers(2) else neg
                    else:
                        # A naturally single-class first shot can still teach its
                        # observed class; no invented examples or labels are used.
                        ends = pos if len(pos) else neg
                    end = int(rng.choice(ends))
                    s = shots[group]
                    x,mask = data_api.window(s,end,model.cfg.window)
                    y = float(s['y'][end])
                    positive += int(y > .5)
                    examples.append((x,mask,y))
                    w = float(weights[group][end]) if weights is not None else 1.
                    batch_weights.append(w)
                    excluded += int(w == 0)
                    digest.update(f'{s["shot"]}:{end};'.encode())
                x,mask,y = [torch.as_tensor(np.stack(v),device=device) for v in zip(*examples)]
                effective += sum(w > 0 for w in batch_weights)
                # If no precursor or safe window was observed in this batch,
                # there is no supervised target. Do not fabricate negatives or
                # apply Adam momentum on an empty loss; retain ended history.
                if self.objective == 'safe_negative' and not any(batch_weights):
                    losses.append(0.)
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(x,mask)['logit']
                if weights is None:
                    loss = F.binary_cross_entropy_with_logits(logits,y.float())
                else:
                    per_window = F.binary_cross_entropy_with_logits(logits,y.float(),reduction='none')
                    w = torch.as_tensor(batch_weights,device=device)
                    loss = (per_window*w).mean()
                    if self.objective == 'safe_negative':
                        loss = (per_window*w).sum()/w.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite trained model')
        except Exception:
            model.load_state_dict(before)
            raise
        finally:
            for p,enabled in flags:
                p.requires_grad_(enabled)
        self.history.append(shot)
        self.seen.add(sid)
        self.round += 1
        return dict(accepted=True, reason='unguarded_retraining_diagnostic', mode=self.mode,
                    history_ids=[int(s['shot']) for s in shots], steps=self.steps,
                    losses=losses, sample_sha256=digest.hexdigest(),
                    positive_windows=positive, negative_windows=self.steps*self.batch_size-positive,
                    objective=self.objective,
                    excluded_late_windows=excluded if self.objective == 'actionable' else 0,
                    excluded_early_windows=excluded if self.objective == 'safe_negative' else 0,
                    effective_windows=effective, optimizer_steps=optimizer_steps,
                    trainable_scope=self.scope, trainable_parameters=sum(p.numel() for p in selected),
                    optimizer='fresh_AdamW_both_arms', lr=self.lr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--order',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=360)
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--mode',choices=['paired','source_restart','continue'],default='paired')
    parser.add_argument('--objective',choices=['bce','actionable','safe_negative'],default='bce')
    parser.add_argument('--scope',choices=['last_block','full'],default='last_block')
    args = parser.parse_args()
    if not 1 <= args.limit <= 360 or args.steps < 1:
        parser.error('invalid budget')
    if args.output.exists():
        raise FileExistsError('preserve existing run')
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(gate['reasons'])
    order = json.loads(args.order.read_text())
    ids = [int(r['shot']) for r in order]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) != set(io23.cohort_ids('EAST')):
        raise ValueError('locked full360 manifest required')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required for this experiment')
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    parent = Simple23(Config(**checkpoint['model_config']),checkpoint['source_center'],checkpoint['source_scale'])
    parent.load_state_dict(checkpoint['model'])
    parent.cuda().eval()
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json',dict(
        checkpoint_sha256=_hash(args.checkpoint), order_sha256=_hash(args.order),
        source_gate=gate, threshold=float(checkpoint['threshold']), limit=args.limit,
        steps=args.steps,capacity=200,batch_size=64,lr=1e-4,seed=131,
        objective=args.objective, requested_mode=args.mode,
        trainable_scope=args.scope,
        factors='initialization paired only when mode=paired; fresh optimizer; no guard; original inputs',
        scope='DEVELOPMENT_ONLY inter-only diagnostic; not E-CAAD reproduction',
        code_sha256={name:_hash(Path(__file__).with_name(name)) for name in
                     ('replay_retraining.py','simple_online.py','simple23.py','simple_data.py','io23.py','simple_train.py','stage_metrics.py')}))
    result = {}
    modes = ('source_restart','continue') if args.mode == 'paired' else (args.mode,)
    for mode in modes:
        model = deepcopy(parent)
        learner = ReplayRetrainer(model,mode=mode,steps=args.steps,objective=args.objective,scope=args.scope)
        def update(net,ended,**kwargs):
            log = learner(net,ended,**kwargs)
            if learner.round % 10 == 0:
                print(f'{mode}: {learner.round}/{args.limit} sealed, revealed, trained',flush=True)
            return log
        result[mode] = run_arm(model,order=order,arm='inter',threshold=float(checkpoint['threshold']),
                               output_dir=args.output/mode,device='cuda',limit=args.limit,
                               startup_budget=30,persistent_updater=update)
        torch.save(model.state_dict(),args.output/f'{mode}_final.pt')
        key = 'metrics_full' if args.limit == 360 else 'metrics_prefix'
        print(mode,json.dumps({k:v for k,v in result[mode][key].items() if k != 'predictions'}),flush=True)
    if args.mode == 'paired':
        for a,b in zip(result['source_restart']['rows'],result['continue']['rows']):
            if a['persistent']['sample_sha256'] != b['persistent']['sample_sha256']:
                raise AssertionError('paired sampling diverged')
    _durable_json(args.output/'summary.json',result)


if __name__ == '__main__':
    main()
