"""Offline source auxiliary fitting; online adaptation strictly on EAST.

Run matched risk-aligned, forecast-only and shuffled-family source candidates.
Only the original STATIC source qualification is required. Do not make J-TEXT
online results a gate for EAST. No outcome reaches the intra-shot updater and
no persistent/post-shot update is enabled in this bounded intra-only trial.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import simple_data, simple_train
from .context_feedback import train_step
from .context_experiment import replay
from .diagnostic_training import load_parent
from .family_feedback import FAMILIES, FamilyFeedback, signal_target23
from .fast_meta import episode_pools, sample
from .io23 import RAW23
from .source_gate import qualify_source


def train(args):
    if args.output.exists() or min(args.steps, args.batch_size, args.unroll) < 1:
        raise ValueError('fresh output and positive source budget required')
    gate = qualify_source(args.parent)
    if not gate['qualified'] or not torch.cuda.is_available():
        raise ValueError('qualified source and local CUDA required')
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    parent, ckpt = load_parent(args.parent)
    frozen = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    adapter = FamilyFeedback(parent.cfg.dim, patch=parent.cfg.patch, grouping=args.grouping).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.)
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    protocol = dict(parent=str(args.parent.resolve()), parent_sha256=gate['checkpoint_sha256'],
                    candidate='family_patch_input', seed=131, mode=args.mode, grouping=args.grouping,
                    groups=[[RAW23[i] for i in group] for group in adapter.groups],
                    physical_labels=FAMILIES, rank=adapter.rank, patch=parent.cfg.patch,
                    steps=args.steps, batch_size=args.batch_size, inner_unroll=args.unroll,
                    source_train_ids=roles['train'], source_labels='max(25ms,AdvancedTime)',
                    threshold=float(ckpt['threshold']), horizon=adapter.horizon,
                    state_dimension=adapter.state_dim, temporary_step=.5, temporary_radius=8.,
                    parameter_count=sum(p.numel() for p in adapter.parameters()),
                    forecast='direct full window of 23 recorded channels; no autoregressive feedback',
                    coordinate='asinh((arrived value-origin last)/origin past std); minimum 8 valid points',
                    insertion='patch-dependent per-family low-rank residual before first Transformer',
                    source_absolute_coordinates_preserved=True, synthetic_perturbations=False,
                    no_final_risk_correction=True, no_alarm_parameter_updates=True,
                    post_shot_updates=False, all_EAST_labels_before_first_shot=0,
                    online_domain='EAST', jtext_role='offline source/auxiliary training only',
                    jtext_online_evaluation=False,
                    evidence='DEVELOPMENT_ONLY', code_sha256={
                        n: simple_train.sha(Path(__file__).with_name(n)) for n in
                        ('family_feedback.py', 'family_experiment.py', 'context_feedback.py',
                         'context_experiment.py', 'simple23.py', 'simple_data.py', 'fast_meta.py')})
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    start = time.monotonic()
    shots = simple_train.load_shots(roles['train'], advanced)
    cfg = replace(parent.cfg, horizon=args.unroll*(adapter.horizon+1)-1)
    pools = episode_pools(shots, cfg)
    history = []
    for step in range(1, args.steps+1):
        episode = sample(shots, pools, cfg, rng, args.batch_size, 'cuda')
        history.append(train_step(parent, adapter, optimizer, episode, mode=args.mode,
                                  unroll=args.unroll, target_fn=signal_target23))
        if step % 64 == 0:
            print(json.dumps(dict(step=step, elapsed_seconds=time.monotonic()-start, **history[-1])), flush=True)
    if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in parent.state_dict().items()):
        raise AssertionError('qualified source changed')
    protocol['training_seconds'] = time.monotonic()-start
    torch.save(dict(protocol=protocol, adapter={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    parent_bitwise_unchanged=True), args.output/'adapter.pt')
    (args.output/'training.json').write_text(json.dumps(history, indent=2))
    print('offline source auxiliary fit saved; source unchanged; online evaluation is EAST-only', flush=True)


def replay_east(args):
    protocol = torch.load(args.adapter, map_location='cpu', weights_only=False)['protocol']
    if protocol.get('candidate') != 'family_patch_input':
        raise ValueError('family-adapter checkpoint required')
    # User scope: J-TEXT provides the offline source model only. Legacy
    # val.json online diagnostics are neither read nor required. The shared
    # EAST runner still checks the qualified STATIC parent and its file hash.
    args.mask_channel = None

    def factory(dim, horizon):
        return FamilyFeedback(dim, horizon=horizon, patch=protocol['patch'],
                              rank=protocol['rank'], grouping=protocol['grouping'])
    replay(args, adapter_factory=factory, target_fn=signal_target23)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--parent', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--mode', choices=['aligned', 'forecast'], required=True)
    fit.add_argument('--grouping', choices=['physical', 'shuffled'], default='physical')
    fit.add_argument('--steps', type=int, default=512)
    fit.add_argument('--batch-size', type=int, default=16)
    fit.add_argument('--unroll', type=int, default=4)
    run = sub.add_parser('replay')
    run.add_argument('--adapter', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--limit', type=int, default=360)
    run.add_argument('--sample-count', type=int)
    run.add_argument('--arms', nargs='+', choices=['static', 'learned', 'gradient', 'reverse'],
                     default=['static', 'learned', 'gradient', 'reverse'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args) if args.command == 'train' else replay_east(args)


if __name__ == '__main__':
    main()
