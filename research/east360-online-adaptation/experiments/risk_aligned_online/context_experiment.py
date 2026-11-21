"""Bounded source-only training and causal EAST replay of ContextFeedback.

No qualified source weights, threshold, outcomes or alarm rules are modified.
No support110 data. Every EAST outcome is read only after prediction sealing.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import io23, simple_data, simple_train
from .context_feedback import ContextFeedback, predict_shot, train_step, signal_target
from .diagnostic_training import load_parent
from .fast_meta import episode_pools, sample
from .simple_online import build_order, seal_predictions
from .source_gate import qualify_source


def evaluate(parent, adapter, shots, threshold, *, arm, target_fn=signal_target):
    rows, event_count = [], 0
    for shot in shots:
        scores, events = predict_shot(parent, adapter, shot, arm=arm, target_fn=target_fn)
        valid = shot['valid']
        rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                         times=shot['times'][valid], scores=scores[valid]))
        event_count += len(events)
    result = simple_train.metrics(rows, threshold, lead=10.)
    result['signal_updates'] = event_count
    return result


def train(args):
    if args.output.exists() or min(args.steps, args.batch_size, args.unroll) < 1:
        raise ValueError('fresh output and positive source budget required')
    gate = qualify_source(args.parent)
    if not gate['qualified'] or not torch.cuda.is_available():
        raise ValueError('qualified parent and local CUDA required')
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    parent, ckpt = load_parent(args.parent)
    frozen = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    adapter = ContextFeedback(parent.cfg.dim, horizon=8).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.)
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    protocol = dict(parent=str(args.parent.resolve()), parent_sha256=gate['checkpoint_sha256'],
                    mode=args.mode, seed=131, steps=args.steps, batch_size=args.batch_size,
                    source_train_ids=roles['train'], source_labels='max(25ms,AdvancedTime)',
                    threshold=float(ckpt['threshold']), horizon=8, stride=9,
                    state_dimension=parent.cfg.dim, temporary_step=.5, temporary_radius=8.,
                    inner_unroll=args.unroll,
                    no_final_risk_correction=True, no_alarm_parameter_updates=True,
                    perturbation='source-only constant patch-feature shift, half of episodes; norm uniformly 0..4',
                    perturbation_caveat='synthetic latent nuisance stress test; not validated plasma or sensor-unit model',
                    auxiliary_coordinate='asinh of diagnostic change normalized by origin past-window std; minimum 8 points',
                    controller='normalized feature gradient plus bounded learned current-context modulation',
                    evidence='DEVELOPMENT_ONLY',
                    code_sha256={n: simple_train.sha(Path(__file__).with_name(n)) for n in
                                 ('context_feedback.py', 'context_experiment.py', 'diagnostic_targets.py',
                                  'simple23.py', 'simple_data.py', 'fast_meta.py')})
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    start = time.monotonic()
    shots = simple_train.load_shots(roles['train'], advanced)
    cfg = replace(parent.cfg, horizon=args.unroll*9-1)
    pools = episode_pools(shots, cfg)
    history = []
    for step in range(1, args.steps+1):
        e = sample(shots, pools, cfg, rng, args.batch_size, 'cuda')
        noise = torch.randn(args.batch_size, parent.cfg.dim, device='cuda')
        noise = noise/noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        amplitude = 4.*torch.rand(args.batch_size, 1, device='cuda')
        include = (torch.rand(args.batch_size, 1, device='cuda') < .5)
        perturb = noise*amplitude*include
        history.append(train_step(parent, adapter, optimizer, e, mode=args.mode,
                                  perturb=perturb, unroll=args.unroll))
        if step % 64 == 0:
            print(json.dumps(dict(step=step, elapsed_seconds=time.monotonic()-start, **history[-1])), flush=True)
    if any(not torch.equal(v.detach().cpu(), frozen[k]) for k, v in parent.state_dict().items()):
        raise AssertionError('qualified source changed')
    protocol['training_seconds'] = time.monotonic()-start
    torch.save(dict(protocol=protocol, adapter={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    parent_bitwise_unchanged=True), args.output/'adapter.pt')
    (args.output/'training.json').write_text(json.dumps(history, indent=2))
    print('source-only adapter saved; parent unchanged; no EAST result claimed', flush=True)
    del shots, pools
    if args.evaluate_val:
        shots = simple_train.load_shots(roles['val'], advanced)
        adapter.requires_grad_(False)
        report = {arm: evaluate(parent, adapter, shots, protocol['threshold'], arm=arm)
                  for arm in ('static', 'learned')}
        report.update(evidence='DEVELOPMENT_ONLY', protocol=protocol)
        (args.output/'val.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({arm: {k: v for k, v in report[arm].items() if k != 'predictions'}
                          for arm in ('static', 'learned')}), flush=True)


def replay(args, *, adapter_factory=ContextFeedback, target_fn=signal_target):
    if args.output.exists() or not 1 <= args.limit <= 360:
        raise ValueError('fresh output and fixed 1..360-shot prefix required')
    if args.mask_channel is not None and args.arms != ['static']:
        raise ValueError('input masking is a static diagnostic, never a neural adaptation claim')
    stored = torch.load(args.adapter, map_location='cpu', weights_only=False)
    protocol = stored['protocol']
    gate = qualify_source(Path(protocol['parent']))
    if not gate['qualified'] or gate['checkpoint_sha256'] != protocol['parent_sha256']:
        raise ValueError('matched qualified source required')
    parent, _ = load_parent(protocol['parent'])
    adapter = adapter_factory(parent.cfg.dim, horizon=protocol['horizon']).cuda()
    adapter.load_state_dict(stored['adapter'])
    adapter.requires_grad_(False)
    full_order = build_order('EAST', io23.cohort_ids('EAST'))
    if args.sample_count is not None:
        if args.limit != 360 or not 2 <= args.sample_count <= 360:
            raise ValueError('spaced diagnostic requires 2..360 samples and no prefix limit')
        positions = np.linspace(0, len(full_order)-1, args.sample_count, dtype=int).tolist()
        order = [full_order[i] for i in positions]
    else:
        positions = list(range(args.limit))
        order = full_order[:args.limit]
    protocol = dict(protocol, adapter_sha256=simple_train.sha(args.adapter),
                    replay_code_sha256=simple_train.sha(__file__),
                    online_domain='EAST', jtext_role='offline source/auxiliary training only',
                    source_online_gate_used=False,
                    runtime_code_sha256={n: simple_train.sha(Path(__file__).with_name(n)) for n in
                                         ('context_experiment.py', 'context_feedback.py', 'simple_data.py',
                                          'simple23.py', 'family_feedback.py', 'family_experiment.py')},
                    order=order, selected_positions=positions, arms=args.arms, selected_count=len(order),
                    labels='fixed125ms', timely_lead='strict >30ms',
                    outcome_access='only evaluator after prediction seal; never adaptation',
                    post_shot_updates=False, all_EAST_labels_before_first_shot=0,
                    input_mask_diagnostic=args.mask_channel,
                    evidence=('DEVELOPMENT_ONLY' if len(order) == 360 else
                              'DEVELOPMENT_ONLY_SPACED_DIAGNOSTIC' if args.sample_count is not None else
                              'DEVELOPMENT_ONLY_PREFIX'))
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    summary = {}
    for arm in args.arms:
        directory = args.output/arm
        directory.mkdir()
        rows, total_updates = [], 0
        start = time.monotonic()
        for i, ordered in enumerate(order, 1):
            shot = simple_data.read_shot('EAST', ordered['shot'])
            if args.mask_channel is not None:
                # Deliberate counterfactual diagnostic. Do not promote this
                # missing-channel input as an instrument/unit calibration.
                channel = io23.RAW23.index(args.mask_channel)
                shot['available'][:, channel] = False
                shot['values'][:, channel] = 0.
            scores, events = predict_shot(parent, adapter, shot, arm=arm, target_fn=target_fn)
            receipt = seal_predictions(directory, i, shot['shot'], shot['times'], scores,
                                       arm=arm, threshold=protocol['threshold'])
            # Evaluator access ONLY: no y/down/IsDisrupt reaches predict_shot.
            labeled = simple_data.attach_outcome(shot, 'EAST', {})
            valid = labeled['valid']
            rows.append(dict(shot=shot['shot'], is_disrupt=labeled['is_disrupt'], down=labeled['down'],
                             times=shot['times'][valid], scores=scores[valid]))
            total_updates += len(events)
            (directory/f'{i:04d}_{shot["shot"]}_audit.json').write_text(json.dumps(
                dict(seal=receipt, events=events, no_persistent_update=True), indent=2))
            if i % min(30, len(order)) == 0:
                result = simple_train.metrics(rows, protocol['threshold'], lead=30.)
                print(json.dumps(dict(arm=arm, through=i, elapsed_seconds=time.monotonic()-start,
                                      **{k: v for k, v in result.items() if k != 'predictions'})), flush=True)
        summary[arm] = dict(simple_train.metrics(rows, protocol['threshold'], lead=30.),
                            signal_updates=total_updates, elapsed_seconds=time.monotonic()-start)
        (directory/'summary.json').write_text(json.dumps(summary[arm], indent=2))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    if 'static' in summary:
        for arm in args.arms:
            summary[arm]['delta_J_vs_static'] = summary[arm]['J']-summary['static']['J']
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--parent', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--mode', choices=['aligned', 'forecast'], required=True)
    fit.add_argument('--steps', type=int, default=512)
    fit.add_argument('--batch-size', type=int, default=16)
    fit.add_argument('--evaluate-val', action='store_true')
    fit.add_argument('--unroll', type=int, default=1)
    run = sub.add_parser('replay')
    run.add_argument('--adapter', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--limit', type=int, default=360)
    run.add_argument('--sample-count', type=int)
    run.add_argument('--mask-channel', choices=list(io23.RAW23))
    run.add_argument('--arms', nargs='+', choices=['static', 'learned', 'gradient', 'reverse'],
                     default=['static', 'learned'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args) if args.command == 'train' else replay(args)


if __name__ == '__main__':
    main()
