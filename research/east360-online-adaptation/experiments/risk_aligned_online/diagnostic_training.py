"""Source-only fitting of an internal diagnostic-coordinate adaptation rule.

Parent source classifier and threshold never change. Forecast-only and
risk-aligned arms share initialization, episodes and budgets. Risk labels
appear only in offline source outer training, never the within-shot step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data, simple_train
from .diagnostic_adapter import DiagnosticAdapter, bounded_step
from .diagnostic_online import evaluate
from .diagnostic_targets import DIAGNOSTICS, coordinates, linear_extrapolation, trajectory_loss, trajectory_target
from .fast_meta import episode_pools, sample
from .simple23 import Config, Simple23
from .source_gate import qualify_source


def train_step(parent, adapter, optimizer, episode, *, mode, update_rule='gradient', target_mode='increment'):
    if mode not in ('forecast', 'aligned'):
        raise ValueError('matched forecast or risk-aligned arm required')
    optimizer.zero_grad(set_to_none=True)
    fast = adapter.initial_fast(len(episode['x'])).requires_grad_(True)
    prediction = adapter(parent, episode['x'], episode['a'], fast)['trajectory']
    coordinate = coordinates(episode['x'], episode['a'])
    normal = (linear_extrapolation(episode['x'], episode['a'], coordinate)
              if target_mode == 'innovation' else torch.zeros_like(prediction))
    if target_mode not in ('increment', 'innovation'):
        raise ValueError('known diagnostic target mode required')
    prediction = prediction+normal
    target, valid = trajectory_target(coordinate, episode['truth'], episode['mask'])
    signal = trajectory_loss(prediction, target, valid)
    if update_rule == 'gradient':
        gradient, = torch.autograd.grad(signal.sum(), fast, create_graph=True)
        updated = bounded_step(fast, gradient)
    elif update_rule == 'assimilation':
        from .diagnostic_assimilation import assimilate
        updated = assimilate(fast, prediction, target, valid, adapter.time_basis)
    else:
        raise ValueError('known update rule required')
    after = adapter(parent, episode['qx'], episode['qa'], updated)['logit']
    risk = F.binary_cross_entropy_with_logits(after, episode['y'])
    with torch.no_grad():
        before = F.binary_cross_entropy_with_logits(parent(episode['qx'], episode['qa'])['logit'], episode['y'])
        persistence = trajectory_loss(torch.zeros_like(target), target, valid).mean()
        trend = trajectory_loss(normal, target, valid).mean()
    # Both arms must actually predict diagnostics. The aligned arm additionally
    # learns an internal direction whose SIGNAL-only step benefits later risk.
    objective = signal.mean() if mode == 'forecast' else risk+.2*signal.mean()
    if not torch.isfinite(objective):
        raise FloatingPointError('nonfinite diagnostic meta-training objective')
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(signal_loss=float(signal.mean().detach()), persistence_loss=float(persistence), linear_trend_loss=float(trend),
                risk_before=float(before), risk_after=float(risk.detach()), gradient_norm=float(norm),
                fast_change=float(updated.detach().norm(dim=-1).mean()), valid_fraction=float(valid.float().mean()))


def load_parent(path, device='cuda'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    parent = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale'])
    parent.load_state_dict(ckpt['model'])
    parent.to(device).eval().requires_grad_(False)
    return parent, ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['forecast', 'aligned'], required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--update-rule', choices=['gradient', 'assimilation'], default='gradient')
    parser.add_argument('--site', choices=['cls', 'last_patch'], default='cls')
    parser.add_argument('--target-mode', choices=['increment', 'innovation'], default='increment')
    args = parser.parse_args()
    if args.output.exists() or min(args.steps, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('fresh output directory and positive finite budget required')
    gate = qualify_source(args.parent)
    if not gate['qualified'] or not torch.cuda.is_available():
        raise ValueError('qualified source and local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    rng = np.random.default_rng(131)
    parent, ckpt = load_parent(args.parent)
    frozen = {key: value.detach().cpu().clone() for key, value in parent.state_dict().items()}
    adapter = DiagnosticAdapter(parent.cfg.dim, parent.cfg.horizon, site=args.site).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.)
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    args.output.mkdir(parents=True)
    protocol = dict(parent=str(args.parent.resolve()), parent_sha256=gate['checkpoint_sha256'],
                    parent_gate=gate, source_labels='max(25ms,AdvancedTime)',
                    diagnostics=list(DIAGNOSTICS), source_train_ids=roles['train'],
                    evidence='DEVELOPMENT_ONLY', seed=131, mode=args.mode, steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr, threshold=float(ckpt['threshold']),
                    fast_step=.1, fast_radius=2., update_rule=args.update_rule, site=args.site,
                    target_mode=args.target_mode,
                    calibration='origin past-window standard deviation only',
                    selection='no threshold scan; compare full VAL160; TEST only after training fixed',
                    code_sha256={name: simple_train.sha(Path(__file__).with_name(name)) for name in
                                 ('diagnostic_adapter.py', 'diagnostic_targets.py', 'diagnostic_training.py',
                                  'diagnostic_online.py', 'diagnostic_assimilation.py',
                                  'fast_meta.py', 'simple23.py', 'simple_data.py')})
    (args.output/'config.json').write_text(json.dumps(protocol, indent=2))
    train = simple_train.load_shots(roles['train'], advanced)
    pools = episode_pools(train, parent.cfg)
    history = []
    for step in range(1, args.steps+1):
        episode = sample(train, pools, parent.cfg, rng, args.batch_size, 'cuda')
        history.append(train_step(parent, adapter, optimizer, episode, mode=args.mode,
                                  update_rule=args.update_rule, target_mode=args.target_mode))
        if step % 64 == 0:
            print(json.dumps(dict(step=step, **history[-1])), flush=True)
    if any(not torch.equal(value.detach().cpu(), frozen[key]) for key, value in parent.state_dict().items()):
        raise AssertionError('qualified parent parameter/buffer changed')
    torch.save(dict(adapter={key: value.detach().cpu() for key, value in adapter.state_dict().items()},
                    protocol=protocol, parent_bitwise_unchanged=True), args.output/'adapter.pt')
    (args.output/'training.json').write_text(json.dumps(history, indent=2))
    del train, pools
    for role in (('val', 'test') if args.evaluate_test else ('val',)):
        shots = simple_train.load_shots(roles[role], advanced)
        report = evaluate(parent, adapter, shots, float(ckpt['threshold']),
                          update_rule=args.update_rule, target_mode=args.target_mode)
        report.update(protocol=protocol, parent_bitwise_unchanged=True)
        (args.output/f'{role}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(dict(role=role, static={k: v for k, v in report['static'].items() if k != 'predictions'},
                              intra={k: v for k, v in report['intra'].items() if k != 'predictions'})), flush=True)


if __name__ == '__main__':
    main()
