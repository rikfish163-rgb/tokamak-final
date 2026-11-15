"""Optional OFFLINE JTEXT TRAIN-only gradient-aligned forecast-head prior.

This does not train a source classifier or a target-domain branch. Only the
existing direct forecast head is saved, separately from the qualified parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import fast_meta, simple_data
from ..simple23 import Config, Simple23
from ..simple_online import FAST_STEP, FAST_RADIUS


PARENT_SHA256 = 'c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02'
DEFAULT_PARENT = Path(__file__).resolve().parents[1]/'runs/gradient_aligned512_20260929/source.pt'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def classifier_state(model):
    """Snapshot EVERY parameter/buffer except the two forecast-head tensors."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.startswith('forecast_head.')}


def _tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def assert_classifier_unchanged(model, original):
    current = classifier_state(model)
    if set(current) != set(original):
        raise AssertionError('static classifier state keys changed')
    for name, old in original.items():
        new = current[name]
        if new.shape != old.shape or new.dtype != old.dtype or _tensor_bytes(new) != _tensor_bytes(old):
            raise AssertionError(f'static classifier changed: {name}')


def load_parent(checkpoint):
    """Hash-lock the explicitly trusted parent BEFORE loading its NumPy pickle."""
    checkpoint = Path(checkpoint)
    if sha256(checkpoint) != PARENT_SHA256:
        raise ValueError('parent SHA256 does not match the locked qualified source')
    # The trusted, hash-locked existing parent contains NumPy metadata. There
    # is no unsafe-deserialization fallback for any other checkpoint hash.
    parent = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = Config(**parent['model_config'])
    if (cfg.fast_site != 'tokens' or cfg.fast_rule != 'normalized'
            or cfg.input_view != 'source' or cfg.dim != 64
            or cfg.window != 128 or cfg.horizon != 25):
        raise ValueError('unchanged source token/flat23/direct-H25 configuration required')
    if not parent.get('forecast_trained', False):
        raise ValueError('parent must have its existing trained direct forecast head')
    base = Simple23(cfg, parent['source_center'], parent['source_scale'])
    base.load_state_dict(parent['model'], strict=True)
    base.requires_grad_(False)
    base.eval()
    return base, parent


def load_source_train(*, data_api=simple_data, expected_train_count=1353):
    """Read signals/outcomes for TRAIN only; other role lists are identity checks."""
    roles = data_api.source_roles()
    ids = [int(sid) for sid in roles['train']]
    if len(ids) != expected_train_count or len(set(ids)) != len(ids):
        raise ValueError('distinct canonical source TRAIN IDs required')
    if set(ids) & (set(roles['val']) | set(roles['test'])):
        raise ValueError('source TRAIN/holdout role overlap')
    all_advanced = data_api.advanced_times()
    # Only TRAIN values are passed to the existing max(25, AdvancedTime) label
    # function; no holdout value lookup, qualification or target label access.
    advanced = {sid: all_advanced[sid] for sid in ids if sid in all_advanced}
    shots = []
    for sid in ids:
        signals = data_api.read_shot('J-TEXT', sid)
        if signals['device'] != 'J-TEXT' or int(signals['shot']) != sid:
            raise ValueError('source TRAIN signal identity mismatch')
        shots.append(data_api.attach_outcome(signals, 'J-TEXT', advanced))
    return shots, ids


def source_episode_pools(shots, cfg, train_ids):
    """Reuse the source's t -> t+H+1 episode utility, after a label boundary check."""
    allowed = set(train_ids)
    if not shots or any(s['device'] != 'J-TEXT' or int(s['shot']) not in allowed for s in shots):
        raise ValueError('only JTEXT source TRAIN episodes are authorized')
    # Existing utilities balance actual future endpoint labels; they never
    # invent a 30-ms/125-ms positive window or use TPR for optimization.
    return fast_meta.episode_pools(shots, cfg)


def temporary_step_components(model, e, *, create_graph):
    """Exact trained-site source update, absolute fast_origin + zero delta.

    Translation to ContextModel's zero-delta coordinates leaves both tasks'
    fast gradients unchanged. The normalized update is .05, radius 1, per
    episode (not a batch-mean gradient); the source cfg is never replaced.
    """
    if model.cfg.fast_site != 'tokens' or model.cfg.fast_rule != 'normalized':
        raise ValueError('matched normalized token adaptation required')
    if FAST_STEP != .05 or FAST_RADIUS != 1.:
        raise ValueError('source numerical update no longer matches the fixed intra contract')
    return fast_meta.temporary_step_components(model, e, create_graph=create_graph)


def alignment_loss(signal_gradient, risk_gradient, eps=1e-8):
    """Smooth-norm cosine with a detached future-risk target; safe double backward."""
    if signal_gradient.shape != risk_gradient.shape or signal_gradient.ndim != 2:
        raise ValueError('matching batch-by-fast gradient tensors required')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('positive finite gradient-coordinate epsilon required')
    if not torch.isfinite(signal_gradient).all() or not torch.isfinite(risk_gradient).all():
        raise FloatingPointError('nonfinite alignment gradient')
    target = risk_gradient.detach()
    norm2 = target.square().sum(-1)
    valid = norm2 > eps*eps
    # norm() at an exactly zero signal has undefined second derivatives.
    # Smooth both norms; epsilon is NOT a change of diagnostic input scale.
    sn = (signal_gradient.square().sum(-1) + eps*eps).sqrt()
    rn = (norm2 + eps*eps).sqrt()
    cosine = ((signal_gradient*target).sum(-1)/(sn*rn)).clamp(-1., 1.)
    return ((1.-cosine)*valid).sum()/valid.sum().clamp_min(1)


def train_step(model, optimizer, e, *, forecast_weight=.1):
    """Align d(signal loss)/d(fast) to d(future source BCE)/d(fast).

    The risk direction is detached. The outer gradient passes through the
    inner signal derivative into forecast_head ONLY; modest forecast fitting
    preserves diagnostic decoding. Actual risk-after is an audit, not TPR.
    """
    if not np.isfinite(forecast_weight) or not 0 <= forecast_weight <= 1:
        raise ValueError('forecast maintenance weight must be finite in [0,1]')
    head = {id(p) for p in model.forecast_head.parameters()}
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(optimizer_ids) != head or len(optimizer_ids) != len(head):
        raise ValueError('optimizer must contain only forecast_head weight and bias')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    training = model.training
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in head)
        updated, signal, fast, signal_gradient = temporary_step_components(model, e, create_graph=True)
        losses = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'], fast)['logit'],
                                                     e['y'], reduction='none')
        risk_gradient, = torch.autograd.grad(losses.sum(), fast)
        aligned = alignment_loss(signal_gradient, risk_gradient)
        objective = aligned + forecast_weight*signal
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite prior objective')
        # Match the before-step dispatch: no_grad would enable a different
        # Transformer inference path and swamp a tiny actual fast-step effect.
        # A fresh leaf preserves dispatch without retaining the outer graph.
        with torch.enable_grad():
            audit_fast = updated.detach().requires_grad_(True)
            after = F.binary_cross_entropy_with_logits(
                model(e['qx'], e['qa'], audit_fast)['logit'], e['y']).detach()
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return dict(objective=float(objective.detach()), signal_loss=float(signal.detach()),
                    alignment_loss=float(aligned.detach()), head_grad_norm=float(norm),
                    risk_before=float(losses.detach().mean()), risk_after=float(after),
                    risk_improvement=float(losses.detach().mean()-after),
                    signal_grad_norm=float(signal_gradient.detach().norm(dim=-1).mean()),
                    risk_grad_norm=float(risk_gradient.detach().norm(dim=-1).mean()),
                    fast_delta_norm=float((updated.detach()-fast.detach()).norm(dim=-1).mean()),
                    direction_count=int((risk_gradient.detach().norm(dim=-1)>1e-8).sum()),
                    fast_step=FAST_STEP, fast_radius=FAST_RADIUS)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        model.train(training)


def build_prior(model, frozen, *, train_ids, parent_sha256, code_sha256,
                steps, batch_size, seed, lr, forecast_weight, episode_sha256):
    assert_classifier_unchanged(model, frozen)
    if parent_sha256 != PARENT_SHA256:
        raise ValueError('prior parent SHA256 is not the locked source')
    head = {k: v.detach().cpu().clone() for k, v in model.forecast_head.state_dict().items()}
    if set(head) != {'weight', 'bias'} or any(not torch.isfinite(v).all() for v in head.values()):
        raise ValueError('exactly two finite forecast_head tensors required')
    protocol = dict(format='redesign-20261004-forecast-prior-v1', source_only=True,
        static_classifier_bitwise_unchanged=True, parent_sha256=parent_sha256,
        train_ids=[int(sid) for sid in train_ids], code_sha256=code_sha256,
        source_model_config=model.cfg.to_dict(), source_label='max(25ms,AdvancedTime)',
        source_qualification_lead_ms=10., target_labels_accessed=0,
        holdout_outcomes_accessed=0, support110_used=False, features='flat23',
        objective='gradient_cosine_alignment + forecast_weight * direct_H25_SmoothL1',
        alignment_weight=1., forecast_weight=forecast_weight,
        fast_coordinates='source fast_origin + zero temporary delta; trained input token site',
        fast_step=FAST_STEP, fast_radius=FAST_RADIUS, fast_rule=model.cfg.fast_rule,
        query_offset_points=model.cfg.horizon+1, steps=steps, batch_size=batch_size,
        episodes_seen=steps*batch_size, seed=seed, lr=lr, weight_decay=0.,
        episode_sha256=episode_sha256, source_threshold_unchanged=True,
        qualified_source_artifact_unchanged=True, evidence='DEVELOPMENT_ONLY; no online gain established')
    return dict(forecast_head=head, protocol=protocol)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--output', type=Path, required=True, help='new directory for prior.pt and training.json')
    parser.add_argument('--steps', type=int, default=128, help='optimizer episodes/batches; 64 or 128 recommended')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--forecast-weight', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=131)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='GPU launch is owned by the main agent')
    return parser


def validate_args(args):
    if min(args.steps, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive finite training budget and learning rate required')
    if not np.isfinite(args.forecast_weight) or not 0 <= args.forecast_weight <= 1:
        raise ValueError('forecast weight must be finite in [0,1]')


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    if args.output.exists():
        raise FileExistsError('do not overwrite an existing prior directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable; no silent CPU fallback')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, _ = load_parent(args.checkpoint)
    frozen = classifier_state(model)
    model.to(args.device)
    shots, train_ids = load_source_train()
    pools = source_episode_pools(shots, model.cfg, train_ids)
    optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=args.lr, weight_decay=0.)
    names = ('simple23.py', 'simple_data.py', 'io23.py', 'fast_meta.py',
             'fast_direction.py', 'fast_update.py', 'simple_online.py')
    code = {'prepare_prior.py': sha256(__file__),
            **{name: sha256(Path(__file__).resolve().parents[1]/name) for name in names}}
    history, digest = [], hashlib.sha256()
    for step in range(args.steps):
        e = fast_meta.sample(shots, pools, model.cfg, rng, args.batch_size, args.device)
        for key in sorted(e):
            digest.update(key.encode())
            digest.update(_tensor_bytes(e[key]))
        log = train_step(model, optimizer, e, forecast_weight=args.forecast_weight)
        history.append(dict(step=step+1, **log))
        if (step+1) % 16 == 0 or step+1 == args.steps:
            print(json.dumps(history[-1], allow_nan=False), flush=True)
    if sha256(args.checkpoint) != PARENT_SHA256:
        raise AssertionError('qualified source checkpoint changed during offline training')
    if any(sha256(Path(__file__) if name == 'prepare_prior.py' else
                  Path(__file__).resolve().parents[1]/name) != value for name, value in code.items()):
        raise AssertionError('training code changed during this run; do not attribute mixed code to one hash')
    artifact = build_prior(model, frozen, train_ids=train_ids, parent_sha256=PARENT_SHA256,
        code_sha256=code, steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        lr=args.lr, forecast_weight=args.forecast_weight, episode_sha256=digest.hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(artifact, args.output/'prior.pt')
    report = dict(protocol=artifact['protocol'], history=history, prior_sha256=sha256(args.output/'prior.pt'))
    (args.output/'training.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(completed=True, prior=str((args.output/'prior.pt').resolve()),
                         prior_sha256=report['prior_sha256'], classifier_bitwise_unchanged=True)), flush=True)
    return report


if __name__ == '__main__':
    main()
