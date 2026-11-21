"""Optional OFFLINE JTEXT TRAIN-only finite-step risk-after forecast prior.

Only the original forecast head is trained. The qualified source classifier,
its token adaptation coordinates, input scales and threshold remain frozen.
No EAST data, target labels, JTEXT online branch or score correction is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import fast_meta
from . import prepare_prior as common


@torch.enable_grad()
def train_step(model, optimizer, e, *, forecast_weight=.1):
    """Optimize finite future BCE through the ACTUAL source normalized step.

    Before is measured with a fresh gradient-enabled fast leaf. After retains
    the inner create_graph=True path into forecast_head; detaching updated
    would silently reduce this to forecast-only training. Both query forwards
    use eval + enabled autograd + fast.requires_grad, matching dispatch.
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
        updated, signal, initial, signal_gradient = common.temporary_step_components(model, e, create_graph=True)
        labels = e['y'].detach()
        # Detaching this BEFORE-only leaf avoids retaining an unused audit
        # graph, but retains the same grad-enabled Transformer dispatch.
        audit_fast = initial.detach().requires_grad_(True)
        before = F.binary_cross_entropy_with_logits(
            model(e['qx'], e['qa'], audit_fast)['logit'], labels, reduction='none').detach()
        # Do NOT detach updated or risk_after: this is the meta-objective.
        risk_after = F.binary_cross_entropy_with_logits(
            model(e['qx'], e['qa'], updated)['logit'], labels, reduction='none')
        objective = risk_after.mean() + forecast_weight*signal
        if not torch.isfinite(before).all() or not torch.isfinite(objective):
            raise FloatingPointError('nonfinite finite-step prior objective or audit')
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        after = risk_after.detach()
        delta = after-before
        # Preserve per-episode outcomes, not just the helpful-batch count.
        # These are source endpoint labels, never safe-shot identifiers/TPR.
        return dict(objective=float(objective.detach()), signal_loss=float(signal.detach()),
                    risk_before=float(before.mean()), risk_after=float(after.mean()),
                    risk_improvement=float(-delta.mean()), head_grad_norm=float(norm),
                    signal_grad_norm=float(signal_gradient.detach().norm(dim=-1).mean()),
                    fast_delta_norm=float((updated.detach()-initial.detach()).norm(dim=-1).mean()),
                    fast_step=common.FAST_STEP, fast_radius=common.FAST_RADIUS,
                    future_labels=labels.cpu().tolist(), risk_before_per_episode=before.cpu().tolist(),
                    risk_after_per_episode=after.cpu().tolist(), risk_delta_per_episode=delta.cpu().tolist())
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        model.train(training)


def build_prior(model, frozen, **metadata):
    """Reuse all parent/bitwise/tensor invariants, replace cosine-only provenance."""
    artifact = common.build_prior(model, frozen, **metadata)
    artifact['protocol'].update(
        format='redesign-20261004-finite-forecast-prior-v1',
        objective='finite_step_BCE + forecast_weight * direct_H25_SmoothL1',
        risk_weight=1., alignment_weight=0., risk_after_graph_retained=True,
        risk_audit_path='eval; autograd enabled; fast requires_grad; before fresh leaf; after undetached',
        finite_step_scope='one unconditional source normalized level-loss step from zero delta; no online acceptance gate',
        qualified_source_artifact_unchanged=True)
    return artifact


def code_paths():
    source = Path(common.__file__).resolve().parents[1]
    names = ('simple23.py', 'simple_data.py', 'io23.py', 'fast_meta.py',
             'fast_direction.py', 'fast_update.py', 'simple_online.py')
    return {'prepare_finite_prior.py': Path(__file__).resolve(),
            'prepare_prior.py': Path(common.__file__).resolve(),
            **{name: source/name for name in names}}


def make_parser():
    parser = common.make_parser()
    parser.description = __doc__
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    common.validate_args(args)
    if args.output.exists():
        raise FileExistsError('do not overwrite an existing prior directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable; no silent CPU fallback')
    paths = code_paths()
    code = {name: common.sha256(path) for name, path in paths.items()}
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, _ = common.load_parent(args.checkpoint)
    frozen = common.classifier_state(model)
    model.to(args.device)
    shots, train_ids = common.load_source_train()
    pools = common.source_episode_pools(shots, model.cfg, train_ids)
    optimizer = torch.optim.AdamW(model.forecast_head.parameters(), lr=args.lr, weight_decay=0.)
    history, digest = [], hashlib.sha256()
    for step in range(args.steps):
        e = fast_meta.sample(shots, pools, model.cfg, rng, args.batch_size, args.device)
        # Identical sampler/hash convention to the cosine prior: the same
        # seed/budget/data gives the same episode digest, independent of head.
        for key in sorted(e):
            digest.update(key.encode())
            digest.update(common._tensor_bytes(e[key]))
        log = train_step(model, optimizer, e, forecast_weight=args.forecast_weight)
        history.append(dict(step=step+1, **log))
        if (step+1) % 16 == 0 or step+1 == args.steps:
            print(json.dumps(history[-1], allow_nan=False), flush=True)
    if common.sha256(args.checkpoint) != common.PARENT_SHA256:
        raise AssertionError('qualified source checkpoint changed during offline training')
    if any(common.sha256(path) != code[name] for name, path in paths.items()):
        raise AssertionError('training code changed during this run; do not attribute mixed code to one hash')
    artifact = build_prior(model, frozen, train_ids=train_ids, parent_sha256=common.PARENT_SHA256,
        code_sha256=code, steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        lr=args.lr, forecast_weight=args.forecast_weight, episode_sha256=digest.hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(artifact, args.output/'prior.pt')
    report = dict(protocol=artifact['protocol'], history=history,
                  prior_sha256=common.sha256(args.output/'prior.pt'))
    (args.output/'training.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(completed=True, prior=str((args.output/'prior.pt').resolve()),
                         prior_sha256=report['prior_sha256'], classifier_bitwise_unchanged=True)), flush=True)
    return report


if __name__ == '__main__':
    main()
