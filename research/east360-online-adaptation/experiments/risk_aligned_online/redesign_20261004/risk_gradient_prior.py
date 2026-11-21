"""Read-only source gradient diagnostic; opt-in offline forecast-head trainer.

Independent internal-redesign loss iteration, NOT V3C or JTEXT online.
The qualified classifier is immutable. No EAST or source holdout outcomes.
Default CLI execution only measures gradients and prints audits; --train is
required to optimize and create a new riskprior.pt. GPU scheduling is main-owned.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import fast_meta
from . import prepare_prior as common


SIGNAL_WEIGHT = .1
ALPHA_EPS = 1e-12
ALPHA_MIN, ALPHA_MAX = 1., 10000.
EXPECTED_EPISODE_SHA256 = 'bed9e7906427c7c7df194b731c35c311037dde0b62580f571c8d181972110813'


def _validate_weight(forecast_weight):
    if not np.isfinite(forecast_weight) or forecast_weight != SIGNAL_WEIGHT:
        raise ValueError('this bounded objective requires forecast maintenance weight 0.1')


@contextmanager
def _head_scope(model):
    """Autograd scope, not an update: preserve flags, all modes, and .grad."""
    head = tuple(model.forecast_head.parameters())
    ids = {id(p) for p in head}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in ids)
        yield head
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def gradient_balance(risk_gradients, signal_gradients):
    """Detached norms of TWO forecast-head gradients, not fast-direction cosine.

    Float64 audit reductions avoid float32 norm overflow/underflow. Exact zero
    is reported as a finite rejection; it is not made into a unit direction.
    The 1e-12 is only the specified alpha denominator, not a new update epsilon.
    """
    if not risk_gradients or len(risk_gradients) != len(signal_gradients):
        raise ValueError('matching nonempty forecast-head gradient tuples required')
    if any(g is None for g in (*risk_gradients, *signal_gradients)):
        raise FloatingPointError('missing forecast-head gradient; risk-after graph must remain connected')
    if any(r.shape != s.shape for r, s in zip(risk_gradients, signal_gradients)):
        raise ValueError('matching forecast-head gradient shapes required')
    if any(not torch.isfinite(g).all() for g in (*risk_gradients, *signal_gradients)):
        raise FloatingPointError('nonfinite forecast-head gradient')
    r = torch.cat([g.detach().reshape(-1).double() for g in risk_gradients])
    s = torch.cat([g.detach().reshape(-1).double() for g in signal_gradients])
    rn, sn = float(r.norm()), float(s.norm())
    ratio = SIGNAL_WEIGHT*sn/(rn+ALPHA_EPS)
    if not all(math.isfinite(v) for v in (rn, sn, ratio)):
        raise FloatingPointError('nonfinite gradient norm or balancing ratio')
    alpha = min(ALPHA_MAX, max(ALPHA_MIN, ratio))  # A detached Python scalar.
    reason = 'zero_risk_head_gradient' if rn == 0. else 'zero_signal_head_gradient' if sn == 0. else None
    cosine = None if reason else float(((r/rn)*(s/sn)).sum().clamp(-1., 1.))
    return dict(risk_head_grad_norm=rn, signal_head_grad_norm=sn,
        maintenance_head_grad_norm=SIGNAL_WEIGHT*sn,
        maintenance_to_risk_head_ratio=ratio,
        head_gradient_cosine=cosine, head_gradient_conflict=None if cosine is None else cosine < 0.,
        alpha=alpha, alpha_clipped=ratio < ALPHA_MIN or ratio > ALPHA_MAX,
        balanced_risk_head_grad_norm=alpha*rn,
        gradient_status='ready' if reason is None else 'rejected', rejection_reason=reason)


def _losses_and_audit(model, e, head):
    labels = e['y'].detach()
    if not torch.isfinite(labels).all():
        raise FloatingPointError('nonfinite source endpoint label')
    if labels.shape != (len(e['x']),) or not ((labels == 0.) | (labels == 1.)).all():
        raise ValueError('one binary source endpoint label per episode required')
    mask = e['mask'] & torch.isfinite(e['truth'])
    if not mask.flatten(1).any(-1).all():
        raise ValueError('each source episode needs an observed forecast target')
    updated, signal, initial, inner = common.temporary_step_components(model, e, create_graph=True)
    # Same eval/autograd/fast-requires-grad dispatch on both sides. no_grad
    # would change Transformer dispatch and can swamp the small actual effect.
    before = F.binary_cross_entropy_with_logits(
        model(e['qx'], e['qa'], initial.detach().requires_grad_(True))['logit'],
        labels, reduction='none').detach()
    after = F.binary_cross_entropy_with_logits(
        model(e['qx'], e['qa'], updated)['logit'], labels, reduction='none')
    if any(not torch.isfinite(v).all() for v in (before, after, signal, inner, updated)):
        raise FloatingPointError('nonfinite actual finite-step risk or signal audit')
    risk = after.mean()
    # These audits must NOT consume the graph needed by objective.backward().
    # allow_unused permits an explicit fail-closed message for a severed path.
    gr = torch.autograd.grad(risk, head, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(signal, head, retain_graph=True, allow_unused=True)
    audit = gradient_balance(gr, gs)
    inner_norms = inner.detach().double().norm(dim=-1)
    if (inner_norms == 0.).any():
        audit.update(gradient_status='rejected', rejection_reason='zero_inner_signal_gradient')
    delta = after.detach()-before
    audit.update(signal_loss=float(signal.detach()), risk_before=float(before.mean()),
        risk_after=float(risk.detach()), risk_improvement=float(-delta.mean()),
        signal_fast_grad_norm=float(inner_norms.mean()),
        signal_fast_grad_norm_per_episode=inner_norms.cpu().tolist(),
        fast_delta_norm=float((updated.detach()-initial.detach()).double().norm(dim=-1).mean()),
        fast_step=common.FAST_STEP, fast_radius=common.FAST_RADIUS,
        future_labels=labels.cpu().tolist(), risk_before_per_episode=before.cpu().tolist(),
        risk_after_per_episode=after.detach().cpu().tolist(), risk_delta_per_episode=delta.cpu().tolist(),
        alpha_detached=True, optimizer_step_applied=False)
    return risk, signal, audit


@torch.enable_grad()
def diagnostic_step(model, e, *, forecast_weight=SIGNAL_WEIGHT):
    """Read-only on the SAME source episode: actual risk-after vs maintenance.

    No optimizer, backward(), .grad clearing, parameter/buffer writes, or data
    loading. Existing gradients and mixed submodule modes are preserved.
    Zero gradients produce an explicit rejection record; nonfinite raises.
    """
    _validate_weight(forecast_weight)
    with _head_scope(model) as head:
        _, _, audit = _losses_and_audit(model, e, head)
        return audit


@torch.enable_grad()
def train_step(model, optimizer, e, *, forecast_weight=SIGNAL_WEIGHT):
    """alpha * actual risk-after + .1 signal, forecast_head ONLY.

    alpha=clip(.1*||g_signal_head||/(||g_risk_head||+1e-12),1,10000).
    Both gradients use source TRAIN and the current head before this optimizer
    step. alpha is detached, never learned. Keep updated and risk undetached.
    """
    _validate_weight(forecast_weight)
    ids = {id(p) for p in model.forecast_head.parameters()}
    optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(optimized) != ids or len(optimized) != len(ids):
        raise ValueError('optimizer must contain exactly forecast_head weight and bias')
    with _head_scope(model) as head:
        risk, signal, audit = _losses_and_audit(model, e, head)
        if audit['gradient_status'] != 'ready':
            raise FloatingPointError(f"risk prior fails closed: {audit['rejection_reason']}")
        objective = audit['alpha']*risk + SIGNAL_WEIGHT*signal
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite gradient-balanced objective')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head):
            raise FloatingPointError('nonfinite or missing balanced forecast-head gradient')
        norm = torch.nn.utils.clip_grad_norm_(head, 1., error_if_nonfinite=True)
        if float(norm) == 0.:
            raise FloatingPointError('zero balanced forecast-head gradient; no optimizer step')
        optimizer.step()
        return dict(audit, objective=float(objective.detach()), head_grad_norm=float(norm),
                    optimizer_step_applied=True)


def build_prior(model, frozen, *, mode='train', **metadata):
    """Reuse parent, tensor and byte invariants; replace the old objective label."""
    _validate_weight(metadata['forecast_weight'])
    if mode not in ('train', 'diagnostic'):
        raise ValueError('unknown source prior mode')
    artifact = common.build_prior(model, frozen, **metadata)
    artifact['protocol'].update(
        format='redesign-20261004-risk-gradient-forecast-prior-v1',
        lineage='INDEPENDENT_INTERNAL_REDESIGN_NOT_V3C',
        mode='source_only_training' if mode == 'train' else 'read_only_gradient_diagnostic',
        objective='detached_alpha * actual_post_fast_step_future_BCE + 0.1 * direct_H25_SmoothL1',
        risk_weight='detached_per_batch_alpha', alignment_weight=0., forecast_weight=SIGNAL_WEIGHT,
        alpha_rule='clip(0.1 * norm(g_signal_head) / (norm(g_risk_head) + 1e-12), 1, 10000)',
        alpha_bounds=[ALPHA_MIN, ALPHA_MAX], alpha_epsilon=ALPHA_EPS,
        alpha_detached=True, alpha_meta_learned=False, alpha_data='same source TRAIN batch only',
        gradient_cosine_definition='cos(d(actual_post_fast_step_future_BCE)/d(forecast_head), d(signal_loss)/d(forecast_head))',
        gradient_reduction='mean losses; concatenated weight+bias; detached float64 audit norms',
        risk_after_graph_retained=True, forecast_head_only=True,
        risk_audit_path='eval; autograd enabled; before fresh fast leaf; after undetached actual normalized step',
        finite_step_scope='one unconditional source normalized level-loss step; no online acceptance gate',
        zero_gradient_policy='diagnostic records rejection; trainer aborts before optimizer.step; no epsilon direction fallback',
        nonfinite_gradient_policy='abort; no artifact emitted',
        optimizer_steps=metadata['steps'] if mode == 'train' else 0,
        optimizer='AdamW; lr from CLI; weight_decay=0; outer clip norm=1' if mode == 'train' else None,
        jtext_online_branch=False, target_data_accessed=0,
        expected_full_budget_episode_sha256=EXPECTED_EPISODE_SHA256,
        qualified_source_artifact_unchanged=True,
        evidence='SOURCE_TRAIN_DEVELOPMENT_ONLY; no EAST benefit established')
    return artifact


def code_paths():
    source = Path(common.__file__).resolve().parents[1]
    names = ('simple23.py', 'simple_data.py', 'io23.py', 'fast_meta.py',
             'fast_direction.py', 'fast_update.py', 'simple_online.py')
    return {'risk_gradient_prior.py': Path(__file__).resolve(),
            'prepare_prior.py': Path(common.__file__).resolve(),
            **{name: source/name for name in names}}


def load_fixed_prior(model, path):
    """Diagnostic-only head selection. Never load another source classifier.

    Tensor-only deserialization has no unsafe pickle fallback. Validate the
    full head before loading, and hash before/after to reject file drift.
    """
    digest = common.sha256(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {'forecast_head', 'protocol'}:
        raise ValueError('fixed prior must contain only forecast_head and protocol')
    protocol, head = saved['protocol'], saved['forecast_head']
    if (not isinstance(protocol, dict)
            or protocol.get('parent_sha256') != common.PARENT_SHA256
            or protocol.get('source_only') is not True
            or protocol.get('static_classifier_bitwise_unchanged') is not True
            or protocol.get('source_model_config') != model.cfg.to_dict()
            or protocol.get('source_label') != 'max(25ms,AdvancedTime)'
            or protocol.get('target_labels_accessed') != 0
            or protocol.get('holdout_outcomes_accessed') != 0
            or protocol.get('support110_used') is not False
            or protocol.get('jtext_online_branch', False) is not False):
        raise ValueError('fixed prior violates qualified source-only provenance')
    expected = model.forecast_head.state_dict()
    if (not isinstance(head, dict) or set(head) != set(expected)
            or any(not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                   or v.dtype != expected[k].dtype or not torch.isfinite(v).all()
                   for k, v in head.items())):
        raise ValueError('fixed prior requires exactly two finite matching head tensors')
    if common.sha256(path) != digest:
        raise AssertionError('fixed prior changed while loading')
    model.forecast_head.load_state_dict(head, strict=True)
    return dict(sha256=digest, protocol=protocol)


def make_parser():
    parser = common.make_parser()
    parser.description = __doc__
    parser.add_argument('--train', action='store_true',
        help='opt in to offline training; otherwise read-only diagnostic on stdout, no output directory')
    parser.add_argument('--prior', type=Path,
        help='diagnostic ONLY: fixed saved forecast head on the immutable qualified source')
    for action in parser._actions:
        if action.dest == 'output':
            action.help = 'new riskprior.pt/training.json directory with --train; unused legacy argument for diagnostic'
    return parser


def validate_args(args):
    common.validate_args(args)
    _validate_weight(args.forecast_weight)
    if args.steps > 128 or args.batch_size > 8 or args.seed != 131:
        raise ValueError('bounded iteration: at most 128 batches of 8, fixed seed 131')
    if args.train and args.prior is not None:
        raise ValueError('--prior is diagnostic-only; training starts from the qualified source head')


def validate_episode_hash(digest, *, steps, batch_size):
    if steps == 128 and batch_size == 8 and digest != EXPECTED_EPISODE_SHA256:
        raise AssertionError('source episodes changed: full 128x8 digest must match both frozen priors')


def summarize(history):
    """Source TRAIN mechanism evidence, never shot-level or target-domain gain."""
    ratios = [row['maintenance_to_risk_head_ratio'] for row in history]
    cosines = [row['head_gradient_cosine'] for row in history if row['head_gradient_cosine'] is not None]
    deltas = [v for row in history for v in row['risk_delta_per_episode']]
    return dict(batches=len(history), episodes=len(deltas),
        rejected_batches=sum(row['gradient_status'] != 'ready' for row in history),
        maintenance_dominant_batches=sum(r > 1. for r in ratios),
        median_maintenance_to_risk_head_ratio=float(np.median(ratios)),
        min_maintenance_to_risk_head_ratio=min(ratios), max_maintenance_to_risk_head_ratio=max(ratios),
        defined_cosine_batches=len(cosines), conflicting_batches=sum(c < 0. for c in cosines),
        mean_head_gradient_cosine=float(np.mean(cosines)) if cosines else None,
        min_alpha=min(row['alpha'] for row in history), max_alpha=max(row['alpha'] for row in history),
        mean_source_actual_fast_step_BCE_delta=float(np.mean(deltas)),
        evidence='source TRAIN mechanism diagnostic, not proof of failure causality or EAST benefit')


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    if args.train and args.output.exists():
        raise FileExistsError('do not overwrite an existing prior directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA unavailable; no silent CPU fallback')
    paths = code_paths()
    code = {name: common.sha256(path) for name, path in paths.items()}
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, _ = common.load_parent(args.checkpoint)
    frozen = common.classifier_state(model)
    fixed_prior = load_fixed_prior(model, args.prior) if args.prior is not None else None
    original_head = {k: v.detach().cpu().clone() for k, v in model.forecast_head.state_dict().items()}
    model.to(args.device)
    shots, train_ids = common.load_source_train()
    if fixed_prior is not None and fixed_prior['protocol'].get('train_ids') != train_ids:
        raise ValueError('fixed prior TRAIN identities differ from canonical source TRAIN')
    pools = common.source_episode_pools(shots, model.cfg, train_ids)
    optimizer = (torch.optim.AdamW(model.forecast_head.parameters(), lr=args.lr, weight_decay=0.)
                 if args.train else None)
    history, digest = [], hashlib.sha256()
    for step in range(args.steps):
        e = fast_meta.sample(shots, pools, model.cfg, rng, args.batch_size, args.device)
        # IDENTICAL draws/hash convention to both frozen priors. The diagnostic
        # does not consume an extra draw before training or perturb the RNG.
        for key in sorted(e):
            digest.update(key.encode())
            digest.update(common._tensor_bytes(e[key]))
        log = (train_step(model, optimizer, e, forecast_weight=args.forecast_weight) if args.train
               else diagnostic_step(model, e, forecast_weight=args.forecast_weight))
        history.append(dict(step=step+1, **log))
        if (step+1) % 16 == 0 or step+1 == args.steps:
            print(json.dumps(history[-1], allow_nan=False), flush=True)
    episode_hash = digest.hexdigest()
    validate_episode_hash(episode_hash, steps=args.steps, batch_size=args.batch_size)
    if common.sha256(args.checkpoint) != common.PARENT_SHA256:
        raise AssertionError('qualified source checkpoint changed during run')
    if fixed_prior is not None and common.sha256(args.prior) != fixed_prior['sha256']:
        raise AssertionError('fixed prior checkpoint changed during diagnostic')
    if any(common.sha256(path) != code[name] for name, path in paths.items()):
        raise AssertionError('code changed during run; mixed-code provenance rejected')
    if not args.train:
        for name, tensor in model.forecast_head.state_dict().items():
            if common._tensor_bytes(tensor) != common._tensor_bytes(original_head[name]):
                raise AssertionError('read-only diagnostic changed forecast_head')
    artifact = build_prior(model, frozen, mode='train' if args.train else 'diagnostic',
        train_ids=train_ids, parent_sha256=common.PARENT_SHA256, code_sha256=code,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed, lr=args.lr,
        forecast_weight=args.forecast_weight, episode_sha256=episode_hash)
    artifact['protocol'].update(
        fixed_head_prior_sha256=fixed_prior['sha256'] if fixed_prior is not None else None,
        audit_scope='changing head before each optimizer step' if args.train
                    else 'fixed saved head; not changing training-time audits')
    report = dict(protocol=artifact['protocol'], history=history, summary=summarize(history))
    if args.train:
        args.output.mkdir(parents=True, exist_ok=False)
        torch.save(artifact, args.output/'riskprior.pt')
        report['prior_sha256'] = common.sha256(args.output/'riskprior.pt')
        (args.output/'training.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    completion = dict(completed=True, training=args.train,
        artifact_written=args.train, classifier_bitwise_unchanged=True,
        episode_sha256=episode_hash, summary=report['summary'])
    if not args.train:
        completion.update(protocol=report['protocol'], history=history)
    print(json.dumps(completion, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    main()
