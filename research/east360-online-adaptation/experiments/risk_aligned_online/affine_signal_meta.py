"""Source-only learning of signal-error weights for disposable EAST INTRA.

Original classifier, forecast head and scaler stay frozen. Synthetic changes
are source-coordinate stress tests, not simulated physical disruptions.
Online weights never enter the persistent loss or the final risk score.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import fast_meta, simple_data, simple_train
from .fast_update import normalized_step
from .feature_affine import FeatureAffineModel
from .redesign_20261004.run import DEFAULT_SOURCE, load_parent
from .simple_online import _durable_json, _hash

FORMAT = 'source-affine-signal-weights-v1'


def channel_weights(logits):
    if logits.shape != (23,) or not torch.isfinite(logits).all():
        raise ValueError('finite 23 loss logits required')
    positive = (2 * logits.tanh()).exp()
    # Unit mean removes global-scale ambiguity of a normalized gradient.
    # The positive floor prevents hard deletion; these are numerical bounds.
    return 1 + .9 * (positive / positive.mean() - 1)


def weighted_signal_loss(model, prediction, truth, available, weights):
    if (weights.shape != (23,) or not torch.isfinite(weights).all()
            or not (weights > 0).all() or weights.device != prediction.device
            or weights.dtype != prediction.dtype):
        raise ValueError('positive finite 23 weights in prediction dtype/device required')
    if (prediction.ndim != 3 or prediction.shape[1:] != (25, 23)
            or truth.shape != prediction.shape or available.shape != truth.shape
            or available.dtype != torch.bool):
        raise ValueError('matching direct H25 truth and availability required')
    observed = available & torch.isfinite(truth)
    target = torch.asinh((torch.where(observed, truth.detach(), model.center)
                         - model.center) / model.scale)
    error = F.smooth_l1_loss(prediction, target, reduction='none').masked_fill(~observed, 0.)
    mass = observed.to(error.dtype) * weights[None, None]
    return (error * weights[None, None]).sum((1, 2)) / mass.sum((1, 2)).clamp_min(1e-12)


class WeightedFeatureAffineModel(FeatureAffineModel):
    def __init__(self, base, weights):
        super().__init__(base, forecast_coordinates='observed')
        weights = torch.as_tensor(weights, dtype=base.center.dtype, device=base.center.device)
        if weights.shape != (23,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('positive finite source-learned weights required')
        self.register_buffer('signal_weights', weights.detach().clone())

    def intra_forecast_loss(self, prediction, truth, available):
        return weighted_signal_loss(self, prediction, truth, available, self.signal_weights).mean()


def distort_episode(model, episode, rng):
    """One numerical convention shared by past, actual future and later input.

    Source-only offline randomization: no future-fitted mean/scale, no changed
    labels or masks, and no use of EAST distributions to choose perturbations.
    """
    batch = len(episode['x'])
    gain = np.exp(rng.uniform(-.35, .35, (batch, 23)))
    bias = rng.uniform(-1., 1., (batch, 23))
    clean = rng.random(batch) < .3
    gain[clean], bias[clean] = 1., 0.
    # Evaluate the numerical convention in float64, then round once back to
    # source input dtype. Separate W128 and H25 float32 vector kernels otherwise
    # gave differing roundoff for the same value in the consistency regression.
    g = torch.as_tensor(gain, dtype=torch.float64, device=model.center.device)[:, None]
    b = torch.as_tensor(bias, dtype=torch.float64, device=model.center.device)[:, None]
    center, scale = model.center.double(), model.scale.double()
    moved = dict(episode)
    for name, mask_name in [('x', 'a'), ('qx', 'qa'), ('truth', 'mask')]:
        value, mask = episode[name], episode[mask_name]
        raw = value.double()
        level = torch.asinh((torch.where(mask, raw, center) - center) / scale)
        changed = level * g + b
        mapped = raw + scale * (changed.sinh() - level.sinh())
        moved[name] = torch.where(mask, mapped.to(value.dtype), value).detach()
        if not torch.isfinite(moved[name][mask]).all():
            raise FloatingPointError('nonfinite synthetic source coordinate')
    return moved, dict(gain=gain.tolist(), bias=bias.tolist(), clean=clean.tolist())


@contextmanager
def frozen_model(model):
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        for p, _ in flags: p.requires_grad_(False)
        yield
    finally:
        for p, flag in flags: p.requires_grad_(flag)
        for m, mode in modes: m.training = mode


def adaptation_step(model, weights, episode, *, create_graph):
    if (episode['y'].shape != (len(episode['x']),)
            or not ((episode['y'] == 0.) | (episode['y'] == 1.)).all()
            or not episode['mask'].flatten(1).any(-1).all()):
        raise ValueError('binary source labels and real signal targets per episode required')
    fast = model.initial_fast(len(episode['x'])).detach().requires_grad_(True)
    before = model(episode['x'], episode['a'], fast)
    signal = weighted_signal_loss(model, before['forecast'], episode['truth'], episode['mask'], weights)
    gradient, = torch.autograd.grad(signal.sum(), fast, create_graph=create_graph,
                                    retain_graph=create_graph)
    candidate = normalized_step(fast, torch.zeros_like(fast), gradient, step=.05, radius=1.)
    proposed = model(episode['x'], episode['a'], candidate)
    candidate_loss = weighted_signal_loss(model, proposed['forecast'], episode['truth'], episode['mask'], weights)
    tolerance = 8 * torch.finfo(signal.dtype).eps * signal.detach().abs().clamp_min(1e-12)
    accepted = (signal.detach() - candidate_loss.detach()) > tolerance
    updated = torch.where(accepted[:, None], candidate, fast)
    logits = model(episode['qx'], episode['qa'], updated)['logit']
    risk_after = F.binary_cross_entropy_with_logits(logits, episode['y'], reduction='none')
    # Both risks use the SAME later observed input, not a synthetic forecast
    # classified by a teacher, and not the earlier origin's risk label.
    risk_before = F.binary_cross_entropy_with_logits(
        model(episode['qx'], episode['qa'], fast)['logit'], episode['y'], reduction='none')
    return risk_after, dict(risk_before=risk_before.detach(), risk_after=risk_after.detach(),
                           accepted=accepted.detach(), signal=signal.detach())


def train_step(model, logits, optimizer, episode):
    optimizer.zero_grad(set_to_none=True)
    with frozen_model(model):
        risk, measured = adaptation_step(model, channel_weights(logits), episode, create_graph=True)
        objective = risk.mean()
        if not torch.isfinite(objective): raise FloatingPointError('nonfinite source meta objective')
        objective.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise FloatingPointError('missing or nonfinite signal-weight meta gradient')
    norm = float(logits.grad.detach().double().norm())
    torch.nn.utils.clip_grad_norm_([logits], 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(risk_before=float(measured['risk_before'].mean()),
                risk_after=float(measured['risk_after'].mean()),
                accepted=int(measured['accepted'].sum()), batch_size=len(episode['x']),
                signal_loss=float(measured['signal'].mean()), meta_gradient_norm=norm)


def load_weights(path, *, expected_parent_sha256):
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    protocol, weights = artifact['protocol'], artifact['weights']
    if (protocol.get('format') != FORMAT or protocol.get('parent_sha256') != expected_parent_sha256
            or protocol.get('source_only') is not True
            or protocol.get('original_model_bitwise_unchanged') is not True):
        raise ValueError('signal weights violate source-only parent contract')
    if (not isinstance(weights, torch.Tensor) or weights.shape != (23,)
            or not torch.isfinite(weights).all() or not (weights > 0).all()):
        raise ValueError('positive finite source signal weights required')
    return weights, protocol


def evaluate_episodes(model, weights, episodes):
    records = []
    with frozen_model(model), torch.enable_grad():
        for e in episodes:
            _, measured = adaptation_step(model, weights, e, create_graph=False)
            records.extend(dict(label=int(y), risk_before=float(before), risk_after=float(after), accepted=bool(a))
                           for y, before, after, a in zip(e['y'], measured['risk_before'], measured['risk_after'], measured['accepted']))
    return dict(n=len(records), risk_before=float(np.mean([r['risk_before'] for r in records])),
                risk_after=float(np.mean([r['risk_after'] for r in records])),
                classes={str(y): dict(n=sum(r['label'] == y for r in records),
                    risk_after=float(np.mean([r['risk_after'] for r in records if r['label'] == y]))) for y in (0, 1)}, records=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    if args.output.exists() or min(args.steps, args.batch_size) < 1: p.error('new output and positive source budget required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    wrapper, state, gate, _ = load_parent(DEFAULT_SOURCE, 'cuda', persistent_site='embedding')
    model = FeatureAffineModel(wrapper.base, forecast_coordinates='observed').cuda().eval()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    train = simple_train.load_shots(roles['train'], advanced)
    val = simple_train.load_shots(roles['val'], advanced)
    pools = fast_meta.episode_pools(train, model.cfg); val_pools = fast_meta.episode_pools(val, model.cfg)
    vrng = np.random.default_rng(20261009)
    episodes = [distort_episode(model, fast_meta.sample(val, val_pools, model.cfg, vrng, 16, 'cuda'), vrng)[0] for _ in range(16)]
    reference = evaluate_episodes(model, torch.ones(23, device='cuda'), episodes)
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [root / n for n in ('affine_signal_meta.py', 'feature_affine.py', 'fast_meta.py',
        'fast_direction.py', 'fast_update.py', 'simple23.py', 'simple_data.py', 'simple_train.py', 'io23.py')]
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for path in files: z.write(path, path.name)
    protocol = dict(format=FORMAT, parent_sha256=gate['checkpoint_sha256'], source_only=True,
        source_classifier_training=False, jtext_online=False, east_access=False,
        original_model_bitwise_unchanged=True, steps=args.steps, batch_size=args.batch_size,
        seed=131, optimizer='Adam', learning_rate=.03, selected_step='last fixed step, not best EAST result',
        train_ids=roles['train'], val_ids=roles['val'], test_access=False,
        temporary_parameters=46, learned_loss_parameters=23, inner_step=.05, radius=1., horizon=25,
        query_gap=26, synthetic_gain_log_range=[-.35, .35], synthetic_bias_range=[-1., 1.],
        synthetic_clean_probability=.3, synthetic_units='source asinh coordinates, not physical parameters',
        signal_weights_only_intra=True, threshold=float(state['threshold']),
        code_sha256={path.name: _hash(path) for path in files}, code_snapshot_sha256=_hash(args.output / 'code_snapshot.zip'),
        scope='SOURCE DEVELOPMENT; task-loss screening, not shot-level EAST evidence')
    _durable_json(args.output / 'protocol.json', protocol)
    logits = torch.nn.Parameter(torch.zeros(23, device='cuda')); optimizer = torch.optim.Adam([logits], lr=.03)
    rng = np.random.default_rng(131); digest = hashlib.sha256(); history = []
    for index in range(args.steps):
        e, _ = distort_episode(model, fast_meta.sample(train, pools, model.cfg, rng, args.batch_size, 'cuda'), rng)
        for key in sorted(e): digest.update(key.encode()); digest.update(e[key].detach().cpu().numpy().tobytes())
        record = train_step(model, logits, optimizer, e); history.append(record)
        if (index + 1) % 32 == 0:
            _durable_json(args.output / 'training.json', dict(history=history, episode_sha256=digest.hexdigest()))
            print(json.dumps(dict(step=index + 1, **record)), flush=True)
    weights = channel_weights(logits).detach(); learned = evaluate_episodes(model, weights, episodes)
    if any(not torch.equal(model.state_dict()[k], v) for k, v in original.items()): raise RuntimeError('source or persistent affine changed')
    if _hash(DEFAULT_SOURCE) != gate['checkpoint_sha256']: raise RuntimeError('source checkpoint changed')
    protocol['episode_sha256'] = digest.hexdigest()
    torch.save(dict(weights=weights.cpu(), logits=logits.detach().cpu(), protocol=protocol), args.output / 'weights.pt')
    screening = dict(uniform=reference, learned=learned, learned_beats_uniform=learned['risk_after'] < reference['risk_after'],
                     weights=dict(zip(simple_data.io23.RAW23, weights.cpu().tolist())))
    _durable_json(args.output / 'validation.json', screening); _durable_json(args.output / 'protocol.json', protocol)
    _durable_json(args.output / 'training.json', dict(history=history, episode_sha256=digest.hexdigest()))
    print(json.dumps(dict(completed=True, uniform_risk=reference['risk_after'], learned_risk=learned['risk_after'],
                          source_screening_passed=screening['learned_beats_uniform'], weights=screening['weights'])), flush=True)


if __name__ == '__main__': main()
