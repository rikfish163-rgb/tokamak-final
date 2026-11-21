"""EAST-only experiment: adapt an existing temporal block, not risk scores.

The qualified J-TEXT checkpoint is never overwritten. Unlabelled H25 targets
update only the last Transformer block's two LayerNorm affine transforms.
This TTA location is inspired by TENT (arXiv:2006.10726); the objective here is
observed diagnostic forecasting, NOT entropy minimisation or a fusion law.
Every shot starts from persistent weights and discards its temporary weights.
Post-shot labels may update the same internal parameters only after sealing.
All results are development diagnostics, not independent future validation.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data, simple_train
from .simple23 import Config, Simple23
from .simple_online import _durable_json, _hash, _examples, build_order, seal_predictions
from .source_gate import qualify_source

# Observed diagnostics with documented proxy meanings. These are auxiliary
# targets only: all flat23 input fields remain in the original classifier.
# They are not density/q95/temperature or confirmed disruption mechanisms.
FORECAST_CHANNELS = tuple(io23.RAW23.index(c) for c in
                          ('ip', 'dx_a', 'dy_a', 'Mir_avg_amp', 'sxr_mean', 'xuv_var'))
ARMS = ('raw', 'contract', 'stats', 'intra', 'inter', 'both')


class PrefixDataView:
    """Optional numerical alignment, separately attributable from gradients.

    Reference is exactly the first 100 recorded points, available only from
    endpoint99. It never expands into the later trajectory. Constant or
    insufficient channels remain unchanged. The mapping can remove absolute
    level information; it is an explicit diagnostic arm, not physical units.
    """
    def __init__(self, center, scale, *, channels, data_api=simple_data):
        self.center, self.scale, self.channels = center, scale, channels
        self.base = data_api
        self.cache = {}

    def _alignment(self, s, end):
        if end < 99:
            return None
        if s['shot'] not in self.cache:
            from .causal_input import PrefixAlignment
            alignment = PrefixAlignment(self.center, self.scale, channels=self.channels, min_points=100)
            alignment.fit_prefix(s['values'][:100], s['available'][:100])
            self.cache[s['shot']] = alignment
        return self.cache[s['shot']]

    def window(self, s, end, length):
        x, mask = self.base.window(s, end, length)
        alignment = self._alignment(s, end)
        return (alignment.transform(x, mask)[0] if alignment else x), mask

    def forecast_truth(self, s, origin, horizon):
        truth = s['values'][origin+1:origin+1+horizon]
        mask = s['available'][origin+1:origin+1+horizon]
        alignment = self._alignment(s, origin)
        return alignment.transform(truth, mask)[0] if alignment else truth

    def read_shot(self, device, sid):
        return self.base.read_shot(device, sid)

    def attach_outcome(self, s, device, advanced):
        return self.base.attach_outcome(s, device, advanced)


def adaptation_names(model):
    prefix = f'encoder.layers.{len(model.encoder.layers)-1}.'
    return tuple(n for n, _ in model.named_parameters()
                 if n.startswith((prefix+'norm1.', prefix+'norm2.')))


def _scope(model):
    names = adaptation_names(model)
    params = dict(model.named_parameters())
    if len(names) != 4:
        raise ValueError('two affine LayerNorm modules required in last block')
    return names, [params[n] for n in names]


def forecast_objective(model, output, truth, mask):
    selected = torch.zeros_like(mask)
    selected[..., FORECAST_CHANNELS] = mask[..., FORECAST_CHANNELS]
    return model.forecast_loss(output['forecast'], truth, selected)


def _restore(params, saved):
    with torch.no_grad():
        for p, old in zip(params, saved):
            p.copy_(old)


def _move(params, gradients, origin, *, lr, radius):
    norm = torch.sqrt(sum(g.square().sum() for g in gradients))
    if not torch.isfinite(norm):
        raise FloatingPointError('nonfinite internal gradient')
    multiplier = (1./norm.clamp_min(1e-12)).clamp(max=1.)
    with torch.no_grad():
        for p, g in zip(params, gradients):
            p.add_(g, alpha=-lr*float(multiplier))
        offset = torch.sqrt(sum((p-old).square().sum() for p, old in zip(params, origin)))
        if not torch.isfinite(offset):
            raise FloatingPointError('nonfinite internal candidate')
        if offset > radius:
            for p, old in zip(params, origin):
                p.copy_(old+(p-old)*(radius/offset))


def _windows(shot, ends, model, device, data_api):
    pairs = [data_api.window(shot, end, model.cfg.window) for end in ends]
    return (torch.as_tensor(np.stack([v for v, _ in pairs]), device=device),
            torch.as_tensor(np.stack([m for _, m in pairs]), device=device))


def predict_shot(model, shot, *, intra, device='cpu', stride=100, lr=.01,
                 radius=.25, data_api=simple_data):
    """Emit risk first; update after a direct 25-point forecast has matured.

    Update schedule is fixed, not selected using actual future error types.
    Accepting a lower diagnostic loss is not a warning-performance guarantee.
    The whole shot is present on disk for replay, but windows/targets are
    explicitly sliced at their simulated arrival boundaries. No outcome read.
    """
    if shot['device'] != 'EAST':
        raise ValueError('online adaptation is restricted to EAST')
    if (not np.isfinite([lr, radius]).all() or lr <= 0 or radius <= 0
            or stride <= model.cfg.horizon):
        raise ValueError('positive lr/radius and stride > forecast horizon required')
    names, params = _scope(model)
    origin_weights = [p.detach().clone() for p in params]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    chosen = {id(p) for p in params}
    scores = np.empty(len(shot['values']), np.float32)
    audit = dict(attempted_updates=0, accepted_updates=0, rejected_updates=0,
                 max_parameter_change=0., adaptation_parameter_count=sum(p.numel() for p in params),
                 adapted_names=list(names), update_events=[])
    model.eval()
    try:
        for p, _ in flags:
            p.requires_grad_(intra and id(p) in chosen)

        def emit(start, stop):
            if start == stop:
                return  # Empty remainder when stride == horizon + 1.
            x, m = _windows(shot, range(start, stop), model, device, data_api)
            with torch.no_grad():
                prob = model(x, m)['logit'].sigmoid()
            if not torch.isfinite(prob).all():
                raise FloatingPointError('nonfinite emitted risk')
            scores[start:stop] = prob.cpu().numpy()

        for start in range(0, len(scores), stride if intra else 256):
            stop = min(len(scores), start+(stride if intra else 256))
            if not intra or start+model.cfg.horizon >= stop:
                emit(start, stop)
                continue
            maturity = start+model.cfg.horizon
            emit(start, maturity+1)
            target_mask = shot['available'][start+1:maturity+1]
            if target_mask[:, FORECAST_CHANNELS].any():
                x, mask = _windows(shot, [start], model, device, data_api)
                future = (data_api.forecast_truth(shot, start, model.cfg.horizon)
                          if hasattr(data_api, 'forecast_truth')
                          else shot['values'][start+1:maturity+1])
                truth = torch.as_tensor(future[None], device=device)
                observed = torch.as_tensor(target_mask[None], device=device)
                before = [p.detach().clone() for p in params]
                audit['attempted_updates'] += 1
                loss = forecast_objective(model, model(x, mask), truth, observed)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite matured forecast loss')
                gradients = torch.autograd.grad(loss, params)
                _move(params, gradients, origin_weights, lr=lr, radius=radius)
                with torch.no_grad():
                    after = forecast_objective(model, model(x, mask), truth, observed)
                if not torch.isfinite(after):
                    raise FloatingPointError('nonfinite updated forecast loss')
                accepted = bool(after <= loss.detach())
                if not accepted:
                    _restore(params, before)
                audit['accepted_updates' if accepted else 'rejected_updates'] += 1
                change = float(torch.sqrt(sum((p.detach()-old).square().sum()
                                              for p, old in zip(params, origin_weights))))
                audit['max_parameter_change'] = max(audit['max_parameter_change'], change)
                audit['update_events'].append(dict(origin=start, matured_at=maturity,
                    first_affected_endpoint=maturity+1, loss_before=float(loss.detach()),
                    loss_after=float(after), accepted=accepted, parameter_change=change))
            emit(maturity+1, stop)
    finally:
        # Also restore on numerical failure, not only on normal shot completion.
        _restore(params, origin_weights)
        for p, enabled in flags:
            p.requires_grad_(enabled)
        for module, training in modes:
            module.training = training
    return scores, audit


def postshot_update(model, shot, replay, *, device, rng, lr=.01, radius=.05,
                    data_api=simple_data):
    """Supervised internal step after natural reveal; guard on prior labels.

    Guard is historical balanced-window BCE, not a guaranteed FPR bound.
    Neither the risk head nor the forecast head nor source scaler is updated.
    """
    if shot['device'] != 'EAST' or not np.isfinite([lr, radius]).all() or min(lr, radius) <= 0:
        raise ValueError('EAST revealed shot and positive update bounds required')
    _, params = _scope(model)
    saved = [p.detach().clone() for p in params]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    chosen = {id(p) for p in params}
    history = list(replay)[-4:]
    training = _examples([shot, *history], rng, 64, model.cfg, device, data_api)
    guard = _examples(history, rng, 64, model.cfg, device, data_api) if history else None
    def loss_on(batch):
        x, m, y = batch
        return F.binary_cross_entropy_with_logits(model(x, m)['logit'], y)
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in chosen)
        model.eval()
        with torch.no_grad():
            guard_before = float(loss_on(guard)) if guard else None
        loss = loss_on(training)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite supervised loss')
        _move(params, torch.autograd.grad(loss, params), saved, lr=lr, radius=radius)
        with torch.no_grad():
            guard_after = float(loss_on(guard)) if guard else None
        accepted = guard_after is None or guard_after <= guard_before+1e-4
        if guard_after is not None and not np.isfinite(guard_after):
            raise FloatingPointError('nonfinite historical guard')
        change = float(torch.sqrt(sum((p.detach()-old).square().sum() for p, old in zip(params, saved))))
        if not accepted:
            _restore(params, saved)
        return dict(accepted=accepted, guard_before=guard_before, guard_after=guard_after,
                    guard_metric='balanced_window_BCE_not_shot_safety', parameter_change=change if accepted else 0.)
    except Exception:
        _restore(params, saved)
        raise
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)
        for module, training in modes:
            module.training = training


def run_arm(model, *, order, arm, threshold, output_dir, limit=0, device='cuda',
            stride=100, lr=.01, radius=.25, inter_lr=.01, data_api=simple_data):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    from .causal_input import apply_input_contract
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = order[:limit] if limit else order
    replay = deque(maxlen=4)
    rows, audits = [], []
    rng = np.random.default_rng(131)
    for index, item in enumerate(selected, 1):
        s = data_api.read_shot('EAST', int(item['shot']))
        if arm != 'raw':
            s = apply_input_contract(s)
        scores, audit = predict_shot(model, s, intra=arm in ('intra', 'both'),
            device=device, stride=stride, lr=lr, radius=radius, data_api=data_api)
        seal_predictions(output_dir, index, s['shot'], s['times'], scores,
                         arm=arm, threshold=threshold)
        # Natural post-shot outcome reveal: no query-cost or active-query claim.
        revealed = data_api.attach_outcome(s, 'EAST', {})
        valid = revealed['valid'].copy()
        if revealed['is_disrupt']:
            valid &= s['times'] <= revealed['down']
        rows.append(dict(shot=s['shot'], is_disrupt=revealed['is_disrupt'], down=revealed['down'],
                         times=s['times'][valid], scores=scores[valid]))
        m = simple_train.metrics(rows, threshold, lead=30.)
        persistent = (postshot_update(model, revealed, replay, device=device, rng=rng, lr=inter_lr,
                                     data_api=data_api) if arm in ('inter', 'both') else None)
        audit.update(shot=s['shot'], prediction=m['predictions'][-1], persistent=persistent)
        audits.append(audit)
        replay.append(revealed)
        _durable_json(output_dir/f'{index:04d}_{s["shot"]}_audit.json', audit)
        if index % 5 == 0 or index == len(selected):
            print(f'{arm} {index}/{len(selected)} TP={m["tp"]}/{m["n_disrupt"]} '
                  f'FP={m["fp"]}/{m["n_safe"]} internal_updates='
                  f'{sum(a["accepted_updates"] for a in audits)}', flush=True)
    result = dict(arm=arm, metrics=simple_train.metrics(rows, threshold, lead=30.),
                  observed_count=len(selected), roster_count=len(order),
                  accepted_intra_updates=sum(a['accepted_updates'] for a in audits),
                  accepted_inter_updates=sum(bool(a['persistent'] and a['persistent']['accepted']) for a in audits),
                  evidence='DEVELOPMENT_PREFIX' if len(selected) != len(order) else 'DEVELOPMENT_FULL360')
    _durable_json(output_dir/'summary.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--order', type=Path)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['contract', 'intra', 'inter', 'both'])
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--stride', type=int, default=100)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--radius', type=float, default=.25)
    p.add_argument('--inter-lr', type=float, default=.01)
    p.add_argument('--align-channels', nargs='*', choices=io23.RAW23, default=[])
    args = p.parse_args(argv)
    if not 0 <= args.limit <= 360 or len(set(args.arms)) != len(args.arms):
        p.error('limit 0..360 and unique arms required')
    if 'stats' in args.arms and not args.align_channels:
        p.error('stats requires explicitly selected --align-channels')
    torch.set_num_threads(4)
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError('source has not qualified')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not ckpt.get('forecast_trained'):
        raise ValueError('trained direct forecast head required')
    model = Simple23(Config(**ckpt['model_config']), ckpt['source_center'], ckpt['source_scale'])
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(args.device).eval()
    ids = io23.cohort_ids('EAST')
    order = json.loads(args.order.read_text()) if args.order else build_order('EAST', ids)
    if len(order) != 360 or {int(r['shot']) for r in order} != set(ids):
        raise ValueError('locked full EAST360 manifest required before selecting prefix')
    args.output.mkdir(parents=True, exist_ok=False)
    _durable_json(args.output/'order.json', order)
    _durable_json(args.output/'protocol.json', dict(
        source_gate=gate, checkpoint=str(args.checkpoint.resolve()), model_config=ckpt['model_config'],
        threshold=float(ckpt['threshold']), threshold_origin='fixed J-TEXT VAL',
        lead_rule='strict >30ms', input_contract='CIII -10 +/-0.01 mask candidate; no semantic repair of unknown units',
        auxiliary_channels=[io23.RAW23[c] for c in FORECAST_CHANNELS],
        adapted_names=list(adaptation_names(model)), stride=args.stride, lr=args.lr, radius=args.radius,
        inter_lr=args.inter_lr, label_access='automatic post-shot reveal after durable seal',
        numerical_alignment_channels=args.align_channels, numerical_alignment_prefix_points=100,
        numerical_alignment_warning='not physical calibration; may erase absolute-level information',
        interpretation='development experiment; no support110; no new risk head; no alarm adjustment',
        source_sha256=_hash(args.checkpoint), code_sha256={name: _hash(Path(__file__).with_name(name)) for name in
            ('representation_online.py', 'causal_input.py', 'simple23.py', 'simple_data.py', 'io23.py')}))
    results = {}
    for arm in args.arms:
        view = (PrefixDataView(ckpt['source_center'], ckpt['source_scale'], channels=args.align_channels)
                if args.align_channels and arm not in ('raw', 'contract') else simple_data)
        results[arm] = run_arm(deepcopy(model), order=order, arm=arm, threshold=float(ckpt['threshold']),
            output_dir=args.output/arm, limit=args.limit, device=args.device, stride=args.stride,
            lr=args.lr, radius=args.radius, inter_lr=args.inter_lr, data_api=view)
        _durable_json(args.output/'summary.json', results)
    if _hash(args.checkpoint) != gate['checkpoint_sha256']:
        raise RuntimeError('qualified source checkpoint changed during experiment')
    return results


if __name__ == '__main__':
    main()
