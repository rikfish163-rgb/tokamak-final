"""Sealed, single-risk-head EAST online replay for a qualified Simple23 parent.

No old model is imported.  TEST220 qualification is a development gate, not
independent EAST validation.  Ordering is a raw-signal commissioning proxy,
not chronological order or a physical damage measure.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data, simple_train, stage_metrics
from .simple23 import Config, Simple23
from .source_gate import qualify_source
from .fast_update import normalized_step


ARMS = ('static', 'intra', 'inter', 'both')
STARTUP_BUDGET = 20
FAST_STEP = .05
FAST_GRAD_CLIP = 1.
FAST_RADIUS = 1.
GUARD_TOLERANCE = 1e-4


def build_order(domain, ids, *, data_api=simple_data):
    """Read every roster signal first, sort by the first 100 raw Ip/Bt points."""
    if domain != 'EAST':
        raise ValueError('this replay supports the EAST commissioning order only')
    rows = []
    for sid in ids:
        s = data_api.read_shot(domain, int(sid))
        ip = np.asarray(s['planning_ip'][:100], np.float64)
        bt = np.asarray(s['planning_Bt'][:100], np.float64)
        if len(ip) != 100 or len(bt) != 100 or not np.isfinite(ip).all() or not np.isfinite(bt).all():
            raise ValueError(f'{sid}: insufficient or nonfinite initial 100 ms planning fields')
        current, field = float(np.median(np.abs(ip))), float(np.median(np.abs(bt)))
        rows.append(dict(shot=int(sid), initial_ip_A=current,
                         initial_abs_Bt_raw=field,
                         proxy=(current / 100_000.) ** 1.2 * field))
    if len({r['shot'] for r in rows}) != len(rows):
        raise ValueError('duplicate commissioning shot')
    return sorted(rows, key=lambda row: (row['proxy'], row['shot']))


def _window(data_api, shot, end, length, device):
    values, available = data_api.window(shot, end, length)
    return (torch.as_tensor(values[None], device=device),
            torch.as_tensor(available[None], device=device))


def predict_shot(model, shot, *, intra, device, data_api=simple_data, channel_logits=None, loss_rule=None):
    """Predict each endpoint; a full H-step truth matures after that prediction.

    One non-overlapping H-step forecast starts at endpoint 0, then another at
    the first endpoint after its update.  Actual future signal, never a model
    forecast, supplies the loss.  Fast state is local and discarded on return.
    """
    if channel_logits is not None and loss_rule is not None:
        raise ValueError('choose one auxiliary weighting rule')
    model.eval()
    horizon = model.cfg.horizon
    if horizon != 25:
        raise ValueError('online contract requires H25')
    initial = model.initial_fast(1).detach()
    fast = initial.clone()
    scores = np.empty(len(shot['values']), np.float32)
    updates = 0
    origin = 0
    while origin < len(scores):
        # No fast update occurs inside this block. Each constituent window is
        # nevertheless constructed only from that endpoint's observed prefix.
        stop = min(len(scores), origin + (horizon + 1 if intra else 256))
        windows = [data_api.window(shot, end, model.cfg.window)
                   for end in range(origin, stop)]
        x = torch.as_tensor(np.stack([w[0] for w in windows]), device=device)
        mask = torch.as_tensor(np.stack([w[1] for w in windows]), device=device)
        with torch.no_grad():
            logit = model(x, mask, fast.expand(stop-origin, -1))['logit']
            if not torch.isfinite(logit).all():
                raise FloatingPointError(f'nonfinite risk logit at shot {shot["shot"]}, block {origin}')
            scores[origin:stop] = logit.sigmoid().cpu().numpy()
        end = stop - 1
        if not intra or end != origin + horizon:
            origin = stop
            continue
        future = np.asarray(shot['values'][origin+1:end+1], np.float32)
        observed = np.asarray(shot['available'][origin+1:end+1], bool)
        if len(future) != horizon:
            raise AssertionError('forecast matured without H25 observations')
        if observed.any():
            start_x, start_mask = _window(data_api, shot, origin, model.cfg.window, device)
            candidate = fast.detach().requires_grad_(True)
            forecast = model(start_x, start_mask, candidate)['forecast']
            truth = torch.as_tensor(future[None], device=device)
            available = torch.as_tensor(observed[None], device=device)
            loss = model.forecast_loss(forecast, truth, available)
            if channel_logits is not None:
                from .channel_loss import forecast_loss
                loss = forecast_loss(model, forecast, truth, available, channel_logits)
            if loss_rule is not None:
                loss = loss_rule(model, forecast, truth, available).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite H25 forecast loss')
            (gradient,) = torch.autograd.grad(loss, candidate)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('nonfinite fast gradient')
            if model.cfg.fast_rule=='normalized':
                fast=normalized_step(candidate.detach(),initial,gradient,step=FAST_STEP,radius=FAST_RADIUS)
            else:
                norm = gradient.norm()
                gradient = gradient * min(1., FAST_GRAD_CLIP / max(float(norm), 1e-12))
                proposed = candidate.detach() - FAST_STEP * gradient
                offset = proposed - initial
                radius = float(offset.norm())
                fast = initial + offset * min(1., FAST_RADIUS / max(radius, 1e-12))
            updates += 1
        origin = end + 1  # This endpoint's sealed prediction cannot change.
    return scores, updates


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _durable_json(path, payload):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def seal_predictions(output_dir, index, shot, times, scores, *, arm, threshold):
    """Persist raw predictions and SHA receipt, then fsync directory."""
    output_dir = Path(output_dir)
    stem = f'{index:04d}_{shot}'
    npz_path = output_dir / f'{stem}_sealed.npz'
    with npz_path.open('wb') as handle:
        np.savez(handle, shot=int(shot), times=np.asarray(times), scores=np.asarray(scores))
        handle.flush()
        os.fsync(handle.fileno())
    receipt = dict(shot=int(shot), arm=arm, threshold=float(threshold),
                   npz=npz_path.name, sha256=_hash(npz_path))
    _durable_json(output_dir / f'{stem}_seal.json', receipt)
    directory_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt


def _examples(shots, rng, count, cfg, device, data_api):
    """Balance valid positive/negative endpoints within each revealed shot.

    The history guard uses this same *window-balanced BCE* sampling protocol;
    it is not a shot-level safety or FPR guarantee.
    """
    eligible = []
    for s in shots:
        ends = np.flatnonzero(np.asarray(s['valid'], bool))
        if len(ends):
            positive = ends[np.asarray(s['y'])[ends] > .5]
            negative = ends[np.asarray(s['y'])[ends] <= .5]
            eligible.append((s, negative, positive))
    if not eligible:
        raise ValueError('no revealed valid endpoints for BCE update')
    windows, masks, labels = [], [], []
    # Round-robin coverage ensures the just-revealed shot participates even
    # when the four replay shots are much longer.
    per_shot = [0] * len(eligible)
    for index in range(count):
        group = index % len(eligible)
        s, negative, positive = eligible[group]
        ends = (positive if per_shot[group] % 2 else negative)
        if not len(ends):
            ends = negative if len(negative) else positive
        per_shot[group] += 1
        end = int(rng.choice(ends))
        x, mask = data_api.window(s, end, cfg.window)
        windows.append(x)
        masks.append(mask)
        labels.append(float(s['y'][end]))
    return (torch.as_tensor(np.stack(windows), device=device),
            torch.as_tensor(np.stack(masks), device=device),
            torch.as_tensor(labels, dtype=torch.float32, device=device))


def _bce(model, examples):
    x, mask, labels = examples
    return F.binary_cross_entropy_with_logits(model(x, mask)['logit'], labels)


def postshot_update(model, shot, replay, *, lr, rng, device, data_api=simple_data):
    """One scoped persistent BCE step, guarded on already revealed earlier shots."""
    if not (np.isfinite(lr) and lr > 0):
        raise ValueError('positive finite persistent learning rate required')
    history = list(replay)[-4:]
    train = _examples([shot, *history], rng, 64, model.cfg, device, data_api)
    guard = _examples(history, rng, 64, model.cfg, device, data_api) if history else None
    selected = [*model.risk_head.parameters(), model.fast_origin,
                *model.encoder.layers[-1].parameters()]
    chosen = {id(p) for p in selected}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    before_weights = [p.detach().clone() for p in selected]
    model.eval()
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in chosen)
        with torch.no_grad():
            before = float(_bce(model, guard)) if guard else None
        optimizer = torch.optim.AdamW(selected, lr=lr, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        loss = _bce(model, train)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite persistent BCE')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in selected):
            raise FloatingPointError('nonfinite persistent parameter after optimizer step')
        with torch.no_grad():
            after = float(_bce(model, guard)) if guard else None
        if after is not None and not np.isfinite(after):
            raise FloatingPointError('nonfinite historical guard BCE')
        if before is not None and after - before > GUARD_TOLERANCE:
            for p, old in zip(selected, before_weights):
                p.data.copy_(old)
            return dict(accepted=False, reason='guard_bce_worse',
                        guard_bce_before=before, guard_bce_after=after,
                        guard_metric='balanced_window_bce_not_shot_safety', lr=float(lr))
        return dict(accepted=True, reason='no_history' if guard is None else 'guard_pass',
                    guard_bce_before=before, guard_bce_after=after,
                    guard_metric='balanced_window_bce_not_shot_safety' if guard else None,
                    lr=float(lr))
    except Exception:
        # A broken candidate must not leak a partial optimizer step.
        for p, old in zip(selected, before_weights):
            p.data.copy_(old)
        raise
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)


def run_arm(model, *, order, arm, threshold, output_dir, data_api=simple_data,
            device='cpu', advanced=None, limit=0, startup_budget=STARTUP_BUDGET,
            seed=131, persistent_updater=None, channel_logits=None, loss_rule=None):
    """Replay one arm in fixed manifest order; never reveal before seal."""
    if arm not in ARMS or not 0 < threshold <= 1 or limit < 0 or startup_budget < 0:
        raise ValueError('invalid arm, threshold, limit, or startup budget')
    if persistent_updater is not None and not (
            arm == 'inter' or (arm == 'both' and getattr(persistent_updater, 'supports_intra', False))):
        raise ValueError('custom both requires maintained forecast head')
    manifest = list(order)
    if len({int(r['shot']) for r in manifest}) != len(manifest):
        raise ValueError('duplicate manifest shot')
    selected = manifest if limit == 0 else manifest[:limit]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    replay = deque(maxlen=4)
    metric_rows, decisions = [], []
    for index, item in enumerate(selected, 1):
        sid = int(item['shot'])
        s = data_api.read_shot('EAST', sid)
        if s['shot'] != sid or s['device'] != 'EAST':
            raise ValueError('signal identity differs from locked order')
        scores, fast_updates = predict_shot(model, s, intra=arm in ('intra', 'both'),
                                             device=device, data_api=data_api, channel_logits=channel_logits,
                                             loss_rule=loss_rule)
        seal_predictions(output_dir, index, sid, s['times'], scores,
                         arm=arm, threshold=threshold)
        revealed = data_api.attach_outcome(s, 'EAST', advanced or {})
        # Score *existing sealed* predictions only up to DownTime.  Never
        # rescore or rewrite their pre-reveal NPZ after seeing the outcome.
        valid = np.asarray(revealed['valid'], bool)
        if revealed['is_disrupt']:
            valid &= np.asarray(s['times']) <= revealed['down']
        scored = dict(shot=sid, is_disrupt=bool(revealed['is_disrupt']),
                      down=revealed['down'], times=np.asarray(s['times'])[valid],
                      scores=scores[valid])
        metric_rows.append(scored)
        measured = simple_train.metrics(metric_rows, threshold, lead=30.)
        phase = 'startup' if index <= startup_budget else 'post_startup'
        if arm in ('inter', 'both'):
            lr = 3e-4 if index <= startup_budget else 1e-4
            if persistent_updater is None:
                persistent = postshot_update(model, revealed, replay, lr=lr, rng=rng,
                                             device=device, data_api=data_api)
            else:
                persistent = persistent_updater(model, revealed, device=device, data_api=data_api)
        else:
            persistent = dict(accepted=False, reason='static_or_intra')
        decision = dict(measured['predictions'][-1], phase=phase,
                        fast_updates=fast_updates, persistent=persistent,
                        cumulative={k: measured[k] for k in ('tp', 'fp', 'n_disrupt', 'n_safe', 'tpr', 'fpr')})
        decisions.append(decision)
        replay.append(revealed)
        _durable_json(output_dir / f'{index:04d}_{sid}_audit.json', decision)
    measured = simple_train.metrics(metric_rows, threshold, lead=30.)
    stage = stage_metrics.summarize_commissioning(decisions, manifest, startup_budget)
    result = dict(arm=arm, rows=decisions, manifest_count=len(manifest),
                  observed_count=len(selected), threshold=float(threshold),
                  threshold_source='J-TEXT VAL; not EAST 125 ms calibrated probability',
                  stage=stage)
    result['metrics_full' if len(selected) == len(manifest) else 'metrics_prefix'] = measured
    _durable_json(output_dir / 'summary.json', result)
    return result


def run_four_arms(model, *, order, threshold, output_dir, data_api=simple_data,
                  device='cpu', advanced=None, limit=0, startup_budget=STARTUP_BUDGET,
                  seed=131):
    """Run independent arms from the same qualified parent weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    _durable_json(output_dir / 'order.json', list(order))
    _durable_json(output_dir / 'protocol.json', dict(
        seed=seed, startup_budget=startup_budget, requested_limit=limit,
        fast_step=FAST_STEP, fast_radius=FAST_RADIUS, fast_grad_clip=FAST_GRAD_CLIP,
        fast_rule=model.cfg.fast_rule,
        guard_tolerance=GUARD_TOLERANCE, lead_rule='strict >30ms',
        causal_batching='independent past-only windows; no update within a batch',
        code_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                     ('simple_online.py', 'simple23.py', 'simple_data.py', 'io23.py',
                      'simple_train.py', 'stage_metrics.py', 'source_gate.py','fast_update.py')}))
    for arm in ARMS:
        result[arm] = run_arm(deepcopy(model), order=order, arm=arm,
                              threshold=threshold, output_dir=output_dir / arm,
                              data_api=data_api, device=device, advanced=advanced,
                              limit=limit, startup_budget=startup_budget, seed=seed)
    _durable_json(output_dir / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 = complete locked 360; else sorted prefix')
    parser.add_argument('--compute-device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=131)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.limit < 0 or args.limit > 360:
        parser.error('--limit must be 0..360')
    if args.output.exists():
        raise FileExistsError('do not overwrite a previous replay')
    gate = qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(f'unqualified source checkpoint: {gate["reasons"]}')
    # torch.load is restricted to trusted, locally produced project artifacts.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['config']['format'] != 'simple23-v1' or not checkpoint.get('forecast_trained'):
        raise ValueError('qualified, forecast-trained simple23-v1 source.pt required')
    cfg = Config(**checkpoint['model_config'])
    model = Simple23(cfg, checkpoint['source_center'], checkpoint['source_scale'])
    model.load_state_dict(checkpoint['model'])
    model.to(args.compute_device).eval()
    roster = io23.cohort_ids('EAST')
    order = build_order('EAST', roster)  # Always build all 360 before limit.
    result = run_four_arms(model, order=order, threshold=float(checkpoint['threshold']),
                           output_dir=args.output, device=args.compute_device,
                           limit=args.limit, seed=args.seed)
    _durable_json(args.output / 'provenance.json',
                  dict(checkpoint=str(args.checkpoint.resolve()),
                       checkpoint_sha256=gate['checkpoint_sha256'], source_gate=gate,
                       model_format='simple23-v1', roster_count=len(order),
                       threshold=float(checkpoint['threshold']),
                       interpretation='development replay; no independent EAST calibration'))
    return result


if __name__ == '__main__':
    main()
