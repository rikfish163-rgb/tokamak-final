"""Direct joint signal trajectories with matured, shot-local neural learning.

The qualified source classifier is immutable. A separate temporal predictor
forecasts whole H25 signal paths; the same classifier evaluates these paths.
Static and intra arms share that predictive readout, preparation and threshold.
"""

import argparse
from copy import deepcopy
import inspect
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from experiments.risk_aligned_online import simple_data
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    PARENT_SHA256, load_source_train,
)
from joint_trajectory_20261107 import JointTrajectory
from run_layer0_query_memory_20261105 import raw_for
from run_rolling_safety_20261106 import cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


FIT = r.ROOT/'probe_outputs/joint_trajectory_source_20261107_v1'
ARMS = ('static', 'hold', 'mean_frozen', 'joint_frozen', 'mean_intra',
        'joint_intra', 'joint_probability_only', 'joint_wrong_time')
ADAPT = ('intra', 'probability_only', 'wrong_time')
HORIZON, WINDOW, STRIDE, BATCH = 25, 128, 32, 128
SOURCE_STEPS, COLD_STEPS, FIT_BATCH, FIT_LR = 512, 128, 32, 3e-4
INTRA_LR, PRIOR_WEIGHT, SEED = 1e-4, .01, 131


def eligible_origins(shots):
    """Uniform shot/origin sampling, without selecting future risk classes."""
    pools = []
    width = WINDOW + HORIZON
    for i, shot in enumerate(shots):
        valid = np.asarray(shot.get('valid', np.ones(len(shot['values']), bool)), bool)
        if len(valid) < width:
            continue
        starts = np.flatnonzero(np.convolve(valid.astype(np.int32),
            np.ones(width, np.int32), 'valid') == width)
        if len(starts):
            pools.append((i, starts + WINDOW - 1))
    if not pools:
        raise ValueError('no complete observed-context/direct-H25 episodes')
    return pools


def sample_signals(shots, pools, rng, size, device):
    values, available, truths, masks, origins = [], [], [], [], []
    for _ in range(size):
        i, eligible = pools[int(rng.integers(len(pools)))]
        origin = int(rng.choice(eligible))
        shot = shots[i]
        x, a = simple_data.window(shot, origin, WINDOW)
        values.append(x)
        available.append(a)
        truths.append(shot['values'][origin+1:origin+HORIZON+1])
        masks.append(shot['available'][origin+1:origin+HORIZON+1])
        origins.append(origin)
    return dict(x=torch.as_tensor(np.stack(values), device=device),
        a=torch.as_tensor(np.stack(available), device=device),
        truth=torch.as_tensor(np.stack(truths), device=device),
        mask=torch.as_tensor(np.stack(masks), device=device),
        origin=torch.tensor(origins, device=device))


def features(base, x, available):
    """Read actual encoder tokens, honoring the caller's parameter/grad scope."""
    captured = []
    handle = base.forecast_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    try:
        logits = base(x, available)['logit']
    finally:
        handle.remove()
    tokens = captured[0].reshape(len(x), WINDOW//base.cfg.patch+1, base.cfg.dim)
    valid = (available & torch.isfinite(x)).reshape(len(x), WINDOW//base.cfg.patch, -1).any(-1)
    valid = torch.cat((valid, torch.ones(len(x), 1, dtype=torch.bool, device=x.device)), 1)
    return tokens, valid, logits


def energy(model, e, *, wrong_time=False):
    output = model(e['tokens'], e['token_valid'], e['x'][:, -1], e['a'][:, -1])
    truth, observed = e['truth'], e['mask']
    if wrong_time:
        truth, observed = truth.flip(1), observed.flip(1)
    observed = observed & output['available'] & torch.isfinite(truth)
    target = torch.asinh((torch.where(observed, truth, model.center)-model.center)/model.scale)
    return model.loss(output['encoded'], output['probabilities'], target, observed).mean()


def update(model, optimizer, e, prior, *, wrong_time=False):
    optimizer.zero_grad(set_to_none=True)
    loss = energy(model, e, wrong_time=wrong_time)
    penalty = sum((p-prior[name]).square().sum() for name, p in model.named_parameters() if p.requires_grad)
    objective = loss + PRIOR_WEIGHT*penalty
    if not torch.isfinite(objective):
        raise FloatingPointError('nonfinite matured trajectory learning objective')
    objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(energy=float(loss.detach()), prior=float(penalty.detach()), gradient_norm=float(norm))


@torch.no_grad()
def virtual_risks(base, x, available, raw, predicted_available):
    if not torch.isfinite(raw).all():
        raise FloatingPointError('nonfinite generated signal is not a missing diagnostic')
    batch, paths = raw.shape[:2]
    history = x[:, None, HORIZON:].expand(-1, paths, -1, -1)
    history_mask = available[:, None, HORIZON:].expand(-1, paths, -1, -1)
    windows = torch.cat((history, raw), 2).reshape(batch*paths, WINDOW, 23)
    mask = torch.cat((history_mask, predicted_available[:, None].expand(-1, paths, -1, -1)), 2)
    mask = mask.reshape(batch*paths, WINDOW, 23)
    probabilities = [base(windows[lo:lo+BATCH], mask[lo:lo+BATCH])['logit'].sigmoid()
                     for lo in range(0, len(windows), BATCH)]
    return torch.cat(probabilities).reshape(batch, paths)


@torch.no_grad()
def trajectory_risks(base, x, available, output):
    path_risk = virtual_risks(base, x, available, output['raw'], output['available'])
    joint = (path_risk*output['probabilities']).sum(1)
    mean_signal = (output['raw']*output['probabilities'][:, :, None, None]).sum(1, keepdim=True)
    mean = virtual_risks(base, x, available, mean_signal, output['available'])[:, 0]
    return dict(joint=joint, mean=mean)


def prediction_context(base, signal):
    """Cache only per-endpoint causal windows and immutable source features."""
    base.eval()
    x, available = r.latent.make_views(signal, base.center.device)
    token_parts, valid_parts, logit_parts = [], [], []
    for lo in range(0, len(x), BATCH):
        token, valid, logits = features(base, x[lo:lo+BATCH], available[lo:lo+BATCH])
        token_parts.append(token)
        valid_parts.append(valid)
        logit_parts.append(logits)
    tokens, valid = torch.cat(token_parts), torch.cat(valid_parts)
    static = torch.cat(logit_parts).sigmoid()
    return x, available, tokens, valid, static


def adaptive_curves(base, predictor, signal, context, *, arms=ADAPT, warm=False):
    """Both readouts for the selected feedback branches; no unexecuted arms."""
    x, available, tokens, valid, static = context
    curves = {f'{kind}_{arm}': static.cpu().numpy().copy()
              for arm in arms for kind in ('mean', 'joint')}
    info = dict(updates={k: 0 for k in arms}, release_indices=[], parameter_change={})
    if warm or len(x) < WINDOW:
        return curves, info
    models = {k: deepcopy(predictor).eval().requires_grad_(True) for k in arms}
    if 'probability_only' in models:
        models['probability_only'].requires_grad_(False).probability.requires_grad_(True)
    prior = {k: v.detach().clone() for k, v in predictor.named_parameters()}
    optimizers = {k: torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=INTRA_LR, weight_decay=0.) for k, model in models.items()}
    values = torch.as_tensor(signal['values'], device=x.device)
    masks = torch.as_tensor(signal['available'], device=x.device)

    releases = list(range(WINDOW+HORIZON, len(x), STRIDE))
    lo = WINDOW-1
    for end in releases + [len(x)]:
        with torch.no_grad():
            for arm, model in models.items():
                output = model(tokens[lo:end], valid[lo:end], x[lo:end, -1], available[lo:end, -1])
                risks = trajectory_risks(base, x[lo:end], available[lo:end], output)
                for kind in ('mean', 'joint'):
                    curves[f'{kind}_{arm}'][lo:end] = torch.maximum(static[lo:end], risks[kind]).cpu().numpy()
        if end != len(x):
            origin = end-HORIZON-1
            # No update occurs between issuance at origin and this maturity,
            # so recomputation uses the very parameters that issued the path.
            e = dict(tokens=tokens[origin:origin+1], token_valid=valid[origin:origin+1],
                x=x[origin:origin+1], a=available[origin:origin+1],
                truth=values[origin+1:origin+HORIZON+1][None],
                mask=masks[origin+1:origin+HORIZON+1][None])
            for arm, model in models.items():
                update(model, optimizers[arm], e, prior, wrong_time=arm == 'wrong_time')
                info['updates'][arm] += 1
            info['release_indices'].append(end)
        lo = end
    for arm, model in models.items():
        info['parameter_change'][arm] = {group: float(torch.stack([
            (p-prior[name]).square().sum() for name, p in model.named_parameters()
            if name.startswith(group+'.')]).sum().sqrt().detach())
            for group in ('temporal', 'trajectory', 'probability')}
    if any(not np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite direct/predictive risk curve')
    return curves, info


def joint_curves(base, predictor, signal, *, warm=False):
    """Seal through origin+25, update, then first affect origin+26 onward."""
    context = prediction_context(base, signal)
    x, available, tokens, valid, static = context
    curves = {k: static.cpu().numpy().copy() for k in ARMS}
    if not warm and len(x) >= WINDOW:
        # Both predictive-static controls use the identical prepared predictor.
        with torch.no_grad():
            for lo in range(WINDOW-1, len(x), BATCH):
                end = min(len(x), lo+BATCH)
                current = predictor(tokens[lo:end], valid[lo:end], x[lo:end, -1], available[lo:end, -1])
                risks = trajectory_risks(base, x[lo:end], available[lo:end], current)
                for kind in ('mean', 'joint'):
                    curves[f'{kind}_frozen'][lo:end] = torch.maximum(static[lo:end], risks[kind]).cpu().numpy()
                held = x[lo:end, -1:, :].expand(-1, HORIZON, -1)[:, None]
                held_mask = available[lo:end, -1:, :].expand(-1, HORIZON, -1)
                hold = virtual_risks(base, x[lo:end], available[lo:end], held, held_mask)[:, 0]
                curves['hold'][lo:end] = torch.maximum(static[lo:end], hold).cpu().numpy()
    adapted, info = adaptive_curves(base, predictor, signal, context, warm=warm)
    curves.update({k: v for k, v in adapted.items() if k in curves})
    return curves, info


def fit_predictor(base, model, shots, steps, seed, label):
    pools = eligible_origins(shots)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIT_LR, weight_decay=0.)
    history = []
    for step in range(steps):
        e = sample_signals(shots, pools, rng, FIT_BATCH, base.center.device)
        e['tokens'], e['token_valid'], _ = features(base, e['x'], e['a'])
        optimizer.zero_grad(set_to_none=True)
        objective = energy(model, e)
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite source/cold signal objective')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append(float(objective.detach()))
        if (step+1) % 64 == 0:
            print(label, step+1, 'energy', history[-1], flush=True)
    return dict(history=history, eligible_shots=len(pools), steps=steps)


def fit(out):
    r.configure()
    torch.manual_seed(SEED)
    started = time.monotonic()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    frozen = {k: v.detach().clone() for k, v in base.state_dict().items()}
    model = JointTrajectory(base.center, base.scale, dim=base.cfg.dim, scenarios=4).cuda().eval()
    _, cfg, _, (_, _, warm) = cohort('fixed360')
    mapping = r.read(r.COLD/'input_map.json')
    if warm != 20 or mapping['calibration_ids'] != cfg['order'][:warm]:
        raise ValueError('locked cold20/map required')
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='JOINT_TRAJECTORY_SOURCE_AND_ENDED20', source=str(r.DEFAULT_PARENT),
        parent_sha256=PARENT_SHA256, source_steps=SOURCE_STEPS, cold_steps=COLD_STEPS,
        batch=FIT_BATCH, lr=FIT_LR, seed=SEED, scenarios=4, horizon=HORIZON,
        source_sampling='uniform eligible TRAIN shot, then uniform complete origin; not balanced on future y',
        cold20=cfg['order'][:warm], map=str(r.COLD/'input_map.json'),
        cold_use='all signals after those shots ended; no EAST outcome lookup, including no outcome-based trim',
        objective='joint L2 energy score in fixed source-asinh coordinates',
        verified_physical_support='only sxr_var/xuv_var nonnegative; no other physical law asserted',
        source_classifier_training=False, source_val_or_test_selection=False,
        source_files={str(p): p.read_text() for p in (Path(__file__),
            r.ROOT/'joint_trajectory_20261107.py', r.ROOT/'experiments/risk_aligned_online/simple23.py')})
    r.write_new(out/'protocol.json', protocol)
    train, ids = load_source_train()
    source = fit_predictor(base, model, train, SOURCE_STEPS, SEED, 'source')
    with (out/'source_predictor.pt').open('xb') as handle:
        torch.save(model.state_dict(), handle)
    cold = []
    with patch.object(r.io23, 'read_outcome', side_effect=RuntimeError('EAST outcome unavailable in signal preparation')):
        for sid in cfg['order'][:warm]:
            raw = raw_for(sid, cfg['raw_sha256'])
            # At shot21 all20 ended; use their observations in the same fixed
            # coordinates as subsequent shots, not their earlier unaligned view.
            cold.append(r.prepare_actual(raw, warm+1, mapping))
        preparation = fit_predictor(base, model, cold, COLD_STEPS, SEED+1, 'ended20')
    if any(not torch.equal(v, frozen[k]) for k, v in base.state_dict().items()):
        raise ValueError('signal fitting changed the qualified source classifier')
    with (out/'prepared_predictor.pt').open('xb') as handle:
        torch.save(model.state_dict(), handle)
    r.write_new(out/'fit.json', dict(status='FIT_COMPLETE', source=source, cold=preparation,
        source_train_ids=ids, source_classifier_unchanged=True, seconds=time.monotonic()-started))
    print('completed joint trajectory fit', round(time.monotonic()-started, 2), 's', flush=True)


def predict(out, *, fit_root=FIT, predictor_type=JointTrajectory):
    r.configure()
    started = time.monotonic()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    fit_protocol, done = (r.read(fit_root/f'{k}.json') for k in ('protocol', 'fit'))
    if done['status'] != 'FIT_COMPLETE' or fit_protocol['parent_sha256'] != PARENT_SHA256:
        raise ValueError('completed joint trajectory preparation required')
    predictor = predictor_type(base.center, base.scale, dim=base.cfg.dim, scenarios=4).cuda().eval()
    predictor.load_state_dict(torch.load(fit_root/'prepared_predictor.pt', map_location='cuda', weights_only=True))
    reference, cfg, archive, (real, stale, warm) = cohort('fixed360')
    mapping = r.read(r.COLD/'input_map.json')
    if (warm != 20 or parent['threshold'] != r.THRESHOLD
            or fit_protocol['cold20'] != cfg['order'][:warm]):
        raise ValueError('same qualified parent, cold20 and fixed threshold required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'shots').mkdir()
    r.write_new(out/'protocol.json', dict(status='FIXED_JOINT_TRAJECTORY360', order=cfg['order'],
        parent_sha256=PARENT_SHA256, preparation=str(fit_root), threshold=r.THRESHOLD,
        predictor_type=type(predictor).__name__, learning_objective=fit_protocol['objective'],
        arms=ARMS, warm=warm, stride=STRIDE, lr=INTRA_LR, prior_weight=PRIOR_WEIGHT,
        readout='max(current original risk, anticipated risk); unchanged threshold; first127 points direct',
        anticipated_joint='one probability per entire H25x23 trajectory; expected original classifier probability',
        anticipated_mean='original classifier on probability-weighted physical-file-unit mean trajectory',
        feedback='entire real H25 matured; first affected prediction origin+26; cumulative until shot end',
        controls='probability-head-only and time-reversed matured H25; same step count/lr, not norm-matched',
        neural_update='temporal predictor and trajectory/probability heads; original classifier frozen',
        inter_updates=0, target_label_fitting=False, jtext_online=False, independent_confirmation=False,
        baseline=str(reference), preparation_protocol=fit_protocol,
        source_files={str(p): p.read_text() for p in (Path(__file__),
            r.ROOT/'joint_trajectory_20261107.py',Path(inspect.getfile(predictor_type)))}))
    frozen = {k: v.detach().clone() for k, v in base.state_dict().items()}
    records, witnesses = [], []
    with patch.object(r.io23, 'read_outcome', side_effect=RuntimeError('EAST outcomes unavailable before all predictions sealed')):
        for rank, note in enumerate(archive['sealed'], 1):
            sid = note['shot']
            times, baseline, _, _ = recorded(reference, note, rank, real, stale)
            raw = raw_for(sid, cfg['raw_sha256'])
            np.testing.assert_array_equal(times, raw['times'])
            signal = r.prepare_actual(raw, rank, mapping)
            traces, info = joint_curves(base, predictor, signal, warm=rank <= warm)
            error = require_parity(traces['static'], baseline)
            if rank in (21, 181, 359):
                stop = min(len(times), max(WINDOW+HORIZON+1, len(times)//2))
                partial = {k: signal[k][:stop] for k in ('values', 'available')}
                truncated, _ = joint_curves(base, predictor, partial)
                witnesses.append(dict(shot=sid, points=stop,
                    max_error=max(require_parity(truncated[k], traces[k][:stop]) for k in ARMS)))
            r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz', **traces)
            records.append(dict(shot=sid, rank=rank, static_error=error, adaptation=info))
            if rank % 10 == 0:
                print('joint predicted', rank, '/360; seconds', round(time.monotonic()-started, 1), flush=True)
    if any(not torch.equal(v, frozen[k]) for k, v in base.state_dict().items()):
        raise ValueError('online trajectory fitting changed the qualified source classifier')
    r.write_new(out/'completed.json', dict(status='SEALED_JOINT_TRAJECTORY360', records=records,
        witnesses=witnesses, seconds=time.monotonic()-started, source_classifier_unchanged=True))
    print('sealed joint trajectory360', flush=True)


def recount(out):
    protocol, done = (r.read(out/f'{k}.json') for k in ('protocol', 'completed'))
    reference, cfg, archive, (real, stale, warm) = cohort('fixed360')
    if done['status'] != 'SEALED_JOINT_TRAJECTORY360' or protocol['order'] != cfg['order']:
        raise ValueError('all360 predictions must be sealed before outcome access')
    if [v['shot'] for v in done['records']] != cfg['order']:
        raise ValueError('incomplete sealed stream')
    rows = {k: [] for k in ARMS}
    for rank, note in enumerate(archive['sealed'], 1):
        sid = note['shot']
        times, _, _, _ = recorded(reference, note, rank, real, stale)
        truth, clock = raw_truth(r.io23.path_for('EAST', sid), sid)
        np.testing.assert_array_equal(times, clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz', allow_pickle=False) as data:
            for arm in ARMS:
                rows[arm].append(score(times, data[arm], truth, protocol['threshold']))
    metrics = {k: counts(v) for k, v in rows.items()}
    if (metrics['static']['tp'], metrics['static']['fp']) != (116, 29):
        raise ValueError('original static reproduction changed')
    for value in metrics.values():
        value['fpr_cap_met'] = {str(cap): value['fpr'] <= cap for cap in (.10, .15, .20)}
    paired = {kind: pair(rows[f'{kind}_intra'], rows[f'{kind}_frozen']) for kind in ('mean', 'joint')}
    gains = {kind: metrics[f'{kind}_intra']['fpr'] <= .20
                  and metrics[f'{kind}_intra']['tp'] > metrics[f'{kind}_frozen']['tp']
                  for kind in ('mean', 'joint')}
    r.write_new(out/'result.json', dict(status='HDF5_JOINT_TRAJECTORY360', metrics=metrics,
        paired_own_frozen=paired, independent_intra_gain=gains,
        paired_direct={k: pair(v, rows['static']) for k, v in rows.items() if k != 'static'},
        prefix20={k: counts(v[:20]) for k, v in rows.items()},
        suffix340={k: counts(v[20:]) for k, v in rows.items()}, rows=rows,
        independent_confirmation=False))
    print({k: (v['tp'], v['fp'], v['tp_30_to_3000']) for k, v in metrics.items()}, flush=True)
    print('paired intra gain', gains, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'predict', 'score'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    {'fit': fit, 'predict': predict, 'score': recount}[args.phase](args.out)
