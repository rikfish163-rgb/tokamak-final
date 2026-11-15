"""Replay causal forecast comparison with or without within-shot dynamics learning."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_dual_dynamics_20261110 as dynamics
from . import commissioning_source_dynamics_20261110 as source
from . import io23, simple_data
from .commissioning_calibration_20261108 import held_safe_cutoffs, prefix_mapping
from .commissioning_event_roughness_20261110 import CUE
from .commissioning_intra_20261110 import CALIBRATION, CAPS, PARENT_ARM, PARENT_RUN, STARTUP, load_frozen
from .commissioning_latent_rate_20261110 import read_parent_curve, supervised_rows
from .commissioning_transfer_20261108 import alarm_loss, capacity, features, shot_weights
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


STEPS = 256
FIRST_ADAPTED_EVIDENCE = 127 + 32 + 25 + 1
MODES = ('fixed', 'real', 'reverse')
NAMES = ('parent', 'no_forecast', *MODES)


def joined(raw, evidence, scale):
    transformed = evidence.copy()
    transformed[:, :len(scale)] = np.clip(transformed[:, :len(scale)] / scale, -4., 4.)
    return np.concatenate((raw, transformed), axis=1)


def encode(base, signal, center, scale, forecasts):
    raw, _ = features(base, signal, 'cuda')
    blocks, _ = source.shot_blocks(signal, center, scale)
    cues, audit = dynamics.replay(len(raw), blocks, *forecasts)
    return raw, cues, audit


def fit(base, raws, outcomes, ids, mapping, center, scale, forecasts, original):
    rows = []
    for sid in ids:
        raw, cue, _ = encode(base, prepare_actual(raws[sid], STARTUP + 1, mapping), center, scale, forecasts)
        times = raws[sid]['times']
        _, keep = io23.labels_at(times, *outcomes[sid], 125.)
        f, y, timely, safe = supervised_rows(cue['fixed'][None], times, *outcomes[sid])
        rows.append((raw[keep], f[0], y, timely, safe))
    size = 2 * len(CUE)
    # Scale only the two evidence terms, from fitting shots with equal shot
    # mass. Coverage is already a fraction and left in its original units.
    scale = np.maximum(np.sqrt(np.mean([np.mean(f[:, :size]**2, axis=0)
                                        for _, f, *_ in rows], axis=0)), .02)
    x, masks, labels, groups = [], [], [], []
    offset = 0
    for i, (raw, cue, y, timely, safe) in enumerate(rows):
        x.append(joined(raw, cue, scale))
        labels.append(y)
        groups.append(np.full(len(y), i, int))
        masks.append([np.flatnonzero(v) + offset for v in (timely, safe)])
        offset += len(y)
    target = torch.from_numpy(np.concatenate(labels).astype(np.float32)).cuda()
    weights = torch.from_numpy(shot_weights(np.concatenate(groups), np.concatenate(labels)).astype(np.float32)).cuda()
    timely, safe = ([torch.from_numpy(r[i]).cuda() for r in masks if len(r[i])] for i in (0, 1))
    mat = torch.from_numpy(np.concatenate(x)).cuda()
    n_evidence = rows[0][1].shape[1]
    control = mat.clone()
    control[:, -n_evidence:] = 0.
    origin = torch.cat((base.risk_head.weight.flatten(), mat.new_zeros(23+n_evidence),
                        base.risk_head.bias))
    coef = torch.nn.Parameter(origin.repeat(2, 1))
    optimizer = torch.optim.Adam([coef], lr=.02)
    boundary = float(np.log(original / (1. - original)))
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        z = torch.stack((control @ coef[0, :-1], mat @ coef[1, :-1])) + coef[:, -1:]
        bce = F.binary_cross_entropy_with_logits(z, target.expand_as(z), reduction='none') @ weights
        warn = torch.stack([alarm_loss(v, timely, safe, boundary) for v in z])
        (bce.sum() + warn.sum() + .05 * (coef - origin).square().sum()).backward()
        optimizer.step()
    if not torch.isfinite(coef).all():
        raise FloatingPointError('commissioning readout is not finite')
    return scale, coef.detach(), dict(
        bce=bce.detach().cpu().tolist(), warning=warn.detach().cpu().tolist(),
        order=['no_forecast', 'fixed'], steps=STEPS)


@torch.no_grad()
def risks(raw, cue, scale, coef, modes=MODES):
    # No-forecast head has identically zero coefficients on the forecast inputs.
    base = torch.from_numpy(raw).cuda()
    result = {'no_forecast': (base @ coef[0, :raw.shape[1]] + coef[0, -1]).sigmoid().cpu().numpy()}
    for name in modes:
        x = torch.from_numpy(joined(raw, cue[name], scale)).cuda()
        result[name] = (x @ coef[1, :-1] + coef[1, -1]).sigmoid().cpu().numpy()
    return result


def calibrate(base, raws, outcomes, source_stats, center, scale, forecasts, original, started):
    folds = []
    for sid, (d, _) in outcomes.items():
        if d:
            continue
        ids = [k for k in raws if k != sid]
        mapping = prefix_mapping(source_stats, [raws[k] for k in ids])
        norm, coef, _ = fit(base, raws, outcomes, ids, mapping, center, scale, forecasts, original)
        h, x, _ = encode(base, prepare_actual(raws[sid], STARTUP + 1, mapping), center, scale, forecasts)
        maxima = {k: float(v.max()) for k, v in risks(h, x, norm, coef, ('fixed',)).items()}
        folds.append(dict(held_safe=sid, fit_ids=ids, maxima=maxima))
        if len(folds) % 3 == 0:
            print(dict(stage='calibration', folds=len(folds), seconds=time.monotonic()-started), flush=True)
    cutoffs = {k: held_safe_cutoffs([f['maxima'][k] for f in folds]) for k in ('no_forecast', 'fixed')}
    return dict(scope='COLD20_LEAVE_SAFE_SHOT_OUT', folds=folds, thresholds=cutoffs)


def run(out):
    started = time.monotonic()
    configure()
    inputs, prior_head, original, parent_cutoffs, bound = load_frozen()
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().numpy().copy() for v in (base.center, base.scale))
    base.cuda().requires_grad_(False)
    source_state = state_digest(base)
    if ckpt['threshold'] != original or source_state != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source changed')
    ids = inputs['ids']
    parent_seals = read(PARENT_RUN / 'result.json')['curve_sha256']
    out.mkdir(parents=True)
    (out / 'shots').mkdir()
    write(out / 'protocol.json', dict(
        scope='EAST360_COLD20_DUAL_SOURCE_DYNAMICS_INTRA_ONLY_DEVELOPMENT', order=ids, startup=STARTUP,
        source_sha256=PARENT_SHA256, **bound, original=original, parent_thresholds=parent_cutoffs,
        arms=NAMES, source_fit='TRAIN_only_same_activity_safe_vs_disruptive',
        fit_steps=STEPS, lead_strictly_gt_ms=30., training_label_ms=125.,
        origin=127, horizon=25, stride=32, first_evidence=153,
        first_adapted_evidence=FIRST_ADAPTED_EVIDENCE,
        shared_cutoffs_modes=MODES, persistent_updates=0,
        producer_sha256=sha(Path(__file__)), source_dynamics_sha256=sha(Path(source.__file__)),
        core_sha256=sha(Path(dynamics.__file__))))
    forecasts, source_fit = source.fit_source(center, scale)
    write(out / 'source_activity.json', source_fit)
    print(dict(stage='source_fit', pairs=len(source_fit['pairs']),
               held_preference=[round(f['paired_preference'], 4) for f in source_fit['held_source']],
               seconds=time.monotonic()-started), flush=True)

    def raw_shot(sid, times):
        if sha(io23.path_for('EAST', sid)) != inputs['hashes'][str(sid)]:
            raise ValueError(f'EAST waveform changed: {sid}')
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        np.testing.assert_array_equal(times, raw['times'])
        return raw

    raws, outcomes = {}, {}
    for i, sid in enumerate(ids[:STARTUP], 1):
        times, _ = read_parent_curve(f'{i:04d}_{sid}.npz', parent_seals)
        raws[sid] = raw_shot(sid, times)
        outcomes[sid] = io23.read_outcome('EAST', sid)
    if sum(d for d, _ in outcomes.values()) != 8:
        raise ValueError('expected eight disruptive startup shots after the seal')
    cal = calibrate(base, raws, outcomes, inputs['mapping']['source_statistics'],
                    center, scale, forecasts, original, started)
    write(out / 'calibration.json', cal)
    norm, coef, train = fit(base, raws, outcomes, ids[:STARTUP], inputs['mapping'],
                           center, scale, forecasts, original)
    with (out / 'models.npz').open('xb') as f:
        np.savez(f, benign=forecasts[0], disruptive=forecasts[1], risk_coef=coef.cpu().numpy(),
                 evidence_scale=norm)
    binding = {k: sha(out / k) for k in ('models.npz', 'calibration.json', 'source_activity.json')}
    print(dict(stage='fit20_complete', seconds=time.monotonic()-started), flush=True)
    thresholds = {k: cal['thresholds']['fixed'] for k in MODES}
    thresholds.update(parent=parent_cutoffs, no_forecast=cal['thresholds']['no_forecast'])
    seals, audit, witnesses = {}, {}, {}

    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before prediction seal')):
        for i, sid in enumerate(ids, 1):
            name = f'{i:04d}_{sid}.npz'
            times, parent = read_parent_curve(name, parent_seals)
            if i <= STARTUP:
                curves = dict.fromkeys(NAMES, parent)
            else:
                signal = prepare_actual(raw_shot(sid, times), i, inputs['mapping'])
                raw, cue, audit[str(sid)] = encode(base, signal, center, scale, forecasts)
                previous = (torch.from_numpy(raw).cuda() @ prior_head[:-1] + prior_head[-1]).sigmoid().cpu().numpy()
                np.testing.assert_array_equal(previous, parent)
                curves = dict(parent=parent, **risks(raw, cue, norm, coef))
                for k in MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:FIRST_ADAPTED_EVIDENCE],
                                                  curves['fixed'][:FIRST_ADAPTED_EVIDENCE])
                if i in (21, 120, 360):
                    prefix = {k: signal[k][:243] for k in ('values', 'available')}
                    x, e, _ = encode(base, prefix, center, scale, forecasts)
                    witnesses[str(sid)] = {}
                    for k, v in risks(x, e, norm, coef).items():
                        held = curves[k][:len(v)]
                        np.testing.assert_allclose(v, held, rtol=0., atol=1e-5)
                        for cutoff in thresholds[k].values():
                            np.testing.assert_array_equal(v >= cutoff, held >= cutoff)
                        witnesses[str(sid)][k] = float(np.max(abs(v-held)))
            path = out / 'shots' / name
            with path.open('xb') as f:
                np.savez_compressed(f, times=times, **curves)
            seals[name] = sha(path)
            if i % 60 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if source_state != state_digest(base) or any(sha(out / k) != v for k, v in binding.items()):
        raise ValueError('frozen model or calibration changed during replay')
    write(out / 'seal.json', dict(curve_sha256=seals, artifacts_sha256=binding, witnesses=witnesses))
    write(out / 'dynamics_by_shot.json', audit)

    per = {c: {k: [] for k in NAMES} for c in CAPS}
    maxima, labels = {k: [] for k in NAMES}, []
    for i, sid in enumerate(ids, 1):
        path = out / 'shots' / f'{i:04d}_{sid}.npz'
        if sha(path) != seals[path.name]:
            raise ValueError('post-seal curve change')
        d, down = io23.read_outcome('EAST', sid)
        labels.append(d)
        with np.load(path, allow_pickle=False) as f:
            t = f['times']
            eligible = np.round((down-t)*1000., 9) > 30. if d else np.ones(len(t), bool)
            for k in NAMES:
                v = f[k]
                maxima[k].append(float(v[eligible].max(initial=-np.inf)))
                for c, g in per.items():
                    score = score_curve(t, v, d, down, original if i <= STARTUP else thresholds[k][c])
                    g[k].append(dict(shot=sid, disruptive=d, down_s=down, **score))
    archived = read(CALIBRATION / 'per_shot.json')
    if any(g['parent'] != archived[c][PARENT_ARM] for c, g in per.items()):
        raise ValueError('archived incumbent scoring changed')
    metrics = {c: {k: timing_decomposition(v) for k, v in g.items()} for c, g in per.items()}
    paired = {c: {k: flips(g['fixed'], g[k]) for k in MODES[1:]} for c, g in per.items()}
    paired_reverse = {c: flips(g['reverse'], g['real']) for c, g in per.items()}
    write(out / 'per_shot.json', per)
    write(out / 'result.json', dict(status='COMPLETED_DUAL_DYNAMICS_EAST360_DEVELOPMENT', metrics=metrics,
        paired_vs_fixed=paired, paired_real_vs_reverse=paired_reverse, training=train,
        posthoc_capacity_not_deployable={k: capacity(v, labels, original, startup=STARTUP)
                                         for k, v in maxima.items()},
        seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output.resolve())
