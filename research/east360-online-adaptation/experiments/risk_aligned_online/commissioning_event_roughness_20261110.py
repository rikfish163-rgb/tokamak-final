"""Source event-matched trajectory cue with a causal per-shot roughness reference."""

import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_event_pairs_20261110 as pairs
from . import commissioning_roughness_20261110 as roughness
from . import io23, simple_data
from .commissioning_calibration_20261108 import held_safe_cutoffs, prefix_mapping
from .commissioning_intra_20261110 import CALIBRATION, CAPS, PARENT_ARM, PARENT_RUN, STARTUP, load_frozen
from .commissioning_latent_rate_20261110 import read_parent_curve, score_warning, supervised_rows
from .commissioning_transfer_20261108 import contrast, shot_weights
from .event_prefix_probe import LABEL_SHA256, assign_events
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition
from .source_event_tasks_20261102.data import load_events


CUE = ('Mir_avg_amp', 'Mir_avg_fre', 'n=1 amplitude', 'dx_a', 'dy_a', 'v_loop',
       'sxr_var', 'xuv_var', 'sxr_skew', 'xuv_skew')
COLUMNS = [io23.RAW23.index(k) for k in CUE]
CONTEXT = [io23.RAW23.index(k) for k in ('ip', 'Bt')]
REQUIRED = [io23.RAW23.index(k) for k in ('Mir_avg_amp', 'sxr_var', 'xuv_var')]
MODES = ('static', 'real', 'first', 'lag')
READOUTS = ('event', 'shuffle')
VARIANTS = {**{f'event_{m}': ('event', m) for m in MODES},
            **{f'shuffle_{m}': ('shuffle', m) for m in MODES[:2]}}
FIT_MODES = ('static', 'real')
STEPS = 256
FIRST_EFFECT = 153


def directional_features(z):
    """Current signed change and the part already moving that way 32 points ago."""
    p = np.zeros_like(z)
    a, b = z[32:], z[:-32]
    p[32:] = np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(abs(a), abs(b)), 0.)
    return np.concatenate((z, p), axis=1)


def prefix_context(z, mask):
    """Present Ip, Bt and signed current motion. Only used to match source rows."""
    now = np.arange(1, len(z) + 1)
    mid = np.maximum(now - 16, 0)
    old = np.maximum(now - 64, 0)
    n = np.r_[0, np.cumsum(mask[:, 0])]
    v = np.r_[0., np.cumsum(np.where(mask[:, 0], z[:, 0], 0.))]
    fresh, earlier = n[now] - n[mid], n[mid] - n[old]
    delta = (v[now] - v[mid]) / np.maximum(fresh, 1) - (v[mid] - v[old]) / np.maximum(earlier, 1)
    valid = mask.all(1) & (fresh >= 8) & (earlier >= 24)
    return np.column_stack((z, delta)), valid


def event_endpoints(times, family, active, *, disruptive, down):
    """Auxiliary selection only: never substitute the event tag for a risk label."""
    i = np.arange(len(times))
    keep = active & (i >= 127) & (family >= 0)
    if not disruptive:
        return [(int(family[k]), int(k)) for k in np.flatnonzero(keep & (i % 8 == 0))]
    lead = np.round((down - times) * 1000., 9)
    keep &= (lead > 30.) & (lead <= 125.)
    out = []
    for f in np.unique(family[keep]):
        take = np.flatnonzero(keep & (family == f))
        out.append((int(f), int(take[np.abs(lead[take] - 77.5).argmin()])))
    return out


def source_cues(center, scale):
    """Learn only from TRAIN; no annotated interval is called normal by absence."""
    events, _ = load_events()
    training = io23.source_roles()['train']
    references, records = [], []
    for sid in training:
        d, down = io23.read_outcome('J-TEXT', sid)
        if d and sid not in events:
            continue
        s = simple_data.read_shot('J-TEXT', sid)
        z, m = roughness.coordinates(s['values'], s['available'], center, scale)
        dynamics = (z[:, COLUMNS], m[:, COLUMNS])
        if sid not in events:
            references.append(dynamics)
        else:
            records.append((s['shot'], d, down, s['times'], dynamics,
                            prefix_context(z[:, CONTEXT], m[:, CONTEXT]), m[:, REQUIRED].all(1)))
    prior = roughness.fit_prior(references)
    by_outcome = {False: [], True: []}
    for sid, d, down, times, dynamics, (ctx, valid), observed in records:
        family, _ = assign_events(times, events[sid])
        endpoints = event_endpoints(times, family, observed & valid, disruptive=d, down=down)
        if not endpoints:
            continue
        ratio, _ = roughness.summarize(*dynamics, prior, mode='static')
        x = directional_features(ratio)
        by_outcome[d].extend(dict(shot=sid, family=f, endpoint=t, context=ctx[t], feature=x[t])
                             for f, t in endpoints)
    pos, safe = by_outcome[True], by_outcome[False]
    matches = pairs.match(pos, safe)
    axes = {k: pairs.fit_axis(matches, shuffled=(k == 'shuffle')) for k in READOUTS}

    # The waveform prior excluded all event shots; both members of every held
    # pair are unseen by the direction fit, including safe donors.
    assignment = {r: j % 4 for group in (pos, safe)
                  for j, r in enumerate(sorted({x['shot'] for x in group}))}

    def partition(rows, fold, is_held):
        return [x for x in rows if (assignment[x['shot']] == fold) == is_held]

    held = []
    for fold in range(4):
        fitted = pairs.match(partition(pos, fold, False), partition(safe, fold, False))
        retained = pairs.match(partition(pos, fold, True), partition(safe, fold, True))
        if not fitted or not retained:
            raise ValueError(f'empty event-matched source fold {fold}')
        fit_shots = {x['shot'] for p in fitted for x in p}
        held_shots = {x['shot'] for p in retained for x in p}
        if fit_shots & held_shots:
            raise AssertionError('both members of a source pair must be held out')
        held.append(dict(fold=fold, fitted_pairs=len(fitted), held_pairs=len(retained),
                         held_shots=sorted(held_shots), fit_shots=sorted(fit_shots),
                         agreement={k: pairs.pair_agreement(retained, pairs.fit_axis(fitted, shuffled=(k == 'shuffle')))
                                    for k in READOUTS}))
    receipt = dict(scope='JTEXT_TRAIN_ONLY_EVENT_MATCHED_AUXILIARY_NO_LABEL_REPLACEMENT',
                   train_shots=len(training), source_reference_shots=len(references),
                   candidates={'disruptive': len(pos), 'safe': len(safe)},
                   paired_shots={'disruptive': len({p['shot'] for p, _ in matches}),
                                 'safe': len({p['shot'] for _, p in matches})},
                   supported_channels=[k for k, supported in zip(CUE, prior[1], strict=True) if supported],
                   pairs=[[[r['shot'], r['family'], r['endpoint']] for r in p] for p in matches],
                   source_held_folds=held, event_csv_sha256=LABEL_SHA256)
    return axes, receipt


def encode(base, signal, center, scale):
    h, _ = extract_shot(base, signal, 'cuda')
    raw = np.concatenate((h, contrast(signal['values'], signal['available'])), axis=1)
    z, valid = roughness.coordinates(signal['values'][:, COLUMNS], signal['available'][:, COLUMNS],
                                     center[COLUMNS], scale[COLUMNS])
    return raw, (z, valid)


def cue_views(record, prior, modes):
    rows, audits = {}, {}
    for mode in modes:
        ratio, audits[mode] = roughness.summarize(*record[1], prior, mode=mode)
        rows[mode] = directional_features(ratio)
    return rows, audits


def fit_readout(base, records, axes, original):
    origin = torch.cat((base.risk_head.weight.detach().flatten(), torch.zeros(24, device='cuda'),
                        base.risk_head.bias.detach()))
    labels, shot, timely, safe = [], [], [], []
    offset = 0
    for i, (_, _, y, t, s) in enumerate(records):
        labels.append(y)
        shot.append(np.full(len(y), i, int))
        for selected, seq in ((t, timely), (s, safe)):
            z = np.flatnonzero(selected)
            if len(z):
                seq.append(slice(offset + int(z[0]), offset + int(z[-1]) + 1))
        offset += len(y)
    y = np.concatenate(labels)
    weights = torch.from_numpy(shot_weights(np.concatenate(shot), y).astype(np.float32)).cuda()
    target = torch.from_numpy(y.astype(np.float32)).cuda().expand(len(FIT_MODES), -1)
    x = torch.from_numpy(np.concatenate([r[0] for r in records])).cuda()
    features = np.concatenate([r[1] for r in records], axis=1)
    boundary = float(np.log(original / (1. - original)))
    heads, losses = {}, {}
    for k, axis in axes.items():
        cue = torch.from_numpy(features @ axis).cuda()
        coef = torch.nn.Parameter(origin.clone())
        opt = torch.optim.Adam([coef], lr=.02)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            z = (x @ coef[:-2] + coef[-1])[None, :] + coef[-2] * cue
            point = (F.binary_cross_entropy_with_logits(z, target, reduction='none') @ weights).mean()
            alarm = score_warning(z, timely, safe, boundary, deadline=False)
            (point + alarm + .05 * (coef - origin).square().sum()).backward()
            opt.step()
        if not torch.isfinite(coef).all():
            raise FloatingPointError('nonfinite fit')
        heads[k] = coef.detach()
        losses[k] = dict(bce=float(point.detach()), warning=float(alarm.detach()),
                         cue_coefficient=float(coef[-2].detach()))
    return heads, losses


def fit(base, raws, outcomes, ids, mapping, center, scale, axes, original):
    records = {s: encode(base, prepare_actual(raws[s], STARTUP + 1, mapping), center, scale) for s in ids}
    prior = roughness.fit_prior([records[s][1] for s in ids if not outcomes[s][0]])
    rows = []
    for s, record in records.items():
        cue, _ = cue_views(record, prior, FIT_MODES)
        f = np.stack([cue[k] for k in FIT_MODES])
        f, y, t, safe = supervised_rows(f, raws[s]['times'], *outcomes[s])
        _, mask = io23.labels_at(raws[s]['times'], *outcomes[s], 125.)
        rows.append((record[0][mask], f, y, t, safe))
    heads, losses = fit_readout(base, rows, axes, original)
    return prior, heads, losses


@torch.no_grad()
def predict(record, features, axes, heads, variants):
    x = torch.from_numpy(record[0]).cuda()
    results = {}
    for name, (head, mode) in variants.items():
        c = heads[head]
        cue = torch.from_numpy(features[mode] @ axes[head]).cuda()
        results[name] = ((x @ c[:-2] + c[-1]) + c[-2] * cue).sigmoid().cpu().numpy()
    return results


def calibrate(base, raws, outcomes, stats, center, scale, axes, original, out, started):
    held = []
    for sid in (s for s in raws if not outcomes[s][0]):
        train = [s for s in raws if s != sid]
        mapping = prefix_mapping(stats, [raws[s] for s in train])
        prior, heads, _ = fit(base, raws, outcomes, train, mapping, center, scale,
                              {'event': axes['event']}, original)
        record = encode(base, prepare_actual(raws[sid], STARTUP + 1, mapping), center, scale)
        v, _ = cue_views(record, prior, ('static',))
        y = predict(record, v, axes, heads, {'static': ('event', 'static')})['static']
        held.append(dict(held_safe=sid, fit_ids=train, maximum=float(y.max())))
        if len(held) % 3 == 0:
            print(dict(stage='calibration', held=len(held), seconds=time.monotonic()-started), flush=True)
    thresholds = held_safe_cutoffs([r['maximum'] for r in held])
    write(out / 'calibration.json', dict(shared_cutoffs=thresholds, held_safe=held))
    return thresholds


def verify_prefix(base, signal, curves, prior, axes, heads, center, scale, thresholds):
    n = min(211, len(signal['values']))
    prefix = {k: signal[k][:n] for k in ('values', 'available')}
    record = encode(base, prefix, center, scale)
    v, _ = cue_views(record, prior, MODES)
    estimates = predict(record, v, axes, heads, VARIANTS)
    for k, p in estimates.items():
        np.testing.assert_allclose(p, curves[k][:n], rtol=0, atol=1e-5)
        for cutoff in thresholds.values():
            np.testing.assert_array_equal(p >= cutoff, curves[k][:n] >= cutoff)
    return {k: float(np.max(abs(p - curves[k][:n]))) for k, p in estimates.items()}


def run(out):
    configure()
    started = time.monotonic()
    inputs, coef, original, parent_cutoffs, binding = load_frozen()
    base, ckpt = load_parent(DEFAULT_PARENT)
    center, scale = (a.detach().cpu().numpy().copy() for a in (base.center, base.scale))
    base.cuda()
    source_state = state_digest(base)
    if ckpt['threshold'] != original or source_state != inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('source contract mismatch')
    ids, names = inputs['ids'], ('parent', *VARIANTS)
    old_seals = read(PARENT_RUN / 'result.json')['curve_sha256']
    out.mkdir(parents=True)
    (out / 'shots').mkdir()
    write(out / 'protocol.json', dict(scope='EAST360_DEVELOPMENT_EVENT_CUE_AND_ROUGHNESS',
        source_sha256=PARENT_SHA256, **binding, order=ids, startup=STARTUP, fit_shots=ids[:STARTUP],
        original_threshold=original, parent_cutoffs=parent_cutoffs, variants=VARIANTS, steps=STEPS,
        cue_channels=CUE, first_update_index=FIRST_EFFECT, train_views=FIT_MODES,
        no_persistent_update=True, producer_sha256=sha(Path(__file__)),
        core_sha256=sha(Path(roughness.__file__)), pair_sha256=sha(Path(pairs.__file__))))
    axes, source_audit = source_cues(center, scale)
    write(out / 'source_event_pairs.json', source_audit)
    print(dict(stage='source_pairs', pairs=len(source_audit['pairs']), seconds=time.monotonic()-started), flush=True)

    raws, outcomes = {}, {}
    for i, s in enumerate(ids[:STARTUP], 1):
        if sha(io23.path_for('EAST', s)) != inputs['hashes'][str(s)]:
            raise ValueError(f'startup signal changed: {s}')
        raw = simple_data.read_shot('EAST', s, rate_contract='nominal')
        times, _ = read_parent_curve(f'{i:04d}_{s}.npz', old_seals)
        np.testing.assert_array_equal(times, raw['times'])
        raws[s], outcomes[s] = raw, io23.read_outcome('EAST', s)
    if sum(d for d, _ in outcomes.values()) != 8:
        raise ValueError('expected 8 disruptive in ended20')
    thresholds = calibrate(base, raws, outcomes, inputs['mapping']['source_statistics'],
                           center, scale, axes, original, out, started)
    prior, heads, training = fit(base, raws, outcomes, ids[:STARTUP], inputs['mapping'],
                                 center, scale, axes, original)
    with (out / 'parameters.npz').open('xb') as f:
        np.savez(f, prior=prior[0], supported=prior[1], **{f'{k}_axis': v for k, v in axes.items()},
                 **{f'{k}_risk_head': v.cpu().numpy() for k, v in heads.items()})
    bindings = {k: sha(out / k) for k in ('parameters.npz', 'calibration.json', 'source_event_pairs.json')}
    print(dict(stage='fit20_complete', seconds=time.monotonic()-started), flush=True)

    sealed, witnesses, audits = {}, {}, {}
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome before seal')):
        for i, s in enumerate(ids, 1):
            name = f'{i:04d}_{s}.npz'
            times, parent = read_parent_curve(name, old_seals)
            if i <= STARTUP:
                curves = dict.fromkeys(VARIANTS, parent)
            else:
                if sha(io23.path_for('EAST', s)) != inputs['hashes'][str(s)]:
                    raise ValueError(f'EAST signal changed: {s}')
                raw = simple_data.read_shot('EAST', s, rate_contract='nominal')
                np.testing.assert_array_equal(times, raw['times'])
                signal = prepare_actual(raw, i, inputs['mapping'])
                record = encode(base, signal, center, scale)
                x = torch.from_numpy(record[0]).cuda()
                np.testing.assert_array_equal(parent, (x @ coef[:-1] + coef[-1]).sigmoid().cpu().numpy())
                v, audits[str(s)] = cue_views(record, prior, MODES)
                curves = predict(record, v, axes, heads, VARIANTS)
                for k, (h, _) in VARIANTS.items():
                    np.testing.assert_array_equal(curves[k][:FIRST_EFFECT], curves[f'{h}_static'][:FIRST_EFFECT])
                if i in (21, 120, 360):
                    witnesses[str(s)] = verify_prefix(base, signal, curves, prior, axes, heads, center, scale, thresholds)
            path = out / 'shots' / name
            with path.open('xb') as f:
                np.savez_compressed(f, times=times, parent=parent, **curves)
                f.flush()
                os.fsync(f.fileno())
            sealed[name] = sha(path)
            if i % 40 == 0:
                print(dict(stage='sealed', shots=i, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != source_state or any(sha(out / k) != v for k, v in bindings.items()):
        raise AssertionError('persistent state changed')
    write(out / 'seal.json', dict(curve_sha256=sealed, artifact_sha256=bindings, prefix=witnesses))
    per = {c: {n: [] for n in names} for c in CAPS}
    for i, s in enumerate(ids, 1):
        path = out / 'shots' / f'{i:04d}_{s}.npz'
        if sha(path) != sealed[path.name]:
            raise ValueError('curve changed after seal')
        d, down = outcomes[s] if i <= STARTUP else io23.read_outcome('EAST', s)
        with np.load(path, allow_pickle=False) as record:
            for c, by_mode in per.items():
                for k in names:
                    cutoff = original if i <= STARTUP else (parent_cutoffs[c] if k == 'parent' else thresholds[c])
                    by_mode[k].append(dict(shot=s, disruptive=d, down_s=down,
                                           **score_curve(record['times'], record[k], d, down, cutoff)))
    reference = read(CALIBRATION / 'per_shot.json')
    if any(per[c]['parent'] != reference[c][PARENT_ARM] for c in CAPS):
        raise AssertionError('sealed historical father differs')
    metrics = {c: {k: timing_decomposition(v) for k, v in groups.items()} for c, groups in per.items()}
    changes = {c: {k: flips(groups[f'{h}_static'], groups[k]) for k, (h, m) in VARIANTS.items() if m != 'static'}
               for c, groups in per.items()}
    sums = {m: {k: sum(v[m][k] for v in audits.values()) for k in next(iter(audits.values()))[m]} for m in MODES}
    write(out / 'per_shot.json', per)
    write(out / 'roughness_by_shot.json', audits)
    write(out / 'result.json', dict(status='COMPLETED_EVENT_CUE_ROUGHNESS_DEVELOPMENT', metrics=metrics,
                                   changes=changes, roughness=sums, training=training,
                                   seconds=time.monotonic()-started))
    print({c: {k: (v['tp'], v['fp']) for k, v in g.items()} for c, g in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
