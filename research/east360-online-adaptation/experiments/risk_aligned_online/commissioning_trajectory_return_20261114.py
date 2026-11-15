"""Compare ordered departure and return with a matched unordered feedback target."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_feedback_precision_20261113 as precision
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag = precision.group, precision.lag
ARCHIVE, WAIT = precision.ARCHIVE, precision.WAIT
NEW = ('current_5', 'earlier_5', 'ordered_return', 'middle_reversed', 'unordered_magnitude')
ARMS = (*group.OLD, *NEW)
LOCK = ('order', 'startup', 'original', 'calibration', 'source_sha256')
CASES = (70320, 73405, 79649, 71076)


def reverse_middle(level):
    """Reverse only the three middle bins, preserving both ends and every 8D value."""
    return torch.cat((level[:, :1], level[:, 1:-1].flip(1), level[:, -1:]), dim=1)


def soft_magnitude(level):
    return (level.square()+.05**2).sqrt()-.05


def excursion_statistics(level):
    """Expected trend of distance to the pre-issue range, and its unordered mean."""
    distance = soft_magnitude(level)
    order = distance.new_tensor([-2., -1., 0., 1., 2.])[:, None]/6.
    return (distance*order).sum(dim=1), distance.mean(dim=1)


def interior_contrast(level):
    """The only asymmetric component under reversal of the three interior bins."""
    distance = soft_magnitude(level[:, (1, 3), :])
    return distance[:, 1]-distance[:, 0]


def contrast_force(target):
    # In half the squared-distance difference, the entire forecast-independent
    # norm term cancels. A symmetric target gives exactly zero evidence.
    return (target/group.BETA).clamp(-1., 1.)


def feedback_force(prediction, truth):
    return ((truth-prediction.detach())/group.BETA).clamp(-1., 1.)


def blend_directions(directions, *, strength=None):
    """One part current force plus one part auxiliary in unit coefficient coordinates."""
    original = F.normalize(directions[:1].flatten(2), dim=-1, eps=group.EPS)
    auxiliary = F.normalize(directions[1:].flatten(2), dim=-1, eps=group.EPS)
    if strength is not None:
        auxiliary = auxiliary*strength[..., None]
    return torch.cat((directions[:1], (original+auxiliary).reshape_as(directions[1:])))


def estimate(signal, args, decoder, path, teacher, *, conditional_order=False):
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    eligible, fields, truth = group.feedback_waveform(signal, q, args[2], z.device)
    base = args[0]
    states = prior.new_zeros((len(NEW), len(q)+1, base.cfg.window//base.cfg.patch-1, len(group.TRUSTED)))
    force = prior.new_zeros((len(q), lag.H//group.BIN, len(group.USE)))
    _, _, _, common = group.past_force_parts(q, force, eligible)
    geometry = np.zeros((len(q), 3), dtype=np.float32)
    positions = np.flatnonzero(eligible)
    for start in range(0, len(positions), group.BATCH):
        i = positions[start:start+group.BATCH]
        issued, released, state, wave, center, spread = group.signed_window(
            base, signal, q, i, phase, correction, prior, fields, decoder)
        pred, force[i] = precision.signed_residual(wave, truth[i], center, spread)
        _, now, older, _ = group.past_force_parts(q, force, eligible)
        observed = group.signed_levels(truth[i], center, spread)
        _, magnitude = excursion_statistics(pred)
        target, level = excursion_statistics(observed)
        reversed_target, _ = excursion_statistics(reverse_middle(observed))
        gate = torch.as_tensor(common[i], device=z.device)[:, None]
        if conditional_order:
            ordered = interior_contrast(pred)
            measured = interior_contrast(observed)
            real_force = contrast_force(measured)
            size = real_force.square().mean(-1).sqrt()
            strength = torch.stack((size, size, torch.ones_like(size)))
            order_specs = ((ordered, real_force*gate), (ordered, -real_force*gate))
            order_delta = 2.*measured.abs().mean(-1)
        else:
            ordered, _ = excursion_statistics(pred)
            order_specs = ((ordered, feedback_force(ordered, target)*gate),
                           (ordered, feedback_force(ordered, reversed_target)*gate))
            strength = None
            order_delta = (target-reversed_target).abs().mean(-1)
        raw = group.signed_directions(issued, released, state, (
            (pred, group.shape_force(now[i], teacher)*gate[..., None]),
            (pred, group.shape_force(older[i], teacher)*gate[..., None]),
            *order_specs,
            (magnitude, feedback_force(magnitude, level)*gate)))
        ordered_writes = blend_directions(raw[[0, 2, 3, 4]], strength=strength)
        states[:, i+1] = torch.cat((ordered_writes[:1], raw[1:2], ordered_writes[1:]))
        geometry[i] = torch.stack((order_delta,
                                    F.cosine_similarity(raw[2].flatten(1), raw[3].flatten(1)),
                                    F.cosine_similarity(states[2, i+1].flatten(1),
                                                        states[3, i+1].flatten(1))), dim=1).cpu().numpy()
    if not torch.isfinite(states).all():
        raise FloatingPointError('nonfinite diagnostic write')
    return (q, h, z, beta, phase, correction, states, common), geometry


def seal_run(out, *, conditional_order=False):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    ref, _, args, decoder, path = group.load_reference(lock)
    previous, seal = (read(ARCHIVE/f'{k}.json') for k in ('protocol', 'seal'))
    asset = precision.delay.HISTORY/'teaching.json'
    if (any(ref[k]!=previous[k] for k in LOCK) or len(seal['curve_sha256'])!=360
            or previous['history_seal']!=sha(precision.delay.HISTORY/'seal.json')
            or previous['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or read(asset.parent/'protocol.json')['teaching_sha256']!=sha(asset)):
        raise ValueError('locked 360 parent and historical teacher required')
    (out/'shots').mkdir(parents=True)
    copyfile(__file__, out/'executed.py')
    base = args[0]
    frozen = [state_digest(m) for m in args[:2]]
    teacher = torch.as_tensor(read(asset)['weights'], device=base.center.device, dtype=torch.float32)
    tau = tuple(previous['calibration'].values())
    feedback = ('Use the difference of squared distances from the forecast to the true target and the '
                'interior reversal. All shared terms cancel; compare opposite preference gradients.'
                if conditional_order else 'Use the temporal trend of absolute deviation; reverse the '
                'three interior bins only for sham.')
    write(out/'protocol.json', dict(scope='EAST360_ORDERED_RETURN_FEEDBACK_DEVELOPMENT',
        **{k:previous[k] for k in LOCK}, arms=ARMS, archive=str(ARCHIVE), archive_seal=sha(ARCHIVE/'seal.json'),
        labels='untouched before full-stream seal; no feedback budget used or cross-shot learning',
        conditional_order=conditional_order,
        feedback='25 measurements in five consecutive bins. Same preissue reference for forecast and target. '
                 f'{feedback} Unordered mean of the same deviations is the control for overall magnitude.',
        blend='Unit-coefficient directions for existing and auxiliary forces, fixed archived actual token L2. '
              'The conditional preference has equal strength on both sides, scaled by observed contrast RMS.',
        release_offset=lag.H+1+WAIT, handoff=base.cfg.patch))
    files, witnesses = {}, {}
    max_error = dict(parent=0., archived_control=0., dose=0., prefix=0.)
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('outcome forbidden before curve seal')):
        for rank, sid in enumerate(previous['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            reference = ARCHIVE/'shots'/name
            if sha(reference)!=seal['curve_sha256'][name]:
                raise ValueError('changed matched reference curve')
            with np.load(reference, allow_pickle=False) as f:
                fixed, parent, release, eligible = (f[k] for k in ('fixed', 'bias_only', 'release', 'eligible'))
                dose = f['dose'][precision.delay.WAITS.index(WAIT)]
                controls = np.stack([f[k] for k in NEW[:2]])
            if rank<=previous['startup']:
                curves = np.broadcast_to(parent, (len(NEW), len(parent)))
                geometry = np.zeros((len(release), 3), dtype=np.float32)
            else:
                if sha(io23.path_for('EAST', sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('changed EAST input')
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                track, geometry = estimate(signal, args, decoder, path, teacher, conditional_order=conditional_order)
                np.testing.assert_array_equal(track[0]+lag.H+1, release)
                np.testing.assert_array_equal(track[-1], eligible)
                curves, pe, de = precision.render(base, signal, track, parent, dose)
                np.testing.assert_allclose(curves[:2], controls, atol=3e-6, rtol=0.)
                for t in tau:
                    np.testing.assert_array_equal(curves[:2]>=t, controls>=t)
                max_error['parent'] = max(max_error['parent'], pe)
                max_error['dose'] = max(max_error['dose'], de)
                max_error['archived_control'] = max(max_error['archived_control'],
                                                   float(np.abs(curves[:2]-controls).max()))
                if rank in (21, 120, 360) or sid in CASES:
                    active_alarm = np.flatnonzero((curves[2]>=previous['calibration']['0.2']) & (dose>0.))
                    n = min(len(parent), int(active_alarm[0])+2 if len(active_alarm) else 550)
                    prefix = {k:signal[k][:n] for k in ('values', 'available')}
                    short, _ = estimate(prefix, args, decoder, path, teacher,
                                        conditional_order=conditional_order)
                    reduced, *_ = precision.render(base, prefix, short, parent[:n], dose[:n])
                    np.testing.assert_allclose(reduced, curves[:, :n], atol=3e-6, rtol=0.)
                    for t in tau:
                        np.testing.assert_array_equal(reduced>=t, curves[:, :n]>=t)
                    max_error['prefix'] = max(max_error['prefix'], float(np.abs(reduced-curves[:, :n]).max()))
                    witnesses[str(sid)] = n
            dest = out/'shots'/name
            with dest.open('xb') as stream:
                np.savez_compressed(stream, fixed=fixed, bias_only=parent, release=release, eligible=eligible,
                    dose=dose, geometry=geometry, **dict(zip(NEW, curves, strict=True)))
            files[name] = sha(dest)
            if rank==21 or rank%60==0:
                print(dict(stage='seal_trajectory_return', rank=rank, seconds=round(time.monotonic()-start, 2)),
                      flush=True)
    if frozen!=[state_digest(m) for m in args[:2]]:
        raise ValueError('the fixed parent changed')
    write(out/'seal.json', dict(curves_sha256=files, witnesses=witnesses, max_error=max_error,
                              elapsed_s=time.monotonic()-start))


def score(out):
    protocol, seal = (read(out/f'{k}.json') for k in ('protocol', 'seal'))
    if len(seal['curves_sha256'])!=360 or protocol['archive_seal']!=sha(ARCHIVE/'seal.json'):
        raise ValueError('complete matched stream must be sealed before scoring')
    archived = read(ARCHIVE/'per_shot.json')
    pr = {band:{k:[] for k in ARMS} for band in protocol['calibration']}
    witnesses, geometry, late = {}, [], []
    dose_count = 0
    tau = protocol['calibration']['0.2']
    for rank, sid in enumerate(protocol['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        file = out/'shots'/name
        if sha(file)!=seal['curves_sha256'][name]:
            raise ValueError('changed score artifact')
        y, down, times = group.hdf._raw_truth(sid)
        with np.load(file, allow_pickle=False) as data:
            release, eligible, dose = (data[k] for k in ('release', 'eligible', 'dose'))
            active = dose>0.
            dose_count += int(active.sum())
            index = np.searchsorted(release+WAIT, np.arange(len(times)), side='right')
            if not np.r_[False, eligible][index[active]].all():
                raise ValueError('write used unavailable measurement')
            geometry.append(data['geometry'][eligible])
            for k in NEW:
                np.testing.assert_array_equal(data[k][~active], data['bias_only'][~active])
            for band, threshold in protocol['calibration'].items():
                threshold = protocol['original'] if rank<=protocol['startup'] else threshold
                for key in ARMS:
                    r = dict(shot=sid, disruptive=y, down_s=down,
                             **score_curve(times, data[key], y, down, threshold))
                    if key in (*group.OLD, *NEW[:2]) and r!=archived[band][key][rank-1]:
                        raise ValueError('reference first-alarm score changed')
                    pr[band][key].append(r)
            if sid in CASES:
                witnesses[str(sid)] = {k:pr['0.2'][k][-1]['lead_ms'] for k in ARMS}
            if y and rank>protocol['startup'] and not pr['0.2']['current_5'][-1]['timely']:
                t = np.round((down-times)*1000., 9)
                j = np.flatnonzero(active & (t>30.) & (t<=125.))
                if len(j):
                    levels = logit(np.clip(np.stack([data[k][j] for k in NEW]), 1e-6, 1.-1e-6))
                    late.append(dict(shot=sid, candidates=len(j), gap=float(logit(tau)-levels[0].max()),
                        max_lift={k:float((v-levels[0]).max()) for k, v in zip(NEW[2:], levels[2:], strict=True)},
                        near_cross={k:bool((v>=logit(tau)).any()) for k,v in zip(NEW, levels, strict=True)}))
                else:
                    late.append(dict(shot=sid, candidates=0))
    metrics = {b:{k:lead.summarize(v) for k,v in rows.items()} for b, rows in pr.items()}
    pairs = [(a, b) for a in ('bias_only', 'current_5', 'middle_reversed', 'unordered_magnitude')
             for b in NEW[2:] if a!=b]
    paired = {b:{f'{a}__to__{k}':dict(flips=flips(rows[a], rows[k]), lead=lead.paired(rows[a], rows[k]))
                 for a,k in pairs} for b,rows in pr.items()}
    values = np.concatenate(geometry)
    write(out/'per_shot.json', pr)
    write(out/'result.json', dict(metrics=metrics, paired=paired, write_endpoints=dose_count,
        raw_verified=360*len(ARMS)*len(metrics), released_windows=len(values),
        geometry=dict(columns=('order_target_delta', 'auxiliary_gradient_cosine', 'blended_gradient_cosine'),
                      q10_median_q90=np.quantile(values, [.1,.5,.9], axis=0).tolist()),
        witnesses=witnesses, original_late_or_missed=late))
    print({b:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in rows.items()} for b, rows in metrics.items()},
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditional-order', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if not args.score_only:
        seal_run(args.output.resolve(), conditional_order=args.conditional_order)
    score(args.output.resolve())
