"""Start on the first complete feedback and isolate the subsequent handoff."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_group_token_20261112 as group
from . import commissioning_innovation_metric_20261112 as metric
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest
from .source_missing_diagnostics_20261103 import diagnostic_view


lag, layer = group.lag, group.layer
HISTORY = io23.ROOT/'probe_outputs/commissioning_signed_innovations_east360_20261112_v1'
SOURCE = io23.ROOT/'probe_outputs/commissioning_gradient_metric_east360_20261112_v1'
ARMS = (*group.OLD, group.HISTORY, *group.FIRST)
RISK_MASS = {'hard': .25, 'easy': .25, 'safe': .50}


def bootstrap_slots(eligible, common):
    """Use early feedback only until the first historical innovation is possible."""
    return eligible & ~np.maximum.accumulate(common)


def bootstrap_endpoints(release, early, common, length, patch_size):
    t = np.arange(length)
    slots = np.searchsorted(release, t, side='right')
    from_early = np.r_[False, early][np.maximum(slots-1, 0)]
    elapsed = t-np.r_[0, release][slots]
    return np.r_[False, early][slots] | (from_early & np.r_[False, common][slots] & (elapsed<patch_size))


def stratum_mass(records):
    names = [r['stratum'] for r in records]
    counts = {k: names.count(k) for k in RISK_MASS}
    missing = [k for k, n in counts.items() if not n]
    if missing:
        raise ValueError(f'no eligible source examples in strata: {missing}')
    return np.array([RISK_MASS[k]/counts[k] for k in names], dtype=np.float32)


def directions(signal, args, decoder, path, weights, *, source=False):
    """Compute only the legal early moves and the first mature handoff move."""
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path, source=source)
    eligible, fields, waveform = group.feedback_waveform(signal, q, args[2], beta.device)
    force = prior.new_zeros((len(q), lag.H//group.BIN, len(group.USE)))
    _, _, _, common = group.past_force_parts(q, force, eligible)
    early = bootstrap_slots(eligible, common)
    boundary = np.r_[False, early][:-1] & common
    indices = np.flatnonzero(early | boundary)
    states = prior.new_zeros((2, len(q)+1, base.cfg.window//base.cfg.patch-1, len(group.TRUSTED)))
    for start in range(0, len(indices), group.BATCH):
        i = indices[start:start+group.BATCH]
        issued, released, state, wave, center, spread = group.signed_window(
            base, signal, q, i, phase, correction, prior, fields, decoder)
        pred = group.signed_levels(wave, center, spread)
        sf = (group.signed_levels(waveform[i], center, spread)-pred.detach()).div(group.BETA).clamp(-1., 1.)
        force[i] = sf
        _, innovation, _, _ = group.past_force_parts(q, force, eligible)
        initial = torch.as_tensor(early[i], device=state.device)[:, None, None]
        exact = torch.where(initial, sf, innovation[i])
        wrong = torch.where(initial, sf.roll(1, dims=1), innovation[i])
        states[:, i+1] = group.signed_directions(issued, released, state, (
            (pred, group.shape_force(exact, weights)), (pred, group.shape_force(wrong, weights))))
    if not torch.isfinite(states).all():
        raise FloatingPointError('nonfinite first-feedback direction')
    return q, h, z, beta, phase, correction, states, early, common


def historical_episode(raw, truth, args, decoder, path, weights, stratum):
    view = diagnostic_view({k: raw[k] for k in ('values', 'available')}, 'no_MNM_CIII')
    q, h, z, beta, phase, correction, states, early, _ = directions(
        view, args, decoder, path, weights, source=True)
    base, positive = args[0], truth['is_disrupt']
    lead = (truth['down']-raw['times'])*1000. if positive else None
    valid = truth['valid'] & truth['y'].astype(bool) if positive else truth['valid']
    ends = metric.source_peaks(z.cpu().numpy(), q+lag.H+1, early, valid, base.cfg.patch, lead_ms=lead)
    if not len(ends):
        return None
    slot, _ = lag.delayed_slots(q, ends)
    x, mask = lag._windows(base, view, ends)
    a, b = states[:, slot]
    return (x, mask, phase[slot], correction[slot], z[ends]-h[ends]@beta, a, b), dict(
        shot=raw['shot'], positive=positive, stratum=stratum, endpoints=ends.tolist(),
        release=(q[slot-1]+lag.H+1).tolist(), lead_ms=lead[ends].tolist() if positive else None)


def teach(args, decoder, path, weights, threshold):
    clock_path, static_path = (SOURCE/k for k in ('source_clock_coverage.json', 'source_positive_support.json'))
    clock = read(clock_path)
    if (clock['audited_static_support_sha256'] != sha(static_path)
            or read(static_path)['threshold'] != threshold):
        raise ValueError('source hard-shot coverage is not bound to the original threshold')
    history = read(group.REFERENCE/'historical.json')['orientation']
    roles, advanced = io23.source_roles(), io23.advanced_times()
    collections, audit = {}, {}
    for role in ('train', 'val'):
        positive = [(r['shot'], 'easy' if r['timely']['no_MNM_CIII'] else 'hard')
                    for r in clock['positive_shots'][role] if r['clock']['single_complete25_with_reference']]
        safe = [(r['shot'], 'safe') for r in history[role]['manifest'] if not r['positive']]
        requested = positive+safe
        allowed, used, missing = set(roles[role]), [], []
        if len({sid for sid, _ in requested}) != len(requested):
            raise ValueError('source teaching shot duplicated')
        for sid, stratum in requested:
            if sid not in allowed:
                raise ValueError('first feedback source shot outside the named split')
            raw = simple_data.read_shot('J-TEXT', sid)
            truth = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
            if truth['is_disrupt'] != (stratum != 'safe'):
                raise ValueError('source stratum disagrees with raw HDF5')
            episode = historical_episode(raw, truth, args, decoder, path, weights, stratum)
            if episode is None:
                missing.append(dict(shot=sid, stratum=stratum))
            else:
                used.append(episode)
        records = [r for _, r in used]
        collections[role] = used
        audit[role] = dict(requested=len(requested), missing=missing, used=records,
                           strata={k: sum(r['stratum']==k for r in records) for k in RISK_MASS})
        print(dict(stage='first_feedback_source_support', role=role, strata=audit[role]['strata']), flush=True)
    base = args[0]
    masses = stratum_mass(audit['train']['used'])
    fit, losses = metric.fit(base, collections['train'], float(logit(threshold)),
                             mass=torch.as_tensor(masses, device=base.center.device))
    for role, rows in collections.items():
        audit[role]['evaluation'] = metric.validate(base, rows, fit, float(logit(threshold)),
                                                    arms=('parent', *group.FIRST))
        hard = [r for r in rows if r[1]['stratum'] == 'hard']
        audit[role]['hard_evaluation'] = metric.validate(base, hard, fit, float(logit(threshold)),
                                                         arms=('parent', *group.FIRST))
    receipt = dict(source_coverage_sha256=sha(clock_path), source_positive_support_sha256=sha(static_path),
                   evidence=audit, mass=RISK_MASS, factors=fit.cpu().tolist(), loss_trace=losses,
                   optimizer=dict(rank=metric.RANK, epochs=metric.EPOCHS, batch=metric.BATCH,
                                  seed=metric.SEED, lr=metric.LR, temperature=metric.TEMPERATURE,
                                  penalty=metric.PENALTY), fitter_sha256=sha(Path(metric.__file__)))
    print(dict(stage='first_feedback_source_frozen', validation=audit['val']['evaluation']), flush=True)
    return fit, receipt


def predict(signal, args, decoder, path, parent, old, weights, factors):
    q, h, z, beta, phase, correction, states, early, common = directions(
        signal, args, decoder, path, weights)
    release = q+lag.H+1
    np.testing.assert_array_equal(release, old['release'])
    np.testing.assert_array_equal(common, old['eligible'])
    base = args[0]
    area = bootstrap_endpoints(release, early, common, len(parent), base.cfg.patch)
    curves = np.broadcast_to(old['innovation_current'], (len(group.FIRST), len(parent))).copy()
    dose = old['dose'].copy()
    learner = torch.as_tensor(np.r_[False, early], device=states.device)[None, :, None, None]
    expanded = torch.cat((states, torch.where(learner, group.gradient_transform(states, factors), states)))
    changed = np.flatnonzero(area)
    parent_error = dose_error = 0.
    with torch.no_grad():
        for start in range(0, len(changed), group.BATCH):
            i = changed[start:start+group.BATCH]
            slot, _ = lag.delayed_slots(q, i)
            x, mask = lag._windows(base, signal, i)
            fraction = group.handoff_fraction(release[slot-1], i, base.cfg.patch)
            handoff = expanded[:, slot-1], torch.as_tensor(fraction, device=x.device, dtype=x.dtype)
            hidden, actual, error = group.forward_hidden(base, x, mask, layer.patcher(phase[slot], correction[slot]),
                                                        writes=expanded[:, slot], handoff=handoff)
            p = (z[i] + ((hidden-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
            parent_error = max(parent_error, float(np.max(np.abs(p[0]-parent[i]))))
            dose_error = max(dose_error, float(error))
            dose[i] = actual.cpu().numpy()
            curves[:, i] = np.where(dose[i]>0., p[1:], parent[i])
    if parent_error>3e-6 or dose_error>1e-5 or not np.isfinite(curves).all():
        raise ValueError('first-feedback curve or matched dose check failed')
    return curves, dose, early|common, area, parent_error, dose_error


def run(out):
    lag.configure()
    lock = lag.load_inputs()
    parent, parent_seal, args, decoder, path = group.load_reference(lock)
    history, hist_seal = (read(HISTORY/k) for k in ('protocol.json', 'seal.json'))
    teacher = read(HISTORY/'teaching.json')
    fixed = [state_digest(m) for m in args[:2]]
    fixed_keys = ('order', 'startup', 'original', 'calibration', 'source_sha256')
    if (any(parent[k] != history[k] for k in fixed_keys)
            or history['reference_seal'] != sha(group.REFERENCE/'seal.json')
            or history['scope'] != 'EAST360_SAME_SHOT_FORCE_INNOVATION_DEVELOPMENT'
            or tuple(history['arms']) != (*group.OLD, *group.INNOVATION)
            or history['teaching_sha256'] != sha(HISTORY/'teaching.json')
            or len(hist_seal['curve_sha256']) != 360):
        raise ValueError('first-feedback replay requires the exact sealed innovation baseline')
    if (read(SOURCE/'protocol.json')['source_sha256'] != parent['source_sha256']
            or read(SOURCE/'source_positive_support.json')['source_sha256'] != parent['source_sha256']):
        raise ValueError('source difficulty was measured on a different frozen base')
    weights = torch.as_tensor(teacher['weights'], device=args[0].center.device, dtype=torch.float32)
    factors, receipt = teach(args, decoder, path, weights, parent['original'])
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed_probe.py')
    copyfile(Path(group.__file__), out/'executed_group.py')
    copyfile(Path(metric.__file__), out/'executed_fitter.py')
    write(out/'teaching.json', teacher)
    write(out/'bootstrap_fit.json', receipt)
    p = dict(scope='EAST360_FIRST_COMPLETE_FEEDBACK_DEVELOPMENT', **{k: parent[k] for k in fixed_keys},
             arms=ARMS, reference=str(group.REFERENCE), reference_seal=history['reference_seal'],
             historical_reference=str(HISTORY), historical_seal=sha(HISTORY/'seal.json'),
             executed_sha256=sha(out/'executed_probe.py'), group_sha256=sha(out/'executed_group.py'),
             bootstrap_fit_sha256=sha(out/'bootstrap_fit.json'), teaching_sha256=sha(out/'teaching.json'),
             horizon=lag.H, stride=lag.STRIDE, handoff=args[0].cfg.patch,
             availability='complete25, existing observed 125-point reference, q+26; no earlier measurements used',
             early='before first computable historical innovation; unchanged historical curve after one handoff',
             control='cyclically shift five temporal bins of the same completed signed force; same dose and clock',
             truth='no EAST outcomes in prediction; original HDF5 used after sealing all 360')
    write(out/'protocol.json', p)
    hashes, witnesses, errors = {}, {}, []
    began = time.monotonic()
    with patch.object(io23, 'read_outcome', side_effect=RuntimeError('EAST outcome unavailable before sealing')):
        for rank, sid in enumerate(p['order'], 1):
            name = f'{rank:04d}_{sid}.npz'
            f, j = (r/'shots'/name for r in (group.REFERENCE, HISTORY))
            if (sha(f) != parent_seal['curve_sha256'][name] or sha(j) != hist_seal['curve_sha256'][name]
                    or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]):
                raise ValueError('fixed data or historical curves changed')
            with np.load(f, allow_pickle=False) as ref, np.load(j, allow_pickle=False) as data:
                t, archive = ref['times'], ref[group.bias.BIAS]
                old = {k: data[k] for k in ('release', 'eligible', 'dose', 'innovation_current')}
                np.testing.assert_array_equal(old['release'], ref['release'])
            if rank <= p['startup']:
                curves, dose, eligible = np.broadcast_to(archive, (len(group.FIRST), len(t))), old['dose'], old['eligible']
                area = np.zeros(len(t), dtype=bool)
            else:
                raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
                np.testing.assert_array_equal(t, raw['times'])
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                curves, dose, eligible, area, err, d = predict(signal, args, decoder, path, archive, old, weights, factors)
                errors.append((err, d))
                if rank in (21, 120, 360):
                    end = min(len(t), 390)
                    short = {k: signal[k][:end] for k in ('values', 'available')}
                    keep = old['release']<end
                    clipped = {k: v[keep] if k in ('release', 'eligible') else v[:end] for k, v in old.items()}
                    a, b, c, d, *_ = predict(short, args, decoder, path, archive[:end], clipped, weights, factors)
                    np.testing.assert_allclose(a, curves[:, :end], rtol=0., atol=3e-6)
                    np.testing.assert_allclose(b, dose[:end], rtol=0., atol=2e-6)
                    np.testing.assert_array_equal(c, eligible[keep])
                    np.testing.assert_array_equal(d, area[:end])
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(a>=tau, curves[:, :end]>=tau)
                    witnesses[str(sid)] = end
            legal = np.r_[False, eligible][np.searchsorted(old['release'], np.arange(len(t)), side='right')]
            if (dose[~legal]>0.).any():
                raise ValueError('first state applied before an eligible release')
            f = out/'shots'/name
            with f.open('xb') as handle:
                np.savez_compressed(handle, release=old['release'], eligible=eligible, dose=dose, bootstrap=area,
                                    history_dose=old['dose'], history_innovation=old['innovation_current'],
                                    auxiliary=np.empty((0,len(ARMS)-1), np.float32),
                                    **dict(zip(group.FIRST, curves, strict=True)))
            hashes[name] = sha(f)
            if rank==21 or rank%60==0:
                print(dict(stage='first_feedback_seal', rank=rank, seconds=round(time.monotonic()-began, 2)), flush=True)
    if fixed != [state_digest(m) for m in args[:2]]:
        raise ValueError('frozen source or inherited adapter changed')
    err, dose = np.max(errors, axis=0)
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses, max_parent_error=err,
                               max_token_norm_error=dose, seconds=time.monotonic()-began))
    group.score(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out.resolve())
