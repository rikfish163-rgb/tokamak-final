"""Replay support-gated spectral motion with a joint, equal-dose time control."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_record_dynamics_east_20261112 as parent
from .paper131_alarm_audit import alarm_endpoints


lag, guard, two, bias = parent.lag, parent.guard, parent.two, parent.bias
REFERENCE, PARENT = parent.REFERENCE, parent.PARENT
NEW = ('spectral_true', 'spectral_half_shift', 'other_same_gate', 'all_same_gate')
ARMS = (*parent.OLD, *NEW)
PAIRS = tuple((PARENT, k) for k in NEW) + tuple((k, NEW[0]) for k in NEW[1:])
read, sha, write = parent.read, parent.sha, parent.write


def spectral_counterfactual(motion, supported):
    """Rotate paired frequency/amplitude values halfway in supported-sample order."""
    joint = supported.all(-1)
    t = motion.shape[1]
    i = torch.arange(t, device=motion.device)[None, :]
    count = joint.sum(1, keepdim=True)
    order = torch.where(joint, i, i + t).argsort(1)[..., None].expand_as(motion)
    source = torch.where(i < count, (i + count // 2) % count.clamp_min(1), i)
    shifted = motion.gather(1, order).gather(1, source[..., None].expand_as(motion))
    moved = torch.empty_like(motion).scatter(1, order, shifted)
    contrast = (count[:, 0] >= 2) & ((moved != motion) & joint[..., None]).any((1, 2))
    return moved, joint, contrast


def active_slots(q, eligible, ends):
    slots, _ = lag.delayed_slots(q, ends)
    points = np.flatnonzero(np.r_[False, eligible][slots])
    return points, slots[points] - 1


def proposals(raw, signal, q, keys, predicted, measured, args, decoder, path, fitted):
    base = args[0]
    context, target = two.observation(raw, signal, q, args)
    spec, other = list(two.SPECTRAL), list(two.OTHER)
    arrays = {k: torch.as_tensor(v, device=base.center.device) for k, v in target.items()}
    y, obs = arrays['motion'], arrays['motion_observed']
    wrong, joint, contrast = spectral_counterfactual(y[..., spec], obs[..., spec])
    eligible = contrast & obs[..., other].any((1, 2))
    slot_indices = eligible.nonzero().flatten().cpu().numpy()
    matrices, residues = [], []
    for start in range(0, len(q), 128):
        sl = slice(start, start + 128)
        x, mask = lag._windows(base, signal, q[sl])
        theta = x.new_zeros((len(x), two.C, two.C)).requires_grad_()
        h = lag._features(base, x, mask, theta)

        def to(a):
            return torch.as_tensor(a[sl], device=x.device)

        yh, yt = lag._predict(h, to(predicted), decoder), to(measured)
        residues.append((yt-yh.detach()).mean(1))
        use = eligible[sl]
        if not use.any():
            continue
        old = parent.teacher.grouped_error(
            lag.ordered_signature(lag.motion(yh[use])), lag.ordered_signature(lag.motion(yt[use])))
        grads = torch.stack([torch.autograd.grad(old[:, j].sum(), theta, retain_graph=True)[0][use]
                             for j in range(parent.teacher.GROUPS)], 1)
        old_mat = -lag.RADIUS * guard.signed_direction(*guard.basis(grads), to(keys)[use], path)
        logits, value = two.forecast(h[use], to(context)[use], fitted)
        record, _ = two._mean(F.binary_cross_entropy_with_logits(
            logits, arrays['status'][sl][use], reduction='none'), arrays['status_observed'][sl][use])
        actual = F.smooth_l1_loss(value, y[sl][use], beta=.2, reduction='none')
        spectrum, _ = two._mean(actual[..., spec], joint[sl][use][..., None].expand_as(actual[..., spec]))
        nonspec, _ = two._mean(actual[..., other], obs[sl][use][..., other])
        placebo, _ = two._mean(
            F.smooth_l1_loss(value[..., spec], wrong[sl][use], beta=.2, reduction='none'),
            joint[sl][use][..., None].expand_as(actual[..., spec]))
        gs, ga, gb, gc = [torch.autograd.grad(loss.sum(), theta, retain_graph=i < 3)[0][use]
                          for i, loss in enumerate((record, spectrum, placebo, nonspec))]
        directions = (ga, gb, gc, .5*(ga+gc))
        matrices.append(torch.stack((old_mat, *(-lag.RADIUS * guard._unit(two.isolate(g, gs))
                                                for g in directions))).detach())
    mat = torch.cat(matrices, 1) if matrices else base.center.new_empty((len(NEW)+1, 0, two.C, two.C))
    errors = torch.cat(residues) if residues else base.center.new_empty((0, two.C))
    return mat, errors.cpu().numpy(), slot_indices, joint.sum(1).cpu().numpy()


@torch.no_grad()
def forward(base, x, mask, mats, correction):
    n, arms = len(x), len(mats)
    outputs, trace = [], []

    def adjust(_layer, inputs):
        raw = inputs[0][:n]
        old = lag.patch_input(raw, mats[0])
        delta = old-raw
        directions = torch.stack([lag.patch_input(raw, z)-raw for z in mats[1:]])
        moved, use = two.equal_rotation(delta, directions)
        old_bias = bias.bias_patches(old, correction, delta.flatten(1).square().mean(1).sqrt())
        common = old_bias-old
        new = raw+moved+common
        original = (old_bias-raw).flatten(1).square().mean(1).sqrt()
        new_dose = (new-raw).flatten(2).square().mean(2).sqrt()
        trace.extend((use, (new_dose-original).abs().amax()))
        return (torch.cat((old_bias, new.flatten(0, 1))),)

    a = base.embedding.register_forward_pre_hook(adjust)
    b = base.risk_head.register_forward_pre_hook(lambda _m, z: outputs.append(z[0]))
    try:
        base(x.repeat(arms, 1, 1), mask.repeat(arms, 1, 1))
    finally:
        b.remove()
        a.remove()
    return outputs[0].reshape(arms, n, -1), *trace


def predict(raw, signal, args, decoder, path, fitted, archived):
    base = args[0]
    tr, h, z, beta = guard._track(signal, args)
    q, key, initial, target = parent.forecast_track(tr, args)
    mat, residual, indices, counts = proposals(raw, signal, q, key, initial, target, args, decoder, path, fitted)
    eligible = np.zeros(len(q), bool)
    eligible[indices] = True
    corrected, _ = bias.past_bias(q, residual)
    b = torch.as_tensor(corrected, device=beta.device)*torch.as_tensor(args[2][3], device=beta.device)
    active, origin = active_slots(q, eligible, np.arange(len(z)))
    h, z = (torch.as_tensor(a, device=beta.device) for a in (h, z))
    curves = np.broadcast_to(archived, (len(NEW), len(z))).copy()
    used = np.zeros(len(z), bool)
    err, dose = 0., 0.
    for start in range(0, len(active), parent.eos.BATCH):
        sl = slice(start, start + parent.eos.BATCH)
        i, j = active[sl], origin[sl]
        x, mask = lag._windows(base, signal, i)
        hidden, apply, e = forward(base, x, mask, mat[:, np.searchsorted(indices, j)], b[j])
        v = (z[i]+((hidden-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        use = apply.cpu().numpy()
        curves[:, i[use]] = v[1:, use]
        used[i] = use
        err, dose = max(err, float(np.max(np.abs(v[0]-archived[i])))), max(dose, float(e))
    if err > 3e-6 or dose > parent.TOL or not np.isfinite(curves).all():
        raise ValueError(f'legacy reproduction / equal dose failed: {err:g} / {dose:g}')
    return curves, q + lag.H + 1, used, counts, eligible, err, dose


def run(fit, out):
    began = time.monotonic()
    lock, p, s, args, decoder, path, fitted = parent._load(fit)
    before = [parent.state_digest(v) for v in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__), out/'executed_probe.py')
    write(out/'protocol.json', dict(scope='EAST360_SPECTRAL_SUPPORT_CONTROL', order=lock['ids'],
        startup=p['startup'], calibration=p['calibration'], original=p['original'], arms=ARMS,
        source_sha256=p['source_sha256'], reference_seal=sha(REFERENCE/'seal.json'), reference=str(REFERENCE),
        ref_score=sha(REFERENCE/'per_shot.json'), fit_manifest=sha(fit/'fit.json'),
        fit_checkpoint=sha(fit/'readout.npz'), executed_sha256=sha(Path(__file__)),
        gate='complete H25 at q+26; at least two paired observed spectral changes with a distinct half shift; '
             'other observed motion available; same realized nondegenerate direction gate for all arms',
        change='only temporary input patch direction; same realized angle and RMS; otherwise original parent',
        control='rotate paired spectral values together in observed time order, preserving locations and pairs',
        outcomes='full stream seal before HDF5 outcomes; no persistent training'))
    hashes, witnesses, err, dose = {}, {}, 0., 0.
    with patch.object(parent.io23, 'read_outcome', side_effect=RuntimeError('unsealed outcome')):
        for rank, sid in enumerate(lock['ids'], 1):
            name = f'{rank:04d}_{sid}.npz'
            archive = REFERENCE/'shots'/name
            if sha(archive) != s['curve_sha256'][name] or sha(parent.io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
                raise ValueError('changed raw HDF5 or reference curve')
            with np.load(archive, allow_pickle=False) as f:
                t, r, v = (f[k] for k in ('times', 'release', PARENT))
            if rank <= p['startup']:
                a, used = np.broadcast_to(v, (len(NEW), len(t))), np.zeros(len(t), bool)
                count, gate = np.zeros(len(r), int), np.zeros(len(r), bool)
            else:
                raw = parent.simple_data.read_shot('EAST', sid, rate_contract='nominal')
                signal = lag.prepare_actual(raw, rank, lock['mapping'])
                np.testing.assert_array_equal(raw['times'], t)
                a, rr, used, count, gate, e, d = predict(raw, signal, args, decoder, path, fitted, v)
                np.testing.assert_array_equal(rr, r)
                err, dose = max(err, e), max(dose, d)
                if rank in (21, 120, 360):
                    end = min(len(t), 330)
                    short = [{k: src[k][:end] for k in ('values', 'available')} for src in (raw, signal)]
                    aa, rr, uu, cc, gg, _, _ = predict(*short, args, decoder, path, fitted, v[:end])
                    keep = r < end
                    for short_a, long_a in ((rr, r[keep]), (uu, used[:end]), (cc, count[keep]), (gg, gate[keep])):
                        np.testing.assert_array_equal(short_a, long_a)
                    np.testing.assert_allclose(aa, a[:, :end], atol=3e-6, rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(aa >= tau, a[:, :end] >= tau)
                    witnesses[str(sid)] = end
            output = out/'shots'/name
            with output.open('xb') as handle:
                np.savez_compressed(handle, release=r, paired_samples=count, gate=gate, rotated=used,
                                    **dict(zip(NEW, a, strict=True)))
            hashes[name] = sha(output)
            if rank in (21, 360) or rank % 60 == 0:
                print(dict(stage='seal_spectral', rank=rank, seconds=round(time.monotonic()-began, 2)), flush=True)
    if before != [parent.state_digest(v) for v in args[:2]]:
        raise ValueError('the frozen network weights changed')
    write(out/'seal.json', dict(curve_sha256=hashes, prefix_witnesses=witnesses,
        seconds=time.monotonic()-began, max_legacy_error=err, max_dose_error=dose))


def score(out):
    p, s = [read(out/k) for k in ('protocol.json', 'seal.json')]
    if (p['executed_sha256'] != sha(out/'executed_probe.py') or p['reference_seal'] != sha(REFERENCE/'seal.json')
            or p['ref_score'] != sha(REFERENCE/'per_shot.json') or len(s['curve_sha256']) != 360):
        raise ValueError('sealed predictions changed')
    ref = read(REFERENCE/'seal.json')['curve_sha256']
    legacy = read(REFERENCE/'per_shot.json')
    rows = {band: {k: [] for k in ARMS} for band in p['calibration']}
    confirmation = {b: {k: {n: [0, 0] for n in (2, 3, 5)} for k in (PARENT, *NEW)} for b in rows}
    activity = dict(mature_blocks=0, paired_ge2=0, eligible_blocks=0, rotated_points=0, rotated_shots=0)
    reach, cases = {b: [] for b in rows}, {b: {} for b in rows}
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        a, b = (root/'shots'/name for root in (out, REFERENCE))
        if sha(a) != s['curve_sha256'][name] or sha(b) != ref[name]:
            raise ValueError('changed sealed curve')
        d, down, t = parent.hdf._raw_truth(sid)
        with np.load(a, allow_pickle=False) as new, np.load(b, allow_pickle=False) as old:
            np.testing.assert_array_equal(t, old['times'])
            np.testing.assert_array_equal(new['release'], old['release'])
            u, r, count, gate = (new[k] for k in ('rotated', 'release', 'paired_samples', 'gate'))
            activity['mature_blocks'] += len(r) if rank > p['startup'] else 0
            activity['paired_ge2'] += int((count >= 2).sum())
            activity['eligible_blocks'] += int(gate.sum())
            activity['rotated_points'] += int(u.sum())
            activity['rotated_shots'] += int(u.any())
            for band, tau in p['calibration'].items():
                threshold = p['original'] if rank <= p['startup'] else tau
                for k in ARMS:
                    v = new[k] if k in NEW else old[k]
                    if k in NEW:
                        np.testing.assert_array_equal(v[~u], old[PARENT][~u])
                    item = dict(shot=sid, disruptive=d, down_s=down,
                                **parent.score_curve(t, v, d, down, threshold))
                    if k not in NEW and item != legacy[band][k][rank-1]:
                        raise ValueError('legacy result differs from original HDF5')
                    rows[band][k].append(item)
                    if k in confirmation[band]:
                        for n, counts in confirmation[band][k].items():
                            hit = alarm_endpoints(t, v, threshold, points=n)
                            if hit is not None:
                                counts[0] += int(d and round(float((down-t[hit[1]])*1000.), 9) > 30.)
                                counts[1] += int(not d)
                if sid in (91689, 90321, 64835, 76956, 80801):
                    cases[band][str(sid)] = {k: rows[band][k][-1] for k in (PARENT, *NEW)}
                if rank > p['startup'] and d and not rows[band][PARENT][-1]['timely']:
                    lead = np.round((down-t)*1000., 9)
                    scope = u & (lead > 30.) & (lead <= 125.)
                    gap = lift = None
                    if scope.any():
                        def logit(x):
                            x = np.clip(x.astype(np.float64), 1e-7, 1.-1e-7)
                            return np.log(x)-np.log1p(-x)
                        orig = logit(old[PARENT][scope])
                        gap = float(logit(np.asarray(tau))-orig.max())
                        lift = float((logit(new[NEW[0]][scope])-orig).max())
                    reach[band].append(dict(shot=sid, supported_points=int(scope.sum()), logit_gap=gap, logit_lift=lift))
    write(out/'per_shot.json', rows)
    write(out/'lead_sensitivity.json', parent.report([out/'per_shot.json'], ARMS))
    metrics = {k: {n: parent.timing_decomposition(x) for n, x in v.items()} for k, v in rows.items()}
    changes = {k: {f'{a}__to__{b}': parent.flips(v[a], v[b]) for a, b in PAIRS} for k, v in rows.items()}
    write(out/'result.json', dict(status='COMPLETED_SPECTRAL_SUPPORT_CONTROL', metrics=metrics, paired=changes,
        raw_verified=360*len(ARMS)*len(metrics), activity=activity, reach=reach, cases=cases, confirmation=confirmation))
    print({k: {n: (x['tp'], x['fp']) for n, x in v.items()} for k, v in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path)
    parser.add_argument('out', type=Path)
    cli = parser.parse_args()
    run(cli.fit.resolve(), cli.out.resolve())
    score(cli.out.resolve())
