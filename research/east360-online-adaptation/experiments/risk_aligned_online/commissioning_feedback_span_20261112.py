"""Postseal test of writable, teachable, and realized first-block directions."""

import argparse
from pathlib import Path
from shutil import copyfile
import time

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


lag, layer, bias = group.lag, group.layer, group.bias
TRACE = io23.ROOT/'probe_outputs/commissioning_group_token_east360_20261112_v1'
PROBES = ('real', 'swapped', 'parallel', 'orthogonal', 'group_up', 'feedback_up', 'free_up')
GEOMETRY = ('write_rank', 'feedback_rank', 'write_risk_fraction', 'feedback_risk_fraction_of_write',
            'real_risk_cosine', 'swapped_risk_cosine')
RTOL, BATCH = 1e-5, 64


def edge_matrix(edges):
    """Linear map from observed group-edge coefficients to first-block time tokens."""
    n = edges.shape[1]+1
    ident = torch.eye(n, dtype=edges.dtype, device=edges.device)
    return torch.einsum('te,begd->btdeg', ident[:, 1:]-ident[:, :-1], edges).flatten(1, 2).flatten(2, 3)


def project(vector, matrix):
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = s > RTOL*s[:, :1]
    a = (u.transpose(1, 2) @ vector[..., None]).squeeze(-1).masked_fill(~keep, 0.)
    return (u @ a[..., None]).squeeze(-1), keep.sum(1)


def split_update(real, sensitivity):
    u = F.normalize(sensitivity, dim=1, eps=group.EPS)
    parallel = (real*u).sum(1, keepdim=True)*u
    return parallel, real-parallel


def feedback_jacobian(pred, state):
    """Return one map per independent endpoint, from residual weights to coefficient updates."""
    outputs = pred.flatten(1)
    return torch.stack([torch.autograd.grad(a.sum(), state, retain_graph=i+1<outputs.shape[1])[0].flatten(1)
                        for i, a in enumerate(outputs.unbind(1))], dim=2)


def issued_map(base, signal, q, selected, prior, phase, correction, fields, target, decoder):
    slot, _ = lag.delayed_slots(q, q[selected])
    recent, center, spread, noise = fields
    jac, force = [], []
    for block in range(0, len(selected), BATCH):
        i = selected[block:block+BATCH]
        t = slot[block:block+BATCH]
        x, m = lag._windows(base, signal, q[i])
        state = x.new_zeros((len(x), base.cfg.window//base.cfg.patch-1, len(group.TRUSTED)),
                            requires_grad=True)
        h = group.forward_hidden(base, x, m, layer.patcher(phase[t], correction[t]), state=state)
        pred = group._radius_from_hidden(h, prior[i], recent[i], center[i], spread[i], noise, decoder)
        force.append(((target[i]-pred.detach())/group.BETA).clamp(-1., 1.))
        jac.append(feedback_jacobian(pred, state))
    return torch.cat(jac), torch.cat(force)


def compare(base, x, m, phase, correction, beta, h, z, jac, force):
    written, radii = [], []
    transform = layer.patcher(phase, correction)
    def capture(raw):
        corrected = transform(raw)
        written.append(edge_matrix(group.group_edges(corrected, base.embedding.weight)))
        radii.append(F.linear(corrected-raw, base.embedding.weight).flatten(1).norm(dim=1))
        return corrected
    sites, r = layer.hidden_pass(base, x, m, patcher=capture, differentiable=True, beta=beta)
    g = torch.autograd.grad(r.sum(), sites[0])[0][:, :-1]
    with torch.no_grad():
        b, dose = written[0], radii[0]
        valid = (m & torch.isfinite(x)).reshape(len(x), g.shape[1], -1).any(-1)
        g = g.masked_fill(~valid[..., None], 0.)
        g = (g-g.sum(1, keepdim=True)/valid.sum(1).clamp_min(1)[:, None, None])
        free = g.masked_fill(~valid[..., None], 0.).flatten(1)
        reachable = b @ jac
        write_risk, write_rank = project(free, b)
        feedback_risk, feedback_rank = project(free, reachable)
        real, swapped = [(reachable @ e.flatten(1)[..., None]).squeeze(-1)
                         for e in (force, group.sham_error(force))]
        parallel, orthogonal = split_update(real, feedback_risk)
        geometry = torch.stack((write_rank, feedback_rank,
            write_risk.norm(dim=1)/free.norm(dim=1).clamp_min(group.EPS),
            feedback_risk.norm(dim=1)/write_risk.norm(dim=1).clamp_min(group.EPS),
            F.cosine_similarity(real, free, dim=1, eps=group.EPS),
            F.cosine_similarity(swapped, free, dim=1, eps=group.EPS)))
        directions = (real, swapped, parallel, orthogonal, write_risk, feedback_risk, free)
        moves = torch.stack([layer.match_dose(v.reshape_as(g), dose) for v in directions])
        before = sites[0][:, :-1]
        intended = moves.flatten(2).norm(dim=2)
        realized = (before+moves-before).flatten(2).norm(dim=2)
        n = len(moves)
        _, out = layer.hidden_pass(base, x.repeat(n, 1, 1), m.repeat(n, 1, 1), beta=beta, site=0,
            change=moves.flatten(0, 1), patcher=layer.patcher(phase.repeat(n, 1, 1), correction.repeat(n, 1)))
        delta = out.reshape(n, -1)-r
        err = float((realized-intended).abs().max())
        return (z + r - h@beta).cpu().numpy(), dose.cpu().numpy(), delta.cpu().numpy(), geometry.cpu().numpy(), err


def inspect(signal, args, decoder, path, indices, *, source=False):
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path, source=source)
    slots, _ = lag.delayed_slots(q, indices)
    eligible, fields, target = group._feedback_data(signal, q, args[2], beta.device)
    keep = np.r_[False, eligible][slots]
    if not keep.any():
        return None
    ends, slots = indices[keep], slots[keep]
    unique, inverse = np.unique(slots-1, return_inverse=True)
    jac, force = issued_map(base, signal, q, unique, prior, phase, correction, fields, target, decoder)
    base_scores, dose, delta, geometry = [], [], [], []
    errors = []
    for start in range(0, len(ends), BATCH):
        sl = slice(start, start+BATCH)
        i, t, j = ends[sl], slots[sl], inverse[sl]
        x, m = lag._windows(base, signal, i)
        a, b, c, d, e = compare(base, x, m, phase[t], correction[t], beta, h[i], z[i], jac[j], force[j])
        base_scores.append(a)
        dose.append(b)
        delta.append(c)
        geometry.append(d)
        errors.append(e)
    return dict(ends=ends, release=q[slots-1]+lag.H+1, z=np.concatenate(base_scores),
        dose=np.concatenate(dose), delta=np.concatenate(delta, axis=1),
        geometry=np.concatenate(geometry, axis=1), error=max(errors))


def summarize(sid, role, found, tau=None):
    if found is None:
        return dict(shot=sid, role=role, endpoints=0)
    z, delta = found['z'], found['delta']
    limit = logit(tau) if tau is not None else None
    return dict(shot=sid, role=role, endpoints=len(z), dose_p50=float(np.median(found['dose'])),
        gap=None if limit is None else float(limit-z.max()),
        geometry=dict(zip(GEOMETRY, np.median(found['geometry'], axis=1).tolist(), strict=True)),
        probes={name: dict(max_delta=float(a.max()), median_delta=float(np.median(a)),
            change_to_peak=float((z+a).max()-z.max()),
            crosses=None if limit is None else bool((z+a).max() >= limit),
            peak_endpoint=int(found['ends'][(z+a).argmax()]))
            for name, a in zip(PROBES, delta, strict=True)})


def summary(rows):
    out = {}
    for role in dict.fromkeys(x['role'] for x in rows):
        all_rows = [x for x in rows if x['role'] == role]
        group_rows = [x for x in all_rows if x['endpoints']]
        item = dict(shots=len(all_rows), used_shots=len(group_rows), endpoints=sum(x['endpoints'] for x in group_rows))
        if group_rows:
            item['geometry'] = {name: float(np.median([x['geometry'][name] for x in group_rows])) for name in GEOMETRY}
            item['probes'] = {name: dict(
                median_max_delta=float(np.median([x['probes'][name]['max_delta'] for x in group_rows])),
                median_peak_change=float(np.median([x['probes'][name]['change_to_peak'] for x in group_rows])),
                crossings=[x['shot'] for x in group_rows if x['gap'] is not None and x['gap']>0.
                           and x['probes'][name]['crosses']])
                for name in PROBES}
        out[role] = item
    return out


def held_pairs(manifest, source, args, decoder, path, lock):
    role = 'val' if source else 'cold'
    device = 'J-TEXT' if source else 'EAST'
    allowed = set(io23.source_roles()['val'] if source else lock['ids'][:lag.STARTUP])
    advanced = io23.advanced_times()
    rows = []
    for r in manifest:
        sid, q, end, pos = (r[k] for k in ('shot', 'origin', 'action', 'positive'))
        if sid not in allowed or end<q+lag.H+1:
            raise ValueError('historical witness outside the allowed partition or release')
        if not source and sha(io23.path_for(device, sid)) != lock['hashes'][str(sid)]:
            raise ValueError('historical witness changed')
        raw = simple_data.read_shot(device, sid, rate_contract='strict' if source else 'nominal')
        outcome = simple_data.attach_outcome(raw, device, advanced)
        if outcome['is_disrupt'] != pos or not outcome['valid'][end]:
            raise ValueError('historical witness disagrees with HDF5')
        signal = raw if source else lag.prepare_actual(raw, lag.STARTUP+1, lock['mapping'])
        found = inspect(signal, args, decoder, path, np.array([end]), source=source)
        if found is not None:
            np.testing.assert_array_equal(found['release'], [q+lag.H+1])
        rows.append(summarize(sid, role, found))
    pairs = [(a, b) for a, b in zip(rows[::2], rows[1::2], strict=True)
             if a['endpoints'] and b['endpoints'] and min(a['dose_p50'], b['dose_p50'])>group.EPS]
    return dict(pairs=len(rows)//2, comparable=len(pairs), rows=rows,
        scores={name: dict(both_correct=sum(a['probes'][name]['max_delta']>0.>b['probes'][name]['max_delta']
                                            for a, b in pairs),
                           positive_higher=sum(a['probes'][name]['max_delta']>b['probes'][name]['max_delta']
                                               for a, b in pairs))
                for name in ('real', 'swapped', 'parallel', 'orthogonal')})


def run(out):
    lag.configure()
    lock = lag.load_inputs()
    p, base_seal, args, decoder, path = group.load_reference(lock)
    trace_p, trace_s = (read(TRACE/k) for k in ('protocol.json', 'seal.json'))
    if (trace_p['reference_seal'] != sha(group.REFERENCE/'seal.json')
            or trace_p['order'] != p['order'] or len(trace_s['curve_sha256']) != len(p['order'])):
        raise ValueError('the group trace is not the sealed matched comparison')
    hist = read(group.REFERENCE/'historical.json')['orientation']
    base = state_digest(args[0])
    out.mkdir()
    copyfile(Path(__file__), out/'executed_probe.py')
    write(out/'protocol.json', dict(scope='POSTSEAL_FEEDBACK_SPAN_DIAGNOSTIC_NOT_ONLINE_EFFICACY',
        frozen_source=p['source_sha256'], group_seal=sha(TRACE/'seal.json'), code=sha(Path(__file__)),
        method='SVD of group write operator and its composition with the 20-coordinate issue Jacobian; '
               'full nonlinear forwards at the archived token dose; SVD relative cutoff 1e-5',
        cohorts='fixed authorized source VAL and completed cold witnesses, then all suffix parent-late '
                'timely-near endpoints and up to four parent-risk peaks per safe shot after sealing',
        no_fitting=True, probes=PROBES))
    start = time.monotonic()
    held = {key: held_pairs(hist[key]['manifest'], key == 'val', args, decoder, path, lock)
            for key in ('val', 'cold')}
    print({k: {a: v[a] for a in ('pairs', 'comparable', 'scores')} for k, v in held.items()}, flush=True)
    old = read(group.REFERENCE/'per_shot.json')['0.2'][bias.BIAS]
    tau, rows = p['calibration']['0.2'], []
    replay_error, dose_error, verified = 0., 0., 0
    for rank, sid in enumerate(p['order'], 1):
        name = f'{rank:04d}_{sid}.npz'
        a, b = (d/'shots'/name for d in (group.REFERENCE, TRACE))
        if (sha(a) != base_seal['curve_sha256'][name] or sha(b) != trace_s['curve_sha256'][name]
                or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]):
            raise ValueError('sealed shot or raw inputs changed')
        d, down, timebase = group.hdf._raw_truth(sid)
        with np.load(a, allow_pickle=False) as old_trace, np.load(b, allow_pickle=False) as actual:
            prob, dose, real = old_trace[bias.BIAS], actual['dose'], actual[group.TRUE]
            np.testing.assert_array_equal(old_trace['times'], timebase)
            np.testing.assert_array_equal(old_trace['release'], actual['release'])
        baseline = dict(shot=sid, disruptive=d, down_s=down, **score_curve(
            timebase, prob, d, down, p['original'] if rank <= p['startup'] else tau))
        if baseline != old[rank-1]:
            raise ValueError('raw HDF5 failed the reference first-alarm check')
        verified += 1
        if rank<=p['startup'] or (d and baseline['timely']):
            continue
        active = dose>0.
        if d:
            lead = np.round((down-timebase)*1000., 9)
            indices = np.flatnonzero(active & (lead>30.) & (lead<=125.))
            role = 'late_or_missed'
        else:
            indices = np.asarray(layer.peaks(active, prob), dtype=np.int64)
            role = 'false_alarm' if baseline['alarm'] else 'safe_no_alarm'
        found = None
        if len(indices):
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            signal = lag.prepare_actual(raw, rank, lock['mapping'])
            found = inspect(signal, args, decoder, path, indices)
            np.testing.assert_array_equal(found['ends'], indices)
            check = torch.sigmoid(torch.from_numpy(found['z']+found['delta'][0])).numpy()
            original = torch.sigmoid(torch.from_numpy(found['z'])).numpy()
            replay_error = max(replay_error, float(np.abs(check-real[indices]).max()),
                               float(np.abs(original-prob[indices]).max()))
            dose_error = max(dose_error, found['error'], float(np.abs(found['dose']-dose[indices]).max()))
        rows.append(summarize(sid, role, found, tau))
        if len(rows)%40 == 0:
            print(dict(stage='postseal_span', rank=rank, examined=len(rows), seconds=round(time.monotonic()-start, 1)),
                  flush=True)
    if replay_error>3e-6 or dose_error>1e-5 or base != state_digest(args[0]):
        raise ValueError(f'frozen forward, replay or dose mismatch: {replay_error:g}/{dose_error:g}')
    info = summary(rows)
    write(out/'result.json', dict(status='POSTSEAL_DIAGNOSTIC_ONLY', verified_east=verified, held=held, cases=rows,
        summary=info, max_probability_error=replay_error, max_dose_error=dose_error, seconds=time.monotonic()-start))
    print({k: (v['used_shots'], v['geometry'], {a: len(b['crossings']) for a, b in v['probes'].items()})
           for k, v in info.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out.resolve())
