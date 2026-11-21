"""Matched-dose, postseal transport through two time-token write sites."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
import time

import numpy as np
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from . import commissioning_bias_dynamics_20261112 as bias
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


lag, guard, teacher = bias.lag, bias.guard, bias.teacher
REFERENCE = io23.ROOT/'probe_outputs/commissioning_bias_dynamics_east360_20261112_v1'
SITES = ('before_block_1', 'before_block_2')
PROBES = ('same_past_direction', 'edge_up', 'edge_down', 'any_centered_up')
BATCH = 96


def edge_vectors(tokens, valid):
    delta = tokens[:, 1:] - tokens[:, :-1]
    allowed = valid[:, 1:] & valid[:, :-1] & (delta.norm(dim=-1) > 1e-8)
    return F.normalize(delta, dim=-1, eps=1e-8).masked_fill(~allowed[..., None], 0.)


def project_edges(vector, edges):
    """Project onto adjacent equal-and-opposite exchanges of observed token differences."""
    # This is B^T B for the path incidence operator with vector-valued edges.
    square = edges.square().sum(-1)
    diagonal = torch.where(square > 0., 2.*square, 1.)
    off = -(edges[:, 1:]*edges[:, :-1]).sum(-1)
    gram = torch.diag_embed(diagonal) + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    rhs = ((vector[:, :-1]-vector[:, 1:])*edges).sum(-1, keepdim=True)
    shifts = edges*torch.linalg.solve(gram, rhs)
    return F.pad(shifts, (0, 0, 0, 1))-F.pad(shifts, (0, 0, 1, 0))


def match_dose(vector, radius):
    n = vector.flatten(1).norm(dim=1)
    return vector*torch.where(n > 1e-8, radius/n.clamp_min(1e-8), 0.)[:, None, None]


def add_to_patches(tokens, shift):
    return torch.cat((tokens[:, :-1]+shift, tokens[:, -1:]), dim=1)


def hidden_pass(base, x, mask, *, patcher=None, site=None, change=None, differentiable=False, beta=None):
    """Use the production forward and touch only the declared input activation."""
    sites, hidden = [], []

    def hook(index):
        def capture(_layer, args):
            z = args[0]
            if index == 0 and differentiable:
                z = z.detach().requires_grad_(True)
            if index == site:
                z = add_to_patches(z, change)
            sites.append(z)
            return (z, *args[1:])
        return capture

    handles = [layer.register_forward_pre_hook(hook(i)) for i, layer in enumerate(base.encoder.layers)]
    handles.append(base.risk_head.register_forward_pre_hook(lambda _layer, p: hidden.append(p[0])))
    if patcher is not None:
        handles.append(base.embedding.register_forward_pre_hook(lambda _layer, p: (patcher(p[0]),)))
    try:
        pred = base(x, mask)
    finally:
        for handle in handles:
            handle.remove()
    return tuple(sites), pred['logit'] if beta is None else hidden[0] @ beta


def patcher(phase, correction):
    def make(x):
        moved = lag.patch_input(x, phase)
        cap = (moved-x).flatten(1).square().mean(1).sqrt()
        return bias.bias_patches(moved, correction, cap)
    return make


def probes(base, x, mask, phase, correction, beta):
    valid = (mask & torch.isfinite(x)).reshape(len(x), base.cfg.window//base.cfg.patch, -1).any(-1)
    with torch.no_grad():
        untouched, readout0 = hidden_pass(base, x, mask, beta=beta)
    current, readout = hidden_pass(base, x, mask, patcher=patcher(phase, correction),
                                   differentiable=True, beta=beta)
    gradients = torch.autograd.grad(readout.sum(), current)
    with torch.no_grad():
        direct = [(a[:, :-1]-b[:, :-1]).masked_fill(~valid[..., None], 0.) for a, b in zip(current, untouched)]
        dose = direct[1].flatten(1).norm(dim=1)
        # Same plane at both sites, defined without the fixed position embedding.
        edges = edge_vectors(current[0][:, :-1]-base.position[:, :-1], valid)
        common = match_dose(project_edges(direct[0], edges), dose)
        scores, geometry, dose_error = [], [], 0.
        for site, g in enumerate(gradients):
            g = g[:, :-1].masked_fill(~valid[..., None], 0.)
            g = (g-g.sum(1, keepdim=True)/valid.sum(1).clamp_min(1)[:, None, None]).masked_fill(~valid[..., None], 0.)
            restricted = project_edges(g, edges)
            up = match_dose(restricted, dose)
            change = torch.stack((common, up, -up, match_dose(g, dose)))
            n = len(change)
            before = current[site][:, :-1]
            realized = (before+change-before).flatten(2).norm(dim=2)
            desired = change.flatten(2).norm(dim=2)
            matched = torch.where(desired > 1e-8, dose, 0.)
            dose_error = max(dose_error, float((realized-matched).abs().max()))
            _, res = hidden_pass(base, x.repeat(n, 1, 1), mask.repeat(n, 1, 1),
                patcher=patcher(phase.repeat(n, 1, 1), correction.repeat(n, 1)),
                site=site, change=change.flatten(0, 1), beta=beta)
            scores.append(res.reshape(n, -1)-readout)
            geometry.append(torch.stack((restricted.flatten(1).norm(dim=1), g.flatten(1).norm(dim=1),
                                         (restricted*common).sum((1, 2)))))
    if dose_error > 1e-5:
        raise ValueError(f'actual token dose was not matched ({dose_error:g})')
    return ((readout-readout0).detach().cpu().numpy(), dose.cpu().numpy(),
            torch.stack(scores).cpu().numpy(), torch.stack(geometry).cpu().numpy(), dose_error)


def inspect_shot(signal, args, decoder, path, pick, *, source=False):
    base = args[0]
    tr, h, z, beta = guard._track(signal, args, source=source)
    q, key, prior, target = teacher._mature(tr, args[2][2], args[2][3])
    slots, _ = lag.delayed_slots(q, np.arange(len(z)))
    ends = np.asarray(pick(slots), dtype=np.int64)
    if not len(ends):
        return None
    if (slots[ends] == 0).any():
        raise ValueError('test endpoint is earlier than its diagnostic feedback')
    j = slots[ends]-1
    used = np.unique(j)
    real, _ = guard._bases(base, signal, q[used], prior[used], target[used], decoder)
    phase = -lag.RADIUS*guard.signed_direction(*real, torch.as_tensor(key[used], device=beta.device), path)
    def tensor(a):
        return torch.as_tensor(a, device=beta.device)
    with torch.no_grad():
        forecast = lag._predict(tensor(h[q]), tensor(prior), decoder)
        residual = (tensor(target)-forecast).mean(1).cpu().numpy()
        b, _ = bias.past_bias(q, residual)
        b = tensor(b)*tensor(args[2][3])
    moves, slopes, dose, ref = [], [], [], []
    err = 0.
    for start in range(0, len(ends), BATCH):
        s = slice(start, start+BATCH)
        x, m = lag._windows(base, signal, ends[s])
        change, r, values, jac, error = probes(base, x, m, phase[np.searchsorted(used, j[s])], b[j[s]], beta)
        err = max(err, error)
        ref.append(z[ends[s]]+change)
        dose.append(r)
        moves.append(values)
        slopes.append(jac)
    return dict(ends=ends, release=q[j]+lag.H+1, parent=np.concatenate(ref), dose=np.concatenate(dose),
                moves=np.concatenate(moves, axis=-1), slopes=np.concatenate(slopes, axis=-1), max_dose_error=err)


def peaks(active, probability):
    return [int(ids[np.argmax(probability[ids])])
            for ids in (part[active[part]] for part in np.array_split(np.arange(len(active)), 4)) if len(ids)]


def summarize_case(sid, role, category, found, tau=None):
    if found is None:
        return dict(shot=sid, role=role, category=category, endpoints=0)
    z, radius = found['parent'], found['dose']
    margin = None if tau is None else z-logit(tau)
    row = dict(shot=sid, role=role, category=category, endpoints=len(z),
               n_nonzero=int((radius>1e-8).sum()), dose_p50=float(np.median(radius)),
               min_gap=None if margin is None else float(-margin.max()),
               max_dose_error=found['max_dose_error'], sites={})
    for i, site in enumerate(SITES):
        a, b, signed = found['slopes'][i]
        row['sites'][site] = dict(
            edge_norm_p50=float(np.median(a)), free_norm_p50=float(np.median(b)),
            projected_fraction_p50=float(np.median(a/np.maximum(b, 1e-9))),
            signed_p50=float(np.median(signed)),
            **{k: dict(max_delta=float(v.max()), median_delta=float(np.median(v)),
                       max_margin=None if margin is None else float((margin+v).max()),
                       endpoint=int(found['ends'][v.argmax() if margin is None else (margin+v).argmax()]))
               for k, v in zip(PROBES, found['moves'][i], strict=True)})
    return row


def rollup(records):
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['role']}/{r['category']}"].append(r)
    out = {}
    for k, all_rows in groups.items():
        rows = [r for r in all_rows if r['endpoints']]
        out[k] = dict(shots=len(all_rows), with_endpoints=len(rows),
                      endpoints=sum(r['endpoints'] for r in rows), sites={})
        if not rows:
            continue
        for site in SITES:
            s = [r['sites'][site] for r in rows]
            out[k]['sites'][site] = dict(
                median_edge_norm=float(np.median([r['edge_norm_p50'] for r in s])),
                median_free_norm=float(np.median([r['free_norm_p50'] for r in s])),
                median_fraction=float(np.median([r['projected_fraction_p50'] for r in s])),
                median_dose=float(np.median([r['dose_p50'] for r in rows])),
                **{a: dict(median_max=float(np.median([r[a]['max_delta'] for r in s])),
                           median_of_medians=float(np.median([r[a]['median_delta'] for r in s])),
                           nonalarm_to_cross=[r['shot'] for r in rows if r['min_gap'] is not None and r['min_gap']>0.
                                              and r['sites'][site][a]['max_margin']>=0.]) for a in PROBES})
    return out


def run(out):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p, seal, hist = (read(REFERENCE/k) for k in ('protocol.json', 'seal.json', 'historical.json'))
    signed = Path(p['reference'])
    gp, _, _, args, decoder = bias.eos._load(signed, lock)
    if (p['order'] != lock['ids'] or p['reference_seal'] != sha(signed/'seal.json')
            or p['source_sha256'] != gp['source_sha256'] or p['calibration'] != gp['calibration']
            or len(seal['curve_sha256']) != 360 or sha(signed/'teacher.json') != gp['teacher_sha256']):
        raise ValueError('unsealed or changed warning ancestry')
    base = args[0]
    if len(base.encoder.layers) != 2:
        raise ValueError('the audited encoder has exactly two blocks')
    frozen = state_digest(base)
    path = torch.as_tensor(read(signed/'teacher.json')['path'], device='cuda', dtype=torch.float32)
    out.mkdir()
    copyfile(Path(__file__), out/'executed_probe.py')
    write(out/'protocol.json', dict(scope='POSTSEAL_MATCHED_LAYER_TRANSPORT_NOT_AN_ONLINE_RESULT',
        order=lock['ids'], cold_count=lag.STARTUP, reference_seal=sha(REFERENCE/'seal.json'),
        source_sha256=p['source_sha256'], code_sha256=sha(out/'executed_probe.py'),
        sites=SITES, methods=PROBES, time='q+26 or later',
        dose='Frobenius norm of realized valid time-token change at block 2 input in the existing biased path',
        plane='same oriented adjacent physical-patch contrasts from block 1 input, excluding position and CLS',
        cases='15 source VAL pairs; cold20 post-commissioning and all suffix shots, disruptive (30,125] ms or safe four segment peaks',
        no_learning='no labels or risk used to train any adapter; both signs and all layer probes retained'))
    rows, witnesses, replay_error = [], [], 0.
    allowed = set(io23.source_roles()['val'])
    adv = io23.advanced_times()
    historical = defaultdict(list)
    for r in hist['orientation']['val']['manifest']:
        if r['shot'] not in allowed or r['action'] < r['origin']+lag.H+1:
            raise ValueError('wrong source VAL witness')
        historical[r['shot']].append(r)
    for sid, entries in historical.items():
        raw = simple_data.read_shot('J-TEXT', sid)
        truth = simple_data.attach_outcome(raw, 'J-TEXT', adv)
        ends = [r['action'] for r in entries]
        if any(r['positive'] != truth['is_disrupt'] for r in entries) or not truth['valid'][ends].all():
            raise ValueError('wrong historical truth')
        sample = inspect_shot(raw, args, decoder, path, lambda _: ends, source=True)
        np.testing.assert_array_equal(sample['release'], [r['origin']+lag.H+1 for r in entries])
        rows.append(summarize_case(sid, 'source_val', 'disruptive' if truth['is_disrupt'] else 'safe', sample))
    old = read(REFERENCE/'per_shot.json')['0.2'][bias.BIAS]
    tau = p['calibration']['0.2']
    for rank, (sid, archived) in enumerate(zip(lock['ids'], old, strict=True), 1):
        name = f'{rank:04d}_{sid}.npz'
        f = REFERENCE/'shots'/name
        if sha(f) != seal['curve_sha256'][name] or sha(io23.path_for('EAST', sid)) != lock['hashes'][str(sid)]:
            raise ValueError('raw or sealed curve changed')
        d, down, t = bias.hdf._raw_truth(sid)
        raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        with np.load(f, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['times'], t)
            prob = a[bias.BIAS]
            rel = a['release']
        np.testing.assert_array_equal(raw['times'], t)
        logged_tau = p['original'] if rank <= lag.STARTUP else tau
        item = dict(shot=sid, disruptive=d, down_s=down, **score_curve(t, prob, d, down, logged_tau))
        if item != archived:
            raise ValueError('reference first alarm no longer matches HDF5')
        # Prepare the cold shots in the already completed commissioning coordinates for retrospective comparison.
        cold = rank <= lag.STARTUP
        role = 'cold_completed' if cold else 'suffix_postseal'
        sig = lag.prepare_actual(raw, max(rank, lag.STARTUP+1), lock['mapping'])
        if d:
            lead = np.round((down-t)*1000., 9)
            def pick(slots):
                return np.flatnonzero((slots>0) & (lead>30.) & (lead<=125.))
            group = 'timely' if item['timely'] else 'late_or_missed'
        else:
            def pick(slots):
                return peaks(slots>0, prob)
            group = 'false_alarm' if item['alarm'] else 'safe_no_alarm'
        x = inspect_shot(sig, args, decoder, path, pick)
        if x is not None and not cold:
            indices = x['ends']
            np.testing.assert_array_equal(x['release'], rel[np.searchsorted(rel, indices, side='right')-1])
            prev = expit(x['parent'])
            replay_error = max(replay_error, float(np.max(np.abs(prev-prob[indices]))))
            np.testing.assert_allclose(prev, prob[indices], atol=3e-6, rtol=0.)
            np.testing.assert_array_equal(prev>=tau, prob[indices]>=tau)
            if len(witnesses) < 3:
                ends = indices[:min(6, len(indices))]
                n = int(ends[-1])+1
                short = {k: v[:n] for k, v in sig.items() if k in ('values', 'available')}
                y = inspect_shot(short, args, decoder, path, lambda _: ends)
                np.testing.assert_array_equal(y['release'], x['release'][:len(ends)])
                np.testing.assert_allclose(y['moves'], x['moves'][..., :len(ends)], atol=2e-5, rtol=0.)
                witnesses.append(dict(shot=sid, prefix=n, endpoints=len(ends)))
        rows.append(summarize_case(sid, role, group, x, tau))
        if rank % 60 == 0 or rank == lag.STARTUP:
            print(dict(stage='layer_transport', rank=rank, seconds=round(time.monotonic()-start, 2)), flush=True)
    if state_digest(base) != frozen:
        raise ValueError('frozen source weights changed')
    result = dict(status='COMPLETED_DIAGNOSTIC_NOT_ALARM_EFFICACY', verified_east=len(old),
        prefix=witnesses, max_reference_probability_error=replay_error, seconds=time.monotonic()-start,
        summary=rollup(rows), records=rows)
    write(out/'result.json', result)
    print({k: {site: v['edge_up']['median_max'] for site, v in x['sites'].items()}
           for k, x in result['summary'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out.resolve())
