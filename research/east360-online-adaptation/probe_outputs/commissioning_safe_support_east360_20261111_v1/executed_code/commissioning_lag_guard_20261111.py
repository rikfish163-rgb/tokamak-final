"""Train shot-local signed feedback with point or trajectory-peak protection."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .latent_features import extract_shot
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


POINT, POINT_SHIFT, PATH, PATH_SHIFT, CONTEXT = NEW = (
    'signed_point_true', 'signed_point_shifted', 'signed_path_true',
    'signed_path_shifted', 'signed_context_only',
)
ARMS = (*teacher.ARMS, *NEW)
COMPARISONS = (
    (lag.BASES[-1], PATH), (lag.TRUE, PATH), (teacher.TRUE, POINT),
    (POINT, PATH), (POINT_SHIFT, PATH_SHIFT), (POINT_SHIFT, POINT),
    (PATH_SHIFT, PATH), (CONTEXT, PATH),
)
RIDGE, SIGN_CAP = .04, 2.


def _unit(x):
    n = x.norm(dim=(-2, -1), keepdim=True)
    return torch.where(n > 1e-8, x/n.clamp_min(1e-8), 0.)


def basis(gradient):
    """Keep the actual group-gradient span and shrink its weak directions."""
    g = gradient.flatten(2).double()
    lam, u = torch.linalg.eigh(g @ g.mT)
    inv = (lam.clamp_min(0.) + RIDGE*lam[:, -1:].clamp_min(1e-16)).rsqrt()
    modes = u @ ((u.mT @ g)*inv.unsqueeze(-1))
    return _unit(gradient.sum(1)), modes.reshape_as(gradient).to(gradient.dtype)


def signed_direction(anchor, modes, keys, params):
    correction = torch.einsum('bg,bgij->bij', SIGN_CAP*(keys @ params.T).tanh(), modes)
    return _unit(anchor+correction)


def path_endpoints(margin, q, anchor):
    """Retain the frozen peak of each constant fast-state interval and the anchor."""
    bounds = np.r_[0, q+lag.H+1, len(margin)]
    peaks = [int(a+np.argmax(margin[a:b])) for a,b in zip(bounds[:-1], bounds[1:], strict=True)]
    return np.unique([*peaks, anchor])


def shot_peaks(values, owner, size):
    peaks = values.new_full((size,), -torch.inf).scatter_reduce(0, owner, values, 'amax')
    row = torch.arange(len(values), device=values.device)
    tied = torch.where(values == peaks[owner], row, len(values))
    first = torch.full_like(peaks, len(values), dtype=torch.long).scatter_reduce(
        0, owner, tied, 'amin')
    return peaks, first


def guarded_loss(pair_delta, safe_margin, safe_delta):
    up, down = pair_delta.T
    positive = F.softplus(-up/.10)
    safe = F.softplus(safe_delta/.10)
    protect = F.relu(safe_delta - (-safe_margin).clamp_min(0.))/.10
    separation = F.softplus((down-up)/.20)
    return (.5*(positive+safe) + .50*protect + .25*separation).mean()


def _track(signal, args, *, source=False):
    base, net, params, old, quiet, mapper, coef, _ = args
    ends = np.arange(len(signal['values']))
    if source:
        tr = lag.direct.track(net, signal, params[0], params[1], ends)
        h, z = extract_shot(base, signal, 'cuda')
        beta = base.risk_head.weight.detach().flatten()
    else:
        tr, inherited, ready = lag.history.observed_track(net, signal, ends, params, old)
        state, _ = lag.persistent.replay(*ready, quiet)
        latest = inherited + torch.einsum('...k,...f,kfw->...w', torch.from_numpy(tr['key']),
                    4.*torch.from_numpy(state['current_equal_norm']), mapper)
        x, _ = lag.features(base, signal, 'cuda')
        h = x[:, :base.cfg.dim]
        z = (torch.from_numpy(np.concatenate((x, latest.tanh().numpy()), 1)).cuda()
             @ coef[:-1]+coef[-1]).cpu().numpy()
        beta = coef[:base.cfg.dim]
    return tr, h, z, beta


def _bases(base, signal, q, prior, target, decoder):
    real_parts, sham_parts = [], []
    for a in range(0, len(q), 128):
        sl = slice(a, a+128)
        x, m = lag._windows(base, signal, q[sl])
        prior_t, target_t = (torch.as_tensor(k[sl], device=x.device) for k in (prior, target))
        theta, real, sham = teacher._signature_problem(base, x, m, prior_t, decoder, target_t)
        both = torch.cat((real, sham), dim=1)
        jac = [torch.autograd.grad(both[:, i].sum(), theta, retain_graph=i+1<both.shape[1])[0]
               for i in range(both.shape[1])]
        real_g, sham_g = torch.stack(jac, 1).split(teacher.GROUPS, dim=1)
        real_parts.append(basis(real_g))
        sham_parts.append(basis(sham_g))
    if not real_parts:
        # A shot shorter than the first complete forecast window never updates.
        n = len(lag.CHANNELS)
        empty = base.center.new_empty((0, n, n)), base.center.new_empty((0, teacher.GROUPS, n, n))
        return empty, empty
    return tuple(tuple(torch.cat(v) for v in zip(*parts, strict=True))
                 for parts in (real_parts, sham_parts))


def _record(role, entry, *, positive, args, decoder, lock, allowed, advanced, threshold):
    sid, q0, t = (entry[k] for k in ('shot', 'origin', 'action'))
    if entry['role'] != role or sid not in allowed:
        raise ValueError('direction supervision outside the sealed training split')
    source = role != 'cold'
    device = 'J-TEXT' if source else 'EAST'
    if not source and sha(io23.path_for(device, sid)) != lock['hashes'][str(sid)]:
        raise ValueError('cold-start signal changed')
    raw = simple_data.read_shot(device, sid, rate_contract='strict' if source else 'nominal')
    truth = simple_data.attach_outcome(raw, device, advanced)
    if truth['is_disrupt'] != positive or not truth['valid'][t]:
        raise ValueError('historical pair does not match HDF5')
    if positive:
        lead = float((truth['down']-raw['times'][t])*1000.)
        if not truth['y'][t] or round(lead, 9) <= (10. if source else 30.):
            raise ValueError('positive anchor outside timely supervised region')
        np.testing.assert_allclose(lead, entry['lead_ms'], atol=1e-7, rtol=0.)
    signal = raw if source else lag.prepare_actual(raw, lag.STARTUP+1, lock['mapping'])
    tr, h, z, beta = _track(signal, args, source=source)
    q, key, prior, target = teacher._mature(tr, args[2][2], args[2][3])
    slot, _ = lag.delayed_slots(q, [t])
    if slot[0] == 0 or q[slot[0]-1] != q0:
        raise ValueError('paired anchor no longer uses its sealed forecast origin')
    ctx = tr['context'][t, :3].astype(float)
    ctx[2] *= 4.
    m = z - logit(threshold)
    np.testing.assert_allclose(np.r_[m[t], np.tanh(ctx)], entry['match'], atol=2e-6, rtol=0.)
    endpoints = np.array([t]) if positive else path_endpoints(m, q, t)
    slots, _ = lag.delayed_slots(q, endpoints)
    use = np.unique(slots[slots>0])-1
    ix = np.where(slots == 0, 0, 1+np.searchsorted(use, slots-1))
    base = args[0]
    x, mask = lag._windows(base, signal, endpoints)
    (a, b), (sa, sb) = _bases(base, signal, q[use], prior[use], target[use], decoder)
    kt = torch.as_tensor(key[use], device=x.device)

    def at_endpoint(v):
        return torch.cat((torch.zeros_like(v[:1]), v), 0)[ix]

    data = dict(x=x, mask=mask, h=torch.as_tensor(h[endpoints], device=x.device),
                beta=beta.expand(len(x), -1), m=torch.as_tensor(m[endpoints], device=x.device),
                key=at_endpoint(kt), a=at_endpoint(a), b=at_endpoint(b),
                sa=at_endpoint(sa), sb=at_endpoint(sb))
    manifest = dict(shot=sid, role=role, positive=positive, anchor=int(t),
                    endpoints=endpoints.tolist(), released_origins=q[slots[slots>0]-1].tolist())
    return data, int(np.searchsorted(endpoints, t)), manifest


def _pack(roles, archive, args, decoder, lock, source_roles, ids, original, cutoff):
    allowed = {'train': set(source_roles['train']), 'val': set(source_roles['val']),
               'cold': set(ids[:lag.STARTUP])}
    advanced = io23.advanced_times()
    data, anchors, paths, windows = [], [], [], []
    offset = 0
    for role in roles:
        used = set()
        for pair in archive['pairs'][role]:
            indices = []
            for i, entry in enumerate(pair):
                if entry['shot'] in used:
                    raise ValueError('historical pairing must be disjoint by shot')
                used.add(entry['shot'])
                row, j, manifest = _record(role, entry, positive=i==0, args=args, decoder=decoder,
                    lock=lock, allowed=allowed[role], advanced=advanced,
                    threshold=cutoff if role=='cold' else original)
                if i==1:
                    paths.append(torch.arange(offset, offset+len(row['m']), device=row['m'].device))
                indices.append(offset+j)
                offset += len(row['m'])
                data.append(row)
                windows.append(manifest)
            anchors.append(indices)
    cols = {k:torch.cat([d[k] for d in data]) for k in data[0]}
    return cols, torch.as_tensor(anchors, device=cols['m'].device), paths, windows


def _direction(data, p, rows, *, shifted=False):
    ka, kb = ('sa','sb') if shifted else ('a','b')
    return signed_direction(data[ka][rows], data[kb][rows], data['key'][rows], p)


def _gain(base, data, p, rows, *, shifted=False, context=None):
    if context is None:
        d = _direction(data, p, rows, shifted=shifted)
    else:
        d = _unit((data['key'][rows] @ context).reshape(-1,len(lag.CHANNELS),len(lag.CHANNELS)))
    h = lag._features(base, data['x'][rows], data['mask'][rows], -lag.RADIUS*d)
    delta = ((h-data['h'][rows])*data['beta'][rows]).sum(-1)
    return torch.where(d.ne(0.).any((1,2)), delta, 0.)


def _path_gains(base, data, p, paths, *, shifted=False, context=None):
    rows = torch.cat(paths)
    owner = torch.cat([torch.full_like(path, i) for i,path in enumerate(paths)])
    with torch.no_grad():
        m = data['m'][rows]
        old, _ = shot_peaks(m, owner, len(paths))
        gain = torch.cat([_gain(base, data, p, r, shifted=shifted, context=context)
                          for r in rows.split(256)])
        _, arg = shot_peaks(m+gain, owner, len(paths))
        worst = rows[arg]
    # Recompute the selected maximum with gradients; this is the exact
    # subgradient on the fixed support, even when the maximum moves.
    new = data['m'][worst]+_gain(base, data, p, worst, shifted=shifted, context=context)
    return old, new-old


def _fit(base, bundle, *, path):
    data, anchors, paths, _ = bundle
    p = torch.nn.Parameter(data['key'].new_zeros((teacher.GROUPS, data['key'].shape[1])))
    opt = torch.optim.Adam([p], lr=.025)
    rng = np.random.default_rng(teacher.SEED)
    for step in range(teacher.STEPS):
        pairs = rng.choice(len(anchors), min(len(anchors), 24), replace=False)
        rows = anchors[pairs]
        opt.zero_grad(set_to_none=True)
        point = _gain(base, data, p, rows.flatten()).reshape(-1, 2)
        safe_margin, safe_delta = (_path_gains(base, data, p, [paths[i] for i in pairs])
                                   if path else (data['m'][rows[:,1]], point[:,1]))
        loss = guarded_loss(point, safe_margin, safe_delta)+.025*p.square().mean()
        loss.backward()
        opt.step()
        if (step+1) % 48 == 0:
            print(dict(stage='fit_path' if path else 'fit_point', step=step+1,
                       loss=float(loss.detach())), flush=True)
    if not torch.isfinite(p).all():
        raise FloatingPointError('nonfinite signed teacher')
    return p.detach()


@torch.no_grad()
def _context_fit(bundle, fitted):
    data, anchors, _, _ = bundle
    rows = anchors.flatten()
    key = data['key'][rows]
    target = _direction(data, fitted, rows).flatten(1)
    # A fixed prefix-to-state regression, with no measured target signal or
    # backpropagation at replay time. Use only the fitted historical anchors.
    return torch.linalg.solve(key.T @ key+torch.eye(key.shape[1], device=key.device), key.T @ target)


@torch.no_grad()
def _describe(base, bundle, point, path, context):
    data, anchors, paths, _ = bundle
    out = {}
    for name, p, shifted in (('uniform', 0.*point, False), (POINT, point, False),
                            (POINT_SHIFT, point, True), (PATH, path, False),
                            (PATH_SHIFT, path, True), (CONTEXT, path, False)):
        ctx = context if name==CONTEXT else None
        d = _gain(base, data, p, anchors.flatten(), shifted=shifted, context=ctx).reshape(-1,2)
        m = data['m'][anchors]
        old, down = _path_gains(base, data, p, paths, shifted=shifted, context=ctx)
        out[name] = dict(pairs=len(d), both_right=int(((d[:,0]>0.)&(d[:,1]<0.)).sum()),
                         timely_reaches=int(((m[:,0]<0.)&(m[:,0]+d[:,0]>=0.)).sum()),
                         path_new_cross=int(((old<0.)&(old+down>=0.)).sum()),
                         path_removed_cross=int(((old>=0.)&(old+down<0.)).sum()),
                         mean_positive=float(d[:,0].mean()),
                         mean_safe_path_change=float(down.mean()),
                         median_safe_path_change=float(down.median()))
    return out


def _replay(signal, args, decoder, point, path, context):
    base = args[0]
    tr, h, z, beta = _track(signal, args)
    q, keys, prior, target = teacher._mature(tr, args[2][2], args[2][3])
    real, sham = _bases(base, signal, q, prior, target, decoder)
    k = torch.as_tensor(keys, device=beta.device)
    directions = [signed_direction(*b, k, p) for p in (point, path) for b in (real, sham)]
    n = len(lag.CHANNELS)
    directions.append(_unit((k @ context).reshape(len(q),n,n)))
    mats = (-lag.RADIUS*torch.stack(directions)).detach().cpu().numpy()
    slots, _ = lag.delayed_slots(q, np.arange(len(z)))
    baseline = torch.from_numpy(z).cuda().sigmoid().cpu().numpy()
    out = {}
    for name, state in zip(NEW, mats, strict=True):
        effective = np.concatenate((np.zeros((1,n,n),np.float32), state), axis=0)[slots]
        active = effective.any((1, 2))
        a = lag._embed_active(base, signal, h, effective)
        dz = (torch.from_numpy(a-h).cuda()*beta).sum(-1)
        pred = (torch.from_numpy(z).cuda()+dz).sigmoid().cpu().numpy()
        pred[~active] = baseline[~active]
        out[name] = pred
    return baseline, out, q+lag.H+1, np.linalg.norm(mats, axis=(2,3)).T


def run(previous, output):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ids = lock['ids']
    pp, seal, archive = (read(previous/k) for k in ('protocol.json','seal.json','teacher.json'))
    lp = read(Path(pp['previous_lag'])/'protocol.json')
    reference = Path(lp['reference'])
    coordinates = next(Path(a).parent for a in pp['assets']
                       if Path(a).parent.name.startswith('commissioning_local_cross_response_'))
    forecast = Path(lp['source_decoder']).parent
    p, _, args, assets = lag.parent._frozen(reference, coordinates, lock)
    roles = io23.source_roles()
    if (tuple(pp['arms'])!=teacher.ARMS or ids!=pp['order'] or pp['assets']!=assets
            or pp['original']!=p['original'] or pp['calibration']!=p['calibration']
            or pp['fit_roles']!={'J-TEXT/train':roles['train'],'EAST/cold':ids[:lag.STARTUP]}
            or pp['held_source']!=roles['val'] or len(seal['curve_sha256'])!=360
            or sha(previous/'teacher.json')!=pp['teacher_sha256']
            or sha(Path(pp['previous_lag'])/'seal.json')!=pp['previous_seal']):
        raise ValueError('sealed point-teaching reference required')
    for rank, sid in enumerate(ids[:lag.STARTUP], 1):
        f = previous/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=seal['curve_sha256'][f.name]:
            raise ValueError('EAST historical forecast must seal before outcomes')
    decoder = lag._bind_forecast(forecast, p)
    producer = Path(__file__)
    code = dict(pp['code_sha256'], **{producer.name:sha(producer)})
    (output/'shots').mkdir(parents=True)
    (output/'executed_code').mkdir()
    for name, digest in code.items():
        src = producer.parent/name
        if sha(src)!=digest:
            raise ValueError('frozen experiment dependency changed: '+name)
        copyfile(src, output/'executed_code'/name)
    frozen = tuple(state_digest(m) for m in args[:2])
    original, thresholds = p['original'], p['calibration']
    common = (archive, args, decoder, lock, roles, ids, original, thresholds['0.2'])
    fit = _pack(('train','cold'), *common)
    print(dict(stage='historical_support', pairs=len(fit[1]), endpoints=len(fit[0]['m']),
               seconds=round(time.monotonic()-started, 2)), flush=True)
    point, path = (_fit(args[0], fit, path=guard) for guard in (False,True))
    context = _context_fit(fit, path)
    trained, fit_windows = _describe(args[0], fit, point, path, context), fit[-1]
    del fit
    held = _pack(('val',), *common)
    validation, val_windows = _describe(args[0], held, point, path, context), held[-1]
    del held
    write(output/'teacher.json', dict(point=point.cpu().tolist(), path=path.cpu().tolist(),
        context=context.cpu().tolist(), fit=trained, held_source=validation,
        fit_windows=fit_windows, val_windows=val_windows,
        seed=teacher.SEED, steps=teacher.STEPS, ridge=RIDGE, signed_cap=SIGN_CAP))
    write(output/'protocol.json', dict(scope='EAST360_SIGNED_POINT_VS_PATH_GUARD_DEVELOPMENT',
        order=ids, startup=lag.STARTUP, arms=ARMS, new=NEW, original=original,
        source_sha256=p['source_sha256'], assets=assets, calibration=thresholds,
        previous_teacher=str(previous), previous_seal=sha(previous/'seal.json'),
        previous_pairs_sha256=pp['teacher_sha256'], teacher_sha256=sha(output/'teacher.json'),
        code_sha256=code, fit_roles=pp['fit_roles'], held_source=pp['held_source'],
        comparisons=COMPARISONS, radius=lag.RADIUS, input_rms=lag.INPUT_RMS,
        first_effect=lag.FIRST_EFFECT, stride=lag.STRIDE,
        path_support='each historical safe release interval frozen peak plus original paired endpoint',
        feedback='exact full-window observed lag signatures, and matched group phase shift',
        context_control='fitted prefix-key regression on path teacher anchor directions; same release clock',
        outcomes='no suffix HDF5 outcome until all predictions seal', persistent_updates=0))
    print(dict(stage='fit_sealed', held=validation,
               seconds=round(time.monotonic()-started, 2)), flush=True)
    hashes, witnesses = {}, {}
    updates = 0
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome')):
        for rank, sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            f = previous/'shots'/name
            if sha(f)!=seal['curve_sha256'][name]:
                raise ValueError('sealed reference curve changed')
            with np.load(f, allow_pickle=False) as saved:
                times, parent = saved['times'], saved[lag.BASES[-1]]
                if rank <= lag.STARTUP:
                    curves, trace = dict.fromkeys(NEW,saved['fixed']), {}
                else:
                    if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                        raise ValueError('locked signal changed')
                    raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                    np.testing.assert_array_equal(raw['times'],times)
                    signal = lag.prepare_actual(raw,rank,lock['mapping'])
                    baseline, curves, release, radius = _replay(signal,args,decoder,point,path,context)
                    np.testing.assert_array_equal(parent,baseline)
                    np.testing.assert_array_equal(release,saved['release'])
                    np.testing.assert_allclose(radius[:,:4],np.tile(saved['radius'],(1,2)),
                                               atol=3e-7,rtol=1e-6)
                    np.testing.assert_allclose(radius[:,4],saved['radius'][:,0],atol=3e-7,rtol=1e-6)
                    stop = int(release[0]) if len(release) else len(times)
                    for v in curves.values():
                        np.testing.assert_array_equal(v[:stop],parent[:stop])
                    if rank in (21,120,360):
                        end = lag.FIRST_EFFECT+3*lag.STRIDE+5
                        prefix = {k:signal[k][:end] for k in ('values','available')}
                        _, short, rel, _ = _replay(prefix,args,decoder,point,path,context)
                        for k in NEW:
                            np.testing.assert_allclose(short[k],curves[k][:end],atol=3e-6,rtol=0.)
                            for tau in thresholds.values():
                                np.testing.assert_array_equal(short[k]>=tau,curves[k][:end]>=tau)
                        np.testing.assert_array_equal(rel,release[release<end])
                        witnesses[str(sid)] = end
                    trace = dict(release=release,radius=radius)
                    updates += int((radius[:,0]>0.).sum())
            if not all(np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite adaptive risk')
            target = output/'shots'/name
            with target.open('xb') as fd:
                np.savez_compressed(fd,times=times,**curves,**trace)
            hashes[name] = sha(target)
            if rank==21 or rank%40==0:
                print(dict(stage='sealed',rank=rank,seconds=round(time.monotonic()-started,2)),flush=True)
    if tuple(state_digest(m) for m in args[:2])!=frozen:
        raise ValueError('frozen model changed')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses))
    per = read(previous/'per_shot.json')
    for band in thresholds:
        per[band].update({name:[] for name in NEW})
    for rank, sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        f = output/'shots'/name
        if sha(f)!=hashes[name]:
            raise ValueError('unsealed adaptive curve')
        d, down = io23.read_outcome('EAST',sid)
        with np.load(f,allow_pickle=False) as z:
            for band, table in per.items():
                tau = original if rank<=lag.STARTUP else thresholds[band]
                for k in NEW:
                    table[k].append(dict(shot=sid,disruptive=d,down_s=down,
                                         **score_curve(z['times'],z[k],d,down,tau)))
    write(output/'per_shot.json',per)
    metrics = {band:{k:timing_decomposition(v) for k,v in table.items()}
               for band,table in per.items()}
    paired = {band:{f'{a}__to__{b}':flips(table[a],table[b]) for a,b in COMPARISONS}
              for band,table in per.items()}
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],ARMS))
    write(output/'result.json',dict(status='COMPLETED_SIGNED_GUARD_DEVELOPMENT',
        metrics=metrics,paired=paired,updates=updates,seconds=time.monotonic()-started))
    print({band:{k:(m['tp'],m['fp']) for k,m in table.items()}
           for band,table in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    run(cli.previous.resolve(), cli.output.resolve())
