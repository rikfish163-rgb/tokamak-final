"""Paired EAST replay: vary the ended examples, then isolate same-shot gradients."""

import argparse
import os
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
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


FIXED, RECENT, MATCHED, TRUE, SHAM, JOINT, JOINT_SHAM = ARMS = (
    'fixed', 'inter_recent', 'inter_matched', 'intra_true', 'intra_shifted',
    'joint_true', 'joint_shifted',
)
PAIRS = ((FIXED, TRUE), (SHAM, TRUE), (FIXED, RECENT), (FIXED, MATCHED),
         (RECENT, MATCHED), (MATCHED, JOINT), (JOINT_SHAM, JOINT), (TRUE, JOINT))
N = len(lag.CHANNELS)
SUPPORT, BATCH, STEP, TEMPERATURE = 9, 256, .10, .35
CALIPERS = teacher.CALIPER
REFERENCE = io23.ROOT/'probe_outputs/commissioning_lag_guard_east360_20261111_v1'


def select_history(history, current):
    """Take one previously ended discharge per class, changing only which ones."""
    if any(r['rank'] >= current['rank'] for r in history):
        raise ValueError('only previous sealed discharges can be replayed')
    recent, matched = [], []
    for positive in (True, False):
        candidates = [r for r in history if r['disruptive'] == positive]
        if not candidates:
            raise ValueError('the locked preparation history must contain both outcomes')
        recent.append(max(candidates, key=lambda r: r['rank']))
        matched.append(min(candidates, key=lambda r: (
            np.square((r['match']-current['match'])/CALIPERS).sum(), -r['rank'])))
    return recent, matched


def support(times, z, disruptive, down, boundary):
    """Whole safe-shot support, or the supervised early part of the positive window."""
    points = np.arange(len(times))
    if disruptive:
        lead = np.round((down-times)*1000., 9)
        points = points[(lead > 30.) & (lead <= 125.+1e-6)]
    if not len(points):
        raise ValueError('ended shot lacks legal training support')
    hits = points[z[points] >= boundary]
    anchor = int(hits[0] if not disruptive and len(hits) else points[z[points].argmax()])
    # Peaks from eight chronological parts protect more than a single alarm point.
    peaks = [part[z[part].argmax()] for part in np.array_split(points, SUPPORT-1) if len(part)]
    return np.unique([anchor, *peaks]), anchor


def alarm_loss(peaks, disruptive, boundary):
    return .5*(F.softplus((boundary-peaks[disruptive])/TEMPERATURE).mean()
               + F.softplus((peaks[~disruptive]-boundary)/TEMPERATURE).mean())


def release_slots(q, endpoints):
    return lag.delayed_slots(q, endpoints)[0]


def equalized_inputs(start, real, shifted):
    """Equalize actual RMS around the same persistent normalized input."""
    a, b = real-start, shifted-start
    dims = tuple(range(1, a.ndim))
    r1, r2 = (d.square().mean(dims, keepdim=True).sqrt() for d in (a,b))
    rms = torch.minimum(r1,r2)
    return (start+a*(rms/r1.clamp_min(1e-12)),
            start+b*(rms/r2.clamp_min(1e-12)), rms.flatten())


def seal_and_reveal(path, arrays, sid, outcome_reader):
    """The unblinded reader is called only after the complete shot is durable."""
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = sha(path)
    return digest, outcome_reader('EAST', sid)


def matrices(key, state):
    theta = torch.einsum('bk,kij->bij', key, state)
    return theta*(lag.RADIUS/theta.norm(dim=(1,2), keepdim=True).clamp_min(lag.RADIUS))


@torch.no_grad()
def _track(signal, args):
    base, net, params, _, _, _, coef, _ = args
    ends = np.arange(len(signal['values']))
    tr = lag.direct.track(net, signal, params[0], params[1], ends)
    x, _ = lag.features(base, signal, 'cuda')
    # Keep all non-adapted components of the fixed cold-prepared risk readout.
    z = (torch.from_numpy(np.c_[x, torch.from_numpy(tr['pre']).tanh().numpy()]).cuda()
         @ coef[:-1] + coef[-1])
    return tr, torch.from_numpy(x[:,:base.cfg.dim]).cuda(), z, coef[:base.cfg.dim]


def _history_row(signal, track, times, rank, sid, disruptive, down, boundary):
    tr, h, z, _ = track
    z = z.detach().cpu().numpy()
    ends, anchor = support(times, z, disruptive, down, boundary)
    context = 2.*tr['key'][anchor, 1:].astype(float)
    # Repeated endpoints are only batching padding; maxima and distinct shots are unchanged.
    batch = np.pad(ends, (0, SUPPORT-len(ends)), mode='edge')
    views = [simple_data.window(signal, int(t), lag.W) for t in batch]
    x, mask = (np.stack(v) for v in zip(*views, strict=True))
    return dict(rank=rank, shot=sid, disruptive=disruptive, match=np.r_[z[anchor], context],
                anchor=anchor, support=ends.tolist(), batch=batch, x=x, mask=mask,
                h=h[batch].cpu().numpy(), z=z[batch], key=tr['key'][batch])


def _update(base, beta, state, rows, boundary):
    # Exact nonlinear forward of the updated patch transform on the same three shots.
    device = state.device
    def batch(key):
        return torch.from_numpy(np.concatenate([r[key] for r in rows])).to(device)

    x, mask, h, z, key = (batch(k) for k in ('x','mask','h','z','key'))
    d = torch.as_tensor([r['disruptive'] for r in rows], device=device)
    state = state.detach().requires_grad_(True)
    trans = lag._features(base, x, mask, matrices(key, state))
    peaks = (z + (trans-h)@beta).reshape(len(rows), SUPPORT).amax(1)
    loss = alarm_loss(peaks, d, boundary)
    g, = torch.autograd.grad(loss, state)
    if not torch.isfinite(g).all() or not torch.isfinite(loss):
        raise FloatingPointError('nonfinite supervised lag gradient')
    with torch.no_grad():
        proposal = state-STEP*g/g.norm().clamp_min(1e-6)
        new = proposal*(lag.RADIUS/proposal.norm().clamp_min(lag.RADIUS))
        new_h = lag._features(base, x, mask, matrices(key, new))
        new_peaks = (z+(new_h-h)@beta).reshape(len(rows), SUPPORT).amax(1)
        change = new_peaks-peaks
    return new.detach(), dict(loss=float(loss.detach()), grad=float(g.norm()),
        step=float((new-state).detach().norm()), pre=peaks.detach().tolist(), post=new_peaks.tolist(),
        safe_up=int(((change>0.) & ~d).sum()),
        new_safe_cross=int(((peaks<boundary) & (new_peaks>=boundary) & ~d).sum()))


def _feedback(base, signal, q, key, pred, truth, decoder, state):
    device = state.device
    out = torch.zeros((2,len(q)+1,N,N), device=device)
    norm = torch.zeros((len(q),2), device=device)
    for start in range(0, len(q), 128):
        sl = slice(start, start+128)
        x, m = lag._windows(base, signal, q[sl])
        k, p, y = (torch.from_numpy(v[sl]).to(device) for v in (key,pred,truth))
        theta = matrices(k,state).detach().requires_grad_(True)
        forecast = lag._predict(lag._features(base,x,m,theta),p,decoder)
        signature = lag.ordered_signature(lag.motion(forecast))
        actual, sham = (lag.ordered_signature(lag.motion(y,shift=b)) for b in (False,True))
        g, g_wrong = (torch.autograd.grad(F.smooth_l1_loss(signature, v, beta=.2,
                         reduction='sum'),theta,retain_graph=i==0)[0]
                      for i,v in enumerate((actual,sham)))
        norm[sl] = torch.stack((g.norm(dim=(1,2)),g_wrong.norm(dim=(1,2))),dim=1)
        out[:,start+1:min(start+129,len(q)+1)] = torch.stack(lag.matched_steps(g,g_wrong))
    if not torch.isfinite(norm).all():
        raise FloatingPointError('nonfinite within-shot diagnostic gradient')
    return out, norm


def _paired_hidden(base, x, m, slow, real, sham):
    """Both branches make a nonlinear encoder pass, not a logit correction."""
    hidden, doses = [], []

    def transform(_layer, args):
        a, _ = args[0].chunk(2)
        start = lag.patch_input(a,slow)
        p, q, rms = equalized_inputs(start, lag.patch_input(a,slow+real),
                                     lag.patch_input(a,slow+sham))
        doses.append(rms)
        return (torch.cat((p,q)),)

    h1 = base.embedding.register_forward_pre_hook(transform)
    h2 = base.risk_head.register_forward_pre_hook(lambda _m,p: hidden.append(p[0]))
    try:
        base(torch.cat((x,x)),torch.cat((m,m)))
    finally:
        h2.remove()
        h1.remove()
    return hidden[0].chunk(2), doses[0]


@torch.no_grad()
def _curves(base, signal, track, q, states, updates):
    tr, h, z, beta = track
    ends = np.arange(len(z))
    device = z.device
    keys = torch.from_numpy(tr['key']).to(device)
    slots = release_slots(q, ends)
    curves = {k: torch.empty_like(z) for k in ARMS}
    curves[FIXED] = z.sigmoid()
    actual_rms = torch.zeros((len(z), 2), device=device)
    zero = torch.zeros_like(states[0])
    for a in range(0, len(ends), BATCH):
        i = ends[a:a+BATCH]
        x, m = lag._windows(base, signal, i)
        slow = [matrices(keys[i], w) for w in states]
        for name, theta in zip((RECENT,MATCHED),slow,strict=True):
            v = lag._features(base,x,m,theta) if theta.any() else h[i]
            curves[name][i] = (z[i]+(v-h[i])@beta).sigmoid()
        active = np.flatnonzero(slots[i])
        for col, (names, w, baseline, delta) in enumerate((
                ((TRUE,SHAM),zero,FIXED,updates[0]),
                ((JOINT,JOINT_SHAM),states[1],MATCHED,updates[1]))):
            for name in names:
                curves[name][i] = curves[baseline][i]
            if not len(active):
                continue  # No matured target is available on this prefix.
            j = i[active]
            ks = matrices(keys[j], w)
            (p,s), rms = _paired_hidden(base,x[active],m[active],ks,*delta[:,slots[j]])
            actual_rms[j,col] = rms
            for name, v in zip(names,(p,s),strict=True):
                curves[name][j] = torch.where(rms>0., (z[j]+(v-h[j])@beta).sigmoid(),
                                               curves[baseline][j])
    return {k:v.cpu().numpy() for k,v in curves.items()}, actual_rms.cpu().numpy()


def _predict(signal, args, decoder, states):
    track = _track(signal,args)
    tr = track[0]
    q, keys, pred, truth = teacher._mature(tr,args[2][2],args[2][3])
    w0 = torch.zeros_like(states[0])
    (a,ga), (b,gb) = [_feedback(args[0],signal,q,keys,pred,truth,decoder,w)
                       for w in (w0,states[1])]
    curves, rms = _curves(args[0],signal,track,q,states,(a,b))
    return curves, track, dict(release=q+lag.H+1, grad=torch.stack((ga,gb),1).cpu().numpy(),
                               rms=rms)


def _load(reference, lock):
    gp, gs = (read(reference/k) for k in ('protocol.json','seal.json'))
    previous = Path(gp['previous_teacher'])
    pp, ps = (read(previous/k) for k in ('protocol.json','seal.json'))
    lp = read(Path(pp['previous_lag'])/'protocol.json')
    coordinates = next(Path(k).parent for k in pp['assets']
                       if Path(k).parent.name.startswith('commissioning_local_cross_response_'))
    forecast = Path(lp['source_decoder']).parent
    p, _, args, assets = lag.parent._frozen(Path(lp['reference']), coordinates, lock)
    if (gp['scope']!='EAST360_SIGNED_POINT_VS_PATH_GUARD_DEVELOPMENT'
            or gp['order']!=lock['ids'] or gp['startup']!=lag.STARTUP
            or gp['calibration']!=p['calibration'] or gp['original']!=p['original']
            or sha(previous/'seal.json')!=gp['previous_seal'] or gp['assets']!=assets
            or len(ps['curve_sha256'])!=360 or len(gs['curve_sha256'])!=360):
        raise ValueError('fixed source, preparation or score reference changed')
    return gp, ps, previous, args, lag._bind_forecast(forecast,p)


def run(reference, output):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    ids = lock['ids']
    p, seal, previous, args, decoder = _load(reference,lock)
    base, net = args[:2]
    frozen = state_digest(base), state_digest(net)
    thresholds, original = p['calibration'], p['original']
    boundary = float(logit(thresholds['0.2']))
    states = tuple(base.center.new_zeros(4,N,N) for _ in range(2))
    (output/'shots').mkdir(parents=True)
    copyfile(Path(__file__),output/'executed_probe.py')
    write(output/'protocol.json',dict(scope='EAST360_NATURAL_EOS_MATCHED_VS_RECENT_REPLAY',
        order=ids, startup=lag.STARTUP, calibration=thresholds, original=original, arms=ARMS,
        reference=str(reference), reference_seal=sha(reference/'seal.json'),
        producer_sha256=sha(Path(__file__)), source_sha256=p['source_sha256'],
        feedback='whole-shot seal then natural outcome, no label queries; training only affects later shots',
        replay='one newest ended shot plus one previous positive and one previous safe',
        recency='latest per class', match='closest per class to just-ended anchor in four scaled fields',
        match_calipers=CALIPERS.tolist(), supports_per_shot=SUPPORT, supervised_updates_per_shot=1,
        step=STEP, parameter_radius=lag.RADIUS, temperature=TEMPERATURE, stride=lag.STRIDE,
        first_effect=lag.FIRST_EFFECT, equalized='real and shifted input RMS per evaluated window',
        frozen='source encoder, auxiliary forecast, pre context, risk coefficients, thresholds and map'))
    reader = io23.read_outcome
    history, prepared, fit, hashes, witnesses = [], [], [], {}, {}
    per = {band:{k:[] for k in ARMS} for band in thresholds}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden in prediction')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            f = previous/'shots'/name
            raw_file = io23.path_for('EAST',sid)
            if sha(f)!=seal['curve_sha256'][name] or sha(raw_file)!=lock['hashes'][str(sid)]:
                raise ValueError('sealed reference or current HDF5 changed')
            raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
            times = raw['times']
            with np.load(f,allow_pickle=False) as saved:
                np.testing.assert_array_equal(times,saved['times'])
                old = saved[FIXED]
            if rank<=lag.STARTUP:
                curves, trace = dict.fromkeys(ARMS,old), {}
            else:
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                curves, track, trace = _predict(signal,args,decoder,states)
                np.testing.assert_array_equal(curves[FIXED],old)
                for a,b in ((TRUE,FIXED),(SHAM,FIXED),(JOINT,MATCHED),(JOINT_SHAM,MATCHED)):
                    stop = int(trace['release'][0]) if len(trace['release']) else len(times)
                    np.testing.assert_array_equal(curves[a][:stop],curves[b][:stop])
                if rank in (21,120,360):
                    end = min(len(times),lag.FIRST_EFFECT+3*lag.STRIDE+5)
                    prefix = {k:signal[k][:end] for k in ('values','available')}
                    short,_,st = _predict(prefix,args,decoder,states)
                    for k in ARMS:
                        np.testing.assert_allclose(short[k],curves[k][:end],atol=3e-6,rtol=0.)
                        for tau in thresholds.values():
                            np.testing.assert_array_equal(short[k]>=tau,curves[k][:end]>=tau)
                    np.testing.assert_array_equal(st['release'],trace['release'][trace['release']<end])
                    witnesses[str(sid)] = end
            if not all(np.isfinite(v).all() for v in curves.values()):
                raise FloatingPointError('nonfinite prediction')
            hashes[name], (d, down) = seal_and_reveal(output/'shots'/name,
                dict(times=times,**curves,**trace),sid,reader)
            for band,tau in thresholds.items():
                for k,v in curves.items():
                    per[band][k].append(dict(shot=sid,disruptive=d,down_s=down,
                         **score_curve(times,v,d,down,original if rank<=lag.STARTUP else tau)))
            if rank<=lag.STARTUP:
                prepared.append((rank,sid,raw,d,down))
                if rank==lag.STARTUP:
                    for r,s,data,dy,dt in prepared:
                        signal = lag.prepare_actual(data,lag.STARTUP+1,lock['mapping'])
                        history.append(_history_row(signal,_track(signal,args),data['times'],r,s,dy,dt,boundary))
                    prepared.clear()
            else:
                current = _history_row(signal,track,times,rank,sid,d,down,boundary)
                groups = select_history(history,current)
                updated = [_update(base,track[-1],w,[current,*group],boundary)
                           for w,group in zip(states,groups,strict=True)]
                states = tuple(w for w,_ in updated)
                fit.append(dict(rank=rank,shot=sid,disruptive=d,anchor=current['anchor'],
                    support=current['support'], seal=hashes[name],
                    policies={k:dict(ranks=[r['rank'] for r in group],shots=[r['shot'] for r in group],
                         distances=[float(np.linalg.norm((r['match']-current['match'])/CALIPERS))
                                    for r in group], **stats)
                         for k,group,(_,stats) in zip((RECENT,MATCHED),groups,updated,strict=True)}))
                history.append(current)
            if rank in (20,21) or rank%40==0:
                print(dict(stage='shot_sealed',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    if (state_digest(base),state_digest(net))!=frozen:
        raise ValueError('source model changed')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses))
    write(output/'eos_replay.json',fit)
    write(output/'per_shot.json',per)
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],ARMS))
    result = dict(status='COMPLETED_EAST360_MATCHED_VS_RECENT_DEVELOPMENT',
        seconds=time.monotonic()-started,
        metrics={band:{k:timing_decomposition(v) for k,v in arms.items()} for band,arms in per.items()},
        paired={band:{f'{a}__to__{b}':flips(arms[a],arms[b]) for a,b in PAIRS}
                for band,arms in per.items()})
    write(output/'result.json',result)
    print({band:{k:(m['tp'],m['fp']) for k,m in table.items()}
           for band,table in result['metrics'].items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    cli = parser.parse_args()
    run(cli.reference,cli.output)
