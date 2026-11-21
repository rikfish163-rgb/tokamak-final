"""Adapt the old part of the encoder's causal history to a shot's observed dynamics."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_cross_diagnostic_20261113 as cross
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


group, lag, layer = cross.group, cross.lag, cross.layer
ARCHIVE, WAIT, LOCK = cross.precision.ARCHIVE, cross.WAIT, cross.LOCK_KEYS
WINDOW, PATCH, RECENT = 128, 8, 32
SPANS = (64, 144)
TRUSTED = tuple(c for k in group.TRUSTED for c in k)
OLD = (*group.OLD, 'current_5', 'earlier_5')
DIALS = ('fixed_short', 'fixed_long', 'fixed_mix', 'span_now', 'span_earlier', 'span_shifted')
ARMS = (*OLD, 'legacy_matched', *DIALS)
WITNESS = ('no_added_write', *DIALS)
CASES = (70320, 73405, 79649, 71076)


def span_coordinates(ends, *, device):
    """Anchor both donor views to the end of the historical part, never to shot length."""
    end = torch.as_tensor(ends, device=device)
    anchors = end-RECENT
    history = WINDOW-RECENT
    received = anchors[:, None]-torch.arange(history-1, -1, -1, device=device)
    steps = torch.linspace(1., 0., history, device=device, dtype=torch.float64)
    width = torch.as_tensor(SPANS, device=device)
    return received, anchors[:, None, None]-(width[None, :, None]-1)*steps


def span_changes(x, observed, ends):
    received, donor = span_coordinates(ends, device=x.device)
    lo = donor.floor().long()
    hi = donor.ceil().long()
    within = (lo >= 0).all(dim=(1, 2))
    a, b = lo.clamp_min(0), hi.clamp_min(0)
    portion = (donor-lo).to(x.dtype)[..., None]
    both = ((observed[a] & observed[b]).all(dim=1) &
            observed[received.clamp_min(0)] & within[:, None, None])
    both[:, -1] = False  # Same old endpoint in both views; do not move its anchor.
    d = (x[a].lerp(x[b], portion)-x[received.clamp_min(0)][:, None]).masked_fill(~both[:, None], 0.)
    # Center separately for each within-patch sample position. The linear
    # embedding then preserves the sum of the old time-token changes.
    d = d.unflatten(2, (-1, PATCH))
    keep = both.unflatten(1, (-1, PATCH))[:, None]
    return (d-d.sum(dim=2, keepdim=True)/keep.sum(dim=2, keepdim=True).clamp_min(1)).masked_fill(
        ~keep, 0.).flatten(2, 3)


def normalized_history(base, signal):
    x = torch.as_tensor(signal['values'], device=base.center.device)
    m = torch.as_tensor(signal['available'], device=base.center.device) & torch.isfinite(x)
    m = m[:, TRUSTED]
    z = ((torch.where(m, x[:, TRUSTED], base.center[list(TRUSTED)])
          -base.center[list(TRUSTED)])/base.scale[list(TRUSTED)]).asinh()
    return z, m


def history_basis(base, x, obs, ends):
    raw = span_changes(x, obs, ends).unflatten(2, (-1, PATCH))
    weight = base.embedding.weight.unflatten(-1, (PATCH, 46))[:, :, TRUSTED]
    return F.pad(torch.einsum('bspic,dic->bspd', raw, weight), (0, 0, 0, RECENT//PATCH))


def forward_scale(base, x, mask, transform, basis, dial, dose=None):
    """Change time tokens, preserve the recent four, CLS, and the frozen head."""
    batch, arms = len(x), len(dial)
    stash, ret = {}, []

    def prepare(_m, args):
        raw = args[0][:batch]
        adjusted = transform(raw)
        cap = F.linear(adjusted-raw, base.embedding.weight).flatten(1).norm(dim=1)
        stash['cap'] = cap
        return (adjusted.repeat(arms+1, 1, 1),)

    def inject(_m, args):
        tokens = args[0]
        early = tokens[batch:, :-1].reshape(arms, batch, WINDOW//PATCH, -1)
        cap = stash['cap']
        requested = cap if dose is None else dose
        delta = dial[..., None, None]*basis[None, :, 0]+(1.-dial[..., None, None])*basis[None, :, 1]
        supported = (delta.flatten(2).norm(dim=-1)>group.EPS).all(dim=0)
        got_target = requested.masked_fill(~supported, 0.)
        delta = group.matched_writes(delta, got_target)
        got = ((early+delta)-early).flatten(2).norm(dim=2)
        error = torch.maximum((got-got_target).abs().max(), (got-cap).clamp_min(0.).max())
        stash.update(dose=got_target, error=error)
        joined = torch.cat((torch.zeros_like(delta[:1]), delta)).flatten(0, 1)
        return (layer.add_to_patches(tokens, joined), *args[1:])

    hooks = (base.embedding.register_forward_pre_hook(prepare),
             base.encoder.layers[0].register_forward_pre_hook(inject),
             base.risk_head.register_forward_pre_hook(lambda _m, a: ret.append(a[0])))
    try:
        base(x.repeat(arms+1, 1, 1), mask.repeat(arms+1, 1, 1))
    finally:
        for handle in hooks:
            handle.remove()
    return ret[0].unflatten(0, (arms+1, batch)), stash['dose'], stash['cap'], float(stash['error'])


def fixed_dials(count):
    return np.broadcast_to(np.array([[1.], [0.], [.5]], dtype=np.float32), (3, count))


def choices_from_losses(q, eligible, qualified, actual, wrong):
    """Solve the same one-parameter two-view choice using current, older, or shuffled truth."""
    # Demand the previous clock slot, not an arbitrarily older well-supported row.
    prev = np.searchsorted(q, q-lag.STRIDE).clip(max=len(q)-1) if len(q) else q
    ok = eligible & qualified
    shared = (ok & ok[prev] & (q[prev]+lag.STRIDE==q) &
              (q[prev]+lag.H+1<=q))

    def choose(loss):
        margin = np.sign(loss[:, 1]-loss[:, 0])
        return np.where(ok, .5+.5*margin, .5).astype(np.float32)

    now = choose(actual)
    return np.vstack((fixed_dials(len(q)), now, now[prev], choose(wrong))), shared


def action_clock(q, shared, endpoints):
    release = q+lag.H+1+WAIT
    slot = np.searchsorted(release, endpoints, side='right')
    feasible = np.r_[False, shared]
    current = feasible[slot]
    previous = feasible[np.maximum(slot-1, 0)]
    onset = np.r_[-1, release][slot]
    return slot, current & ((endpoints>onset) | previous)


def _center(t):
    return t-t.mean(dim=-2, keepdim=True)


def loss_by_view(pred, observed):
    return F.smooth_l1_loss(_center(pred)-_center(observed), torch.zeros_like(pred),
                           beta=group.BETA, reduction='none').mean(dim=(-2, -1))


def fields_at(fields, prior, i):
    return (prior[i], *(e[i] for e in fields[:3]), fields[3])


def encoder_at(base, signal, coords, ends, q, phase, correction, dials, dose=None):
    x, mask = lag._windows(base, signal, ends)
    k = lag.delayed_slots(q, ends)[0]
    basis = history_basis(base, *coords, ends)
    dial = torch.tensor(dials, device=x.device, dtype=x.dtype)
    return forward_scale(base, x, mask, layer.patcher(phase[k], correction[k]), basis, dial, dose)


def track_shot(signal, args, decoder, path, archive):
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(signal, args, decoder, path)
    with torch.no_grad():
        observed, fields, waveform = group.feedback_waveform(signal, q, args[2], z.device)
        outputs = group.signed_levels(waveform, fields[1], fields[2])
        np.testing.assert_array_equal(q+lag.H+1, archive['release'])
        raw = outputs.new_zeros(outputs.shape)
        _, _, _, original = group.past_force_parts(q, raw, observed)
        np.testing.assert_array_equal(original, archive['eligible'])
        coord = normalized_history(base, signal)
        actual, shifted, qualified = np.zeros((len(q),2)), np.zeros((len(q),2)), np.zeros(len(q), bool)
        obs_loss = np.zeros((len(q),4), dtype=np.float32)
        issued = np.flatnonzero(observed & (q>=RECENT+max(SPANS)-1))
        maximum = np.zeros(3)
        for o in range(0, len(issued), group.BATCH):
            i = issued[o:o+group.BATCH]
            state, dose, _, err = encoder_at(base, signal, coord, q[i], q, phase, correction,
                                             fixed_dials(len(i)))
            qualified[i] = (dose>0.).cpu().numpy()
            pred = cross.decode(state, fields_at(fields, prior, i), decoder)
            actual[i] = loss_by_view(pred[1:3],outputs[i]).T.cpu().numpy()
            shifted[i] = loss_by_view(pred[1:3],outputs[i].roll(2, dims=-2)).T.cpu().numpy()
            obs_loss[i] = loss_by_view(pred,outputs[i]).T.cpu().numpy()
            maximum[0] = max(maximum[0], err)

        # The learner can consume a forecast before its first archived write,
        # but all compared alarm arms start only where the archive could write.
        weights, available = choices_from_losses(q, observed, qualified, actual, shifted)
        parent, original_dose = archive['bias_only'], archive['dose']
        curves = np.broadcast_to(parent, (len(DIALS), len(parent))).copy()
        legacy = parent.copy()
        on = np.flatnonzero(original_dose>0.)
        slot, legal = action_clock(q, available, on)
        t, cur = on[legal], slot[legal]-1
        dose = np.zeros_like(parent)
        for o in range(0, len(t), group.BATCH):
            i, k = t[o:o+group.BATCH], cur[o:o+group.BATCH]
            p = np.maximum(k-1, 0)
            # At a transition the previous weight stays in effect until the
            # inherited handoff. Before the first learned weight use the
            # explicitly tested fixed equal mixture, not a fitted prior.
            prev = np.where((k>0) & available[p], weights[:,p], fixed_dials(len(i))[2])
            prior_weights = np.vstack((weights[:3,k], prev[3:]))
            alpha = group.handoff_fraction(q[k]+lag.H+1+WAIT, i, PATCH).astype(np.float32)
            dials = prior_weights+(weights[:,k]-prior_weights)*alpha
            requested = torch.as_tensor(original_dose[i], device=z.device)
            vh, used, _, err = encoder_at(base, signal, coord, i, q, phase, correction, dials, requested)
            prob = (z[i]+((vh-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
            valid = used.cpu().numpy()>0.
            chosen = i[valid]
            curves[:, chosen] = prob[1:,valid]
            legacy[chosen] = archive['current_5'][chosen]
            dose[chosen] = original_dose[chosen]
            maximum[1:] = np.maximum(maximum[1:], (err, abs(prob[0]-parent[i]).max()))

        later_origin = q + 2*lag.STRIDE
        later = np.searchsorted(q,later_origin).clip(max=max(len(q)-1,0))
        accepted = np.flatnonzero(available)
        i = accepted[(q[later[accepted]]==later_origin[accepted]) & qualified[later[accepted]]]
        # Require a realized action at the fully handed-off warning endpoint.
        action = q[i]+lag.H+1+WAIT+PATCH
        i = i[dose[action]>0.]
        j = later[i]
        follow = np.empty((len(i), len(WITNESS)), dtype=np.float32)
        for o in range(0, len(i), group.BATCH):
            sl = slice(o,o+group.BATCH)
            k = j[sl]
            v, d, _, err = encoder_at(base, signal, coord, q[k], q, phase, correction, weights[:,i[sl]])
            if not (d>0.).all():
                raise ValueError('holdout must support all replayed history weights')
            pred = cross.decode(v, fields_at(fields, prior, k), decoder)
            follow[sl] = loss_by_view(pred,outputs[k]).T.cpu().numpy()
            # Fixed endpoint weights must exactly reproduce their issue-time losses.
            np.testing.assert_allclose(follow[sl,:4], obs_loss[k], rtol=0., atol=7e-6)
            maximum[0] = max(maximum[0], err)
        frames = np.c_[q[i],q[i]+lag.H+1+WAIT+PATCH,q[j],q[j]+lag.H+1]
        if (maximum[:2]>1e-5).any() or maximum[2]>3e-6 or not np.isfinite(curves).all():
            raise ValueError(f'invalid span state vs frozen parent or realized token dose: {maximum}')
        detail = dict(q=q, qualified=qualified, available=available, weights=weights, dose=dose,
                      frames=frames, follow=follow)
        return dict(legacy_matched=legacy,**dict(zip(DIALS, curves, strict=True))), detail, maximum


def inputs():
    lag.configure()
    lock = lag.load_inputs()
    ref, _, args, decoder, path = group.load_reference(lock)
    previous, seal = (read(ARCHIVE/f'{k}.json') for k in ('protocol','seal'))
    if (any(ref[k]!=previous[k] for k in LOCK) or len(seal['curve_sha256'])!=360
            or previous['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or (args[0].cfg.window,args[0].cfg.patch)!=(WINDOW,PATCH)):
        raise ValueError('same source, thresholds, reference stream and encoder window required')
    return lock, previous, seal, args, decoder, path


def archived_shot(rank, sid, s):
    name = f'{rank:04d}_{sid}.npz'
    p = ARCHIVE/'shots'/name
    if sha(p)!=s['curve_sha256'][name]:
        raise ValueError('locked reference curve changed')
    with np.load(p, allow_pickle=False) as f:
        return {k:f[k] for k in (*OLD,'release','eligible')} | {
            'dose': f['dose'][cross.precision.delay.WAITS.index(WAIT)]}


def replay(out, *, smoke_rank=None):
    start = time.monotonic()
    lock, p, s, args, decoder, path = inputs()
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    copyfile(__file__, out/'executed.py')
    write(out/'protocol.json', dict(scope='EAST360_CAUSAL_ENCODER_HISTORY_SPAN_DEVELOPMENT',
        **{k:p[k] for k in LOCK}, arms=ARMS, witness=WITNESS, smoke_rank=smoke_rank,
        reference=str(ARCHIVE), reference_seal=sha(ARCHIVE/'seal.json'),
        transform=dict(recipient_older_samples=WINDOW-RECENT, recent_unchanged=RECENT, spans=SPANS,
                       channels=[io23.RAW23[c] for c in TRUSTED],
                       only_paired_observed_support=True, per_within_patch_position_centered=True),
        clock=dict(stride=lag.STRIDE, forecast_length=lag.H, available=lag.H+1, wait=WAIT,
                   handoff=PATCH, witness_issue_after=2*lag.STRIDE),
        learning='choose one of two internal receptive fields by direct 25-point diagnostic forecasting '
                 'error scored only after maturity; no training labels, new fitted weights or alarm changes',
        reset='shot-local two-way choice; unchanged cold20; no inter-shot updates'))
    frozen = [state_digest(m) for m in args[:2]]
    files, witnesses, maxima = {}, {}, np.zeros(3)
    for rank, sid in enumerate(p['order'],1):
        if smoke_rank is not None and rank!=smoke_rank:
            continue
        arc = archived_shot(rank,sid,s)
        parent = arc['bias_only']
        if rank<=p['startup']:
            curves = dict.fromkeys(ARMS[len(OLD):], parent)
            detail = dict(q=np.empty(0,np.int64), qualified=np.empty(0,bool), available=np.empty(0,bool),
                weights=np.empty((len(DIALS),0),np.float32), dose=np.zeros_like(parent),
                frames=np.empty((0,4),np.int64), follow=np.empty((0,len(WITNESS)),np.float32))
        else:
            with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome access')):
                if sha(io23.path_for('EAST',sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signals changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                curves, detail, maximum = track_shot(signal,args,decoder,path,arc)
                maxima = np.maximum(maxima, maximum)
                if rank in (21,120,360) or sid in CASES or smoke_rank==rank:
                    n = min(590,len(parent))
                    short = {k: signal[k][:n] for k in ('values','available')}
                    sub = {k: arc[k][:n] for k in (*OLD,'dose')}
                    matured = arc['release']<n
                    sub.update(release=arc['release'][matured], eligible=arc['eligible'][matured])
                    pc, pd, _ = track_shot(short,args,decoder,path,sub)
                    for key in pc:
                        np.testing.assert_allclose(pc[key],curves[key][:n],atol=3e-6,rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(pc[key]>=tau,curves[key][:n]>=tau)
                    horizon = detail['frames'][:,-1]<n
                    for k in ('dose','q','available','weights'):
                        np.testing.assert_array_equal(pd[k],detail[k][...,:pd[k].shape[-1]])
                    np.testing.assert_array_equal(pd['frames'], detail['frames'][horizon])
                    np.testing.assert_allclose(pd['follow'],detail['follow'][horizon],atol=7e-6,rtol=0.)
                    witnesses[str(sid)] = dict(prefix=n, follow=len(pd['frames']),
                        max_error=max(float(abs(pc[k]-curves[k][:n]).max()) for k in pc))
        name = f'{rank:04d}_{sid}.npz'
        with (out/'shots'/name).open('xb') as f:
            np.savez_compressed(f,**{k:arc[k] for k in OLD},**curves,**detail)
        files[name] = sha(out/'shots'/name)
        if smoke_rank or rank in (21,60,120,180,240,300,360):
            print(dict(stage='sealed_history_span',rank=rank,write_endpoints=int((detail['dose']>0.).sum()),
                       seconds=round(time.monotonic()-start,2)),flush=True)
    if frozen != [state_digest(m) for m in args[:2]]:
        raise ValueError('source model or inherited network changed')
    write(out/'seal.json',dict(curves=files,witnesses=witnesses,maximum_error=maxima.tolist(),
                              elapsed_s=time.monotonic()-start))


def score(out):
    p, seal = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if p['smoke_rank'] is not None or len(seal['curves'])!=360 or p['reference_seal']!=sha(ARCHIVE/'seal.json'):
        raise ValueError('score only a sealed full stream')
    archived = read(ARCHIVE/'per_shot.json')
    per = {b:{k:[] for k in ARMS} for b in p['calibration']}
    sets = {g:[] for g in ('all','late_available','late_target_mature','safe_actual_false_before_first')}
    late, activity = [], []
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        if sha(out/'shots'/name)!=seal['curves'][name]:
            raise ValueError('sealed prediction changed')
        y, down, times = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            active = f['dose']>0.
            for b,tau in p['calibration'].items():
                for a in ARMS:
                    if a not in OLD:
                        np.testing.assert_array_equal(f[a][~active],f['bias_only'][~active])
                    row = dict(shot=sid,disruptive=y,down_s=down,**score_curve(times,f[a],y,down,
                        p['original'] if rank<=p['startup'] else tau))
                    if a in OLD and archived[b][a][rank-1]!=row:
                        raise ValueError('reference first alarm changed')
                    per[b][a].append(row)
            use = f['available']
            now, prev, wrong = f['weights'][3:,use]
            activity.append(dict(shot=sid, active=int(active.sum()), forecast=int(f['qualified'].sum()),
                actions=int(use.sum()), short=int((now==1).sum()), long=int((now==0).sum()),
                changed_vs_earlier=int((now!=prev).sum()), changed_vs_shift=int((now!=wrong).sum())))
            frame, witness = f['frames'], f['follow']
            if len(frame):
                sets['all'].append(witness)
            r = per['0.2']['current_5'][-1]
            tau = p['calibration']['0.2']
            if rank<=p['startup']:
                continue
            if y and not r['timely']:
                dt = np.round((down-times)*1000., 9)
                valid = (dt[frame[:,1]]>30.) & (dt[frame[:,1]]<=125.)
                mature = valid & (dt[frame[:,3]]>30.)
                if valid.any():
                    sets['late_available'].append(witness[valid])
                if mature.any():
                    sets['late_target_mature'].append(witness[mature])
                take = (dt>30.) & (dt<=125.) & active
                if take.any():
                    orig = logit(f['current_5'][take].clip(1e-6,1-1e-6))
                    late.append(dict(shot=sid, points=int(take.sum()), gap=float(logit(tau)-orig.max()),
                        uplift={k:float((logit(f[k][take].clip(1e-6,1-1e-6))-orig).max()) for k in DIALS},
                        crossing={k:bool((f[k][take]>=tau).any()) for k in DIALS}))
            elif not y and r['alarm']:
                hit = int(np.flatnonzero(f['current_5']>=tau)[0])
                valid = frame[:,3]<=hit
                if valid.any():
                    sets['safe_actual_false_before_first'].append(witness[valid])
    compare = [('legacy_matched','span_now'),('current_5','span_now'),('bias_only','span_now'),
               ('legacy_matched','current_5'),*((k,'span_now') for k in DIALS if k!='span_now')]
    paired = {b:{f'{a}__to__{c}':dict(flips=flips(r[a],r[c]),lead=lead.paired(r[a],r[c]))
                 for a,c in compare} for b,r in per.items()}
    report = {}
    for name,samples in sets.items():
        if not samples:
            report[name] = dict(shots=0,blocks=0)
            continue
        by_shot = np.stack([v.mean(0, dtype=np.float64) for v in samples])
        run = by_shot[:,WITNESS.index('span_now')]
        report[name] = dict(shots=len(samples),blocks=sum(map(len,samples)),
            motion_huber=dict(zip(WITNESS,by_shot.mean(0).tolist(),strict=True)),
            current_better={k:int((run<by_shot[:,j]).sum()) for j,k in enumerate(WITNESS) if k!='span_now'})
    metrics = {b:{k:lead.summarize(v) for k,v in rows.items()} for b,rows in per.items()}
    write(out/'per_shot.json', per)
    write(out/'result.json',dict(metrics=metrics,paired=paired,followup=report,late=late,
        coverage=activity,raw_verified=360*len(ARMS)*len(per)))
    print({b:{k:(row['by_deadline']['30']['tp'],row['fp']) for k,row in rr.items()} for b,rr in metrics.items()},
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke-rank',type=int)
    ap.add_argument('--score-only',action='store_true')
    config = ap.parse_args()
    if not config.score_only:
        replay(config.output.resolve(),smoke_rank=config.smoke_rank)
    if config.smoke_rank is None:
        score(config.output.resolve())
