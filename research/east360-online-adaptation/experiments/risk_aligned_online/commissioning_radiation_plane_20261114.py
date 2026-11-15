"""EAST-only, measured-radiation-preserving writes before the first attention block."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_weak_signal_probe_20261114 as probe
from . import io23, simple_data, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest


lag, group, layer = probe.lag, probe.group, probe.layer
history = probe.history
RAD = tuple(io23.RAW23.index(c) for c in ('sxr_var', 'xuv_var', 'sxr_skew', 'xuv_skew'))
POSITIONS = tuple(lag.CHANNELS.index(c) for c in RAD)
NEW = ('fixed_separation', 'current_feedback', 'earlier_feedback')
ARMS = (*history.OLD, *NEW)
WIDTH, PATCH = probe.WIDTH, history.PATCH
CASES = (70320, 73405, 69813, 73790, 74970, 85422)
EPS = 1e-8


class Projection:
    def __init__(self, weight):
        w = weight.detach().unflatten(1, (PATCH, 46))
        self.axes = w[..., list(RAD)].sum(1)
        self.other = tuple(c for c in range(23) if c not in RAD)
        self.coords_weight = (torch.linalg.pinv(self.axes) @ w[..., list(self.other)].flatten(1))

    def coords(self, patches):
        """Other observed VALUE lanes only, in each selected axis's token coordinates."""
        x = patches.unflatten(-1, (PATCH, 46))
        val = x[..., list(self.other)]
        other_known = x[..., [23+c for c in self.other]].bool()
        known = x[..., [23+c for c in RAD]].bool().all(-2)
        return F.linear(val.masked_fill(~other_known, 0.).flatten(2), self.coords_weight).masked_fill(~known, 0.)

    def change(self, patches, gain):
        # Direct radiation, mask contributions and the CLS are untouched.
        return (self.coords(patches)*gain[:, None, :]) @ self.axes.T


def matched(proposed, ceiling, *, origin):
    size = ((origin+proposed)-origin).flatten(2).norm(dim=-1)
    want = torch.minimum(size.amin(0), ceiling).masked_fill(~(size > EPS).all(0), 0.)
    return proposed*(want/size.clamp_min(EPS))[..., None, None]


def slots(q, t, enabled, *, patch):
    r = q+lag.H+1+history.WAIT
    s = np.searchsorted(r, t, side='right')
    available = np.r_[False, enabled]
    previous = np.maximum(s-1, 0)
    portion = group.handoff_fraction(np.r_[0,r][s], t, patch)
    return s*available[s], previous*available[previous]*available[s], portion


def as_patches(base, x, mask, t, q, phase, correction):
    # The bound source uses a single asinh value view and separate mask lanes.
    shift = lag.delayed_slots(q, t)[0]
    z = ((torch.where(mask, x, base.center)-base.center)/base.scale).asinh()
    p = torch.cat((z, mask.to(z.dtype)), dim=-1).reshape(len(x), WIDTH//PATCH, PATCH*46)
    return layer.patcher(phase[shift], correction[shift])(p)


def heads(base, values, masks, patches, gain, proj, *, ceiling=None):
    """One real frozen nonlinear forward; changes apply only before attention."""
    n = len(values)
    k = 1 if ceiling is None else 1+len(gain)
    hidden, errors = [], []

    def inject(_module, inputs):
        tokens = inputs[0]
        if ceiling is None:
            shift = proj.change(patches, gain)
        else:
            before = tokens[n:, :-1].reshape(k-1,n,WIDTH//PATCH,base.cfg.dim)
            shape = proj.coords(patches)
            candidate = (shape[None]*gain[:, :, None, :]) @ proj.axes.T
            active = matched(candidate,ceiling,origin=before)
            got = ((before+active)-before).flatten(2).norm(dim=-1)
            errors.append(torch.stack((got[0], (got-got[0]).abs().amax(0),
                                       (got-ceiling).clamp_min(0.).amax(0))))
            shift = torch.cat((torch.zeros_like(active[:1]),active)).flatten(0,1)
        return (layer.add_to_patches(tokens,shift), *inputs[1:])

    handles = (base.embedding.register_forward_pre_hook(lambda _m,_a:(patches.repeat(k,1,1),)),
               base.encoder.layers[0].register_forward_pre_hook(inject),
               base.risk_head.register_forward_pre_hook(lambda _m,a:hidden.append(a[0])))
    try:
        base(values.repeat(k,1,1),masks.repeat(k,1,1))
    finally:
        for handle in handles:
            handle.remove()
    return (hidden[0].reshape(k,n,-1), errors[0]) if errors else hidden[0]


def block_increments(wave, anchor):
    """Each five-point mean minus the previous one, starting at the issued anchor."""
    level = wave.unflatten(1, (-1, group.BIN)).mean(2)
    return level-torch.cat((anchor[:,None],level[:,:-1]),dim=1)


def estimate(signal,args,decoder,path,proj):
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    base = args[0]
    eligible,fields,truth = group.feedback_waveform(signal,q,args[2],z.device)
    shape = h.new_zeros((len(q),lag.H//group.BIN,len(RAD)))
    _, _, _, common = group.past_force_parts(q,shape,eligible)
    gain = h.new_zeros((len(NEW),len(q)+1,len(RAD)))
    recent, _, spread, noise = fields
    forecast_metrics = h.new_zeros((len(q),4))
    eligible_positions = np.flatnonzero(eligible)
    for j in range(0,len(eligible_positions),group.BATCH):
        i = eligible_positions[j:j+group.BATCH]
        tx,mx = lag._windows(base,signal,q[i])
        pp = as_patches(base,tx,mx,q[i],q,phase,correction)
        a = h.new_zeros((len(i),len(RAD)),requires_grad=True)
        hidden = heads(base,tx,mx,pp,a,proj)
        wave = lag._predict(hidden,prior[i],decoder)*noise+recent[i,None]
        guess = wave[...,list(POSITIONS)]
        obs = truth[i][...,list(POSITIONS)]
        last = recent[i][:,list(POSITIONS)]
        den = spread[i][:,None,list(POSITIONS)]
        prediction, target = (block_increments(v,last)/den for v in (guess,obs))
        shape[i] = ((target-prediction.detach())/group.BETA).clamp(-1.,1.)
        _, now, earlier, _ = group.past_force_parts(q,shape,eligible)
        g1,g2 = (torch.autograd.grad(prediction,a,grad_outputs=v[i],retain_graph=(k==0))[0]
                 for k,v in enumerate((now,earlier)))
        use = torch.as_tensor(common[i],device=z.device)
        gain[:,i+1] = torch.stack((torch.full_like(g1,-1.),g1.sign(),g2.sign()))*use[:,None]
        with torch.no_grad():
            forecast_metrics[i] = torch.stack(
                (((prediction-target).abs()).mean((1,2)),target.abs().mean((1,2)),
                 ((guess-obs)/den).abs().mean((1,2)),((last[:,None]-obs)/den).abs().mean((1,2))),dim=1)
    return q,h,z,beta,phase,correction,gain,common,forecast_metrics.cpu().numpy()


@torch.no_grad()
def predict(signal,base,track,archive):
    q,h,z,beta,phase,correction,gain,available,_ = track
    fixed = archive['bias_only']
    dose = archive['dose']
    curves = np.broadcast_to(fixed,(len(NEW),len(fixed))).copy()
    actual = np.zeros_like(dose)
    err = np.zeros(3)
    proj = Projection(base.embedding.weight)
    active = np.flatnonzero(dose>0.)
    for j in range(0,len(active),group.BATCH):
        t = active[j:j+group.BATCH]
        current, old, f = slots(q,t,available,patch=PATCH)
        now = gain[:,current].lerp(gain[:,old], z.new_tensor(1.-f)[None,:,None])
        x,m = lag._windows(base,signal,t)
        pp = as_patches(base,x,m,t,q,phase,correction)
        outs, radii = heads(base,x,m,pp,now,proj,ceiling=z.new_tensor(dose[t]))
        ref = (z[t]+((outs-h[t])*beta).sum(-1)).sigmoid().cpu().numpy()
        err = np.maximum(err,[np.abs(ref[0]-fixed[t]).max(), float(radii[1].max()),float(radii[2].max())])
        actual[t] = radii[0].cpu().numpy()
        curves[:,t] = np.where(actual[t]>0.,ref[1:],fixed[t])
    if err[0]>3e-6 or err[1:].max()>1e-5:
        raise ValueError(f'baseline reconstruction or matched edit failed: {err}')
    return curves,actual,err


def run(out):
    begun = time.monotonic()
    lock,p,seal,args,decoder,path = history.inputs()
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(scope='EAST360_RADIATION_PRESERVING_TOKEN_ADAPTATION_DEVELOPMENT',
        **{k:p[k] for k in history.LOCK}, arms=ARMS, archive=str(history.ARCHIVE),
        archive_seal=sha(history.ARCHIVE/'seal.json'), feedback='direct 25 samples; five adjacent '
        'signed bin increments; subtract past expected error and test equal-clock earlier feedback',
        position='actual first attention input; only other value lanes projecting along the four '
        'radiation axes; do not modify radiation values, missingness contribution or CLS input',
        direction='four temporary signed coefficients, per-channel nonzero sign of the true or older '
        'negative diagnostic gradient; fixed control removes the projected contribution',
        dose='all three arms share the smallest achievable real token norm, capped by the old dose',
        clock=dict(issue_stride=lag.STRIDE, complete_samples=lag.H, extra_wait=history.WAIT, handoff=PATCH),
        labels='unchanged cold20; all target labels blocked before full predicted curve seal'))
    base = args[0]
    frozen = [state_digest(m) for m in args[:2]]
    proj = Projection(base.embedding.weight)
    checks, files, errors = {}, {}, np.zeros(3)
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('prediction cannot access outcomes')):
        for rank,sid in enumerate(p['order'],1):
            arc = history.archived_shot(rank,sid,seal)
            curves = np.broadcast_to(arc['bias_only'],(len(NEW),len(arc['bias_only'])))
            actual = np.zeros_like(arc['dose'])
            forecast = np.zeros((len(arc['release']),4),np.float32)
            if rank>p['startup']:
                if sha(io23.path_for('EAST',sid)) != lock['hashes'][str(sid)]:
                    raise ValueError('same original signal required')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                track = estimate(signal,args,decoder,path,proj)
                forecast = track[-1]
                np.testing.assert_array_equal(track[0]+lag.H+1,arc['release'])
                np.testing.assert_array_equal(track[-2],arc['eligible'])
                curves,actual,error = predict(signal,base,track,arc)
                errors = np.maximum(errors,error)
                if rank in (21,120,360) or sid in CASES:
                    hit = np.flatnonzero((curves[1]>=p['calibration']['0.2']) & (actual>0.))
                    n = min(len(actual),max(512,int(hit[0])+2) if len(hit) else 512)
                    prefix = {k:signal[k][:n] for k in ('values','available')}
                    short = estimate(prefix,args,decoder,path,proj)
                    again,d,_ = predict(prefix,base,short,{k:arc[k][:n] for k in ('bias_only','dose')})
                    probe.check_thresholds(again,curves[:,:n],tuple(p['calibration'].values()))
                    np.testing.assert_allclose(d,actual[:n],atol=5e-6,rtol=0.)
                    checks[str(sid)] = dict(prefix_points=n,max_score_delta=float(abs(again-curves[:,:n]).max()))
            dest = out/'shots'/f'{rank:04d}_{sid}.npz'
            with dest.open('xb') as f:
                np.savez_compressed(f,**dict(zip(NEW,curves,strict=True)),actual=actual,forecast=forecast)
            files[dest.name] = sha(dest)
            if rank in (21,60,120,180,240,300,360):
                print(dict(stage='seal_radiation_plane', rank=rank, seconds=round(time.monotonic()-begun,2)),
                      flush=True)
    if frozen != [state_digest(m) for m in args[:2]]:
        raise ValueError('parameter change in frozen networks')
    write(out/'seal.json',dict(curves=files,prefix=checks,max_errors=errors.tolist(),seconds=time.monotonic()-begun))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    archive = read(history.ARCHIVE/'seal.json')
    old = read(history.ARCHIVE/'per_shot.json')
    if len(s['curves'])!=360 or p['archive_seal']!=sha(history.ARCHIVE/'seal.json'):
        raise ValueError('sealed matched complete stream required')
    rows = {band:{k:[] for k in ARMS} for band in p['calibration']}
    support = dict(available=0,written=0,discharges=0,eligible=0)
    regular, late = [], []
    unsafe = {}
    for rank,sid in enumerate(p['order'],1):
        f = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=s['curves'][f.name]:
            raise ValueError('sealed method curves changed')
        with np.load(f,allow_pickle=False) as g:
            view = dict(g)
        arc = history.archived_shot(rank,sid,archive)
        y,down,t = group.hdf._raw_truth(sid)
        active = view['actual']>0.
        if (active & (arc['dose']<=0.)).any():
            raise ValueError('extra write outside the matched inherited clock')
        support['available'] += int((arc['dose']>0.).sum())
        support['written'] += int(active.sum())
        support['discharges'] += bool(active.any())
        support['eligible'] += int(arc['eligible'].sum())
        for k in NEW:
            np.testing.assert_array_equal(view[k][~active],arc['bias_only'][~active])
        for band,tau in p['calibration'].items():
            if rank<=p['startup']:
                tau=p['original']
            for k,series in ((k,(arc if k in history.OLD else view)[k]) for k in ARMS):
                row=dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,series,y,down,tau))
                if k in history.OLD and row!=old[band][k][rank-1]:
                    raise ValueError('original HDF5 result does not reproduce')
                rows[band][k].append(row)
        if rank>p['startup'] and arc['eligible'].any():
            regular.append(view['forecast'][arc['eligible']].mean(0))
            if y and not old['0.2']['current_5'][rank-1]['timely']:
                idx = np.flatnonzero(arc['eligible'] & (arc['release']+history.WAIT<len(t)))
                lead_at_write = np.round((down-t[arc['release'][idx]+history.WAIT])*1e3,9)
                idx = idx[(lead_at_write>30.) & (lead_at_write<=125.)]
                if len(idx):
                    late.append(view['forecast'][idx].mean(0))
                unsafe[str(sid)] = int((active & (np.round((down-t)*1e3, 9)>30.) &
                                        (np.round((down-t)*1e3,9)<=125.)).sum())
    metrics = {b:{k:lead.summarize(v) for k,v in r.items()} for b,r in rows.items()}
    pair = {b:{f'{a}__to__{k}':dict(flips=flips(r[a],r[k]),lead=lead.paired(r[a],r[k]))
               for a in ('bias_only','current_5','fixed_separation','earlier_feedback')
               for k in NEW if a!=k} for b,r in rows.items()}
    columns = ('five_increment_forecast','five_increment_hold','waveform_forecast','waveform_hold')
    diag = dict(columns=columns, all_discharges=len(regular), means=np.mean(regular,axis=0).tolist(),
                late_discharges=len(late), late_means=np.mean(late,axis=0).tolist())
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,paired=pair,support=support,forecast=diag,
        late_near_window_write_points=unsafe,raw_verified=360*len(ARMS)*len(metrics)))
    for b,r in metrics.items():
        print(b,{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in r.items()},flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--score',action='store_true')
    opts = cli.parse_args()
    (score if opts.score else run)(opts.output.resolve())
