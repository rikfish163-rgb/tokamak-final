"""Adapt the same input matrix using the arrived size of a diagnostic deviation."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_lag_clock_20261111 as clock
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_recovery_clock_20261111 as recovery
from . import io23, simple_data
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


OLD = clock.ARMS[:4]
TRUE, WRONG = 'radius_feedback','misassigned_feedback'
ARMS = (*OLD, TRUE, WRONG)
PAIRS = ((clock.STATIC,TRUE),(clock.STATIC,WRONG),(WRONG,TRUE),(clock.OLD,TRUE))
GROUPS = tuple(np.flatnonzero(lag.GROUP.numpy()==g).tolist() for g in range(4))
DIAGNOSTIC = ('static','radius_feedback','misassigned_feedback')
WINDOW, BETA = recovery.MEAN_POINTS, .2


def normalized_radius(level, mean, spread):
    """Log radius of each five-point level from its origin's known reference."""
    n,h,c = level.shape
    avg = level.reshape(n,h//WINDOW,WINDOW,c).mean(2)
    dev = (avg-mean[:,None])/spread[:,None]
    return torch.stack([torch.log1p(torch.sqrt(dev[...,g].square().mean(-1)+1e-12))
                        for g in GROUPS],dim=-1)


def residual_targets(pred, observed):
    residual = observed-pred.detach()
    return residual,residual.roll(1,dims=-1)


def mature(q, visible, reference_known, length):
    return visible.all((1,2)) & reference_known.all(1) & (q+lag.H+1<length)


def _pack(signal, track, params):
    center,scale,forecast,_,_ = params
    v,m,_,_ = lag.direct.weak.shot_coordinates(signal,center,scale)
    ref = recovery.rolling_reference(v,m)
    q = track['q']
    use = mature(q,track['observed'],ref['reference_known'][q],len(v))
    q = q[use]
    if not len(q):
        return q,()
    recent,n = _window_mean(v,m,q,-15,16)
    if not (n>=8).all():
        raise ValueError('inconsistent valid forecast origin')
    obs = v[q[:,None]+np.arange(1,lag.H+1)]
    return q,tuple(np.asarray(x,dtype=np.float32) for x in
                   (lag.direct.base_forecast(track,forecast)[use],recent,
                    ref['center'][q],ref['spread'][q],obs))


def _estimate(base, signal, q, packed, noise, decoder):
    device = base.center.device
    n = len(lag.CHANNELS)
    states = torch.zeros((2,len(q)+1,n,n),device=device)
    norms = torch.zeros((len(q),2),device=device)
    for start in range(0,len(q),128):
        block = slice(start,start+128)
        x,m = lag._windows(base,signal,q[block])
        p,recent,mu,sd,y = (a[block] for a in packed)
        theta = torch.zeros((len(x),n,n),device=device,requires_grad=True)
        future = lag._predict(lag._features(base,x,m,theta),p,decoder)*noise+recent[:,None]
        estimate,obs = (normalized_radius(a,mu,sd) for a in (future,y))
        g,w = (torch.autograd.grad(F.smooth_l1_loss(estimate-estimate.detach(),e,beta=BETA,
                        reduction='sum'),theta,retain_graph=j==0)[0]
               for j,e in enumerate(residual_targets(estimate,obs)))
        norms[block] = torch.stack((g.norm(dim=(1,2)),w.norm(dim=(1,2))),dim=1)
        states[:,start+1:min(start+129,len(q)+1)] = torch.stack(lag.matched_steps(g,w))
    if not torch.isfinite(norms).all():
        raise FloatingPointError('nonfinite feedback gradient')
    return states,norms


@torch.no_grad()
def _replay(base, signal, h, z, beta, q, states):
    ends = np.arange(len(z))
    slots,_ = lag.delayed_slots(q,ends)
    available = np.flatnonzero(slots)
    views = h.expand(2,-1,-1).clone()
    rms = z.new_zeros(len(ends))
    for j in range(0,len(available),eos.BATCH):
        indices = available[j:j+eos.BATCH]
        x,m = lag._windows(base,signal,indices)
        real,wrong = states[:,slots[indices]]
        (a,b),rms[indices] = eos._paired_hidden(base,x,m,torch.zeros_like(real),real,wrong)
        views[:,indices] = torch.stack((a,b))
    # A zero-dose arm is exactly the immutable parent, without floating-point drift.
    curves = torch.where(rms[None]>0., (z[None]+(views-h)@beta).sigmoid(),z.sigmoid()[None])
    return curves,views,rms


@torch.no_grad()
def _future_errors(h, updated, q, packed, noise, decoder):
    p,recent,mu,sd,obs = packed
    target = normalized_radius(obs,mu,sd)
    errors = []
    for view in (h,*updated):
        x = lag._predict(view[q],p,decoder)*noise+recent[:,None]
        y = normalized_radius(x,mu,sd)
        errors.append(F.smooth_l1_loss(y,target,beta=BETA,reduction='none').mean((1,2)))
    return torch.stack(errors,dim=1).cpu().numpy()


def _predict(signal, args, decoder):
    base = args[0]
    tr,h,z,beta = eos._track(signal,args)
    q,fields = _pack(signal,tr,args[2])
    static = z.sigmoid().cpu().numpy()
    if not len(q):
        return static,np.stack((static,static)),q,np.zeros(len(z),np.float32),(
            np.empty((0,2),np.float32)),np.empty((0,3),np.float32)
    fields = tuple(torch.from_numpy(a).to(z.device) for a in fields)
    noise = torch.as_tensor(args[2][3],device=z.device,dtype=h.dtype)
    steps,grad = _estimate(base,signal,q,fields,noise,decoder)
    curves,updated,rms = _replay(base,signal,h,z,beta,q,steps)
    errors = _future_errors(h,updated,q,fields,noise,decoder)
    return static,curves.cpu().numpy(),q+lag.H+1,rms.cpu().numpy(),grad.cpu().numpy(),errors


def predict(reference, out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p,s,previous,args,decoder = eos._load(reference,lock)
    fixed = read(reference/'seal.json')['curve_sha256']
    frozen = [state_digest(a) for a in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_NORMALIZED_RADIUS_FEEDBACK_DEVELOPMENT',
        reference=str(reference),reference_seal=sha(reference/'seal.json'),order=lock['ids'],
        producer_sha256=sha(Path(__file__)),source_sha256=p['source_sha256'],
        startup=p['startup'],calibration=p['calibration'],original=p['original'],
        arms=ARMS,stride=lag.STRIDE,horizon=lag.H,
        prior_points=recovery.REFERENCE_POINTS,prior_lag=recovery.REFERENCE_LAG,
        objective='Huber on five by four log standardized diagnostic radii; baseline known at issue; '
                  'signed diagnostic error is circularly reassigned across groups in the control',
        timing='same zero-state direct H25 forecast; replace temporary matrix only after q+25 '
               'arrives and only from q+26; no outcome used; no cumulative/persistent update',
        dose='exact nonlinear forwards through existing lag patch; match matrix norm and '
             'equalize realized input RMS at every scored endpoint',
        auxiliary='compare forecasts at later origins from the adapted representation already '
                  'available at issue, not the forecast used to fit its own target'))
    hashes,witnesses = {},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome unavailable')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            prepared = rank<=p['startup']
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('locked signal changed')
            t,curves,old_release = clock.read_archived_curves(reference,previous,name,fixed,
                                                             s['curve_sha256'],prepared)
            if prepared:
                curves.update(dict.fromkeys((TRUE,WRONG),curves[clock.STATIC]))
                release,rms,grad,errors = (np.empty(0,dtype=np.int64),np.zeros(len(t),np.float32),
                                           np.empty((0,2)),np.empty((0,3)))
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                static,new,release,rms,grad,errors = _predict(signal,args,decoder)
                np.testing.assert_array_equal(static,curves[clock.STATIC])
                if not np.isin(release,old_release).all():
                    raise ValueError('new feedback is off the legal complete-window clock')
                curves.update(zip((TRUE,WRONG),new,strict=True))
                if rank in (21,120,360):
                    end = min(len(t),lag.FIRST_EFFECT+3*lag.STRIDE+5)
                    prefix = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                    _,cut,r0,dose,_,_ = _predict(prefix,args,decoder)
                    np.testing.assert_allclose(cut,new[:,:end],atol=3e-6,rtol=0.)
                    np.testing.assert_allclose(dose,rms[:end],atol=3e-6,rtol=0.)
                    np.testing.assert_array_equal(r0,release[release<end])
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(cut>=tau,new[:,:end]>=tau)
                    witnesses[str(sid)] = end
            if not all(np.isfinite(a).all() for a in (*curves.values(),rms,grad,errors)):
                raise FloatingPointError('nonfinite curves or diagnostics')
            f = out/'shots'/name
            with f.open('xb') as h:
                np.savez_compressed(h,times=t,release=release,input_rms=rms,grad_norm=grad,
                                    forecast_huber=errors,**curves)
            hashes[name] = sha(f)
            if rank%60==0 or rank==21:
                print(dict(stage='seal_radius_feedback',rank=rank,seconds=round(time.monotonic()-started,2)),
                      flush=True)
    if [state_digest(a) for a in args[:2]]!=frozen:
        raise ValueError('immutable source or transition model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
                              seconds=time.monotonic()-started))


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    ref = Path(p['reference'])
    if (sha(ref/'seal.json')!=p['reference_seal'] or sha(out/'executed_probe.py')!=p['producer_sha256']
            or len(s['curve_sha256'])!=360):
        raise ValueError('sealed experiment identity changed')
    old = read(ref/'per_shot.json')
    scores = {b:{k:[] for k in ARMS} for b in p['calibration']}
    activity = dict(updates=0,nonzero_grad_windows=0,nonzero_input_endpoints=0,
                    before_deadline_updates=0,before_deadline_late_or_miss_shots=0)
    error_sum = {k:np.zeros(3) for k in ('all','eligible_positive','late_or_miss')}
    error_count = dict.fromkeys(error_sum,0)
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path)!=s['curve_sha256'][name]:
            raise ValueError('sealed curve changed')
        d,down,t = hdf._raw_truth(sid)
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            curves = {k:f[k] for k in ARMS}
            q,rms,grad,e = (f[k] for k in ('release','input_rms','grad_norm','forecast_huber'))
        static = old['0.2'][clock.STATIC][rank-1]
        if len(e)!=len(q) or not np.isin(q, np.arange(lag.FIRST_EFFECT+lag.STRIDE,len(t),lag.STRIDE)).all():
            raise ValueError('invalid complete feedback time')
        if rank<=p['startup'] and len(q):
            raise ValueError('prepared shots do not adapt')
        activity['updates'] += len(q)
        activity['nonzero_grad_windows'] += int((grad>1e-8).all(1).sum())
        activity['nonzero_input_endpoints'] += int((rms>0.).sum())
        if np.any(rms[:q[0] if len(q) else len(t)]>0.):
            raise ValueError('adaptation before first complete reference and target')
        for k in (TRUE,WRONG):
            np.testing.assert_array_equal(curves[k][rms==0.],curves[clock.STATIC][rms==0.])
        lead = np.round((down-t[q])*1000.,9) if d else np.zeros(len(q))
        early = (lead>30.) & (lead<=125.)
        activity['before_deadline_updates'] += int(early.sum())
        activity['before_deadline_late_or_miss_shots'] += int(d and not static['timely'] and early.any())
        for key,selector in (('all',np.ones(len(q),bool)),('eligible_positive',early),
                             ('late_or_miss',early & (not static['timely']))):
            error_sum[key] += e[selector].sum(0,dtype=float)
            error_count[key] += int(selector.sum())
        for band,tau in p['calibration'].items():
            tau = p['original'] if rank<=p['startup'] else tau
            for k,a in curves.items():
                row = dict(shot=sid,disruptive=d,down_s=down,**score_curve(t,a,d,down,tau))
                if k in OLD and row!=old[band][k][rank-1]:
                    raise ValueError('archived first warning or outcome differs')
                scores[band][k].append(row)
    write(out/'per_shot.json',scores)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {b:{k:timing_decomposition(v) for k,v in arms.items()} for b,arms in scores.items()}
    paired = {b:{f'{a}__to__{c}':flips(arms[a],arms[c]) for a,c in PAIRS} for b,arms in scores.items()}
    auxiliary = {k:dict(windows=n,means=dict(zip(DIAGNOSTIC,(error_sum[k]/n).tolist(),strict=True))
                        if n else {}) for k,n in error_count.items()}
    write(out/'result.json',dict(status='COMPLETED_EAST360_RADIUS_FEEDBACK_DEVELOPMENT',
        metrics=metrics,paired=paired,activity=activity,auxiliary=auxiliary,
        raw_verified_scores=len(p['order'])*len(ARMS)*len(scores)))
    print({b:{k:(v['tp'],v['fp']) for k,v in arms.items()} for b,arms in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--reference',type=Path,default=clock.REFERENCE)
    a = parser.parse_args()
    predict(a.reference.resolve(),a.out) if a.stage=='predict' else score(a.out)
