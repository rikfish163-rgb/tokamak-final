"""Combine phase and amplitude feedback at the original realized input dose."""

import argparse
import math
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_lag_clock_20261111 as clock
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import commissioning_radius_feedback_20261111 as amplitude
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


TRUE, WRONG = 'complementary_true','complementary_misassigned'
OLD = clock.ARMS[:4]
ARMS = (*OLD,TRUE,WRONG)
PAIRS = ((clock.STATIC,TRUE),(clock.OLD,TRUE),(clock.OLD,WRONG),(WRONG,TRUE))
RATIO, TOL = .5, 3e-7


def equal_angle(old, real, wrong):
    """Use only the new component orthogonal to the original input change."""
    a = old.flatten(1)
    n = a.norm(dim=1,keepdim=True)
    u = a/n.clamp_min(1e-7)
    ok = n>1e-7
    directions = []
    for candidate in (real,wrong):
        b = candidate.flatten(1)
        tangent = b-(u*b).sum(1,keepdim=True)*u
        tn = tangent.norm(dim=1,keepdim=True)
        # Do not magnify a nearly collinear round-off remainder into a full rotation.
        ok = ok & (tn>(1e-3*b.norm(dim=1,keepdim=True)).clamp_min(1e-7))
        directions.append(tangent/tn.clamp_min(1e-7))
    results = [torch.where(ok,(a+RATIO*n*v)/math.sqrt(1.+RATIO**2),a).reshape_as(old)
               for v in directions]
    return *results,ok.flatten()


def common_slots(phase, amplitude_q, endpoints):
    """The companion step must be from the very same completed forecast."""
    s,_ = lag.delayed_slots(phase,endpoints)
    a,_ = lag.delayed_slots(amplitude_q,endpoints)
    use = (s>0) & (a>0) & (np.r_[-1,phase][s]==np.r_[-1,amplitude_q][a])
    return s,np.where(use,a,0)


def _forward(base,x,m,phase,real,wrong):
    hidden,trace = [],[]

    def transform(_layer, inputs):
        raw = inputs[0][:len(x)]
        anchor = lag.patch_input(raw,phase)
        a,b,used = equal_angle(anchor-raw,lag.patch_input(raw,real)-raw,
                                lag.patch_input(raw,wrong)-raw)
        where = used.reshape((-1,)+(1,)*(raw.ndim-1))
        paths = (anchor,torch.where(where,raw+a,anchor),torch.where(where,raw+b,anchor))
        dose = torch.stack([(v-raw).flatten(1).square().mean(1).sqrt() for v in paths])
        trace.extend((used,(dose[1:]-dose[0]).abs().amax(dim=0)))
        return (torch.cat(paths),)

    t = base.embedding.register_forward_pre_hook(transform)
    v = base.risk_head.register_forward_pre_hook(lambda _m,p: hidden.append(p[0]))
    try:
        base(torch.cat((x,x,x)),torch.cat((m,m,m)))
    finally:
        v.remove()
        t.remove()
    return hidden[0].chunk(3),*trace


@torch.no_grad()
def _replay(base, signal, h, z, beta, q, phase, aq, companion):
    n = len(z)
    baseline = z.sigmoid()
    scores = baseline.expand(3,-1).clone()
    changed = torch.zeros(n,dtype=torch.bool,device=z.device)
    error = z.new_zeros(n)
    s,a = common_slots(q,aq,np.arange(n))
    active = np.flatnonzero(phase.any((1,2)).cpu().numpy()[s])
    for start in range(0,len(active),eos.BATCH):
        i = active[start:start+eos.BATCH]
        x,m = lag._windows(base,signal,i)
        values,used,de = _forward(base,x,m,phase[s[i]],*companion[:,a[i]])
        before,one,two = ((z[i]+((v-h[i])*beta).sum(-1)).sigmoid() for v in values)
        scores[:,i] = torch.stack((before,torch.where(used,one,before),torch.where(used,two,before)))
        changed[i],error[i] = used,de
    if float(error.max())>TOL:
        raise ValueError('realized input dose changed across directions')
    return baseline.cpu().numpy(),scores.cpu().numpy(),changed.cpu().numpy(),float(error.max())


def _predict(signal, args, decoder, params):
    base = args[0]
    tr,h,z,beta = guard._track(signal,args)
    q,key,prior,target = teacher._mature(tr,args[2][2],args[2][3])
    real,_ = guard._bases(base,signal,q,prior,target,decoder)
    direction = guard.signed_direction(*real,torch.as_tensor(key,device=beta.device),params)
    phase = torch.cat((direction.new_zeros((1,len(lag.CHANNELS),len(lag.CHANNELS))),
                       -lag.RADIUS*direction),dim=0)
    aq,fields = amplitude._pack(signal,tr,args[2])
    fields = tuple(torch.from_numpy(a).to(beta.device) for a in fields)
    noise = torch.as_tensor(args[2][3],device=beta.device,dtype=beta.dtype)
    new,_ = amplitude._estimate(base,signal,aq,fields,noise,decoder)
    if not np.isin(aq,q).all():
        raise ValueError('amplitude forecast not on the common release schedule')
    old,curves,used,error = _replay(base,signal,torch.from_numpy(h).to(beta.device),
                                    torch.from_numpy(z).to(beta.device),beta,q,phase,aq,new)
    return old,curves,used,error,q+lag.H+1,aq+lag.H+1


def predict(reference, out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p,s,previous,args,decoder = eos._load(reference,lock)
    original = read(reference/'seal.json')['curve_sha256']
    if sha(reference/'teacher.json')!=p['teacher_sha256']:
        raise ValueError('archived direction teacher changed')
    params = torch.tensor(read(reference/'teacher.json')['path'],device='cuda',dtype=torch.float32)
    frozen = [state_digest(a) for a in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_EQUAL_DOSE_COMPLEMENTARY_FEEDBACK_DEVELOPMENT',
        producer_sha256=sha(Path(__file__)),reference=str(reference),
        reference_seal=sha(reference/'seal.json'),order=lock['ids'],startup=p['startup'],
        source_sha256=p['source_sha256'],original=p['original'],calibration=p['calibration'],
        arms=ARMS,angle_degrees=math.degrees(math.atan(RATIO)),ratio=RATIO,
        clock='same phase and amplitude forecast origin, both mature at origin+26; '
              'zero amplitude reference available at first origin, unchanged on unsupported endpoints',
        intervention='one effective lag matrix: realized phase displacement plus a normalized '
                     'orthogonal amplitude component, renormalized to the original phase dose',
        control='same fixed angle, actual RMS and support; signed amplitude errors reassigned across groups',
        forecast='H25 once at issue; both loss signals computed on arrived values; no outcomes or retraining'))
    hashes,witnesses = {},{}
    errors,phase_diffs = [],[]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome unavailable')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            prepared = rank<=p['startup']
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('locked signal changed')
            t,curves,q = clock.read_archived_curves(reference,previous,name,original,
                                                   s['curve_sha256'],prepared)
            if prepared:
                curves.update(dict.fromkeys((TRUE,WRONG),curves[clock.OLD]))
                used,aq,error,delta = np.zeros(len(t),bool),q,0.,0.
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                baseline,new,used,error,r,aq = _predict(signal,args,decoder,params)
                np.testing.assert_array_equal(baseline,curves[clock.INHERITED])
                np.testing.assert_array_equal(r,q)
                np.testing.assert_allclose(new[0],curves[clock.OLD],atol=3e-6,rtol=0.)
                for tau in p['calibration'].values():
                    np.testing.assert_array_equal(new[0]>=tau,curves[clock.OLD]>=tau)
                delta = float(np.max(np.abs(new[0]-curves[clock.OLD])))
                # Numerical round-off in a three-arm forward is not a treatment.
                curves.update({k:np.where(used,v,curves[clock.OLD])
                               for k,v in zip((TRUE,WRONG),new[1:],strict=True)})
                if rank in (21,120,360):
                    end = min(len(t),lag.FIRST_EFFECT+3*lag.STRIDE+5)
                    prefix = {k:v[:end] for k,v in signal.items() if k in ('values','available')}
                    _,short,m,_,r0,a0 = _predict(prefix,args,decoder,params)
                    np.testing.assert_array_equal(m,used[:end])
                    np.testing.assert_allclose(short,new[:,:end],atol=3e-6,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(short>=tau,new[:,:end]>=tau)
                    np.testing.assert_array_equal(r0,q[q<end])
                    np.testing.assert_array_equal(a0,aq[aq<end])
                    witnesses[str(sid)] = end
            if not all(np.isfinite(x).all() for x in curves.values()):
                raise FloatingPointError('nonfinite warning curve')
            f = out/'shots'/name
            with f.open('xb') as h:
                np.savez_compressed(h,times=t,phase_release=q,amp_release=aq,rotated=used,**curves)
            hashes[name] = sha(f)
            errors.append(error)
            phase_diffs.append(delta)
            if rank%60==0 or rank==21:
                print(dict(stage='seal_complementary_feedback',rank=rank,
                           seconds=round(time.monotonic()-started,2)),flush=True)
    if [state_digest(a) for a in args[:2]]!=frozen:
        raise ValueError('immutable source or transition model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
        seconds=time.monotonic()-started,max_dose_rms_error=max(errors),max_phase_curve_error=max(phase_diffs)))


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    ref = Path(p['reference'])
    if (sha(ref/'seal.json')!=p['reference_seal'] or sha(out/'executed_probe.py')!=p['producer_sha256']
            or len(s['curve_sha256'])!=360 or s['max_dose_rms_error']>TOL):
        raise ValueError('sealed experiment identity changed')
    old = read(ref/'per_shot.json')
    scores = {b:{k:[] for k in ARMS} for b in p['calibration']}
    totals = dict(phase_windows=0,amplitude_windows=0,rotated_endpoints=0,
                  rotated_shots=0,old_late_or_miss_with_timely_rotation=0)
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        path = out/'shots'/name
        if sha(path)!=s['curve_sha256'][name]:
            raise ValueError('sealed curve changed')
        d,down,t = hdf._raw_truth(sid)
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            q,aq,used = (f[k] for k in ('phase_release','amp_release','rotated'))
            curves = {k:f[k] for k in ARMS}
        if not np.isin(aq,q).all() or not np.isin(q,np.arange(lag.FIRST_EFFECT,len(t),lag.STRIDE)).all():
            raise ValueError('invalid complete feedback time')
        _,legal = common_slots(q-lag.H-1,aq-lag.H-1,np.arange(len(t)))
        if np.any(used & (legal==0)) or (rank<=p['startup'] and (len(q) or used.any())):
            raise ValueError('adapted outside common matured support')
        for k in (TRUE,WRONG):
            np.testing.assert_array_equal(curves[k][~used],curves[clock.OLD][~used])
        totals['phase_windows'] += len(q)
        totals['amplitude_windows'] += len(aq)
        totals['rotated_endpoints'] += int(used.sum())
        totals['rotated_shots'] += int(used.any())
        if d and rank>p['startup'] and not old['0.2'][clock.OLD][rank-1]['timely']:
            lead = np.round((down-t)*1000.,9)
            totals['old_late_or_miss_with_timely_rotation'] += int(np.any(used & (lead>30.) & (lead<=125.)))
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
    write(out/'result.json',dict(status='COMPLETED_EAST360_EQUAL_DOSE_COMPLEMENTARY_FEEDBACK',
        metrics=metrics,paired=paired,activity=totals,max_dose_rms_error=s['max_dose_rms_error'],
        raw_verified_scores=len(p['order'])*len(ARMS)*len(scores)))
    print({b:{k:(v['tp'],v['fp']) for k,v in arms.items()} for b,arms in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    parser.add_argument('--reference',type=Path,default=clock.REFERENCE)
    a = parser.parse_args()
    predict(a.reference.resolve(),a.out) if a.stage=='predict' else score(a.out)
