"""Isolate additive observation bias from a causal, task-shaped feed-forward update."""

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
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


OLD = clock.ARMS[:4]
BIAS, REAL, WRONG, JOINT, JOINT_WRONG = NEW = (
    'bias_only', 'centered_dynamics_real', 'centered_dynamics_shifted',
    'bias_dynamics_real', 'bias_dynamics_shifted',
)
ARMS = (*OLD, *NEW)
PAIRS = ((clock.STATIC,REAL),(clock.OLD,BIAS),(clock.OLD,REAL),(WRONG,REAL),
         (clock.OLD,JOINT),(BIAS,JOINT),(JOINT_WRONG,JOINT),(REAL,JOINT))
BIN, HISTORY, FIT_STEPS = 5, 8, 96
TOL = 3e-7
CAP_SOURCE = io23.ROOT/'probe_outputs/commissioning_segment_teacher_east360_20261111_v2/cold_dose.json'


def dynamic(x):
    bins = x.reshape(len(x),lag.H//BIN,BIN,x.shape[-1]).mean(2)
    return bins-bins.mean(1,keepdim=True)


def shifted(d):
    return torch.stack([d[...,i].roll(g+1,dims=1) for i,g in enumerate(lag.GROUP.tolist())],-1)


def past_bias(q, mean_error):
    """Use earlier mature blocks in the same shot, excluding the current block."""
    i = np.arange(len(q))
    lo = np.searchsorted(q,q-HISTORY*lag.STRIDE)
    err = np.clip(mean_error,-.5,.5)
    s = np.concatenate((np.zeros((1,err.shape[1]),dtype=err.dtype),np.cumsum(err,axis=0)))
    support = i-lo>=2
    estimate = (s[i]-s[lo])/(i-lo+1)[:,None]
    return np.where(support[:,None],estimate,0.).astype(err.dtype),support


def bias_patches(patches, bias, cap):
    x = patches.reshape(len(patches),lag.W,46)
    cols = list(lag.CHANNELS)
    delta = -bias[:,None].expand(-1,lag.W,-1)
    delta = delta.masked_fill(~x[..., [c+23 for c in cols]].bool(),0.)
    # The cap is the actual RMS of the inherited phase correction on this patch.
    norm = torch.sqrt(delta.square().sum((1,2))/(lag.W*46))
    delta = delta*(cap/norm.clamp_min(1e-12)).clamp_max(1.)[:,None,None]
    out = x.clone()
    out[...,cols] = x[...,cols]+delta
    return out.reshape_as(patches)


def ff_input_change(a, writable, gain):
    v = a.masked_fill(~writable[...,None],0.)
    center = v.sum(1,keepdim=True)/writable.sum(1,keepdim=True).clamp_min(1)[...,None]
    return (a-center).masked_fill(~writable[...,None],0.)*gain[:,None]


def common_dose(d, cap):
    norm = (d.square().mean((2,3))+1e-24).sqrt()
    common = norm.amin(0).clamp_max(cap)
    common = torch.where(common>1e-10,common,0.)
    return d*(common/norm)[...,None,None],common


def forward_hidden(base, x, m, *, gain, patcher=None, cap=None, trace=None):
    """Rescale centered intermediate activations within the first feed-forward layer."""
    valid = (m & torch.isfinite(x)).reshape(len(x),-1,base.cfg.patch*23).any(2)
    writable = F.pad(valid,(0,1))
    writable[:,-2:] = False  # Keep the latest patch and the special token unchanged at this site.
    hidden = []

    def transform(layer, args, out):
        d = F.linear(ff_input_change(args[0],writable,gain),layer.weight)
        if cap is None:
            return out+d
        by_arm = d.reshape(len(NEW)+1,-1,*d.shape[1:])
        matched, target = common_dose(by_arm[2:],cap)
        out_new = out+torch.cat((by_arm[:2],matched)).reshape_as(out)
        actual = (out_new-out).reshape_as(by_arm)[2:].square().mean((2,3)).sqrt()
        trace.extend((target,(actual-target).abs().amax()))
        return out_new

    handles = [base.encoder.layers[0].linear2.register_forward_hook(transform),
               base.risk_head.register_forward_pre_hook(lambda _l,a: hidden.append(a[0]))]
    if patcher is not None:
        handles.append(base.embedding.register_forward_pre_hook(lambda _l,a:(patcher(a[0]),)))
    try:
        base(x,m)
    finally:
        for handle in handles:
            handle.remove()
    return hidden[0]


def width(base):
    return base.encoder.layers[0].linear2.in_features


def feedback(base, signal, q, prior, target, decoder):
    """Differentiate a complete, time-centered diagnostic forecast at its sealed origin."""
    estimates, directions = [], []
    for start in range(0,len(q),128):
        part = slice(start,start+128)
        x,m = lag._windows(base,signal,q[part])
        gain = x.new_zeros(len(x),width(base)).requires_grad_()
        p,y = (torch.as_tensor(a[part],device=x.device) for a in (prior,target))
        forecast = lag._predict(forward_hidden(base,x,m,gain=gain),p,decoder)
        e = y-forecast.detach()
        estimates.append(e.mean(1))
        motion = dynamic(forecast)
        actual = dynamic(e)
        gs = [torch.autograd.grad(F.smooth_l1_loss(motion-motion.detach(),z,beta=.2,reduction='sum'),
                                  gain,retain_graph=i==0)[0]
              for i,z in enumerate((actual,shifted(actual)))]
        g = torch.stack(gs)
        ok = (g.norm(dim=-1)>1e-8).all(0)
        directions.append(torch.where(ok[None,:,None],-F.normalize(g,dim=-1),0.))
    if not len(q):
        return base.center.new_empty(0,len(lag.CHANNELS)),base.center.new_empty(2,0,width(base))
    return torch.cat(estimates),torch.cat(directions,1)


def modulate(direction, param):
    return F.normalize(direction*(1.+.5*param.tanh()),dim=-1,eps=1e-8)


def historical(reference, gp, args, decoder, lock):
    """Fit a bounded diagonal metric on earlier paired risk sensitivities, never on the suffix."""
    path = Path(gp['previous_teacher'])/'teacher.json'
    if sha(path)!=gp['previous_pairs_sha256']:
        raise ValueError('the historical training pairs changed')
    archive = read(path)['pairs']
    base,net,(center,scale,prior_model,noise,_),*_=args
    allowed = {k:set(v) for k,v in io23.source_roles().items()}
    allowed['cold'] = set(lock['ids'][:lag.STARTUP])
    advanced = io23.advanced_times()
    per_role = {}
    for role in ('train','cold','val'):
        device = 'EAST' if role=='cold' else 'J-TEXT'
        source = device=='J-TEXT'
        beta = base.risk_head.weight.detach().flatten() if source else args[6][:base.cfg.dim]
        curves, slopes, manifest = [], [], []
        for pair in archive[role]:
            for j, r in enumerate(pair):
                sid, q, t = (int(r[k]) for k in ('shot','origin','action'))
                if sid not in allowed[role] or r['role']!=role or t<q+lag.H+1:
                    raise ValueError('invalid supervised window identity or timing')
                if not source and sha(io23.path_for(device,sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('cold-shot signal changed')
                raw = simple_data.read_shot(device,sid,rate_contract='strict' if source else 'nominal')
                true = simple_data.attach_outcome(raw,device,advanced)
                if not true['valid'][t] or true['is_disrupt']!=(j==0):
                    raise ValueError('historical supervision must agree with raw outcome')
                if j==0 and (not true['y'][t] or
                        round(float((true['down']-raw['times'][t])*1000),9)<=(10. if source else 30.)):
                    raise ValueError('historical positive outside timely region')
                sig = raw if source else lag.prepare_actual(raw,lag.STARTUP+1,lock['mapping'])
                tr = lag.direct.track(net,sig,center,scale,np.empty(0,np.int64))
                k = np.flatnonzero(tr['q']==q)
                if len(k)!=1 or not tr['observed'][k].all():
                    raise ValueError('historical forecast does not have complete support')
                _,g = feedback(base,sig,np.array([q]),lag.direct.base_forecast(tr,prior_model)[k],
                               lag.direct.normalized_target(tr,noise)[k],decoder)
                x,m = lag._windows(base,sig,[t])
                gain = x.new_zeros(1,width(base)).requires_grad_()
                h = forward_hidden(base,x,m,gain=gain)
                slope, = torch.autograd.grad((h*beta).sum(),gain)
                curves.append(g[:,0])
                slopes.append(F.normalize(slope,dim=-1)[0])
                manifest.append(dict(shot=sid,origin=q,action=t,positive=j==0))
        per_role[role] = (torch.stack(curves,1),torch.stack(slopes),manifest)
    g,a = (torch.cat([per_role[r][i] for r in ('train','cold')],dim=1 if i==0 else 0)
           for i in (0,1))
    param = torch.nn.Parameter(a.new_zeros(a.shape[1]))
    opt = torch.optim.Adam([param],lr=.025)
    for _ in range(FIT_STEPS):
        opt.zero_grad(set_to_none=True)
        change = (modulate(g[0],param)*a).sum(1).reshape(-1,2)
        up,down = change.T
        loss = (F.softplus(-up/.10)+F.softplus(down/.10)+.5*F.softplus((down-up)/.20)).mean()
        (loss+.025*param.square().mean()).backward()
        opt.step()
    if not torch.isfinite(param).all():
        raise FloatingPointError('nonfinite historical orientation metric')
    audit = {}
    with torch.no_grad():
        for role,(d,slope,manifest) in per_role.items():
            scores = {}
            for label,v in (('uniform_real',d[0]),('trained_real',modulate(d[0],param)),
                            ('trained_shifted',modulate(d[1],param))):
                up,down = (v*slope).sum(1).reshape(-1,2).T
                scores[label] = dict(both_correct=int(((up>0.)&(down<0.)).sum()),
                                     positive_higher=int((up>down).sum()),
                                     mean_positive=float(up.mean()),mean_safe=float(down.mean()))
            audit[role] = dict(pairs=len(d[0])//2,scores=scores,manifest=manifest)
    return param.detach(), dict(weights=param.detach().tolist(),steps=FIT_STEPS,
                                source_pairs_sha256=sha(path),orientation=audit)


@torch.no_grad()
def warning_paths(base,x,m,phase,bias,gain,cap):
    n = len(x)
    trace, measured_bias = [], []

    def patches(raw):
        raw = raw[:n]
        anchor = lag.patch_input(raw,phase)
        limit = (anchor-raw).flatten(1).square().mean(1).sqrt()
        corrected = bias_patches(anchor,bias,limit)
        measured_bias.append((corrected-anchor).flatten(1).square().mean(1).sqrt())
        return torch.cat((anchor,corrected,anchor,anchor,corrected,corrected))

    real,wrong = gain
    zeros = torch.zeros_like(real)
    steps = torch.cat((zeros,zeros,real,wrong,real,wrong))
    h = forward_hidden(base,x.repeat(6,1,1),m.repeat(6,1,1),gain=steps,
                       patcher=patches,cap=cap,trace=trace)
    return h.reshape(6,n,-1), trace[0], float(trace[1]), measured_bias[0]


def predict_shot(signal,args,decoder,path_metric,orientation,cap,archived):
    base = args[0]
    tr,h,z,beta = guard._track(signal,args)
    q,key,prior,target = teacher._mature(tr,args[2][2],args[2][3])
    real,_ = guard._bases(base,signal,q,prior,target,decoder)
    phase = -lag.RADIUS*guard.signed_direction(*real,torch.as_tensor(key,device=beta.device),path_metric)
    e,g = feedback(base,signal,q,prior,target,decoder)
    corrected,support = past_bias(q,e.cpu().numpy())
    b = torch.from_numpy(corrected).to(beta.device)*torch.as_tensor(args[2][3],device=beta.device)
    g = modulate(g,orientation)
    supported = np.r_[False,support]
    slots,_ = lag.delayed_slots(q,np.arange(len(z)))
    active = np.flatnonzero(supported[slots])
    scores = np.broadcast_to(archived,(len(NEW),len(z))).copy()
    h,z = (torch.as_tensor(a,device=beta.device) for a in (h,z))
    bias_dose = np.zeros(len(z),np.float32)
    neural_dose = np.zeros(len(z),np.float32)
    dose_err,curve_err = 0.,0.
    with torch.no_grad():
        for start in range(0,len(active),eos.BATCH):
            i = active[start:start+eos.BATCH]
            k = slots[i]-1
            x,m = lag._windows(base,signal,i)
            y,d,e0,b0 = warning_paths(base,x,m,phase[k],b[k],g[:,k],cap)
            p = (z[i]+((y-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
            delta = float(np.max(np.abs(p[0]-archived[i])))
            if delta>3e-6 or e0>TOL:
                raise ValueError('frozen parent replay or equal actual feed-forward dose mismatch')
            b0,d = b0.cpu().numpy(),d.cpu().numpy()
            scores[:,i] = np.where(np.stack((b0>0.,d>0.,d>0.,(b0+d)>0.,(b0+d)>0.)),
                                    p[1:],archived[i][None])
            bias_dose[i],neural_dose[i] = b0,d
            dose_err,curve_err = max(dose_err,e0),max(curve_err,delta)
    return scores,q+lag.H+1,support,bias_dose,neural_dose,dose_err,curve_err


def run(reference,out):
    lag.configure()
    started = time.monotonic()
    lock = lag.load_inputs()
    p,s,previous,args,decoder = eos._load(reference,lock)
    originals = read(reference/'seal.json')['curve_sha256']
    if sha(reference/'teacher.json')!=p['teacher_sha256']:
        raise ValueError('frozen signed parent direction changed')
    path_metric = torch.tensor(read(reference/'teacher.json')['path'],device='cuda',dtype=torch.float32)
    cap = read(CAP_SOURCE)['cap']
    frozen = [state_digest(a) for a in args[:2]]
    orientation,audit = historical(reference,p,args,decoder,lock)
    print(dict(stage='historical_direction',held=audit['orientation']['val']['scores']),flush=True)
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'historical.json',audit)
    write(out/'protocol.json',dict(scope='EAST360_CAUSAL_BIAS_VS_TASK_SHAPED_DYNAMICS',
        order=lock['ids'],startup=p['startup'],source_sha256=p['source_sha256'],
        reference=str(reference),reference_seal=sha(reference/'seal.json'),
        executed_sha256=sha(Path(__file__)),original=p['original'],calibration=p['calibration'],
        arms=ARMS,history=HISTORY,minimum_previous=2,ff_cap=cap,cap_source=str(CAP_SOURCE),
        cap_source_sha256=sha(CAP_SOURCE),history_sha256=sha(out/'historical.json'),
        intervention='constant normalized observation correction or temporary first FF modulation or both',
        supervised='only existing source train pairs and cold EAST pairs; source VAL shadow only',
        control='group-specific cyclic shift of each measured centered error, same actual FF dose',
        clock='same H25 plus one sample release; nuisance fit uses only two or more earlier slots'))
    hashes,witnesses,errors,curve = {},{},[],[]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome unavailable before sealing')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            prepared = rank<=p['startup']
            if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('locked signal changed')
            t,old,release = clock.read_archived_curves(reference,previous,name,originals,
                                                       s['curve_sha256'],prepared)
            v = old[clock.OLD]
            if prepared:
                new = np.broadcast_to(v,(len(NEW),len(v)))
                support = np.zeros(0,bool)
                bd,nd,e,d = np.zeros(len(t),np.float32),np.zeros(len(t),np.float32),0.,0.
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                new,q,support,bd,nd,e,d = predict_shot(sig,args,decoder,path_metric,orientation,cap,v)
                np.testing.assert_array_equal(q,release)
                if rank in (21,120,360):
                    end = min(len(t),330)
                    sig0 = {k:a[:end] for k,a in sig.items() if k in ('values','available')}
                    a,q0,w,b0,n0,_,_ = predict_shot(sig0,args,decoder,path_metric,orientation,cap,v[:end])
                    np.testing.assert_array_equal(q0,q[q<end])
                    np.testing.assert_array_equal(w,support[q<end])
                    np.testing.assert_allclose(a,new[:,:end],atol=3e-6,rtol=0.)
                    np.testing.assert_allclose(np.stack((b0,n0)),np.stack((bd[:end],nd[:end])),atol=3e-7,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(a>=tau,new[:,:end]>=tau)
                    witnesses[str(sid)] = end
            values = old | dict(zip(NEW,new,strict=True))
            if not all(np.isfinite(x).all() for x in values.values()):
                raise FloatingPointError('nonfinite warning curve')
            f = out/'shots'/name
            with f.open('xb') as handle:
                np.savez_compressed(handle,times=t,release=release,eligible=support,
                                    bias_rms=bd,ff_rms=nd,**values)
            hashes[name] = sha(f)
            errors.append(e)
            curve.append(d)
            if rank%60==0 or rank==21:
                print(dict(stage='seal_bias_dynamics',rank=rank,
                           seconds=round(time.monotonic()-started,2)),flush=True)
    if frozen!=[state_digest(a) for a in args[:2]]:
        raise ValueError('source weights or fixed transition model changed')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
                              seconds=time.monotonic()-started,max_ff_rms_error=max(errors),
                              max_parent_curve_error=max(curve)))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    ref = Path(p['reference'])
    if (p['reference_seal']!=sha(ref/'seal.json') or len(s['curve_sha256'])!=360
            or sha(out/'executed_probe.py')!=p['executed_sha256'] or s['max_ff_rms_error']>TOL):
        raise ValueError('incomplete or changed prediction seal')
    old = read(ref/'per_shot.json')
    records = {k:{m:[] for m in ARMS} for k in p['calibration']}
    counts = dict(complete_windows=0,eligible_windows=0,neural_endpoints=0,bias_endpoints=0,
                  late_parent_with_neural_opportunity=0)
    for rank,sid in enumerate(p['order'],1):
        f = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(f)!=s['curve_sha256'][f.name]:
            raise ValueError('changed sealed warning curves')
        d,down,t = hdf._raw_truth(sid)
        with np.load(f,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['times'],t)
            q,eligible,b,n = (a[k] for k in ('release','eligible','bias_rms','ff_rms'))
            values = {k:a[k] for k in ARMS}
        if not np.isin(q,np.arange(lag.FIRST_EFFECT,len(t),lag.STRIDE)).all():
            raise ValueError('invalid feedback release')
        keep = np.r_[False,eligible][np.searchsorted(q,np.arange(len(t)),side='right')]
        if np.any((b+n>0.)&~keep) or (rank<=p['startup'] and keep.any()):
            raise ValueError('unsupported early update')
        for k,v in zip(NEW,(b,n,n,b+n,b+n),strict=True):
            np.testing.assert_array_equal(values[k][v==0.],values[clock.OLD][v==0.])
        if d and not old['0.2'][clock.OLD][rank-1]['timely']:
            lead = np.round((down-t)*1000.,9)
            counts['late_parent_with_neural_opportunity'] += int(np.any((lead>30.)&(lead<=125.)&(n>0.)))
        for k in counts.keys()-{'late_parent_with_neural_opportunity'}:
            counts[k] += dict(complete_windows=len(q),eligible_windows=int(eligible.sum()),
                neural_endpoints=int((n>0.).sum()),bias_endpoints=int((b>0.).sum()))[k]
        for band,tau in p['calibration'].items():
            for k,value in values.items():
                row = dict(shot=sid,disruptive=d,down_s=down,**score_curve(
                    t,value,d,down,p['original'] if rank<=p['startup'] else tau))
                if k in OLD and row!=old[band][k][rank-1]:
                    raise ValueError('archived reference does not reproduce under raw outcomes')
                records[band][k].append(row)
    write(out/'per_shot.json',records)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {k:{m:timing_decomposition(r) for m,r in a.items()} for k,a in records.items()}
    paired = {k:{f'{a}__to__{b}':flips(v[a],v[b]) for a,b in PAIRS} for k,v in records.items()}
    write(out/'result.json',dict(status='COMPLETED_CAUSAL_BIAS_VS_TASK_SHAPED_DYNAMICS',
        metrics=metrics,paired=paired,activity=counts,raw_verified_scores=len(p['order'])*len(ARMS)*len(metrics)))
    print({k:{m:(v['tp'],v['fp']) for m,v in a.items()} for k,a in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    a = parser.parse_args()
    run(eos.REFERENCE,a.out) if a.stage=='predict' else score(a.out)
