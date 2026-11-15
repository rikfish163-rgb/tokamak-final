"""Learn which mature diagnostic discrepancies should change the first-block representation."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_trajectory_return_20261114 as archive
from . import io23, simple_data, warning_lead_metrics as lead
from .commissioning_transition_data_20261110 import _window_mean
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_missing_diagnostics_20261103 import diagnostic_view


group, lag, layer = archive.group, archive.lag, archive.precision.layer
WAIT, LOCK = archive.WAIT, archive.LOCK
BASE = ('fixed', 'bias_only', 'current_5', 'earlier_5')
NEW = ('uniform_current', 'learned_current', 'learned_shifted', 'learned_shifted_fit')
ARMS = (*BASE, *NEW)
CAP_SAMPLES = 12
EPOCHS, RATE = 20, .025
SHIFT = 2


def active_windows(q, eligible, length, patch_size):
    """Issue, observe all targets, wait, then ramp. No state bridges a missing block."""
    release = q+lag.H+1+WAIT
    t = np.arange(length)
    slot = np.searchsorted(release, t, side='right')
    ramp = group.handoff_fraction(np.r_[0, release][slot], t, patch_size)
    active = np.r_[False, eligible][slot] & (ramp>0.)
    return active, slot, ramp


def teaching_window(times, valid, y, down, active):
    if down is None:
        return valid & active
    return valid & active & y & (np.round((down-times)*1000.,9)>10.)


def departure(wave, *, scale):
    return (wave.unflatten(-2, (-1, group.BIN)).mean(-2)/scale).asinh()


def dictionary(force, channel_groups):
    """One level and one ordered residual direction per diagnostic family.

    Subtracting the temporal DC from the motion force prevents an observed
    offset from being reapplied as evidence about ordering.
    """
    dc = force.mean(dim=-2, keepdim=True)
    parts = torch.stack((dc.expand_as(force), force-dc), dim=1)
    return (parts[:, None]*channel_groups.T[None, :, None, None, :]).flatten(1,2)


def transport(new, old, fraction, previous):
    """Blend directions only across adjacent eligible blocks, not across a gap."""
    fraction = torch.where(previous, fraction, 1.)[None, :, None, None]
    return (1.-fraction)*old+fraction*new


def prior_fields(signal, q, parameters, device):
    """Reference is the available 16-sample issue-time mean, never the completed target."""
    center, scale, _, noise, _ = parameters
    v, m, _, _ = lag.direct.weak.shot_coordinates(signal, center, scale)
    recent, count = _window_mean(v, m, q, -15, 16)
    target = v[q[:, None]+np.arange(1, lag.H+1)] - recent[:, None]
    enough = (count[:, group.USE]>=8).all(1)
    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)
    return enough, tensor(target[..., group.USE]), tensor(noise[group.USE])


def forces(real, pred, groups):
    error = ((real-pred.detach())/group.BETA).clamp(-1., 1.)
    # Jointly shift every channel; preserve diagnostic identities, the same
    # numerical residuals, and common shot/time support.
    return torch.stack((dictionary(error,groups),dictionary(error.roll(SHIFT,dims=1),groups)))


def blend(basis, coeff):
    return torch.einsum('...be,...b->...e', basis, coeff)


def rates(context, w):
    return 1.+torch.tanh(context@w)


def flow(signal, q, prior, phase, correction, fields, decoder, base, selected):
    """Actual two-Jacobian backprop at issue and at first complete feedback, never scoring time."""
    issue = q[selected]
    slot = lag.delayed_slots(q, issue)[0]
    x, m = lag._windows(base, signal, issue)
    state_shape = (len(issue), base.cfg.window//base.cfg.patch-1, len(group.TRUSTED))
    issued = group.forward_hidden(base, x, m, layer.patcher(phase[slot], correction[slot]),
                                  state=x.new_zeros(state_shape, requires_grad=True))
    xr, mr = lag._windows(base, signal, issue+lag.H+1)
    state = x.new_zeros(state_shape, requires_grad=True)
    current = group.forward_hidden(base, xr, mr, layer.patcher(phase[selected+1], correction[selected+1]), state=state)
    pred = departure(lag._predict(issued,prior[selected],decoder)[...,group.USE],scale=1.)
    membership = F.one_hot(torch.as_tensor(group.SIGNED_GROUPS,device=x.device),num_classes=len(group.TRUSTED))
    error = forces(departure(fields[0][selected],scale=fields[1]), pred, membership)
    basis = group.signed_directions(issued, current, state, tuple(
        (pred, force) for force in error.transpose(1,2).flatten(0,1).unbind(0)))
    return basis.unflatten(0,(2,-1)).flatten(-2)


def track(signal, args, decoder, path, *, source=False):
    q, h, z, beta, prior, phase, correction = group.parent_track(signal,args,decoder,path,source=source)
    base = args[0]
    eligible, y, noise = prior_fields(signal,q,args[2],z.device)
    states = z.new_zeros((2,len(q)+1,2*len(group.TRUSTED),(base.cfg.window//base.cfg.patch-1)*len(group.TRUSTED)))
    selection = np.flatnonzero(eligible)
    for start in range(0,len(selection),group.BATCH):
        j = selection[start:start+group.BATCH]
        states[:,j+1] = flow(signal,q,prior,phase,correction,(y,noise),decoder,base,j).transpose(1,2)
    _, _, ctx, cm = lag.direct.weak.shot_coordinates(signal,args[2][0],args[2][1])
    latest,n = _window_mean(ctx,cm,q,-15,16)
    previous,count = _window_mean(ctx,cm,q,-31,16)
    motion = np.where((n[:,0]>=8)&(count[:,0]>=8),latest[:,0]-previous[:,0],0.)
    keys = np.column_stack((np.where(n>=8,latest,0.),4.*motion))
    key = torch.as_tensor(np.tanh(keys),dtype=z.dtype,device=z.device)
    context = torch.cat((z.new_zeros((1,4)),torch.column_stack((torch.ones_like(key[:,0]),key))))
    if not torch.isfinite(states).all():
        raise FloatingPointError('nonfinite feedback basis')
    return q,h,z,beta,phase,correction,states,eligible,context


def sampled_peaks(indices, score, count):
    parts = np.array_split(indices,count)
    return np.array([r[score[r].argmax()] for r in parts if len(r)],dtype=int)


def finite_rows(base, signal, tr, ends, clock):
    q,h,z,beta,phase,correction,bases,eligible,context = tr
    active, slot, ramp = clock
    if not active[ends].all():
        raise ValueError('finite rows require legally observed feedback')
    i = slot[ends]
    old = i-1
    x,m = lag._windows(base,signal,ends)
    parent_slot = lag.delayed_slots(q,ends)[0]
    return (x,m,phase[parent_slot],correction[parent_slot],z[ends],h[ends],bases[:,i].transpose(0,1),
            bases[:,old].transpose(0,1),context[i],context[old],
            torch.as_tensor(ramp[ends],dtype=z.dtype,device=z.device),
            torch.as_tensor(np.r_[False,eligible][old],device=z.device))


def directions(rows,w,which):
    basis,previous,x,old,ramp,ready = rows[6:]
    which = [which] if isinstance(which,int) else which
    gain = torch.stack([rates(x,a) for a in w])
    gain_old = torch.stack([rates(old,a) for a in w])
    chosen = torch.stack([basis[:,i] for i in which])
    older = torch.stack([previous[:,i] for i in which])
    new = blend(chosen,gain)
    prior = blend(older,gain_old)
    shape = (-1,len(group.TRUSTED))
    return transport(new.unflatten(-1,shape),prior.unflatten(-1,shape),ramp,ready)


def apply_writes(base, rows, beta, w, which):
    x,mask,phase,correction,z,h = rows[:6]
    t = layer.patcher(phase,correction)
    moves = directions(rows,w,which)
    # Ease in from zero on the first action or after a gap in both domains.
    cap = torch.where(rows[-1],1.,rows[-2])
    common,realized,check = group.forward_hidden(base,x,mask,t,writes=moves,radius_fraction=cap)
    return z+((common-h)*beta).sum(-1),realized,check


def episode(raw, truth, view, args, decoder, path, threshold):
    base = args[0]
    tr = track(view,args,decoder,path,source=True)
    q,_,z,_,_,_,_,eligible,_ = tr
    clock = active_windows(q,eligible,len(z),base.cfg.patch)
    avail,slot,_ = clock
    ends = np.flatnonzero(teaching_window(raw['times'],truth['valid'],truth['y'].astype(bool),truth['down'],avail))
    if not len(ends):
        return None
    scores = z.cpu().numpy()
    if truth['is_disrupt']:
        timely = np.round((truth['down']-raw['times'])*1000.,9)>10.
        hard = not (scores[timely]>=threshold).any()
    else:
        hard = bool((scores>=threshold).any())
    # Use actual near-deadline and safety peaks, not an outcome-derived gate
    # on the diagnostic updates. Only outer, ended-source episodes select endpoints.
    picks = sampled_peaks(ends,scores,CAP_SAMPLES)
    frozen_peak = scores[np.setdiff1d(np.flatnonzero(
        teaching_window(raw['times'],truth['valid'],truth['y'].astype(bool),truth['down'],
                        np.ones(len(scores),bool))),ends)].max(initial=-1e6)
    return finite_rows(base,view,tr,picks,clock), dict(shot=raw['shot'],positive=truth['is_disrupt'],hard=hard,
        sample_ends=picks.tolist(),possible_ends=len(ends),first_update=int(np.flatnonzero(avail)[0]),
        issue=q[slot[picks]-1].tolist(),inactive_peak=float(frozen_peak))


def source_batches(examples, indices, device):
    packed, owner = [], []
    for i in indices:
        rows, record = examples[i]
        packed.append(rows)
        owner.append(record)
    batch = tuple(torch.cat(values).to(device) for values in zip(*packed,strict=True))
    ids = torch.repeat_interleave(torch.arange(len(owner),device=device),
        torch.tensor([len(r['sample_ends']) for r in owner],device=device))
    inactive = torch.tensor([r['inactive_peak'] for r in owner],device=device)
    positive = torch.tensor([r['positive'] for r in owner],device=device,dtype=torch.bool)
    return batch,ids,inactive,positive,owner


def peaks(scores, ids, inactive):
    initial = inactive.expand(len(scores),-1)
    return initial.scatter_reduce(1,ids.expand(len(scores),-1),scores,reduce='amax')


def balanced_mass(samples):
    classes = [(r['positive'],r['hard']) for _,r in samples]
    p = [sum(y==k for y,_ in classes) for k in (False,True)]
    if not all(p):
        raise ValueError('teaching needs both outcomes')
    # Each class has half the mass, with 1:1 ordinary/hard mass if supported.
    counts = {key:classes.count(key) for key in set(classes)}
    return np.array([.5/(sum((y,h) in counts for h in (False,True))*counts[y,hard])
                     for y,hard in classes],dtype=np.float32)


def fit(base,samples,threshold,beta):
    w = torch.zeros(2,4,2*len(group.TRUSTED),device=beta.device,requires_grad=True)
    mass = torch.as_tensor(balanced_mass(samples),device=beta.device)
    gen = torch.Generator().manual_seed(131)
    opt = torch.optim.Adam((w,),lr=RATE)
    trace = []
    for epoch in range(EPOCHS):
        total = 0.
        for j in torch.randperm(len(samples),generator=gen).split(48):
            r,ids,inactive,y,_ = source_batches(samples,j.tolist(),beta.device)
            # Same four directions and support in the training and replay
            # norm calculation; only the two fitted arms enter the loss.
            bank = torch.stack((torch.zeros_like(w[0]),w[0],w[0].detach(),w[1]))
            scores,_,_ = apply_writes(base,r,beta,bank,[0,0,1,1])
            z = peaks(scores[[2,4]],ids,inactive)
            margin = torch.where(y,threshold-z,z-threshold)
            loss = (F.softplus(margin/.25)*mass[j][None,:]).sum()*(len(samples)/len(j)) + .04*w.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(j)/len(samples)
        trace.append(total)
    if not torch.isfinite(w).all():
        raise FloatingPointError('nonfinite source update dictionary')
    return w.detach(),trace


@torch.no_grad()
def measure(base,samples,w,threshold,beta):
    output = []
    if not samples:
        return output
    for i in range(0,len(samples),64):
        r,ids,inactive,_,records = source_batches(samples,range(i,min(i+64,len(samples))),beta.device)
        z,_,_ = apply_writes(base,r,beta,
            torch.stack((torch.zeros_like(w[0]),w[0],w[0],w[1])),[0,0,1,1])
        for record, score in zip(records,peaks(z,ids,inactive).T.cpu().numpy(),strict=True):
            output.append(dict(record,peak=dict(zip(('parent',*NEW),score.tolist(),strict=True)),
                               crossing=dict(zip(('parent',*NEW),(score>=threshold).tolist(),strict=True))))
    return output


def teach(args,decoder,path,threshold,out):
    roles,adv = io23.source_roles(),io23.advanced_times()
    examples = {}
    for role in ('train','val'):
        examples[role] = []
        for i,sid in enumerate(roles[role],1):
            raw = simple_data.read_shot('J-TEXT',sid)
            truth = simple_data.attach_outcome(raw,'J-TEXT',adv)
            signal = diagnostic_view({k:raw[k] for k in ('values','available')},'no_MNM_CIII')
            row = episode(raw,truth,signal,args,decoder,path,threshold)
            if row is not None:
                # Completed source episodes occupy CPU memory between optimization batches.
                samples,record = row
                examples[role].append((tuple(x.cpu() for x in samples),record))
            if i%450==0 or i==len(roles[role]):
                print(dict(stage='source_teaching_examples',role=role,read=i,used=len(examples[role])),flush=True)
    beta = args[0].risk_head.weight.detach().flatten()
    fitted,losses = fit(args[0],examples['train'],logit(threshold),beta)
    summary = {k:measure(args[0],v,fitted,logit(threshold),beta) for k,v in examples.items()}
    write(out/'source_teaching.json',dict(fit=fitted.cpu().tolist(),sampled=summary,
        counts={r:len(roles[r]) for r in examples},loss=losses,
        scope='source sampled eligible peaks only; fixed J-TEXT base and original threshold',
        steps=EPOCHS,learning_rate=RATE,positive_rule='actual AdvancedTime positive window and lead strictly >10ms'))
    return fitted


@torch.no_grad()
def replay(signal,tr,base,factors,parent):
    q,_,z,beta,_,_,_,eligible,_ = tr
    clock = active_windows(q,eligible,len(z),base.cfg.patch)
    active,_,_ = clock
    ends = np.flatnonzero(active)
    scores = np.broadcast_to(parent,(len(NEW),len(parent))).copy()
    doses = np.zeros(len(z),dtype=np.float32)
    w = torch.stack((torch.zeros_like(factors[0]),factors[0],factors[0],factors[1]))
    pe = de = 0.
    for j in range(0,len(ends),group.BATCH):
        i = ends[j:j+group.BATCH]
        rows = finite_rows(base,signal,tr,i,clock)
        values,actual,error = apply_writes(base,rows,beta,w,[0,0,1,1])
        curves = values.sigmoid().cpu().numpy()
        pe,de = max(pe,float(np.abs(curves[0]-parent[i]).max())),max(de,float(error))
        scores[:,i] = np.where(actual.cpu().numpy()>0.,curves[1:],parent[i])
        doses[i] = actual.cpu().numpy()
    if pe>3e-6 or de>1e-5 or not np.isfinite(scores).all():
        raise ValueError(f'parent/dose mismatch: {pe:g}/{de:g}')
    return scores,doses,pe,de


def run(out):
    lag.configure()
    lock = lag.load_inputs()
    ref,_,args,decoder,path = group.load_reference(lock)
    old,seal = (read(archive.ARCHIVE/f'{k}.json') for k in ('protocol','seal'))
    if any(ref[k]!=old[k] for k in LOCK) or len(seal['curve_sha256'])!=len(old['order']):
        raise ValueError('the complete locked parent is required')
    base = args[0]
    before = [state_digest(m) for m in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(__file__,out/'executed.py')
    start = time.monotonic()
    write(out/'protocol.json',dict(scope='EAST360_SOURCE_TAUGHT_DIAGNOSTIC_FEEDBACK_DICTIONARY',
        **{k:old[k] for k in LOCK},arms=ARMS,reference=str(archive.ARCHIVE),reference_seal=sha(archive.ARCHIVE/'seal.json'),
        q='127+32k',loss_support='16 observed-history samples ending at issue; require at least eight per channel',
        maturity=lag.H+1,extra_wait=WAIT,handoff=base.cfg.patch,
        update='Eight level/temporal diagnostic gradient directions, operating coordinates from issue only. '
               'Source TRAIN fits two adapters on chronological and shifted feedback, with the same draws. '
               'No additive output change. The fourth arm uses the fitted shifted adapter with shifted feedback.',
        mode='only entirely completed 25-point forecasts; coast on the frozen parent when unavailable; reset per shot',
        evaluation='EAST outcomes forbidden until all shot curves sealed; cutoff and calibration unchanged'))
    fitted = teach(args,decoder,path,old['original'],out)
    checked,files,maxima = {},{},[0.,0.,0.]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('EAST outcome forbidden before seal')):
        for rank,sid in enumerate(old['order'],1):
            name = f'{rank:04d}_{sid}.npz'
            ref_path = archive.ARCHIVE/'shots'/name
            if sha(ref_path)!=seal['curve_sha256'][name]:
                raise ValueError('locked control curve changed')
            with np.load(ref_path,allow_pickle=False) as f:
                controls = {k:f[k] for k in BASE}
            parent = controls['bias_only']
            if rank<=old['startup']:
                scores=np.broadcast_to(parent,(len(NEW),len(parent)))
                dose=np.zeros(len(parent),np.float32)
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST data changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                tr = track(signal,args,decoder,path)
                scores,dose,pe,de = replay(signal,tr,base,fitted,parent)
                maxima[0],maxima[1] = max(maxima[0],pe),max(maxima[1],de)
                if rank in (21,120,360) or sid in archive.CASES:
                    end = min(550,len(parent))
                    partial = {k:signal[k][:end] for k in ('values','available')}
                    short,dd,*_ = replay(partial,track(partial,args,decoder,path),base,fitted,parent[:end])
                    np.testing.assert_allclose(short,scores[:,:end],rtol=0.,atol=3e-6)
                    np.testing.assert_allclose(dd,dose[:end],rtol=1e-6,atol=3e-6)
                    for tau in old['calibration'].values():
                        np.testing.assert_array_equal(short>=tau,scores[:,:end]>=tau)
                    maxima[2] = max(maxima[2],float(np.abs(short-scores[:,:end]).max()))
                    checked[str(sid)] = end
            dest=out/'shots'/name
            with dest.open('xb') as stream:
                np.savez_compressed(stream,**controls,**dict(zip(NEW,scores,strict=True)),dose=dose)
            files[name]=sha(dest)
            if rank==21 or rank%60==0:
                print(dict(stage='seal_dictionary',rank=rank,elapsed_s=round(time.monotonic()-start,2)),flush=True)
    if before!=[state_digest(m) for m in args[:2]]:
        raise ValueError('base or historical model changed')
    write(out/'seal.json',dict(curves=files,checked=checked,max_parent_error=maxima[0],max_dose_error=maxima[1],
        max_prefix_error=maxima[2],elapsed_s=time.monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    if len(s['curves'])!=360 or p['reference_seal']!=sha(archive.ARCHIVE/'seal.json'):
        raise ValueError('only a complete sealed stream may be scored')
    per = {b:{a:[] for a in ARMS} for b in p['calibration']}
    baseline = read(archive.ARCHIVE/'per_shot.json')
    touch, near, counts = 0,[],[]
    for rank,sid in enumerate(p['order'],1):
        path = out/'shots'/f'{rank:04d}_{sid}.npz'
        if sha(path)!=s['curves'][path.name]:
            raise ValueError('changed sealed prediction')
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(path,allow_pickle=False) as f:
            active=f['dose']>0.
            touch += int(active.sum())
            counts.append(int(active.sum()))
            for b,tau in p['calibration'].items():
                tau = p['original'] if rank<=p['startup'] else tau
                for k in ARMS:
                    curve=f[k]
                    if k in NEW:
                        np.testing.assert_array_equal(curve[~active],f['bias_only'][~active])
                    row=dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,curve,y,down,tau))
                    if k in BASE and row!=baseline[b][k][rank-1]:
                        raise ValueError('changed first alarm in a locked control')
                    per[b][k].append(row)
            if y and rank>p['startup'] and not per['0.2']['current_5'][-1]['timely']:
                dt=np.round((down-t)*1000.,9)
                ok=active&(dt>30.)&(dt<=125.)
                tau=p['calibration']['0.2']
                entry=dict(shot=sid,support=int(ok.sum()))
                if ok.any():
                    probabilities=np.stack([f[k][ok] for k in ('current_5',*NEW)])
                    logits=logit(np.clip(probabilities,1e-6,1.-1e-6))
                    entry.update(gap=float(logit(tau)-logits[0].max()),
                        lift=dict(zip(NEW,(logits[1:]-logits[:1]).max(1).tolist(),strict=True)),
                        crosses=dict(zip(NEW,(probabilities[1:]>=tau).any(1).tolist(),strict=True)))
                near.append(entry)
    metrics={b:{k:lead.summarize(v) for k,v in table.items()} for b,table in per.items()}
    paired={b:{f'{a}__to__{k}':dict(flips=flips(table[a],table[k]),lead=lead.paired(table[a],table[k]))
               for a in ('bias_only','current_5',*NEW) for k in NEW if a!=k} for b,table in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(metrics=metrics,paired=paired,write_endpoints=touch,
        per_shot_writes=counts,original_late=near,raw_verified=360*len(ARMS)*len(per)))
    print({b:{k:(v['by_deadline']['30']['tp'],v['fp']) for k,v in table.items()} for b,table in metrics.items()},
          flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args()
    if not args.score_only:
        run(args.output.resolve())
    score(args.output.resolve())
