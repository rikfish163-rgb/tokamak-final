"""Compare equal-dose time-token subspaces of mature diagnostic gradients on EAST360."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_feedback_span_20261112 as span
from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .commissioning_group_carry_20261112 import HISTORY
from .commissioning_suppression_reach_20261112 import REFERENCE, WAIT
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


lag, layer = group.lag, group.layer
CENTERED_SPACES = ('free','projected')
LEVEL_SPACES = ('free','mean','full')
CENTERED_REFERENCE = io23.ROOT/'probe_outputs/commissioning_centered_feedback_east360_20261112_v1'
OLD = (*group.OLD,'current_5','earlier_5')


def names(spaces):
    return tuple(f'{space}_{source}' for space in spaces for source in ('current','earlier'))


def valid_patches(base,x,mask):
    return (mask&torch.isfinite(x)).reshape(len(x),base.cfg.window//base.cfg.patch,-1).any(-1)


def center(move, valid):
    count = valid.sum(1).clamp_min(1)[...,None,None]
    mask = ~valid[...,None]
    x = move.masked_fill(mask,0.)
    return (x-x.sum(-2,keepdim=True)/count).masked_fill(mask,0.)


def restrict(move, valid, space):
    if space in ('free','projected'):
        return center(move,valid)
    mask = ~valid[...,None]
    x = move.masked_fill(mask,0.)
    if space=='mean':
        n = valid.sum(-1).clamp_min(1)[...,None,None]
        return (x.sum(-2,keepdim=True)/n).expand_as(x).masked_fill(mask,0.)
    return x


def hidden(base,x,mask,transform,*,differentiate=False):
    tokens, result, edges = [], [], []

    def patches(_module, args):
        p = transform(args[0])
        if differentiate:
            edges.append(group.group_edges(p,base.embedding.weight))
        return (p,)

    def capture(_module,args):
        a = args[0].detach().requires_grad_(differentiate)
        tokens.append(a)
        return (a,*args[1:])

    hooks = (base.embedding.register_forward_pre_hook(patches),
             base.encoder.layers[0].register_forward_pre_hook(capture),
             base.risk_head.register_forward_pre_hook(lambda _module,args:result.append(args[0])))
    try:
        base(x,mask)
    finally:
        for h in hooks:
            h.remove()
    return result[0],(tokens[0],edges[0]) if differentiate else None


def feedback_directions(base,signal,q,eligible,phase,correction,prior,fields,waveform,decoder,weights,
                        *,spaces=CENTERED_SPACES):
    """The same issued signed innovation and arrival Jacobian, in time-token coordinates."""
    steps = prior.new_zeros((2*len(spaces),len(q)+1,base.cfg.window//base.cfg.patch,base.cfg.dim))
    force = prior.new_zeros((len(q),lag.H//group.BIN,len(group.USE)))
    *_,common = group.past_force_parts(q,force,eligible)
    seen = np.flatnonzero(eligible)
    recent,ref,spread,noise = fields
    origin,_ = lag.delayed_slots(q,q)
    for start in range(0,len(seen),group.BATCH):
        i = seen[start:start+group.BATCH]
        s = origin[i]
        x,m = lag._windows(base,signal,q[i])
        with torch.no_grad():
            issued,_ = hidden(base,x,m,layer.patcher(phase[s],correction[s]))
        issued.requires_grad_(True)
        wave = lag._predict(issued,prior[i],decoder)*noise+recent[i,None]
        y = group.signed_levels(wave,ref[i],spread[i])
        measured = group.signed_levels(waveform[i],ref[i],spread[i])
        force[i] = ((measured-y.detach())/group.BETA).clamp(-1.,1.)
        _,current,earlier,_ = group.past_force_parts(q,force,eligible)
        delta = (current[i],earlier[i])
        write_x,write_m = lag._windows(base,signal,q[i]+lag.H+1)
        representation,(tokens,edges) = hidden(
            base,write_x,write_m,layer.patcher(phase[i+1],correction[i+1]),differentiate=True)
        valid = valid_patches(base,write_x,write_m)
        if 'projected' in spaces:
            op = span.edge_matrix(edges)
            gram = op.transpose(1,2)@op
            eye = torch.eye(gram.shape[1],device=gram.device,dtype=gram.dtype)
        keep = torch.as_tensor(common[i],device=prior.device)[:,None,None]
        for j,d in enumerate(delta):
            issued_grad, = torch.autograd.grad(y,issued,grad_outputs=group.shape_force(d,weights),
                                                retain_graph=j==0)
            g, = torch.autograd.grad(representation,tokens,grad_outputs=issued_grad,retain_graph=j==0)
            g = g[:,:-1]*keep
            for k,space in enumerate(spaces):
                move = g
                if space=='projected':
                    rhs = op.transpose(1,2)@center(g,valid).flatten(1)[...,None]
                    move = (op@torch.linalg.solve(gram+1e-5*eye,rhs)).reshape_as(g)
                steps[2*k+j,i+1] = restrict(move,valid,space)
    if not torch.isfinite(steps).all():
        raise ValueError('nonfinite diagnostic gradient')
    return steps,common


@torch.no_grad()
def forward_with_writes(base,x,mask,transform,new,previous,fraction,budget,*,spaces=None):
    """Apply both directions to the same valid time patches, at the archived token dose."""
    valid = valid_patches(base,x,mask)
    spaces = ('free',)*len(new) if spaces is None else spaces
    old,new = (torch.stack([restrict(v,valid,space) for v,space in zip(a,spaces,strict=True)])
               for a in (previous,new))
    d = group.blend_writes(new,old,budget,fraction)
    delta = group.matched_writes(d,budget)
    n = len(delta)
    outputs, errors = [], []

    def inject(_module,args):
        before = args[0].unflatten(0,(n,len(x)))
        a = layer.add_to_patches(args[0],delta.flatten(0,1)).unflatten(0,(n,len(x)))
        realized = a[...,:-1,:]-before[...,:-1,:]
        actual = (realized.flatten(2).norm(dim=2)-budget).abs().max()
        subspace = ((restrict(d,valid,space)-d).abs().max() for d,space in zip(realized,spaces,strict=True))
        errors.append(torch.stack((actual,*subspace)).max())
        return (a.flatten(0,1),*args[1:])

    hooks = (base.embedding.register_forward_pre_hook(lambda _module,args:(transform(args[0][:len(x)]).repeat(n,1,1),)),
             base.encoder.layers[0].register_forward_pre_hook(inject,prepend=True),
             base.risk_head.register_forward_pre_hook(lambda _module,args:outputs.append(args[0])))
    try:
        base(x.repeat(n,1,1),mask.repeat(n,1,1))
    finally:
        for h in hooks:
            h.remove()
    err = float(errors[0])
    if err>1e-5 or not torch.isfinite(outputs[0]).all():
        raise ValueError(f'archived norm or token subspace violated: {err:g}')
    return outputs[0].unflatten(0,(n,len(x))),err


def predict(signal,args,decoder,path,weights,parent,release,eligible,budget,*,spaces=CENTERED_SPACES):
    base = args[0]
    q,h,z,beta,prior,phase,correction = group.parent_track(signal,args,decoder,path)
    avail,fields,wave = group.feedback_waveform(signal,q,args[2],z.device)
    steps,common = feedback_directions(base,signal,q,avail,phase,correction,prior,fields,wave,decoder,weights,
                                       spaces=spaces)
    np.testing.assert_array_equal(q+lag.H+1,release)
    np.testing.assert_array_equal(common,eligible)
    i = np.flatnonzero(budget>0.)
    curves = np.broadcast_to(parent,(2*len(spaces),len(parent))).copy()
    dose_error = 0.
    for start in range(0,len(i),group.BATCH):
        e = i[start:start+group.BATCH]
        s = np.searchsorted(release,e,side='right')
        u = np.searchsorted(release+WAIT,e,side='right')
        if (u==0).any() or not common[u-1].all():
            raise ValueError('dose before usable release')
        f = group.handoff_fraction(release[u-1]+WAIT,e,base.cfg.patch)
        x,m = lag._windows(base,signal,e)
        view,err = forward_with_writes(base,x,m,layer.patcher(phase[s],correction[s]),steps[:,u],steps[:,u-1],
                                        z.new_tensor(f),z.new_tensor(budget[e]),
                                        spaces=tuple(v for space in spaces for v in (space,space)))
        curves[:,e] = (z[e]+((view-h[e])*beta).sum(-1)).sigmoid().cpu().numpy()
        dose_error = max(dose_error,err)
    norms = steps[:,1:].flatten(2).norm(dim=2)[:,common]
    denominator = norms[-2:] if 'full' in spaces else norms[:2]
    ratio = (norms[2:4]/denominator.clamp_min(group.EPS)).cpu().numpy()
    return curves,dose_error,ratio


def run(out, spaces):
    lag.configure()
    start = time.monotonic()
    lock = lag.load_inputs()
    p,r,args,decoder,path = group.load_reference(lock)
    reference,seal = (read(REFERENCE/k) for k in ('protocol.json','seal.json'))
    keys = ('order','startup','original','calibration','source_sha256')
    asset = HISTORY/'teaching.json'
    if (any(p[k]!=reference[k] for k in keys) or reference['reference_seal']!=sha(group.REFERENCE/'seal.json')
            or reference['history_seal']!=sha(HISTORY/'seal.json')
            or read(HISTORY/'protocol.json')['teaching_sha256']!=sha(asset)
            or len(seal['curve_sha256'])!=360 or reference['additional_wait_samples'][-1]!=WAIT):
        raise ValueError('wrong frozen reference')
    base = args[0]
    weights = torch.as_tensor(read(asset)['weights'],device=base.center.device,dtype=torch.float32)
    before = [state_digest(m) for m in args[:2]]
    extra = spaces == LEVEL_SPACES
    arms = names(spaces)
    centered_seal = None
    if extra:
        check = read(CENTERED_REFERENCE/'protocol.json')
        centered_seal = read(CENTERED_REFERENCE/'seal.json')
        if (any(check[k]!=p[k] for k in keys) or check['reference_seal']!=sha(REFERENCE/'seal.json')
                or check['teaching_sha256']!=sha(asset) or len(centered_seal['curve_sha256'])!=360):
            raise ValueError('wrong archived centered arm')
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    write(out/'protocol.json',dict(scope='EAST360_REAL_FEEDBACK_TOKEN_SUBSPACES_DEVELOPMENT',
        **{k:p[k] for k in keys},arms=(*OLD,*arms),spaces=spaces,delay=WAIT,reference_seal=sha(REFERENCE/'seal.json'),
        centered_seal=sha(CENTERED_REFERENCE/'seal.json') if extra else None,
        teaching_sha256=sha(asset),executed_sha256=sha(out/'executed_probe.py'),
        contrasts=('real and earlier innovation, zero-mean vs mean-only vs full, transported in the same '
                   'relative time-token coordinates' if extra else 'real and earlier innovation, zero-mean '
                   'vs projected on the group write space at the gradient release window, same transport'),
        clock='same q+26 gradient and 5 additional samples before switching; same 8-point handoff',
        dose='same realized token L2 at each endpoint, only valid time tokens, CLS never written',
        outcomes='none during prediction, independent raw HDF5 after all 360 are sealed'))
    hashes,witnesses,geometry = {},{},[]
    dose_error = control_error = 0.
    endpoints = 0
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('outcome forbidden before sealing')):
        for rank,sid in enumerate(p['order'],1):
            name = f'{rank:04d}_{sid}.npz'
            a,b = REFERENCE/'shots'/name,group.REFERENCE/'shots'/name
            if (sha(a)!=seal['curve_sha256'][name] or sha(b)!=r['curve_sha256'][name]
                    or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]):
                raise ValueError('sealed input changed')
            with np.load(a,allow_pickle=False) as f,np.load(b,allow_pickle=False) as g:
                old = {k:f[k] for k in OLD}
                release,eligible,budget = f['release'],f['eligible'],f['dose'][-1]
                times = g['times']
                np.testing.assert_array_equal(release,g['release'])
            parent = old[group.bias.BIAS]
            if rank<=p['startup']:
                if np.any(budget):
                    raise ValueError('cold dose must be zero')
                curves = np.broadcast_to(parent,(len(arms),len(parent)))
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],times)
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                curves,error,ratio = predict(signal,args,decoder,path,weights,parent,release,eligible,budget,
                                             spaces=spaces)
                dose_error = max(dose_error,error)
                geometry.append(ratio)
                endpoints += int((budget>0.).sum())
                if rank in (21,120,360):
                    end = min(550,len(times))
                    x = {k:signal[k][:end] for k in ('values','available')}
                    keep = release<end
                    short,*_ = predict(x,args,decoder,path,weights,parent[:end],release[keep],eligible[keep],
                                       budget[:end],spaces=spaces)
                    np.testing.assert_allclose(short,curves[:,:end],atol=3e-6,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(short>=tau,curves[:,:end]>=tau)
                    witnesses[str(sid)] = end
            if centered_seal is not None:
                file = CENTERED_REFERENCE/'shots'/name
                if sha(file)!=centered_seal['curve_sha256'][name]:
                    raise ValueError('changed zero-mean control')
                with np.load(file,allow_pickle=False) as f:
                    archived = np.stack([f[k] for k in names(('free',))])
                control_error = max(control_error,float(np.abs(curves[:2]-archived).max()))
                for tau in p['calibration'].values():
                    np.testing.assert_array_equal(curves[:2]>=tau,archived>=tau)
                if control_error>3e-6:
                    raise ValueError('zero-mean control does not match the previous experiment')
            file = out/'shots'/name
            with file.open('xb') as f:
                np.savez_compressed(f,**old,**dict(zip(arms,curves,strict=True)))
            hashes[name] = sha(file)
            if rank==21 or rank%60==0:
                print(dict(stage='seal_token_subspaces',rank=rank,seconds=round(time.monotonic()-start,2)),flush=True)
    if before!=[state_digest(m) for m in args[:2]]:
        raise ValueError('frozen model changed')
    proj = np.concatenate(geometry,axis=1)
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,max_dose_or_subspace_error=dose_error,
        released=int(proj.shape[1]),writable_endpoints=endpoints,
        release_mean_over_full_median=np.median(proj,axis=1).tolist() if extra else None,
        release_group_fraction_median=None if extra else np.median(proj,axis=1).tolist(),
        centered_control_max_error=control_error if extra else None,seconds=time.monotonic()-start))


def score(out):
    p,s = (read(out/k) for k in ('protocol.json','seal.json'))
    if (p['reference_seal']!=sha(REFERENCE/'seal.json') or p['executed_sha256']!=sha(out/'executed_probe.py')
            or (p.get('centered_seal') and p['centered_seal']!=sha(CENTERED_REFERENCE/'seal.json'))
            or len(s['curve_sha256'])!=360):
        raise ValueError('invalid new seal')
    old = read(REFERENCE/'per_shot.json')
    arms,new = p['arms'],names(p['spaces'])
    pairs = ((group.bias.BIAS,'free_current'),('current_5','free_current'),
             *((f'{sp}_earlier',f'{sp}_current') for sp in p['spaces']),
             *((f'free_{age}',f'{sp}_{age}') for sp in p['spaces'][1:] for age in ('current','earlier')),
             *(('current_5',f'{sp}_current') for sp in p['spaces'][1:]))
    per = {b:{k:[] for k in arms} for b in p['calibration']}
    old_seal = read(REFERENCE/'seal.json')['curve_sha256']
    near = []
    from scipy.special import logit

    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        a,b = out/'shots'/name,REFERENCE/'shots'/name
        if sha(a)!=s['curve_sha256'][name] or sha(b)!=old_seal[name]:
            raise ValueError('curve changed before scoring')
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(a,allow_pickle=False) as f,np.load(b,allow_pickle=False) as archive:
            dormant = archive['dose'][-1]==0.
            for k in new:
                np.testing.assert_array_equal(f[k][dormant],f[group.bias.BIAS][dormant])
            for band,tau in p['calibration'].items():
                for k in arms:
                    row = dict(shot=sid,disruptive=y,down_s=d,**score_curve(
                        t,f[k],y,d,p['original'] if rank<=p['startup'] else tau))
                    if k in OLD and row!=old[band][k][rank-1]:
                        raise ValueError('archived first alarm changed')
                    per[band][k].append(row)
            if y and rank>p['startup'] and not old['0.2']['current_5'][rank-1]['timely']:
                lead = np.round((d-t)*1000.,9)
                use = (~dormant)&(lead>30.)&(lead<=125.)
                if use.any():
                    source = logit(f['current_5'][use].astype(float).clip(1e-7,1.-1e-7))
                    after = logit(np.stack([f[k][use] for k in new]).astype(float).clip(1e-7,1.-1e-7))
                    near.append(dict(shot=sid,gap=float(logit(p['calibration']['0.2'])-source.max()),
                                     **dict(zip(new,(after-source).max(1).tolist(),strict=True))))
    write(out/'per_shot.json',per)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],arms))
    metrics = {b:{k:timing_decomposition(v) for k,v in rows.items()} for b,rows in per.items()}
    write(out/'result.json',dict(status='COMPLETE_REAL_FEEDBACK_SUBSPACE_DEVELOPMENT',metrics=metrics,
        raw_verified=360*len(arms)*len(metrics),near_missed=near,
        paired={b:{f'{a}__to__{k}':flips(rows[a],rows[k]) for a,k in pairs} for b,rows in per.items()}))
    print({b:{k:(v['tp'],v['fp']) for k,v in rows.items()} for b,rows in metrics.items()},flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-level',action='store_true',help='compare mean-only and unrestricted time-token writes')
    ap.add_argument('out',type=Path)
    a = ap.parse_args()
    run(a.out.resolve(),LEVEL_SPACES if a.allow_level else CENTERED_SPACES)
    score(a.out.resolve())
