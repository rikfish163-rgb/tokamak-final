"""Full replay of observation-state and continuous-motion feedback at equal input dose."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_bias_dynamics_20261112 as bias
from . import commissioning_coincident_support_audit_20261111 as hdf
from . import commissioning_eos_replay_20261111 as eos
from . import commissioning_frequency_feedback_20261112 as prior
from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import commissioning_record_dynamics_20261112 as two
from . import io23, simple_data
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


REFERENCE = prior.REFERENCE
OLD = prior.OLD
PARENT = 'bias_only'
STATE,STATE_WRONG,MOTION,ISOLATED,ISOLATED_WRONG = NEW = (
    'record_state','record_state_wrong_time','conditional_motion',
    'isolated_motion','isolated_motion_wrong_time',
)
ARMS = (*OLD,*NEW)
PAIRS = tuple((PARENT,k) for k in NEW)+((STATE_WRONG,STATE),(MOTION,ISOLATED),(ISOLATED_WRONG,ISOLATED))
TOL = prior.TOL


def _load(fit):
    lag.configure()
    lock = lag.load_inputs()
    p, s, model = (read(k) for k in (REFERENCE/'protocol.json',REFERENCE/'seal.json',fit/'fit.json'))
    ref = Path(p['reference'])
    gp,_,_,args,decoder = eos._load(ref,lock)
    if (p['reference_seal']!=sha(ref/'seal.json') or model['parent_seal']!=sha(REFERENCE/'seal.json')
            or model['checkpoint']!=sha(fit/'readout.npz') or model['code']!=sha(Path(two.__file__))
            or not set(model['eligible_train_ids'])<=set(io23.source_roles()['train'])
            or p['order']!=lock['ids'] or p['calibration']!=gp['calibration']
            or model['source_sha256']!=p['source_sha256'] or sha(ref/'teacher.json')!=gp['teacher_sha256']):
        raise ValueError('two-part forecasts and legacy path do not share the locked source and stream')
    path = torch.as_tensor(read(ref/'teacher.json')['path'],device='cuda',dtype=torch.float32)
    with np.load(fit/'readout.npz',allow_pickle=False) as f:
        fitted = tuple(torch.as_tensor(f[k],device='cuda') for k in ('center','scale','w','b'))
    return lock,p,s,args,decoder,path,fitted


def directions(raw, signal, q, keys, predicted, measured, args, decoder, path, fitted):
    """Backpropagate independent feedback heads through the same origin and write site."""
    base = args[0]
    context, target = two.observation(raw,signal,q,args)
    residues,matrices,geometry = [],[],[]
    for start in range(0,len(q),128):
        sl = slice(start,start+128)
        x,m = lag._windows(base,signal,q[sl])
        theta = x.new_zeros((len(x),two.C,two.C)).requires_grad_()
        h = lag._features(base,x,m,theta)
        def to(a):
            return torch.as_tensor(a[sl],device=x.device)

        yh,yt = lag._predict(h,to(predicted),decoder),to(measured)
        residues.append((yt-yh.detach()).mean(1))
        old = teacher.grouped_error(lag.ordered_signature(lag.motion(yh)),lag.ordered_signature(lag.motion(yt)))
        gout = torch.stack([torch.autograd.grad(old[:,j].sum(),theta,retain_graph=True)[0]
                            for j in range(teacher.GROUPS)],1)
        old_mat = -lag.RADIUS*guard.signed_direction(*guard.basis(gout),to(keys),path)
        logits,values = two.forecast(h,to(context),fitted)
        truth = {k:to(v) for k,v in target.items()}
        st,motion = two.losses(logits,values,truth)
        sw,mw = two.losses(logits,values,truth,shifted=True)
        gs,gw,gm,gq = [torch.autograd.grad(loss.sum(),theta,retain_graph=i<3)[0]
                       for i,loss in enumerate((st,sw,motion,mw))]
        clean,wrong = two.isolate(gm,gs),two.isolate(gq,gs)
        def norm(z):
            return z.flatten(1).norm(dim=1)

        def dot(a,b):
            return (a*b).flatten(1).sum(1)

        pre = dot(gm,gs)/(norm(gm)*norm(gs)).clamp_min(1e-16)
        post = dot(clean,gs)/(norm(clean)*norm(gs)).clamp_min(1e-16)
        geometry.append(torch.stack((pre,post,norm(clean)/norm(gm).clamp_min(1e-16)),1))
        g = (gs,gw,gm,clean,wrong)
        matrices.append(torch.stack((old_mat,*(-lag.RADIUS*guard._unit(z) for z in g))).detach())
    if not len(q):
        return (base.center.new_empty((len(NEW)+1,0,two.C,two.C)),
                base.center.new_empty((0,two.C)),np.empty((0,3),np.float32))
    return torch.cat(matrices,1),torch.cat(residues),torch.cat(geometry).cpu().numpy()


@torch.no_grad()
def _forward(base, x, m, mats, correction):
    n = len(x)
    outputs, trace = [],[]

    def adjust(_layer, inputs):
        raw = inputs[0][:n]
        old = lag.patch_input(raw,mats[0])
        d = old-raw
        proposals = torch.stack([lag.patch_input(raw,z)-raw for z in mats[1:]])
        moved,use = two.equal_rotation(d,proposals)
        cap = d.flatten(1).square().mean(1).sqrt()
        old_bias = bias.bias_patches(old,correction,cap)
        common = old_bias-old
        all_new = raw+moved+common
        original = (old_bias-raw).flatten(1).square().mean(1).sqrt()
        rms = (all_new-raw).flatten(2).square().mean(2).sqrt()
        trace.extend((use,(rms-original).abs().amax()))
        return (torch.cat((old_bias,all_new.flatten(0,1))),)

    a = base.embedding.register_forward_pre_hook(adjust)
    b = base.risk_head.register_forward_pre_hook(lambda _m,z: outputs.append(z[0]))
    try:
        base(x.repeat(len(NEW)+1,1,1),m.repeat(len(NEW)+1,1,1))
    finally:
        b.remove()
        a.remove()
    return outputs[0].reshape(len(NEW)+1,n,-1),*trace


def forecast_track(tr, args):
    return teacher._mature(tr,args[2][2],args[2][3])


def predict(raw, signal, args, decoder, path, fitted, archived):
    base = args[0]
    tr,h,z,beta = guard._track(signal,args)
    q,key,initial,target = forecast_track(tr,args)
    mat,residual,geometry = directions(raw,signal,q,key,initial,target,args,decoder,path,fitted)
    corrected,_ = bias.past_bias(q,residual.cpu().numpy())
    b = torch.as_tensor(corrected,device=beta.device)*torch.as_tensor(args[2][3],device=beta.device)
    slots,_ = lag.delayed_slots(q,np.arange(len(z)))
    active = np.flatnonzero(slots>0)
    h,z = (torch.as_tensor(a,device=beta.device) for a in (h,z))
    curves = np.broadcast_to(archived,(len(NEW),len(z))).copy()
    used = np.zeros(len(z),bool)
    err, dose = 0.,0.
    for start in range(0,len(active),eos.BATCH):
        i = active[start:start+eos.BATCH]
        k = slots[i]-1
        x,m = lag._windows(base,signal,i)
        u,a,e = _forward(base,x,m,mat[:,k],b[k])
        v = (z[i]+((u-h[i])*beta).sum(-1)).sigmoid().cpu().numpy()
        use = a.cpu().numpy()
        curves[:,i[use]] = v[1:,use]
        used[i] = use
        err,dose = max(err,float(np.max(np.abs(v[0]-archived[i])))),max(dose,float(e))
    if err>3e-6 or dose>TOL or not np.isfinite(curves).all():
        raise ValueError(f'legacy reproduction / equal dose failed: {err:g} / {dose:g}')
    return curves,q+lag.H+1,used,geometry,err,dose


def historical_direction(lock,args,decoder,path,fitted):
    archive = read(REFERENCE/'historical.json')['orientation']
    base,net = args[:2]
    center,scale = args[2][:2]
    allowed = {'val':set(io23.source_roles()['val']),'cold':set(lock['ids'][:lag.STARTUP])}
    advanced = io23.advanced_times()
    result = {}
    for role,group in allowed.items():
        effect,applied = [],[]
        device = 'J-TEXT' if role=='val' else 'EAST'
        for r in archive[role]['manifest']:
            sid,q,t = (r[k] for k in ('shot','origin','action'))
            if sid not in group or t<q+lag.H+1:
                raise ValueError('invalid past risk witness')
            cold = role=='cold'
            raw = simple_data.read_shot(device,sid,rate_contract='nominal' if cold else 'strict')
            truth = simple_data.attach_outcome(raw,device,advanced)
            if r['positive']!=truth['is_disrupt'] or not truth['valid'][t]:
                raise ValueError('past risk witness differs from HDF5')
            if r['positive'] and (not truth['y'][t] or
                    round(float((truth['down']-raw['times'][t])*1000.),9)<=(30. if cold else 10.)):
                raise ValueError('historical positive witness is not in the timely supervision range')
            sig = lag.prepare_actual(raw,lag.STARTUP+1,lock['mapping']) if cold else raw
            # _mature looks up a context key for every complete origin.
            ends = two.origins(len(sig['values']))
            tr = lag.direct.track(net,sig,center,scale,ends)
            q0,key,p,y = forecast_track(tr,args)
            take = q0==q
            if take.sum()!=1:
                raise ValueError('historical forecast not complete')
            mat,_,_ = directions(raw,sig,q0[take],key[take],p[take],y[take],args,decoder,path,fitted)
            x,m = lag._windows(base,sig,[t])
            beta = args[6][:base.cfg.dim] if cold else base.risk_head.weight.detach().flatten()
            h,use,_ = _forward(base,x,m,mat,x.new_zeros(1,two.C))
            effect.append(((h[1:]-h[0])*beta).sum(-1)[:,0].cpu().numpy())
            applied.append(bool(use.item()))
        e = np.asarray(effect).reshape(-1,2,len(NEW))
        result[role] = dict(pairs=len(e),rotated_rows=sum(applied),scores={
            k:dict(both_correct=int(((x>0.)&(y<0.)).sum()),positive_larger=int((x>y).sum()),
                   mean_positive=float(x.mean()),mean_safe=float(y.mean()))
            for j,k in enumerate(NEW) for x,y in (e[:,:,j].T,)})
    return result


def run(fit,out):
    began = time.monotonic()
    lock,p,s,args,decoder,path,fitted = _load(fit)
    before = [state_digest(v) for v in args[:2]]
    (out/'shots').mkdir(parents=True)
    copyfile(Path(__file__),out/'executed_probe.py')
    fit_report = read(fit/'fit.json')
    write(out/'protocol.json',dict(scope='EAST360_TWO_PART_DIAGNOSTIC_FEEDBACK_EQUAL_ANGLE',
        order=lock['ids'],startup=p['startup'],calibration=p['calibration'],original=p['original'],arms=ARMS,
        source_sha256=p['source_sha256'],reference_seal=sha(REFERENCE/'seal.json'),reference=str(REFERENCE),
        ref_score=sha(REFERENCE/'per_shot.json'),fit=str(fit),fit_manifest=sha(fit/'fit.json'),
        fit_checkpoint=fit_report['checkpoint'],executed_sha256=sha(Path(__file__)),
        adjustment='existing temporary lag matrix; directions based on record BCE or conditional motion; '
                   'isolation removes only the first-order record-loss component at the issuing origin',
        wrong='rotate observed targets within each channel without moving observed membership',
        clock='same complete H25 released at q+26; same actual patch RMS and angle for candidates',
        history='held source VAL and cold direction audit does not fit or select anything',
        outcomes='no suffix outcomes before all warnings seal; no persistent learning'))
    hist = historical_direction(lock,args,decoder,path,fitted)
    write(out/'historical_directions.json',hist)
    print(dict(stage='held_directions',audit=hist),flush=True)
    hashes,witnesses,errs,all_geometry = {},{},[],[]
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed outcome')):
        for rank,sid in enumerate(lock['ids'],1):
            name = f'{rank:04d}_{sid}.npz'
            old = REFERENCE/'shots'/name
            if sha(old)!=s['curve_sha256'][name] or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                raise ValueError('changed raw HDF5 or legacy warning')
            with np.load(old,allow_pickle=False) as f:
                t,r,parent = (f[k] for k in ('times','release',PARENT))
            if rank<=p['startup']:
                a = np.broadcast_to(parent,(len(NEW),len(t)))
                used,e,d = np.zeros(len(t),bool),0.,0.
            else:
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                sig = lag.prepare_actual(raw,rank,lock['mapping'])
                np.testing.assert_array_equal(raw['times'],t)
                a,rr,used,g,e,d = predict(raw,sig,args,decoder,path,fitted,parent)
                np.testing.assert_array_equal(rr,r)
                all_geometry.append(g)
                if rank in (21,120,360):
                    end = min(len(t),330)
                    short = [{k:v[k][:end] for k in ('values','available')} for v in (raw,sig)]
                    bb,short_r,short_use,_,_,_ = predict(*short,args,decoder,path,fitted,parent[:end])
                    np.testing.assert_array_equal(short_r,r[r<end])
                    np.testing.assert_array_equal(short_use,used[:end])
                    np.testing.assert_allclose(bb,a[:,:end],atol=3e-6,rtol=0.)
                    for tau in p['calibration'].values():
                        np.testing.assert_array_equal(bb>=tau,a[:,:end]>=tau)
                    witnesses[str(sid)] = end
            output = out/'shots'/name
            with output.open('xb') as handle:
                np.savez_compressed(handle,release=r,rotated=used,**dict(zip(NEW,a,strict=True)))
            hashes[name] = sha(output)
            errs.append((e,d))
            if rank in (21,360) or rank%60==0:
                print(dict(stage='seal_two_part',rank=rank,seconds=round(time.monotonic()-began,2)),flush=True)
    if before!=[state_digest(v) for v in args[:2]]:
        raise ValueError('the frozen network weights changed')
    z = np.concatenate(all_geometry)
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
        seconds=time.monotonic()-began,max_legacy_error=max(v[0] for v in errs),
        max_dose_error=max(v[1] for v in errs),blocks=len(z),
        geometry=dict(pre_cos_median=float(np.median(z[:,0])),
                      pre_cos_negative=int((z[:,0]<0.).sum()),
                      pre_cos_abs_gt_half=int((np.abs(z[:,0])>.5).sum()),
                      post_cos_abs_p99=float(np.quantile(np.abs(z[:,1]),.99)),
                      retained_motion_norm_median=float(np.median(z[:,2])))))


def score(out):
    p,s = [read(out/k) for k in ('protocol.json','seal.json')]
    if (p['executed_sha256']!=sha(out/'executed_probe.py') or p['reference_seal']!=sha(REFERENCE/'seal.json')
            or p['ref_score']!=sha(REFERENCE/'per_shot.json') or len(s['curve_sha256'])!=360):
        raise ValueError('sealed predictions changed')
    ref = read(REFERENCE/'seal.json')['curve_sha256']
    legacy = read(REFERENCE/'per_shot.json')
    rows = {band:{k:[] for k in ARMS} for band in p['calibration']}
    applied, shots, late = 0,0,0
    for rank,sid in enumerate(p['order'],1):
        name = f'{rank:04d}_{sid}.npz'
        a,b = (root/'shots'/name for root in (out,REFERENCE))
        if sha(a)!=s['curve_sha256'][name] or sha(b)!=ref[name]:
            raise ValueError('changed sealed curve')
        d,down,t = hdf._raw_truth(sid)
        with np.load(a,allow_pickle=False) as new,np.load(b,allow_pickle=False) as old:
            np.testing.assert_array_equal(t,old['times'])
            np.testing.assert_array_equal(new['release'],old['release'])
            u = new['rotated']
            applied += int(u.sum())
            shots += int(u.any())
            if d and not legacy['0.2'][PARENT][rank-1]['timely']:
                lead = np.round((down-t)*1000.,9)
                late += int(np.any(u&(lead>30.)&(lead<=125.)))
            for band,tau in p['calibration'].items():
                for k in ARMS:
                    v = old[k] if k in OLD else new[k]
                    if k in NEW:
                        np.testing.assert_array_equal(v[~u],old[PARENT][~u])
                    item = dict(shot=sid,disruptive=d,down_s=down,**score_curve(
                        t,v,d,down,p['original'] if rank<=p['startup'] else tau))
                    if k in OLD and item!=legacy[band][k][rank-1]:
                        raise ValueError('legacy result differs from raw HDF5')
                    rows[band][k].append(item)
    write(out/'per_shot.json',rows)
    write(out/'lead_sensitivity.json',report([out/'per_shot.json'],ARMS))
    metrics = {k:{n:timing_decomposition(x) for n,x in v.items()} for k,v in rows.items()}
    changes = {k:{f'{a}__to__{b}':flips(v[a],v[b]) for a,b in PAIRS} for k,v in rows.items()}
    write(out/'result.json',dict(status='COMPLETED_TWO_PART_DIAGNOSTIC_FEEDBACK',metrics=metrics,
        paired=changes,raw_verified=360*len(ARMS)*len(metrics),geometry=s['geometry'],
        activity=dict(updated_points=applied,updated_shots=shots,late_with_opportunity=late)))
    print({k:{n:(x['tp'],x['fp']) for n,x in v.items()} for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit',type=Path)
    parser.add_argument('out',type=Path)
    a = parser.parse_args()
    run(a.fit.resolve(),a.out.resolve())
    score(a.out.resolve())
