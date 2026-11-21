"""Test risk-aligned forecast-error updates with the sealed shape readout fixed."""

import argparse
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_innovation_metric_20261111 as step
from . import commissioning_weak_shape_data_20261111 as shape
from . import commissioning_weak_shape_source_20261111 as weak
from . import io23, simple_data
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE, WIDTH
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


MODES = ('fixed', 'ordinary', 'aligned', 'shifted')
FIRST_EFFECT = FIRST + HORIZON + 1
RIDGE = .015
RISK_STEPS = 256
RISK_LR = .02
MATCH_STEP = 16


def fit_decoder(h, y, observed, *, ridge=RIDGE):
    """A direct five-bin prediction from the frozen representation."""
    x = np.column_stack((h, np.ones(len(h)))).astype(np.float64)
    penalty = np.diag([1.] * h.shape[1] + [0.]) * ridge
    cols = []
    for j, take in enumerate(observed.T):
        if np.count_nonzero(take) <= h.shape[1]:
            raise ValueError(f'insufficient J-TEXT support for forecast coordinate {j}')
        a, b = x[take], y[take, j].astype(np.float64)
        cols.append(np.linalg.solve(a.T @ a + len(a)*penalty, a.T @ b))
    coef = np.stack(cols).astype(np.float32)
    return coef[:, :-1].copy(), coef[:, -1].copy()


def bounded_metric(a, b):
    """Two signed rank-one changes to a positive definite identity metric."""
    u = a / a.norm().clamp_min(1.)
    v = b / b.norm().clamp_min(1.)
    return torch.eye(len(a),dtype=a.dtype,device=a.device) + .90*(u.outer(u)-v.outer(v))


def separation_loss(delta_positive, delta_safe):
    """Reward increased disruption-safe separation, not a common risk shift."""
    return F.softplus((delta_safe - delta_positive) / .25).mean()


@torch.no_grad()
def coordinates(net, signal, center, scale, *, all_points):
    z,m,c,cm = weak.shot_coordinates(signal,center,scale)
    q = np.arange(FIRST,len(z)-HORIZON,STRIDE)
    ends = np.arange(len(z)) if all_points else q
    x,context = shape.inputs(z,m,c,cm,ends)
    pre,key = net.design(torch.from_numpy(x))
    y,roll,eligible = shape.feedback(z,m,q)
    available = step.observed_mask(z,m,q)
    return pre,key,context,x[:,-3:].astype(bool).all(1),q,y,roll,eligible,available


def source_examples(net, ids, center, scale, role):
    chunks, used = [], []
    for i,sid in enumerate(ids,1):
        raw = simple_data.read_shot('J-TEXT',sid)
        pre,_,_,_,q,y,_,eligible,mask = coordinates(net,raw,center,scale,all_points=False)
        # Only pre-terminal signal windows contribute. Source risk labels are
        # not forecast targets.
        d,down = io23.read_outcome('J-TEXT',sid)
        if d:
            eligible &= raw['times'][q+HORIZON] <= down + 1e-9
        if eligible.any():
            chunks.append((pre.tanh().numpy()[eligible],y[eligible],mask[eligible]))
            used.append(int(sid))
        if i%450==0 or i==len(ids):
            print(dict(stage=f'forecast_{role}',shots=i,total=len(ids)),flush=True)
    return tuple(np.concatenate(c) for c in zip(*chunks,strict=True)),used


def forecast_error(h, y, observed, decoder, mean):
    predicted = h@decoder[0].T + decoder[1]
    residual = np.where(observed,predicted-y,0).astype(np.float64)
    reference = np.where(observed,mean-y,0).astype(np.float64)
    return dict(windows=len(h),
                predicted_sse=float(np.square(residual).sum()),
                mean_sse=float(np.square(reference).sum()))


def track(base,net,signal,center,scale,decoder,mean,*,need_shifted):
    raw,_ = features(base,signal,'cuda')
    p,k,ctx,matchable,q,y,roll,eligible,available = coordinates(net,signal,center,scale,all_points=True)
    p[:FIRST] = 0.
    pre,key = p.numpy(),k.numpy()
    fixed = p.tanh().numpy()
    out, counts = {}, {}
    for name,target in (('real',y),('shifted',roll)) if need_shifted else (('real',y),):
        g = step.anchor_steps(pre[q],key[q],target,available,*decoder)
        out[name], counts[name] = step.released_displacement(key,q,g,eligible)
    report = forecast_error(fixed[q[eligible]],y[eligible],available[eligible],decoder,mean)
    return dict(raw=raw,pre=pre,fixed=fixed,context=ctx,matchable=matchable,
                delta=out,updates=counts,forecast=report)


def make_matches(rows, scale):
    """For each positive anchor, use a context-matched window of each safe shot."""
    groups = []
    for sid,row in rows.items():
        if row['disruptive']:
            candidates = row['timely'] & row['matchable']
            candidates[:FIRST_EFFECT] = False
            eligible = np.flatnonzero(candidates)
            if len(eligible):
                groups.append((sid,np.unique(eligible[[0,len(eligible)//2,-1]])))
    if not groups:
        raise ValueError('no matchable timely EAST cold-start endpoints')
    pos_sid,pos_idx,safe_sid,safe_idx,dist = [],[],[],[],[]
    for sid,anchor in groups:
        a = rows[sid]['context'][anchor].astype(np.float64)/scale
        for nid,negative in rows.items():
            if negative['disruptive']:
                continue
            pool = np.flatnonzero(negative['matchable'] & (np.arange(len(negative['pre']))>=FIRST_EFFECT))
            pool = pool[::MATCH_STEP]
            if not len(pool):
                raise ValueError(f'no matchable windows for ended safe EAST shot {nid}')
            b = negative['context'][pool].astype(np.float64)/scale
            d = ((a[:,None,:]-b[None,:,:])**2).mean(-1)
            nearest = d.argmin(1)
            pos_sid.extend([sid]*len(anchor))
            pos_idx.extend(anchor.tolist())
            safe_sid.extend([nid]*len(anchor))
            safe_idx.extend(pool[nearest].tolist())
            dist.extend(np.sqrt(d[np.arange(len(anchor)),nearest]).tolist())
    return pos_sid,pos_idx,safe_sid,safe_idx,dist


def fit_metric(rows,coef,cutoff,scale):
    p,ip,n,in_,distance = make_matches(rows,scale)
    all_shots = list(rows)
    idx = {sid:i for i,sid in enumerate(all_shots)}
    sizes = [len(rows[s]['pre']) for s in all_shots]
    offsets = np.r_[0,np.cumsum(sizes[:-1])]
    tp = torch.as_tensor([offsets[idx[s]]+i for s,i in zip(p,ip,strict=True)],device='cuda')
    tn = torch.as_tensor([offsets[idx[s]]+i for s,i in zip(n,in_,strict=True)],device='cuda')
    a = torch.from_numpy(np.concatenate([rows[s]['pre'] for s in all_shots])).cuda()
    h = torch.from_numpy(np.concatenate([rows[s]['fixed'] for s in all_shots])).cuda()
    delta = torch.from_numpy(np.concatenate([rows[s]['delta']['real'] for s in all_shots])).cuda()
    raw = torch.from_numpy(np.concatenate([rows[s]['static_logit'] for s in all_shots])).cuda()
    beta = coef[-WIDTH-1:-1]
    eligible = [torch.from_numpy(np.flatnonzero(rows[s]['timely'])+o).cuda()
                for s,o in zip(all_shots,offsets,strict=True) if rows[s]['disruptive']]
    safe = [slice(o,o+k) for s,o,k in zip(all_shots,offsets,sizes,strict=True)
            if not rows[s]['disruptive']]
    if not eligible or not safe or any(not len(i) for i in eligible):
        raise ValueError('ended EAST fit requires timely positive and full safe records')
    base = torch.eye(WIDTH,device='cuda')
    bnd = float(np.log(cutoff/(1.-cutoff)))
    rnd = torch.Generator(device='cuda').manual_seed(131)
    seed = .15*torch.randn(WIDTH,device='cuda',generator=rnd)
    u,v = torch.nn.Parameter(seed.clone()),torch.nn.Parameter(seed.clone())
    opt = torch.optim.Adam([u,v],lr=RISK_LR)

    def components(metric):
        change = ((a + delta@metric.T).tanh() - h) @ beta
        updated = raw + change
        pos = torch.stack([updated[i].amax() for i in eligible])
        neg = torch.stack([updated[i].amax() for i in safe])
        alarm = .5*(F.softplus(bnd-pos).mean() + F.softplus(neg-bnd).mean())
        paired = separation_loss(change[tp],change[tn])
        return alarm,paired,(metric-base).square().sum(),(change[tp]-change[tn]).mean()

    def receipt(metric):
        with torch.no_grad():
            vals = components(metric)
        return dict(zip(('warning_loss','separation_loss','squared_identity_distance','mean_pair_gain'),
                        (float(v) for v in vals),strict=True))

    start = receipt(base)
    for i in range(1,RISK_STEPS+1):
        opt.zero_grad(set_to_none=True)
        alert,pair,penalty,_ = components(bounded_metric(u,v))
        loss = alert + .5*pair + .025*penalty
        loss.backward()
        opt.step()
        if i%128==0:
            print(dict(stage='align_ended20',step=i,loss=float(loss.detach())),flush=True)
    value = bounded_metric(u,v).detach()
    eig = torch.linalg.eigvalsh(value)
    if not torch.isfinite(value).all() or eig.min()<.09999 or eig.max()>1.90001:
        raise ValueError('fitted forecast metric outside declared eigenvalue bounds')
    return value.cpu().numpy(), dict(
        identity=start,learned=receipt(value),eigenvalues=eig.cpu().tolist(),pair_count=len(distance),
        median_match_rms_distance=float(np.median(distance)),training_shots=all_shots,
        matches=[[int(a),int(b),int(c),int(d)] for a,b,c,d in zip(p,ip,n,in_,strict=True)])


@torch.no_grad()
def views(row,metric):
    p = row['pre']
    d = row['delta']['real']
    return dict(fixed=row['fixed'], ordinary=torch.from_numpy(p+d).tanh().numpy(),
                aligned=torch.from_numpy(p+d@metric.T).tanh().numpy(),
                shifted=torch.from_numpy(p+row['delta']['shifted']@metric.T).tanh().numpy())


def run(reference,out):
    started = time.monotonic()
    configure()
    inputs = load_inputs()
    previous,seal,cal = (read(reference/k) for k in ('protocol.json','seal.json','calibration.json'))
    if (previous['order']!=inputs['ids'] or previous['startup']!=STARTUP
            or previous['scope']!='EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or previous['source_sha256']!=PARENT_SHA256
            or len(seal['curve_sha256'])!=360
            or any(sha(reference/k)!=v for k,v in seal['artifacts'].items())):
        raise ValueError('sealed context-matched reference and identical order required')
    source = Path(previous['source_prior'])
    spec,source_result = (read(source/k) for k in ('protocol.json','result.json'))
    if (spec['code_sha256']!=weak.code_bindings()
            or any(sha(source/k)!=v for k,v in previous['source_assets'].items())):
        raise ValueError('shape source assets do not match the sealed parent')
    base,ckpt = load_parent(DEFAULT_PARENT)
    center,scale = (p.detach().numpy().copy() for p in (base.center,base.scale))
    base.cuda().requires_grad_(False)
    original = previous['original']
    base_digest = state_digest(base)
    if original!=ckpt['threshold'] or base_digest!=inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified J-TEXT parent changed')
    net = LagState(source_result['inputs'],source_result['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz',allow_pickle=False) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    shape_digest = state_digest(net)
    with np.load(reference/'readout.npz',allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    thresholds = cal['cutoffs']
    ids = inputs['ids']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(
        scope='EAST360_FIXED_SHAPE_HEAD_RISK_ALIGNED_INNOVATION_DEVELOPMENT',order=ids,startup=STARTUP,
        calibration=thresholds,original=original,arms=MODES,first_effect=FIRST_EFFECT,
        source_sha256=PARENT_SHA256, reference=str(reference),
        reference_assets={k:sha(reference/k) for k in ('protocol.json','readout.npz','calibration.json','seal.json')},
        source_assets=previous['source_assets'],data_bindings=inputs['bound'],
        producer_sha256=sha(Path(__file__)),step_code_sha256=sha(Path(step.__file__)),
        output_curve_code_sha256=sha(Path(unwrap(scores).__code__.co_filename)),
        forecast='frozen J-TEXT TRAIN direct 5x5 ridge in frozen shape coordinates',
        inner='masked forecast innovation gradient at zero fast-state anchor',
        fit_ids=ids[:STARTUP],max_epochs=RISK_STEPS,optimizer_lr=RISK_LR,
        retention=step.RETAIN,base_step=step.STEP,stabilizer=step.STABILIZER,
        persistent_suffix_updates=0))
    roles = io23.source_roles()
    train,train_ids = source_examples(net,roles['train'],center,scale,'train')
    decoder = fit_decoder(*train)
    _,y,obs = train
    mean = np.divide(np.where(obs,y,0).sum(0),obs.sum(0)).astype(np.float32)
    val,val_ids = source_examples(net,roles['val'],center,scale,'val')
    source_fit = dict(train_ids=train_ids,val_ids=val_ids,
                      train=forecast_error(*train,decoder,mean),val=forecast_error(*val,decoder,mean))

    def sealed(i,sid):
        name = f'{i:04d}_{sid}.npz'
        path = reference/'shots'/name
        if sha(path)!=seal['curve_sha256'][name]:
            raise ValueError(f'archived fixed curve changed at {sid}')
        with np.load(path,allow_pickle=False) as f:
            return name,f['times'],f['fixed']

    def raw(sid,t):
        if sha(io23.path_for('EAST',sid))!=inputs['hashes'][str(sid)]:
            raise ValueError(f'locked EAST data changed: {sid}')
        shot = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        np.testing.assert_array_equal(shot['times'],t)
        return shot

    cold={}
    for i,sid in enumerate(ids[:STARTUP],1):
        _,t,_ = sealed(i,sid)
        signal = prepare_actual(raw(sid,t),STARTUP+1,inputs['mapping'])
        data = track(base,net,signal,center,scale,decoder,mean,need_shifted=False)
        d,down = io23.read_outcome('EAST',sid)
        y,_ = io23.labels_at(t,d,down,125.)
        timely = y.astype(bool) & (np.round((down-t)*1000.,9)>30) if d else y.astype(bool)
        logit = torch.from_numpy(np.concatenate((data['raw'],data['fixed']),1)).cuda()@coef[:-1]+coef[-1]
        cold[sid] = dict(data,timely=timely,disruptive=d,static_logit=logit.cpu().numpy())
    if sum(v['disruptive'] for v in cold.values())!=8:
        raise ValueError('same ended 8 disruptive + 12 safe cold-start required')
    metric,fit = fit_metric(cold,coef,thresholds['0.2'],np.asarray(source_result['match_scale']))
    with (out/'fit.npz').open('xb') as f:
        np.savez(f,forecast_weight=decoder[0],forecast_bias=decoder[1],forecast_mean=mean,metric=metric)
    write(out/'fit.json',dict(metric_fit=fit,source_forecast=source_fit))
    fit_assets = {k:sha(out/k) for k in ('fit.npz','fit.json')}
    print(dict(stage='fit_complete',seconds=time.monotonic()-started,pairs=fit['pair_count'],
               identity=fit['identity'],learned=fit['learned']),flush=True)

    hashes,witness,dynamics = {},{},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for i,sid in enumerate(ids,1):
            name,t,fixed = sealed(i,sid)
            if i<=STARTUP:
                curves = dict.fromkeys(MODES,fixed)
            else:
                signal = prepare_actual(raw(sid,t),i,inputs['mapping'])
                row = track(base,net,signal,center,scale,decoder,mean,need_shifted=True)
                curves = scores(row['raw'],views(row,metric),coef,MODES)
                np.testing.assert_array_equal(curves['fixed'],fixed)
                for k in MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:FIRST_EFFECT],fixed[:FIRST_EFFECT])
                dynamics[str(sid)] = dict(updates=row['updates'],forecast=row['forecast'])
                if i in (21,120,360):
                    limited = {k:signal[k][:243] for k in ('values','available')}
                    prefix = track(base,net,limited,center,scale,decoder,mean,need_shifted=True)
                    short = scores(prefix['raw'],views(prefix,metric),coef,MODES)
                    witness[str(sid)] = {}
                    for k,z in short.items():
                        full = curves[k][:len(z)]
                        np.testing.assert_allclose(z,full,rtol=0,atol=2e-6)
                        for cutoff in thresholds.values():
                            np.testing.assert_array_equal(z>=cutoff,full>=cutoff)
                        witness[str(sid)][k] = float(np.max(abs(z-full)))
            path = out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves)
            hashes[name]=sha(path)
            if i%60==0:
                print(dict(stage='sealed',shots=i,seconds=time.monotonic()-started),flush=True)
    if (state_digest(net)!=shape_digest or state_digest(base)!=base_digest
            or any(sha(out/k)!=v for k,v in fit_assets.items())):
        raise ValueError('persistent state changed inside the sealed suffix')
    write(out/'seal.json',dict(curve_sha256=hashes,fit_assets=fit_assets,prefix_witnesses=witness))
    write(out/'dynamics_by_shot.json',dynamics)
    per = {c:{k:[] for k in MODES} for c in CAPS}
    for i,sid in enumerate(ids,1):
        path = out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path)!=hashes[path.name]:
            raise ValueError('post-seal curve changed')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,group in per.items():
                for k in MODES:
                    scored = score_curve(f['times'],f[k],d,down,original if i<=STARTUP else thresholds[c])
                    group[k].append(dict(shot=sid,disruptive=d,down_s=down,**scored))
    ref = read(reference/'per_shot.json')
    if any(g['fixed']!=ref[c]['fixed'] for c,g in per.items()):
        raise ValueError('old frozen reference scores not reproduced')
    metrics = {c:{k:timing_decomposition(rows) for k,rows in group.items()} for c,group in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(status='COMPLETED_FIXED_HEAD_INNOVATION_EAST360_DEVELOPMENT',metrics=metrics,
        paired_vs_fixed={c:{k:flips(g['fixed'],g[k]) for k in MODES[1:]} for c,g in per.items()},
        aligned_vs_shifted={c:flips(g['shifted'],g['aligned']) for c,g in per.items()},
        aligned_vs_ordinary={c:flips(g['ordinary'],g['aligned']) for c,g in per.items()},
        update_windows={k:sum(v['updates'][k] for v in dynamics.values()) for k in ('real','shifted')},
        suffix_forecast={k:sum(v['forecast'][k] for v in dynamics.values())
                         for k in ('windows','predicted_sse','mean_sse')},
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in group.items()} for c,group in metrics.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    args = cli.parse_args()
    run(args.reference.resolve(),args.output.resolve())
