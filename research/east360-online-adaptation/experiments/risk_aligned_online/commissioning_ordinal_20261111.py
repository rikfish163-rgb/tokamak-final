"""Teach a delayed within-shot update to recognize earlier ordered progression."""

import argparse
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_ordinal_pairs_20261111 as pairs
from . import commissioning_surplus_20261111 as surplus
from . import commissioning_weak_shape_source_20261111 as weak
from . import io23, simple_data
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


MODES = ('fixed','late','scrambled','ordered','ordered_lag')
SEED, STEPS, BATCH = 131, 512, 128


def ordered_difference(disruption, safe):
    """Near-minus-far, after subtracting the matched safe change."""
    return (disruption[...,1] - disruption[...,0]) - (safe[...,1] - safe[...,0])


def shotwise_order_control(shots):
    _, inv = np.unique(shots,return_inverse=True)
    count = int(inv.max())+1
    direction = np.ones(count,np.float32)
    direction[:count//2] = -1
    np.random.default_rng(SEED).shuffle(direction)
    return direction[inv]


def cue_direction(rows, matched, mode):
    pos,neg,_,_ = matched
    a,b = rows[mode][pos],rows[mode][neg]
    return ((a[:,1]-a[:,0])-(b[:,1]-b[:,0])).astype(np.float64)


def cue_probe_fit(rows, matched):
    x = cue_direction(rows,matched,'intra')
    _,inv,n = np.unique(rows['shot'][matched[0]],return_inverse=True,return_counts=True)
    w = 1./n[inv]
    w /= w.sum()
    gram = x.T @ (w[:,None]*x) + .002*np.eye(x.shape[1])
    return {k:np.linalg.solve(gram,x.T@(w*y)) for k,y in
            (('ordered',np.ones(len(x))),('scrambled',shotwise_order_control(rows['shot'][matched[0]])))}


def collect(net,ids,advanced,center,scale,noise,role):
    rows = []
    groups = np.asarray(surplus.feedback.GROUPS)
    for i,sid in enumerate(ids,1):
        raw = simple_data.read_shot('J-TEXT',sid)
        data = simple_data.attach_outcome(raw,'J-TEXT',advanced)
        d,down,t = data['is_disrupt'],data['down'],data['times']
        indices = pairs.endpoints(t,data['valid'],d,down,data['horizon'],
                                  first_effect=surplus.FIRST_EFFECT)
        if len(indices):
            trace = surplus.trace(net,raw,center,scale,indices.ravel())
            obs = trace['observed']
            ok = np.stack([obs[:,groups==k].any(1) for k in range(4)],1).sum(1)>=2
            first = trace['origins'][ok][:1]+surplus.HORIZON+1
            # Require the far endpoint to have a matured block as well.
            if len(first):
                mask = (indices[:,0]>=first[0]) & trace['matchable'].reshape(-1,2).all(1)
                if mask.any():
                    ds,_ = surplus.state(trace,noise)
                    near = indices[:,1]
                    ctx = trace['context'].reshape(-1,2,7)
                    tail = (not d) & (t[near]>=t[-1]-weak.TAIL_S) & (ctx[:,1,2]<0)
                    lead = np.round((down-t[near])*1000.,9) if d else np.zeros(len(near))
                    if d:
                        assert not data['y'][indices[mask]].any()
                    rows.append(dict(pre=trace['pre'].reshape(-1,2,surplus.WIDTH)[mask],
                                     intra=ds['intra'].reshape(-1,2,surplus.PHI_DIM)[mask],
                                     lag=ds['lag'].reshape(-1,2,surplus.PHI_DIM)[mask],
                                     context=ctx[mask],lead=lead[mask],tail=tail[mask],
                                     shot=np.full(int(mask.sum()),sid),
                                     disruptive=np.full(int(mask.sum()),d),indices=indices[mask]))
        if i%450==0 or i==len(ids):
            print(dict(stage=f'collect_{role}',shots=i,total=len(ids)),flush=True)
    if not rows:
        raise ValueError(f'no early trajectories in {role}')
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def teach(rows,net,coef,initial,matched):
    pos,neg,_,_ = matched
    beta,target = surplus.directions(net,coef)
    _,inv,n = np.unique(rows['shot'][pos],return_inverse=True,return_counts=True)
    w = torch.from_numpy(1./n[inv]).cuda()
    pp,pn = (torch.from_numpy(i).cuda() for i in (pos,neg))
    p = torch.from_numpy(rows['pre']).cuda()
    delta = torch.from_numpy(rows['intra']).cuda()
    frozen = p.tanh()
    base = frozen@beta+net.risk.bias.detach().cuda()
    initial = initial.cuda()
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    draw = torch.multinomial(w,STEPS*BATCH,True,generator=rng).reshape(STEPS,BATCH)
    trained = {}
    receipt = {}
    null = shotwise_order_control(rows['shot'][pos])
    for name,direction in (('scrambled',null),('ordered',np.ones(len(pos),np.float32))):
        sign = torch.from_numpy(direction).cuda()
        param = torch.nn.Parameter(initial.clone())
        opt = torch.optim.Adam([param],lr=.015)
        for j,i in enumerate(draw,1):
            opt.zero_grad(set_to_none=True)
            model = surplus.bounded_map(param)
            indices = torch.cat((pp[i],pn[i]))
            g = (surplus.project(p[indices],delta[indices],model)-frozen[indices])@target
            gp,gn = g.split(len(i))
            direction_gain = ordered_difference(gp,gn)
            chronology = F.softplus((.20-sign[i]*direction_gain)/.25).mean()
            # Both near and far windows are included in the common loss.
            separation = F.softplus(-(gp-gn)/.25).mean()
            protection = F.relu(gn).square().mean()
            distance = (model-initial).square().sum()
            loss = .50*chronology + .30*separation + .10*protection + .015*distance
            loss.backward()
            opt.step()
            if j%256==0:
                print(dict(stage=f'teach_{name}',steps=j,loss=float(loss.detach())),flush=True)
        m = surplus.bounded_map(param).detach()
        if not torch.isfinite(m).all():
            raise ValueError('nonfinite early progression mapper')
        with torch.no_grad():
            g = (surplus.project(p,delta,m)-frozen)@target
            receipt[name]=dict(norm=float(m.norm()),change_from_late=float((m-initial).norm()),
                               train_mean_order_gain=float(ordered_difference(g[pp],g[pn]).mean()),
                               train_mean_matched_risk_gap=float(((base+g)[pp,1]-(base+g)[pn,1]).mean()))
        trained[name]=m.cpu()
    receipt['scrambled_reversed_shots']=np.unique(rows['shot'][pos[null<0]]).astype(int).tolist()
    return trained,receipt


@torch.no_grad()
def source_hold(rows,net,coef,maps,matched):
    pos,neg,tail,distance = matched
    p = torch.from_numpy(rows['pre']).cuda()
    h = p.tanh()
    beta,target = surplus.directions(net,coef)
    base = h@beta+net.risk.bias.detach().cuda()
    z = {'fixed':base.cpu().numpy()}
    for k in MODES[1:]:
        trajectory = torch.from_numpy(rows['lag' if k=='ordered_lag' else 'intra']).cuda()
        m = maps['ordered' if k=='ordered_lag' else k].cuda()
        z[k] = (base+(surplus.project(p,trajectory,m)-h)@target).cpu().numpy()
    report={}
    for name,sel in (('all',np.ones(len(pos),bool)),('falling_tail',tail),
                     ('lead_60_to_125',rows['lead'][pos]>60.)):
        a,b = pos[sel],neg[sel]
        report[name]=dict(
            n=len(a),
            nearer_vs_matched_safe={k:int((v[a,1]>v[b,1]).sum()) for k,v in z.items()},
            progression_vs_matched_safe={k:int((ordered_difference(v[a],v[b])>0).sum()) for k,v in z.items()},
            mean_order_gain={k:(float(ordered_difference(v[a]-z['fixed'][a],v[b]-z['fixed'][b]).mean())
                                if len(a) else None)
                             for k,v in z.items()})
    return dict(summary=report,positive_shots=np.unique(rows['shot'][pos]).astype(int).tolist(),
                safe_donors=np.unique(rows['shot'][neg]).astype(int).tolist(),
                median_rms_distance=float(np.median(distance)),
                pairs=[[int(rows['shot'][a]),int(rows['shot'][b]),rows['indices'][a].tolist(),
                        rows['indices'][b].tolist(),float(rows['lead'][a]),bool(c),float(d)]
                       for a,b,c,d in zip(pos,neg,tail,distance,strict=True)])


@torch.no_grad()
def encode(base,net,signal,center,scale,noise,maps,coef):
    raw,_ = features(base,signal,'cuda')
    trajectory = surplus.trace(net,signal,center,scale,np.arange(len(raw)))
    ds,audit = surplus.state(trajectory,noise)
    p = torch.from_numpy(trajectory['pre'])
    states = {'fixed':p.tanh().numpy()}
    for k in MODES[1:]:
        move = torch.from_numpy(ds['lag' if k=='ordered_lag' else 'intra'])
        states[k]=surplus.project(p,move,maps['ordered' if k=='ordered_lag' else k]).numpy()
    return scores(raw,states,coef,MODES),audit


def run(reference,out):
    started=time.monotonic()
    configure()
    locked=load_inputs()
    p,seal=(read(reference/k) for k in ('protocol.json','seal.json'))
    ids=locked['ids']
    if (p['scope']!='EAST360_FIXED_SHAPE_HEAD_SIGNED_SURPLUS_DEVELOPMENT' or p['order']!=ids
            or p['source_sha256']!=PARENT_SHA256 or p['startup']!=STARTUP
            or len(seal['curve_sha256'])!=360 or p['producer_sha256']!=sha(Path(surplus.__file__))
            or p['feedback_sha256']!=sha(Path(surplus.feedback.__file__))
            or any(sha(reference/k)!=v for k,v in seal['fit_assets'].items())):
        raise ValueError('sealed surplus source and fixed stream required')
    parent=Path(p['reference'])
    if any(sha(parent/k)!=v for k,v in p['reference_assets'].items()):
        raise ValueError('sealed fixed readout assets changed')
    source=Path(read(parent/'protocol.json')['source_prior'])
    if any(sha(source/k)!=v for k,v in p['source_assets'].items()):
        raise ValueError('sealed source checkpoint changed')
    spec,held_info=(read(source/k) for k in ('protocol.json','result.json'))
    if spec['code_sha256']!=weak.code_bindings():
        raise ValueError('source shape implementation changed')
    net=LagState(held_info['inputs'],held_info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz',allow_pickle=False) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    with np.load(parent/'readout.npz',allow_pickle=False) as f:
        coef=torch.from_numpy(f['coef']).cuda()
    with np.load(reference/'fit.npz',allow_pickle=False) as f:
        late,noise=torch.from_numpy(f['mapper']),f['noise_scale']
    base,ckpt=load_parent(DEFAULT_PARENT)
    center,scale=(v.detach().numpy().copy() for v in (base.center,base.scale))
    base.cuda().requires_grad_(False)
    frozen=state_digest(net)
    original=p['original']
    cutoffs=p['calibration']
    if ckpt['threshold']!=original or state_digest(base)!=locked['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source parent changed')
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(
        scope='EAST360_FIXED_HEAD_EARLY_ORDINAL_TEACHING_DEVELOPMENT',order=ids,startup=STARTUP,
        original=original,calibration=cutoffs,arms=MODES,first_effect=surplus.FIRST_EFFECT,
        reference=str(reference),reference_assets={k:sha(reference/k) for k in ('protocol.json','seal.json','fit.npz')},
        source_sha256=PARENT_SHA256,source_assets=p['source_assets'],data_bindings=locked['bound'],
        source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        code_sha256={f.name:sha(f) for f in (Path(__file__),Path(pairs.__file__),Path(surplus.__file__),
                      Path(surplus.feedback.__file__),Path(unwrap(scores).__code__.co_filename))},
        no_target_fitting=True,persistent_suffix_updates=0,
        fit=dict(seed=SEED,steps=STEPS,batch=BATCH,optimizer_lr=.015,early_lead_ms='(30,125]',
                 before_original_source_window=True,temporal_gap_ms=32)))
    roles,advanced=io23.source_roles(),io23.advanced_times()
    match_scale=np.asarray(held_info['match_scale'])
    train=collect(net,roles['train'],advanced,center,scale,noise,'train')
    matched=pairs.match(train['context'],train['disruptive'],train['tail'],match_scale)
    cue_maps=cue_probe_fit(train,matched)
    new,fit=teach(train,net,coef,late,matched)
    maps=dict(late=late,**new)
    pp,nn,_,dist=matched
    fit.update(train_ids=np.unique(train['shot']).astype(int).tolist(),pairs=len(pp),
               disruptive_shots=np.unique(train['shot'][pp]).astype(int).tolist(),
               safe_donors=np.unique(train['shot'][nn]).astype(int).tolist(),
               median_rms_match=float(np.median(dist)),
               train_lead=[float(train['lead'][pp].min()),float(train['lead'][pp].max())])
    del train
    val=collect(net,roles['val'],advanced,center,scale,noise,'val')
    vm=pairs.match(val['context'],val['disruptive'],val['tail'],match_scale)
    val_result=source_hold(val,net,coef,maps,vm)
    val_result['val_ids']=np.unique(val['shot']).astype(int).tolist()
    val_result['cue_only_progression_probe']={
        k:{j:int((cue_direction(val,vm,j)@w>0).sum()) for j in ('intra','lag')}
        for k,w in cue_maps.items()}
    write(out/'source_validation.json',val_result)
    write(out/'fit.json',fit)
    with (out/'fit.npz').open('xb') as f:
        np.savez(f,**{k:m.numpy() for k,m in maps.items()},noise_scale=noise)
    assets={k:sha(out/k) for k in ('source_validation.json','fit.json','fit.npz')}
    print(dict(stage='source_held',**val_result['summary']),flush=True)
    hashes,witness,dynamics={},{},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for i,sid in enumerate(ids,1):
            name=f'{i:04d}_{sid}.npz'
            archived=reference/'shots'/name
            if sha(archived)!=seal['curve_sha256'][name]:
                raise ValueError(f'sealed reference changed at {sid}')
            with np.load(archived,allow_pickle=False) as f:
                t,fixed,late_curve=(f[k] for k in ('times','fixed','intra'))
            if i<=STARTUP:
                curves=dict.fromkeys(MODES,fixed)
            else:
                if sha(io23.path_for('EAST',sid))!=locked['hashes'][str(sid)]:
                    raise ValueError(f'raw data changed at {sid}')
                raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],t)
                signal=prepare_actual(raw,i,locked['mapping'])
                curves,dynamics[str(sid)]=encode(base,net,signal,center,scale,noise,maps,coef)
                np.testing.assert_array_equal(curves['fixed'],fixed)
                np.testing.assert_array_equal(curves['late'],late_curve)
                for k in MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:surplus.FIRST_EFFECT],fixed[:surplus.FIRST_EFFECT])
                if i in (21,120,360):
                    small={k:signal[k][:243] for k in ('values','available')}
                    short,_=encode(base,net,small,center,scale,noise,maps,coef)
                    witness[str(sid)]={}
                    for k,v in short.items():
                        full=curves[k][:len(v)]
                        np.testing.assert_allclose(v,full,atol=2e-6,rtol=0)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(v>=tau,full>=tau)
                        witness[str(sid)][k]=float(np.max(np.abs(v-full)))
            path=out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves)
            hashes[name]=sha(path)
            if i%60==0:
                print(dict(stage='sealed',shots=i,seconds=time.monotonic()-started),flush=True)
    if (state_digest(net)!=frozen or state_digest(base)!=locked['baseline_protocol']['source_state_sha256']
            or any(sha(out/k)!=v for k,v in assets.items())):
        raise ValueError('persistent state changed inside the sealed suffix')
    write(out/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witness,fit_assets=assets))
    write(out/'dynamics_by_shot.json',dynamics)
    per={c:{k:[] for k in MODES} for c in CAPS}
    for i,sid in enumerate(ids,1):
        path=out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path)!=hashes[path.name]:
            raise ValueError('post-seal curve changed')
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,g in per.items():
                for k in MODES:
                    a=score_curve(f['times'],f[k],d,down,original if i<=STARTUP else cutoffs[c])
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,**a))
    prior=read(reference/'per_shot.json')
    if any(g['fixed']!=prior[c]['fixed'] or g['late']!=prior[c]['intra'] for c,g in per.items()):
        raise ValueError('sealed reference per-shot scores changed')
    metrics={c:{k:timing_decomposition(r) for k,r in g.items()} for c,g in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(
        status='COMPLETED_FIXED_HEAD_EARLY_ORDINAL_EAST360_DEVELOPMENT',metrics=metrics,
        paired_vs_fixed={c:{k:flips(g['fixed'],g[k]) for k in MODES[1:]} for c,g in per.items()},
        ordinal_comparisons={c:{k:flips(g[k],g['ordered']) for k in ('late','scrambled','ordered_lag')}
                             for c,g in per.items()},
        visible_updates={k:sum(v['visible_updates'][k] for v in dynamics.values()) for k in ('intra','lag')},
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()},flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    a=cli.parse_args()
    run(a.reference.resolve(),a.output.resolve())
