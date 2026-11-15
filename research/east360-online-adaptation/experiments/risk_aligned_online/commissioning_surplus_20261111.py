"""Transfer a label-free, delayed surplus-motion update into a fixed risk model."""

import argparse
from inspect import unwrap
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_surplus_feedback_20261111 as feedback
from . import commissioning_weak_shape_data_20261111 as shape
from . import commissioning_weak_shape_source_20261111 as weak
from . import io23, simple_data
from .commissioning_innovation_east_20261111 import make_matches, separation_loss
from .commissioning_innovation_metric_20261111 import RETAIN, released_displacement
from .commissioning_intra_20261110 import CAPS, STARTUP
from .commissioning_lag_state_20261110 import LagState
from .commissioning_transfer_20261108 import features, shot_weights
from .commissioning_transition_east_20261110 import scores
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE, WIDTH
from .commissioning_transition_source_20261110 import decision_points
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


PHI_DIM = 28
MODES = ('fixed','no_background','intra','lag')
UPDATE_STEP = .35
MAPPER_RADIUS = 4.
NOISE_FLOOR = .025
FIRST_EFFECT = FIRST + HORIZON + 1
SEED, STEPS, BATCH = 131, 640, 256


def noise_scale(safe):
    residual, observed = map(np.concatenate,zip(*safe,strict=True))
    scales=[]
    for r,m in zip(residual.T,observed.T,strict=True):
        if not m.any():
            raise ValueError('each cue needs observed safe J-TEXT TRAIN windows')
        scales.append(max(NOISE_FLOOR,float(np.quantile(np.abs(r[m]),.75))))
    return np.asarray(scales,np.float32)


def drive(keys, origins, phi, eligible):
    """State increment is the negative anchored gradient of a squared target error.

    The cue-to-latent map factors out of the linear recurrence, so the stored
    state can be computed in cue coordinates without changing the update.
    """
    gradients = (UPDATE_STEP/np.sqrt(PHI_DIM)) * phi[:,:,None]*keys[origins,None,:]
    return released_displacement(keys,origins,gradients.astype(np.float32),eligible)


def bounded_map(weights):
    return weights * (MAPPER_RADIUS / weights.norm().clamp_min(MAPPER_RADIUS))


def project(pre, displacement, weights):
    return (pre + displacement@weights).tanh()


@torch.no_grad()
def trace(net, signal, center, scale, ends):
    z,m,ctx,cm = weak.shot_coordinates(signal,center,scale)
    q = np.arange(FIRST,len(z)-HORIZON,STRIDE)
    raw,obs = feedback.raw_feedback(z,m,q)
    x,context = shape.inputs(z,m,ctx,cm,ends)
    p = net.stem(torch.from_numpy(x))
    p[ends<FIRST] = 0.
    return dict(pre=p.numpy(), context=context,matchable=x[:,-3:].astype(bool).all(1),
                keys=feedback.physical_keys(ctx,cm),origins=q,raw=raw,observed=obs,ends=ends)


def state(track,scale):
    phi,eligible,audit = feedback.innovations(track['raw'],track['observed'],scale)
    views,counts = {},{}
    for k in MODES[1:]:
        d,counts[k] = drive(track['keys'],track['origins'],phi[k],eligible)
        views[k] = d[track['ends']]
    return views,dict(audit,visible_updates=counts)


def collect_source(net,ids,advanced,center,scale,role):
    rows=[]
    for i,sid in enumerate(ids,1):
        signal = simple_data.read_shot('J-TEXT',sid)
        s = simple_data.attach_outcome(signal,'J-TEXT',advanced)
        t,d = s['times'],s['is_disrupt']
        q = decision_points(s['y'].astype(bool),s['valid'])
        if not d:
            tail = np.flatnonzero(s['valid'] & (t>=t[-1]-weak.TAIL_S) & (np.arange(len(t))>=FIRST+2*STRIDE))
            q = np.union1d(q,tail[::8])
        if len(q):
            a=trace(net,signal,center,scale,q)
            lead=np.round((s['down']-t[q])*1000.,9) if d else np.zeros(len(q))
            terminal=(t[q]>=t[-1]-weak.TAIL_S) & (a['context'][:,2]<0) & (not d)
            rows.append(dict(a,sid=sid,disruptive=d,lead=lead,terminal=terminal,risk=s['y'][q]))
        if i%450==0 or i==len(ids):
            print(dict(stage=f'collect_{role}',shots=i,total=len(ids)),flush=True)
    return rows


def pack(tracks,scale):
    keys=('risk','lead','terminal','matchable')
    rows={k:np.concatenate([r[k] for r in tracks]) for k in keys}
    rows.update(pre=np.concatenate([r['pre'] for r in tracks]),
                pair_context=np.concatenate([r['context'] for r in tracks]),
                end=np.concatenate([r['ends'] for r in tracks]),
                shot=np.concatenate([np.full(len(r['ends']),r['sid']) for r in tracks]),
                disruptive=np.concatenate([np.full(len(r['ends']),r['disruptive']) for r in tracks]))
    states=[state(r,scale)[0] for r in tracks]
    rows['delta']={k:np.concatenate([d[k] for d in states]) for k in MODES[1:]}
    return rows


def directions(net,coef):
    source_beta = net.risk.weight.detach().flatten().cuda()
    target_beta = coef[-WIDTH-1:-1]
    if target_beta.norm() == 0 or source_beta.norm() == 0:
        raise ValueError('both fixed risk directions must be nonzero')
    return source_beta,target_beta*source_beta.norm()/target_beta.norm()


def fit_source(rows,net,coef,match_scale):
    pos,neg,_,distance = weak.make_pairs(rows,match_scale)
    p = torch.from_numpy(rows['pre']).cuda()
    a = torch.from_numpy(rows['delta']['intra']).cuda()
    y = torch.from_numpy(rows['risk']).cuda()
    beta,target = directions(net,coef)
    frozen = p.tanh()
    risk = frozen@beta + net.risk.bias.detach().cuda()
    w = torch.from_numpy(shot_weights(rows['shot'],rows['risk'].astype(bool))).cuda()
    _,inv,n = np.unique(rows['shot'][pos],return_inverse=True,return_counts=True)
    w_pair = torch.from_numpy(1./n[inv]).cuda()
    pair_p,pair_n = (torch.from_numpy(i).cuda() for i in (pos,neg))
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    points = torch.multinomial(w,STEPS*BATCH,True,generator=rng).reshape(STEPS,BATCH)
    pairs = torch.multinomial(w_pair,STEPS*(BATCH//2),True,generator=rng).reshape(STEPS,-1)
    mapper = torch.nn.Parameter(torch.zeros((PHI_DIM,WIDTH),device='cuda'))
    opt = torch.optim.Adam([mapper],lr=.015)
    for step,(i,j) in enumerate(zip(points,pairs,strict=True),1):
        opt.zero_grad(set_to_none=True)
        u = torch.cat((i,pair_p[j],pair_n[j]))
        v = bounded_map(mapper)
        change = (project(p[u],a[u],v)-frozen[u])@target
        c,pos_change,neg_change = change.split((len(i),len(j),len(j)))
        point = F.binary_cross_entropy_with_logits(risk[i]+c,y[i])
        ranked = separation_loss(pos_change,neg_change)
        protection = F.relu(neg_change).square().mean()
        objective = .65*point+.35*ranked+.10*protection+.015*v.square().sum()
        objective.backward()
        opt.step()
        if step%320==0:
            print(dict(stage='source_update_fit',step=step,point=float(point.detach()),
                       pair=float(ranked.detach()),protection=float(protection.detach())),flush=True)
    value=bounded_map(mapper).detach()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite source update map')
    return value.cpu(),dict(steps=STEPS,seed=SEED,source_pairs=len(pos),radius=float(value.norm()),
                            median_pair_rms_distance=float(np.median(distance)))


@torch.no_grad()
def held_source(rows,net,coef,mapper,match_scale):
    pos,neg,tail,distance=weak.make_pairs(rows,match_scale)
    keep=[]
    for sid in np.unique(rows['shot'][pos]):
        ends=pos[::2][rows['shot'][pos[::2]]==sid]
        lead=rows['lead'][ends]
        keep.append(ends[np.argmin(np.abs(lead-np.median(lead)))])
    select=np.isin(pos,keep)
    pos,neg,tail,distance=(a[select] for a in (pos,neg,tail,distance))
    p=torch.from_numpy(rows['pre']).cuda()
    beta,target=directions(net,coef)
    h=p.tanh()
    z=h@beta+net.risk.bias.detach().cuda()
    logits={'fixed':z.cpu().numpy()}
    for k in MODES[1:]:
        value=project(p,torch.from_numpy(rows['delta'][k]).cuda(),mapper.cuda())
        logits[k]=(z+(value-h)@target).cpu().numpy()
    weak_signal=weak.amplitude(rows,match_scale)[pos] <= rows['weak_cutoff']
    summary={}
    for name,m in (('all',np.ones(len(pos),bool)),('falling_tail',tail),('weak',weak_signal),
                   ('weak_falling_tail',weak_signal & tail)):
        summary[name]=dict(pairs=int(m.sum()),
                           wins={k:int((v[pos[m]]>v[neg[m]]).sum()) for k,v in logits.items()})
    return dict(summary=summary,pairs=[[int(rows['shot'][a]),int(rows['end'][a]),
                                        int(rows['shot'][b]),int(rows['end'][b]),bool(c),float(d)]
                                       for a,b,c,d in zip(pos,neg,tail,distance,strict=True)])


@torch.no_grad()
def encode(base,net,signal,center,scale,noise,mapper,coef):
    raw,_=features(base,signal,'cuda')
    track=trace(net,signal,center,scale,np.arange(len(raw)))
    ds,audit=state(track,noise)
    p=torch.from_numpy(track['pre'])
    views={'fixed':p.tanh().numpy()}
    for k in MODES[1:]:
        views[k]=project(p,torch.from_numpy(ds[k]),mapper).numpy()
    return scores(raw,views,coef,MODES),track,audit


def run(reference,out):
    started=time.monotonic()
    configure()
    inputs=load_inputs()
    prior,seal,cal=(read(reference/k) for k in ('protocol.json','seal.json','calibration.json'))
    ids=inputs['ids']
    if (prior['order']!=ids or prior['startup']!=STARTUP
            or prior['scope']!='EAST360_PAIRED_CONTEXT_MATCHED_WEAK_SHAPE_DEVELOPMENT'
            or prior['source_sha256']!=PARENT_SHA256 or len(seal['curve_sha256'])!=360
            or any(sha(reference/k)!=v for k,v in seal['artifacts'].items())):
        raise ValueError('same complete sealed weak-shape reference required')
    source=Path(prior['source_prior'])
    spec,previous=(read(source/k) for k in ('protocol.json','result.json'))
    if (spec['code_sha256']!=weak.code_bindings()
            or any(sha(source/k)!=v for k,v in prior['source_assets'].items())):
        raise ValueError('sealed shape source assets changed')
    base,ckpt=load_parent(DEFAULT_PARENT)
    center,scale=(a.detach().numpy().copy() for a in (base.center,base.scale))
    base.cuda().requires_grad_(False)
    original=prior['original']
    if original!=ckpt['threshold'] or state_digest(base)!=inputs['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified J-TEXT checkpoint changed')
    net=LagState(previous['inputs'],previous['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz',allow_pickle=False) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    with np.load(reference/'readout.npz',allow_pickle=False) as f:
        coef=torch.from_numpy(f['coef']).cuda()
    frozen=state_digest(net)
    cutoffs=cal['cutoffs']
    out.mkdir(parents=True)
    (out/'shots').mkdir()
    write(out/'protocol.json',dict(
        scope='EAST360_FIXED_SHAPE_HEAD_SIGNED_SURPLUS_DEVELOPMENT',order=ids,startup=STARTUP,
        calibration=cutoffs,original=original,arms=MODES,first_effect=FIRST_EFFECT,
        source_sha256=PARENT_SHA256,reference=str(reference),source_assets=prior['source_assets'],
        reference_assets={k:sha(reference/k) for k in ('protocol.json','calibration.json','seal.json','readout.npz')},
        data_bindings=inputs['bound'],producer_sha256=sha(Path(__file__)),
        feedback_sha256=sha(Path(feedback.__file__)),
        recurrence_sha256=sha(Path(released_displacement.__code__.co_filename)),
        scorer_sha256=sha(Path(unwrap(scores).__code__.co_filename)),
        source_training='J-TEXT TRAIN only; locked EAST head direction used without fitting to EAST outcomes',
        source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),retention=RETAIN,step=UPDATE_STEP,
        mapper_norm_cap=MAPPER_RADIUS,noise_floor=NOISE_FLOOR,optimizer_lr=.015,steps=STEPS,
        no_target_fitting=True,persistent_suffix_updates=0))
    roles,advanced=io23.source_roles(),io23.advanced_times()
    tracks=collect_source(net,roles['train'],advanced,center,scale,'train')
    noise=noise_scale([(r['raw'],r['observed']) for r in tracks if not r['disruptive']])
    rows=pack(tracks,noise)
    match_scale=np.asarray(previous['match_scale'])
    mapper,fit=fit_source(rows,net,coef,match_scale)
    fit['train_ids']=[r['sid'] for r in tracks]
    fit['train_rows']=len(rows['pre'])
    del tracks,rows
    held=collect_source(net,roles['val'],advanced,center,scale,'val')
    val=pack(held,noise)
    val['weak_cutoff']=previous['weak_amplitude_cutoff']
    val_result=held_source(val,net,coef,mapper,match_scale)
    val_result['val_ids']=[r['sid'] for r in held]
    write(out/'source_validation.json',val_result)
    with (out/'fit.npz').open('xb') as f:
        np.savez(f,mapper=mapper.numpy(),noise_scale=noise)
    write(out/'fit.json',fit)
    print(dict(stage='source_held',**val_result['summary']),flush=True)

    def sealed(i,sid):
        name=f'{i:04d}_{sid}.npz'
        path=reference/'shots'/name
        if sha(path)!=seal['curve_sha256'][name]:
            raise ValueError(f'archived fixed curve changed at {sid}')
        with np.load(path,allow_pickle=False) as f:
            return name,f['times'],f['fixed']

    def signal(i,sid,t):
        if sha(io23.path_for('EAST',sid))!=inputs['hashes'][str(sid)]:
            raise ValueError(f'locked raw EAST data changed at {sid}')
        data=simple_data.read_shot('EAST',sid,rate_contract='nominal')
        np.testing.assert_array_equal(data['times'],t)
        return prepare_actual(data,max(i,STARTUP+1),inputs['mapping'])

    cold={}
    for i,sid in enumerate(ids[:STARTUP],1):
        _,t,_=sealed(i,sid)
        curves,track,_=encode(base,net,signal(i,sid,t),center,scale,noise,mapper,coef)
        d,down=io23.read_outcome('EAST',sid)
        y,_=io23.labels_at(t,d,down,125.)
        timely=y.astype(bool) & (np.round((down-t)*1000.,9)>30) if d else y.astype(bool)
        cold[sid]=dict(pre=track['pre'],context=track['context'],matchable=track['matchable'],
                       timely=timely,disruptive=d,curves=curves)
    if sum(v['disruptive'] for v in cold.values())!=8:
        raise ValueError('same ended eight disruptive and twelve safe records required')
    p,ip,n,in_,dist=make_matches(cold,match_scale)
    cold_wins={k:sum(bool(cold[a]['curves'][k][b]>cold[c]['curves'][k][d])
                    for a,b,c,d in zip(p,ip,n,in_,strict=True)) for k in MODES}
    write(out/'ended20_diagnostic.json',dict(
        scope='UPDATE_MAP_NOT_FITTED_TO_EAST; ARCHIVED_RISK_HEAD_SAW_ENDED20',
        pairs=len(dist),median_match_rms_distance=float(np.median(dist)),wins=cold_wins,
        identities=[[int(a),int(b),int(c),int(d)] for a,b,c,d in zip(p,ip,n,in_,strict=True)]))
    assets={k:sha(out/k) for k in ('fit.npz','fit.json','source_validation.json','ended20_diagnostic.json')}
    print(dict(stage='ended20_diagnostic',wins=cold_wins,pairs=len(dist),
               seconds=time.monotonic()-started),flush=True)

    hashes,witness,dynamics={},{},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for i,sid in enumerate(ids,1):
            name,t,fixed=sealed(i,sid)
            if i<=STARTUP:
                curves=dict.fromkeys(MODES,fixed)
            else:
                s=signal(i,sid,t)
                curves,_,dynamics[str(sid)]=encode(base,net,s,center,scale,noise,mapper,coef)
                np.testing.assert_array_equal(curves['fixed'],fixed)
                for k in MODES[1:]:
                    np.testing.assert_array_equal(curves[k][:FIRST_EFFECT],fixed[:FIRST_EFFECT])
                if i in (21,120,360):
                    prefix={k:s[k][:243] for k in ('values','available')}
                    short,_,_=encode(base,net,prefix,center,scale,noise,mapper,coef)
                    witness[str(sid)]={}
                    for k,z in short.items():
                        full=curves[k][:len(z)]
                        np.testing.assert_allclose(z,full,atol=2e-6,rtol=0)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(z>=tau,full>=tau)
                        witness[str(sid)][k]=float(np.max(abs(z-full)))
            path=out/'shots'/name
            with path.open('xb') as f:
                np.savez_compressed(f,times=t,**curves)
            hashes[name]=sha(path)
            if i%60==0:
                print(dict(stage='sealed',shots=i,seconds=time.monotonic()-started),flush=True)
    if (state_digest(net)!=frozen or state_digest(base)!=inputs['baseline_protocol']['source_state_sha256']
            or any(sha(out/k)!=v for k,v in assets.items())):
        raise ValueError('persistent state changed inside the sealed suffix')
    write(out/'seal.json',dict(curve_sha256=hashes,fit_assets=assets,prefix_witnesses=witness))
    write(out/'dynamics_by_shot.json',dynamics)
    per={c:{k:[] for k in MODES} for c in CAPS}
    for i,sid in enumerate(ids,1):
        path=out/'shots'/f'{i:04d}_{sid}.npz'
        if sha(path)!=hashes[path.name]:
            raise ValueError('post-seal curve changed')
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,group in per.items():
                for k in MODES:
                    v=score_curve(f['times'],f[k],d,down,original if i<=STARTUP else cutoffs[c])
                    group[k].append(dict(shot=sid,disruptive=d,down_s=down,**v))
    archived=read(reference/'per_shot.json')
    if any(g['fixed']!=archived[c]['fixed'] for c,g in per.items()):
        raise ValueError('sealed reference scores not reproduced')
    metrics={c:{k:timing_decomposition(rows) for k,rows in g.items()} for c,g in per.items()}
    write(out/'per_shot.json',per)
    write(out/'result.json',dict(
        status='COMPLETED_FIXED_HEAD_SIGNED_SURPLUS_EAST360_DEVELOPMENT',metrics=metrics,
        paired_vs_fixed={c:{k:flips(g['fixed'],g[k]) for k in MODES[1:]} for c,g in per.items()},
        intra_vs_no_background={c:flips(g['no_background'],g['intra']) for c,g in per.items()},
        intra_vs_lag={c:flips(g['lag'],g['intra']) for c,g in per.items()},
        visible_updates={k:sum(v['visible_updates'][k] for v in dynamics.values()) for k in MODES[1:]},
        background_updates=sum(v['background_updates'] for v in dynamics.values()),
        seconds=time.monotonic()-started))
    print({c:{k:(v['tp'],v['fp']) for k,v in g.items()} for c,g in metrics.items()},flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--reference',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    run(args.reference.resolve(),args.output.resolve())
