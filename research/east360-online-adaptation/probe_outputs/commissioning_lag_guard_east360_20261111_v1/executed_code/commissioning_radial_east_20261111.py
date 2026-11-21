"""Frozen-head paired alarm replay of a same-shot radial response."""

import argparse
import ast
import json
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_radial_response_20261111 as radial
from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as history
from . import io23, simple_data
from .commissioning_intra_20261110 import STARTUP
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
from .warning_lead_metrics import report


BASES = ('fixed','inherited','legacy_current4')
ARMS = (*BASES, 'response_only','amplitude_only','response_plus_old',
        'amplitude_plus_old','opposite_plus_old','delayed_plus_old')
STEPS, CAP = 256, 1.5


def _bound(x):
    return x * (CAP/x.norm(dim=-1,keepdim=True).clamp_min(CAP))


def _require_hashes(folder, files):
    if any(sha(folder/k)!=v for k,v in files.items()):
        raise ValueError(f'bound assets changed: {folder}')


def _frozen(reference, coordinates, lock):
    p, seal = (read(reference/k) for k in ('protocol.json','seal.json'))
    if (p['scope']!='EAST360_PREQUENTIAL_DIAGNOSTIC_DIRECTION_GATE_DEVELOPMENT'
            or p['order']!=lock['ids'] or p['startup']!=STARTUP or p['source_sha256']!=PARENT_SHA256
            or len(seal['curve_sha256'])!=360):
        raise ValueError('exact sealed alarm reference required')
    _require_hashes(Path(__file__).parent,p['code_sha256'])
    norm = Path(p['reference'])
    _require_hashes(norm,p['reference_sha256'])
    n = read(norm/'protocol.json')
    _require_hashes(Path('/'),n['reference_sha256'])
    prior = Path(n['inherited_reference'])
    earlier = read(prior/'protocol.json')
    operating = Path(earlier['previous'])
    _require_hashes(operating,earlier['previous_assets'])
    op = read(operating/'protocol.json')
    risk = Path(op['reference'])
    _require_hashes(risk,op['reference_assets'])
    source = Path(read(risk/'protocol.json')['source_prior'])
    _require_hashes(source,op['source_assets'])
    forecast = Path(op['forecast_reference'])/'fit.npz'
    if sha(forecast)!=op['forecast_source_fit_sha256']:
        raise ValueError('fixed decoder changed')
    local_protocol = read(coordinates/'protocol.json')
    loc_dir = Path(local_protocol['reference'])
    _require_hashes(loc_dir,local_protocol['assets'])
    cp = read(loc_dir/'protocol.json')
    if (local_protocol['scope']!='SHOT_RESET_LOCAL_CROSS_RESPONSE_SIGNAL_TEST_NO_RISK'
            or cp['order']!=lock['ids'] or local_protocol['order']!=lock['ids']
            or cp['startup']!=STARTUP or tuple(cp['channels'][:4])!=tuple(
                io23.RAW23[i] for i in radial.CHANNELS)):
        raise ValueError('unexpected position coordinate contract')
    with np.load(loc_dir/'normal_fit.npz',allow_pickle=False) as z:
        coords = z['loc'][:4],z['scale'][:4]
    info = read(source/'result.json')
    net = LagState(info['inputs'],info['edges']).eval().requires_grad_(False)
    with np.load(source/'model.npz',allow_pickle=False) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files})
    base, ckpt = load_parent(DEFAULT_PARENT)
    if ckpt['threshold']!=p['original'] or state_digest(base)!=lock['baseline_protocol']['source_state_sha256']:
        raise ValueError('qualified source changed')
    center, scale = (a.detach().numpy().copy() for a in (base.center,base.scale))
    base.cuda().requires_grad_(False)
    with np.load(forecast,allow_pickle=False) as f:
        decoder,noise = (f['decoder_weight'],f['decoder_bias']),f['scale']
    with np.load(operating/'mapper.npz',allow_pickle=False) as f:
        old, ctx = torch.from_numpy(f['mapper']),f['context_scale']
    with np.load(prior/'state.npz',allow_pickle=False) as f:
        quiet = f['location'],f['scale']
    with np.load(norm/'mapper.npz',allow_pickle=False) as f:
        legacy = torch.from_numpy(f['legacy'])
    with np.load(risk/'readout.npz',allow_pickle=False) as f:
        coef = torch.from_numpy(f['coef']).cuda()
    assets = {str(k):sha(k) for k in (
        reference/'protocol.json',reference/'seal.json',norm/'mapper.npz',forecast,
        prior/'state.npz',operating/'mapper.npz',risk/'readout.npz',source/'model.npz',
        loc_dir/'normal_fit.npz',coordinates/'protocol.json')}
    return p,seal,(base,net,(center,scale,decoder,noise,ctx),old,quiet,legacy,coef,coords),assets


@torch.no_grad()
def _parent(raw, signal, args):
    base,net,params,old,quiet,legacy,_,coords = args
    x,_ = features(base,signal,'cuda')
    tr,pre,ready = history.observed_track(net,signal,np.arange(len(x)),params,old)
    states,_ = persistent.replay(*ready,quiet)
    bare = torch.from_numpy(tr['pre'])
    current = pre+torch.einsum('...k,...f,kfw->...w',torch.from_numpy(tr['key']),
                    4.*torch.from_numpy(states['current_equal_norm']),legacy)
    return x,(bare,pre,current),radial.observe(raw,*coords)


def _cold_example(raw, signal, args, cutoff):
    x,(bare,_,current),r = _parent(raw,signal,args)
    coef = args[-2]
    d,down = io23.read_outcome('EAST',raw['shot'])
    offset = torch.from_numpy(x).cuda()@coef[:-25]+coef[-1]
    old = offset+current.cuda().tanh()@coef[-25:-1]
    z0 = offset+bare.cuda().tanh()@coef[-25:-1]
    if d:
        lead = np.round((down-raw['times'])*1000.,9)
        use = np.flatnonzero((lead>30.) & (lead<=125.))
    else:
        use = np.unique(np.r_[np.arange(0,len(x),16),r['release'],
                         old.topk(min(128,len(x))).indices.cpu().numpy(),
                         z0.topk(min(128,len(x))).indices.cpu().numpy()])
    if not len(use):
        raise ValueError('ended preparation shot lacks its legal scoring samples')
    base = torch.stack((bare[use],current[use])).cuda()
    state = torch.from_numpy(r['state'][use]).cuda()
    return dict(shot=raw['shot'],d=d,state=state,p=base,offset=offset[use],n=len(use),
                active=int(state.abs().any(1).sum()),cutoff=cutoff)


def _fit(cold, beta):
    p = torch.cat([r['p'] for r in cold],dim=1)
    state = torch.cat([r['state'] for r in cold])
    x = torch.stack((state,state.abs()))
    offset = torch.cat([r['offset'] for r in cold])
    lengths = [r['n'] for r in cold]
    d = torch.tensor([r['d'] for r in cold],device=p.device,dtype=torch.bool)
    boundary = float(np.log(cold[0]['cutoff']/(1.-cold[0]['cutoff'])))
    mapper = torch.nn.Parameter(torch.zeros(2,radial.N_FEATURES,p.shape[-1],device=p.device))
    optim = torch.optim.Adam([mapper],lr=.02)

    def objective():
        step = _bound(x@mapper)
        y = offset+(p[None]+step[:,None]).tanh()@beta
        peak = torch.stack([z.max(-1).values for z in y.split(lengths,dim=-1)],dim=-1)
        loss = .5*(F.softplus((boundary-peak[...,d])/.35).mean((1,2))
                   + F.softplus((peak[...,~d]-boundary)/.35).mean((1,2)))
        return loss,peak

    before,old = objective()
    for _ in range(STEPS):
        optim.zero_grad(set_to_none=True)
        loss,_ = objective()
        (loss.sum()+.02*mapper.square().sum()).backward()
        optim.step()
    with torch.no_grad():
        after,new = objective()
        if not torch.isfinite(mapper).all():
            raise FloatingPointError('nonfinite radial projection')
        note = dict(objective_before=before.detach().tolist(),objective_after=after.tolist(),
                    old_peak=old.detach().tolist(),new_peak=new.tolist(),mask=d.tolist(),
                    ids=[r['shot'] for r in cold],active_points=[r['active'] for r in cold],
                    norms=mapper.flatten(1).norm(dim=1).tolist())
    return mapper.detach().cpu(),note


@torch.no_grad()
def _encode(raw, signal, args, mapper):
    x,(bare,pre,current),r = _parent(raw,signal,args)
    s = torch.from_numpy(r['state'])
    a = _bound(s@mapper[0])
    u = _bound(s.abs()@mapper[1])
    lag = torch.cat((torch.zeros_like(a[:radial.STRIDE]),a[:-radial.STRIDE]))
    latent = (bare,pre,current,bare+a,bare+u,current+a,current+u,current-a,current+lag)
    h = {k:v.tanh().numpy() for k,v in zip(ARMS,latent,strict=True)}
    return scores(x,h,args[-2],ARMS),r,a.norm(dim=-1).numpy()


def _resume_fit(prefix, producer, protocol):
    """A hard timeout can interrupt writing a shot; retain only an explicit valid prefix."""
    old = read(prefix/'protocol.json')
    old_code = old.pop('code_sha256')
    # Compare the on-disk representation: the live plan uses tuples for arms.
    current = json.loads(json.dumps(protocol))
    new_code = current.pop('code_sha256')
    archived = prefix/'executed_code'/producer.name
    if (old != current or any(new_code.get(k)!=v for k,v in old_code.items() if k!=producer.name)
            or sha(archived)!=old_code[producer.name] or (prefix/'result.json').exists()):
        raise ValueError('cannot continue a different or already completed experiment')
    before,after = (ast.parse(f.read_text()).body for f in (archived,producer))
    after_functions = {n.name:ast.dump(n) for n in after if isinstance(n,ast.FunctionDef)}
    if any(ast.dump(n)!=after_functions.get(n.name) for n in before
           if isinstance(n,ast.FunctionDef) and n.name!='run'):
        raise ValueError('resumption may not change fitted or inference code')
    note = read(prefix/'fit.json')
    if note['ids']!=protocol['fit_ids']:
        raise ValueError('resumption fit has different preparation shots')
    with np.load(prefix/'mapper.npz',allow_pickle=False) as f:
        mapper = torch.from_numpy(np.stack((f['signed'],f['unsigned'])))
    if mapper.shape!=(2,radial.N_FEATURES,24) or not torch.isfinite(mapper).all():
        raise ValueError('incomplete frozen map')
    return mapper


def _retained(path, times, archived):
    """Require a readable, finite, same-parent archive for each retained shot."""
    with np.load(path,allow_pickle=False) as f:
        np.testing.assert_array_equal(f['times'],times)
        for k in ARMS:
            if f[k].shape!=times.shape or not np.isfinite(f[k]).all():
                raise ValueError(f'incomplete retained curve: {path} {k}')
        for k in BASES:
            np.testing.assert_array_equal(f[k],archived[k])


def run(reference, coordinates, output, *, resume=None, retain=0):
    configure()
    started = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p,old_seal,args,assets = _frozen(reference,coordinates,lock)
    frozen = tuple(state_digest(m) for m in args[:2])
    tau,original = p['calibration'],p['original']
    producer = Path(__file__)
    bound = {**p['code_sha256'],producer.name:sha(producer),
             Path(radial.__file__).name:sha(Path(radial.__file__))}
    protocol = dict(
        scope='EAST360_SAME_SHOT_RADIAL_RESPONSE_FROZEN_HEAD_DEVELOPMENT',order=ids,startup=STARTUP,
        reference=str(reference),source_sha256=PARENT_SHA256,calibration=tau,original=original,
        code_sha256=bound,assets=assets,arms=ARMS,fit_ids=ids[:STARTUP],fit_steps=STEPS,cap=CAP,
        first_release=radial.FIRST_RELEASE,stride=radial.STRIDE,window=radial.WINDOW,
        reference_lookback=radial.LOOKBACK,reference_neighbors=radial.NEIGHBORS,
        observation='estimation uses samples no later than release-1; old reference excludes all three current snapshots',
        control='abs features fit at equal capacity; opposite latent shift and one-release delay keep fitted mapper',
        objective='shot peak softplus, only ended cold20; D inside (30,125]ms, safe grid and fixed peaks',
        event_labels_read=False,no_suffix_fit=True,persistent_cross_shot_updates=0)
    if resume is None and retain or resume is not None and not STARTUP <= retain < len(ids):
        raise ValueError('resumption requires an explicit completed prefix count')
    mapper = _resume_fit(resume,producer,protocol) if resume is not None else None
    if resume is not None:
        protocol['resumed'] = dict(directory=str(resume),retained_prefix=retain,
                                   protocol_sha256=sha(resume/'protocol.json'),
                                   map_sha256=sha(resume/'mapper.npz'))
    output.mkdir(parents=True)
    (output/'shots').mkdir()
    write(output/'protocol.json',protocol)

    def load_raw(sid):
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('raw input changed')
        return simple_data.read_shot('EAST',sid,rate_contract='nominal')

    if resume is None:
        cold = []
        for sid in ids[:STARTUP]:
            raw = load_raw(sid)
            cold.append(_cold_example(raw,prepare_actual(raw,STARTUP+1,lock['mapping']),args,tau['0.2']))
        if sum(r['d'] for r in cold)!=8:
            raise ValueError('fixed preparation cohort changed')
        mapper, note = _fit(cold,args[-2][-25:-1])
        write(output/'fit.json',note)
        with (output/'mapper.npz').open('xb') as f:
            np.savez(f,signed=mapper[0].numpy(),unsigned=mapper[1].numpy())
        print(dict(stage='cold_fit',seconds=time.monotonic()-started,
                   objective=note['objective_after'],active_shots=sum(bool(x) for x in note['active_points'])),
              flush=True)
    else:
        for name in ('fit.json','mapper.npz'):
            copyfile(resume/name,output/name)
        if sha(output/'mapper.npz')!=protocol['resumed']['map_sha256']:
            raise ValueError('resumption map changed')
        print(dict(stage='resumed_frozen_fit',retained_prefix=retain),flush=True)
    hashes, witnesses = {},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            path = reference/'shots'/name
            if sha(path)!=old_seal['curve_sha256'][name]:
                raise ValueError('sealed reference curve changed')
            with np.load(path,allow_pickle=False) as f:
                times,arch = f['times'],{k:f[k] for k in BASES}
            target = output/'shots'/name
            if rank<=retain:
                saved = resume/'shots'/name
                _retained(saved,times,arch)
                # The original process writes witnesses only once the whole
                # stream is sealed, so rerun these few shots before scoring.
                if rank not in (21,120,360):
                    copyfile(saved,target)
                    hashes[name] = sha(target)
                    if hashes[name]!=sha(saved):
                        raise ValueError('retained archive changed during copy')
                    continue
            if rank<=STARTUP:
                curves = dict.fromkeys(ARMS,arch['fixed'])
                obs = {}
            else:
                raw = load_raw(sid)
                np.testing.assert_array_equal(raw['times'],times)
                sig = prepare_actual(raw,rank,lock['mapping'])
                curves,r,step = _encode(raw,sig,args,mapper)
                for k in BASES:
                    np.testing.assert_array_equal(curves[k],arch[k])
                n = radial.FIRST_RELEASE
                for k,parent in (('response_only','fixed'),('amplitude_only','fixed'),
                                 *((k,'legacy_current4') for k in ARMS[5:])):
                    end = n+radial.STRIDE if k=='delayed_plus_old' else n
                    np.testing.assert_array_equal(curves[k][:end],curves[parent][:end])
                obs = {k:v for k,v in r.items() if k!='state'}
                obs['step_norm'] = step[r['release']]
                if rank in (21,120,360):
                    witnesses[str(sid)] = []
                    for end in (n+7,n+3*radial.STRIDE+5):
                        rs = {k:raw[k][:end] for k in ('values','available')}
                        ss = {k:sig[k][:end] for k in ('values','available')}
                        cs,rr,_ = _encode(rs,ss,args,mapper)
                        np.testing.assert_array_equal(rr['state'],r['state'][:end])
                        for k,v in cs.items():
                            np.testing.assert_allclose(v,curves[k][:end],atol=2e-6,rtol=0.)
                            for cutoff in tau.values():
                                np.testing.assert_array_equal(v>=cutoff,curves[k][:end]>=cutoff)
                        witnesses[str(sid)].append(end)
            if rank<=retain:
                with np.load(resume/'shots'/name,allow_pickle=False) as old:
                    for k,v in {**curves,**obs}.items():
                        np.testing.assert_array_equal(v,old[k])
            with target.open('xb') as f:
                np.savez_compressed(f,times=times,**curves,**obs)
            hashes[name] = sha(target)
            if rank%60==0:
                print(dict(stage='sealed',shots=rank,seconds=time.monotonic()-started),flush=True)
    if tuple(state_digest(m) for m in args[:2])!=frozen:
        raise ValueError('frozen model changed')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses,
                                  mapper_sha256=sha(output/'mapper.npz')))
    per = {c:{k:[] for k in ARMS} for c in tau}
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        f = output/'shots'/name
        if sha(f)!=hashes[name]:
            raise ValueError('unsealed prediction')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(f,allow_pickle=False) as vals:
            for c,g in per.items():
                threshold = original if rank<=STARTUP else tau[c]
                for k in ARMS:
                    a = score_curve(vals['times'],vals[k],d,down,threshold)
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,**a))
    old_per = read(reference/'per_shot.json')
    if any(per[c][k]!=old_per[c][k] for k in BASES for c in per):
        raise ValueError('archive metric parity failed')
    write(output/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    write(output/'result.json',dict(status='COMPLETED_EAST360_RADIAL_RESPONSE_DEVELOPMENT',
        metrics=metrics,paired={c:{k:flips(g['fixed' if k in ARMS[3:5] else 'legacy_current4'],g[k])
                                   for k in ARMS[3:]} for c,g in per.items()},
        seconds=time.monotonic()-started))
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],('fixed','legacy_current4')))
    print({c:{k:(r['tp'],r['fp']) for k,r in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--coordinates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-prefix',type=Path)
    parser.add_argument('--retain',type=int,default=0)
    a = parser.parse_args()
    run(a.reference.resolve(),a.coordinates.resolve(),a.output.resolve(),
        resume=a.resume_prefix.resolve() if a.resume_prefix else None,retain=a.retain)
