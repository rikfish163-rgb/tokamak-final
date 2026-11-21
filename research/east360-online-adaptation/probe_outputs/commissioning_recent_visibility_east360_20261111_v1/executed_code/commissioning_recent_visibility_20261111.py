"""Same-release tests of residual age and finite-dose visibility in a frozen head."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
import torch

from . import commissioning_donor_control_20261111 as donor
from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as history
from . import commissioning_radial_east_20261111 as reference_model
from . import io23, simple_data
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


BASES, STARTUP = reference_model.BASES, reference_model.STARTUP
LATEST = BASES[-1]
PREVIOUS, AGREEMENT = 'previous_direction_equal_norm', 'agreement_direction_equal_norm'
VISIBLE_LATEST, VISIBLE_PREVIOUS = 'visible_latest_equal_norm', 'visible_previous_equal_norm'
ARMS = (*BASES, PREVIOUS, AGREEMENT, VISIBLE_LATEST, VISIBLE_PREVIOUS)
COMPARISONS = (('fixed',LATEST),('inherited',LATEST),(PREVIOUS,LATEST),
               (AGREEMENT,LATEST),(LATEST,VISIBLE_LATEST),(PREVIOUS,VISIBLE_PREVIOUS),
               (VISIBLE_PREVIOUS,VISIBLE_LATEST))


def previous_direction(ready, quiet, state):
    """Use j-1's residual, with j's masks and *same* older local reference."""
    level,valid,origins,context,usable,n = ready
    result = np.zeros_like(state)
    for j in range(2,len(origins)):
        start = int(origins[j]+persistent.WINDOW+1)
        if start>=n:
            break
        if not state[start].any():
            continue
        center,scale = persistent.local_reference(level,valid,context,usable,j,quiet)
        mask = valid[j-2:j+1].all(0)
        z = np.where(mask,(level[j-1]-center)/scale,0.)
        result[start:start+persistent.STRIDE] = np.sign(z)*np.clip((abs(z)-.50)/1.50,0.,1.)
    return result


def same_radius(own, vector):
    a,b = own.double(),vector.double()
    support = a.abs().any(dim=1)
    step,applied = donor.equalized_delta(a,b,support.numpy())
    if not torch.equal(applied,support):
        raise ValueError('active previous/agreement direction lacks an equal-dose projection')
    return step.to(own.dtype)


def visible_direction(h, d):
    """Reallocate dose along tanh(h+d)-tanh(h), without changing channel signs."""
    active = d.abs().any(dim=1)
    h0,dx = (v[active].double() for v in (h,d))
    h1 = h0+dx
    a = dx.abs()
    # tanh(h+d)-tanh(h) = sinh(d)/(cosh(h+d)*cosh(h)). Work up to a
    # common constant in log-space so saturated but nonzero slopes survive.
    magnitude = a+torch.log(-torch.expm1(-2.*a))-torch.logaddexp(h0,-h0)-torch.logaddexp(h1,-h1)
    v = torch.exp(magnitude-magnitude.amax(dim=1,keepdim=True))*dx.sign()
    result = torch.zeros_like(d)
    result[active] = (v*(dx.norm(dim=1,keepdim=True)/v.norm(dim=1,keepdim=True))).to(d.dtype)
    return result


def head_geometry(h, d, beta):
    """Bound attainable upward logit gain; do not feed these values to prediction."""
    h,d,beta = (v.double() for v in (h,d,beta))
    rho = d.norm(dim=1,keepdim=True)
    sign = beta.sign()
    before = h.tanh()
    box = (((h+rho*sign).tanh()-before)*beta).clamp_min(0.)
    # In maximizing the score every coordinate can move toward sign(beta),
    # since tanh is monotone. The largest slope on that segment is closest to 0.
    nearest = torch.where(h*sign < 0., (h.abs()-rho).clamp_min(0.), h.abs())
    e = torch.exp(-2.*nearest)
    max_slope = (4.*e/(1.+e).square())*beta.abs()
    current_grad = (1.-before.square())*beta.abs()

    def bounds(mask):
        b = torch.where(mask,box,0.)
        deriv = torch.where(mask,max_slope,0.)
        upper = torch.minimum(b.sum(1),rho[:,0]*deriv.norm(dim=1))
        # Both directions are inside the same L2 ball; a single best coordinate
        # is another feasible witness, not a guarantee of successful warning.
        attainable = [b.amax(dim=1)]
        for v in (torch.where(mask,current_grad,0.),b):
            x = rho*sign*(v/v.norm(dim=1,keepdim=True).clamp_min(torch.finfo(h.dtype).tiny))
            attainable.append(((h+x).tanh()-before)@beta)
        return upper,torch.stack(attainable).amax(dim=0)

    upper,feasible = bounds(beta!=0.)
    sign_upper,sign_feasible = bounds((d*beta)>0.)
    return dict(tight=upper, feasible=feasible, same_sign_upper=sign_upper,
                same_sign_feasible=sign_feasible)


@torch.no_grad()
def encode(signal, args):
    base,net,params,old,quiet,mapper,coef,_ = args
    n = len(signal['values'])
    tr,h,ready = history.observed_track(net,signal,np.arange(n),params,old)
    state,_ = persistent.replay(*ready,quiet)
    current = state['current_equal_norm']
    previous = previous_direction(ready,quiet,current)
    key = torch.from_numpy(tr['key'])
    a,prev,ag = (torch.einsum('...k,...f,kfw->...w', key,4.*torch.from_numpy(v),mapper)
                 for v in (current,previous,state['sustained']))
    prev,ag = (same_radius(a,v) for v in (prev,ag))
    av,pv = (visible_direction(h,v) for v in (a,prev))
    steps = (a,prev,ag,av,pv)
    latent = (torch.from_numpy(tr['pre']),h,*(h+v for v in steps))
    x,_ = features(base,signal,'cuda')
    curves = scores(x,{k:v.tanh().numpy() for k,v in zip(ARMS,latent,strict=True)},coef,ARMS)
    if not all(np.isfinite(y).all() for y in curves.values()):
        raise FloatingPointError('nonfinite sealed probability')
    r = np.column_stack([v.double().norm(dim=1).numpy() for v in steps])
    np.testing.assert_allclose(r,np.broadcast_to(r[:,:1],r.shape),atol=1e-8,rtol=5e-7)
    active = r[:,0]>0.
    for k in ARMS[2:]:
        np.testing.assert_array_equal(curves[k][~active],curves['inherited'][~active])
    release = tr['q'][2:]+persistent.WINDOW+1
    release = release[release<n]
    release = release[current[release].any(1)]
    direction = [a[active].double(),prev[active].double(),ag[active].double(),av[active].double()]
    cos = [(direction[0]*v).sum(1).numpy()/(r[active,0]*r[active,col])
           for v,col in zip(direction[1:],(1,2,3),strict=True)]
    g = {k:v.numpy() for k,v in head_geometry(h,a,coef[-25:-1].cpu()).items()}
    telemetry = dict(radius=r, active=active, cos=np.column_stack(cos), release=release,
                     previous_release=release-persistent.STRIDE, **g)
    return curves,telemetry


def run(reference, coordinates, output):
    configure()
    started = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p,old_seal,args,assets = reference_model._frozen(reference,coordinates,lock)
    frozen = tuple(state_digest(m) for m in args[:2])
    cutoffs,original = p['calibration'],p['original']
    producer = Path(__file__)
    code = dict(p['code_sha256'],**{f.name:sha(f) for f in (
        producer,Path(donor.__file__),Path(reference_model.__file__))})
    (output/'shots').mkdir(parents=True)
    archive = output/'executed_code'
    archive.mkdir()
    for name,value in code.items():
        copyfile(producer.parent/name,archive/name)
        if sha(archive/name)!=value:
            raise ValueError(f'archived producer differs: {name}')
    write(output/'protocol.json',dict(
        scope='EAST360_SAME_RELEASE_RECENT_DIRECTION_AND_VISIBILITY_DEVELOPMENT',order=ids,
        startup=STARTUP,reference=str(reference),assets=assets,code_sha256=code,
        source_sha256=p['source_sha256'],arms=ARMS,comparison=COMPARISONS,original=original,
        calibration=cutoffs,first_effect=persistent.FIRST_EFFECT,window=persistent.WINDOW,
        stride=persistent.STRIDE,training_updates=0,outcome_access='all 360 curves sealed first',
        recent_control='j-1 residual in j reference and mask; j gate, recipient key and L2 all retained',
        agreement_control='j,j-1 common signed evidence in the identical 24D L2 budget',
        visibility='renormalize the finite tanh displacement; no risk-head coefficients in policy',
        geometry='label-free telemetry only, upward L2-ball and unchanged-coordinate-sign bounds'))
    hashes,witnesses = {},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            path = reference/'shots'/name
            if sha(path)!=old_seal['curve_sha256'][name]:
                raise ValueError('locked reference curve changed')
            with np.load(path,allow_pickle=False) as f:
                times,arch = f['times'],{k:f[k] for k in BASES}
            obs = {}
            if rank<=STARTUP:
                curves = dict.fromkeys(ARMS,arch['fixed'])
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked raw file changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                np.testing.assert_array_equal(raw['times'],times)
                sig = prepare_actual(raw,rank,lock['mapping'])
                curves,obs = encode(sig,args)
                for k in BASES:
                    np.testing.assert_array_equal(curves[k],arch[k])
                if rank in (21,120,360):
                    end = persistent.FIRST_EFFECT+3*persistent.STRIDE+5
                    cs,short = encode({k:sig[k][:end] for k in ('values','available')},args)
                    for k,v in cs.items():
                        np.testing.assert_allclose(v,curves[k][:end],atol=2e-6,rtol=0.)
                        for tau in cutoffs.values():
                            np.testing.assert_array_equal(v>=tau,curves[k][:end]>=tau)
                    np.testing.assert_array_equal(short['active'],obs['active'][:end])
                    for k in ('radius','tight','feasible','same_sign_upper','same_sign_feasible'):
                        np.testing.assert_allclose(short[k],obs[k][:end],atol=2e-6,rtol=0.)
                    take = obs['release']<end
                    for k in ('release','previous_release'):
                        np.testing.assert_array_equal(short[k],obs[k][take])
                    witnesses[str(sid)] = end
            target = output/'shots'/name
            with target.open('xb') as f:
                np.savez_compressed(f,times=times,**curves,**obs)
            hashes[name] = sha(target)
            if rank%40==0 or rank==STARTUP+1:
                print(dict(stage='sealed',shots=rank,seconds=round(time.monotonic()-started,2)),flush=True)
    if tuple(state_digest(m) for m in args[:2])!=frozen:
        raise ValueError('frozen model changed during replay')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses))
    per = {c:{k:[] for k in ARMS} for c in cutoffs}
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        path = output/'shots'/name
        if sha(path)!=hashes[name]:
            raise ValueError('unsealed curve')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,g in per.items():
                threshold = original if rank<=STARTUP else cutoffs[c]
                for k in ARMS:
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,
                                     **score_curve(f['times'],f[k],d,down,threshold)))
    old = read(reference/'per_shot.json')
    if any(per[c][k]!=old[c][k] for c in cutoffs for k in BASES):
        raise ValueError('frozen parent scoring changed')
    write(output/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    paired = {c:{f'{a}__to__{b}':flips(g[a],g[b]) for a,b in COMPARISONS} for c,g in per.items()}
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],('fixed',PREVIOUS,LATEST)))
    write(output/'result.json',dict(status='COMPLETED_EAST360_RECENT_VISIBILITY_DEVELOPMENT',
        metrics=metrics,paired=paired,seconds=time.monotonic()-started))
    print({c:{k:(r['tp'],r['fp']) for k,r in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('reference','coordinates','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    a = parser.parse_args()
    run(a.reference.resolve(),a.coordinates.resolve(),a.output.resolve())
