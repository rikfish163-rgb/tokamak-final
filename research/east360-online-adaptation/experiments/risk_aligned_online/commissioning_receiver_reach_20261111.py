"""Postseal sensitivity of a first-layer write at the realized patch-token dose."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit
import torch

from . import commissioning_lag_guard_20261111 as guard
from . import commissioning_lag_mixer_20261111 as lag
from . import commissioning_lag_teacher_20261111 as teacher
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, sha, write
from .mask_separated_intra_20261103.run import state_digest


def _norm(x):
    return x.norm(dim=(1,2),keepdim=True)


def project_write(value, valid, radius):
    value = value.masked_fill(~valid[...,None],0.)
    return value*(radius/_norm(value).clamp_min(1e-12)).clamp_max(1.)


def _hidden(base, x, m, theta, offset=None):
    captured = []

    def hook(_layer, _args, out):
        value = out if offset is None else out+offset
        captured.append(value)
        return value

    handle = base.encoder.layers[0].register_forward_hook(hook)
    try:
        h = lag._features(base,x,m,theta)
    finally:
        handle.remove()
    if len(captured)!=1:
        raise ValueError('expected one first-layer activation per forward')
    return captured[0],h


def _reach(base, x, m, theta, beta):
    with torch.no_grad():
        before,h0 = _hidden(base,x,m,torch.zeros_like(theta))
        after,h1 = _hidden(base,x,m,theta)
        delta = after-before
        actual = ((h1-h0)*beta).sum(1)
        valid = m.reshape(len(m),base.cfg.window//base.cfg.patch,-1).any(-1)
        patch = delta[:,:-1].masked_fill(~valid[...,None],0.)
        budget = _norm(patch)
        fixed = delta.clone()
        fixed[:,:-1].masked_fill_(valid[...,None],0.)
    def gain(step):
        # Preserve the original CLS displacement and any masked-token displacement.
        offset = fixed+torch.cat((step,torch.zeros_like(fixed[:,-1:])),1)
        _,h = _hidden(base,x,m,torch.zeros_like(theta),offset)
        return ((h-h0)*beta).sum(1)
    with torch.no_grad():
        torch.testing.assert_close(gain(patch),actual,atol=3e-6,rtol=0.)
    v = torch.zeros_like(patch,requires_grad=True)
    g, = torch.autograd.grad(gain(v).sum(),v)
    g = g.masked_fill(~valid[...,None],0.)
    tangent = budget*g/_norm(g).clamp_min(1e-12)
    with torch.no_grad():
        tg = gain(tangent)
        best = torch.maximum(actual,tg)
        current = torch.where((tg>=actual)[:,None,None],tangent,patch)
    for _ in range(8):
        v = current.detach().requires_grad_(True)
        g, = torch.autograd.grad(gain(v).sum(),v)
        g = g.masked_fill(~valid[...,None],0.)
        with torch.no_grad():
            current = project_write(v+.5*budget*g/_norm(g).clamp_min(1e-12),valid,budget)
            best = torch.maximum(best,gain(current))
    return tuple(t.cpu().numpy() for t in (actual,tg,best,budget.flatten()))


def run(signed, cases, output):
    lag.configure()
    lock = lag.load_inputs()
    p,s,fitted = (read(signed/k) for k in ('protocol.json','seal.json','teacher.json'))
    if p['order']!=lock['ids'] or len(s['curve_sha256'])!=360:
        raise ValueError('completed sealed 360 required')
    if sha(signed/'teacher.json')!=p['teacher_sha256']:
        raise ValueError('signed path parameters changed')
    old = Path(p['previous_teacher'])
    tp,ts = (read(old/k) for k in ('protocol.json','seal.json'))
    if sha(old/'seal.json')!=p['previous_seal']:
        raise ValueError('unbound parent curve')
    mp = read(Path(tp['previous_lag'])/'protocol.json')
    coord = next(Path(a).parent for a in p['assets']
                 if Path(a).parent.name.startswith('commissioning_local_cross_response_'))
    reference,_,args,assets = lag.parent._frozen(Path(mp['reference']),coord,lock)
    if assets!=p['assets']:
        raise ValueError('receiver parent changed')
    decoder = lag._bind_forecast(Path(mp['source_decoder']).parent,reference)
    case_rows = read(cases)['records']
    base = args[0]
    frozen = state_digest(base)
    params = torch.as_tensor(fitted['path'],device=base.center.device,dtype=base.center.dtype)
    tau = p['calibration']['0.2']
    ids = p['order']
    records = []
    for case in case_rows:
        sid = case['shot']
        rank = ids.index(sid)+1
        name = f'{rank:04d}_{sid}.npz'
        path, ancestor = signed/'shots'/name,old/'shots'/name
        if (sha(path)!=s['curve_sha256'][name] or sha(ancestor)!=ts['curve_sha256'][name]
                or sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]):
            raise ValueError('postseal case changed')
        d,down = io23.read_outcome('EAST',sid)
        if d!=case['disruptive'] or rank<=p['startup']:
            raise ValueError('case is not in the sealed diagnostic cohort')
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        signal = lag.prepare_actual(raw,rank,lock['mapping'])
        tr,_,z,beta = guard._track(signal,args)
        q,key,prior,target = teacher._mature(tr,args[2][2],args[2][3])
        slots,_ = lag.delayed_slots(q,np.arange(len(z)))
        with np.load(ancestor,allow_pickle=False) as f, np.load(path,allow_pickle=False) as newer:
            before,actual = f[lag.BASES[-1]],newer[guard.PATH]
            np.testing.assert_array_equal(q+lag.H+1,newer['release'])
        if d:
            lead = np.round((down-raw['times'])*1000.,9)
            ends = np.flatnonzero((slots>0)&(lead>30.)&(lead<=125.))
        else:
            ends = []
            for part in np.array_split(np.arange(len(z)),4):
                active = part[slots[part]>0]
                if len(active):
                    ends.append(int(active[np.argmax(before[active])]))
            ends = np.asarray(ends,dtype=int)
        if len(ends)!=case['points']:
            raise ValueError('changed number of postseal endpoints')
        ix = np.unique(slots[ends]-1)
        a,_ = guard._bases(base,signal,q[ix],prior[ix],target[ix],decoder)
        w = torch.as_tensor(key[ix],device=base.center.device)
        theta = -lag.RADIUS*guard.signed_direction(*a,w,params)
        x,m = lag._windows(base,signal,ends)
        actual_delta,tangent,best,dose = _reach(base,x,m,theta[np.searchsorted(ix,slots[ends]-1)],beta)
        expected = torch.from_numpy(z[ends]+actual_delta).cuda().sigmoid().cpu().numpy()
        np.testing.assert_allclose(expected,actual[ends],atol=3e-6,rtol=0.)
        np.testing.assert_array_equal(expected>=tau,actual[ends]>=tau)
        margin = z[ends]-logit(tau)
        np.testing.assert_allclose(float(margin.max()),case['parent_max_margin'],atol=3e-6,rtol=0.)
        records.append(dict(shot=sid,disruptive=d,points=len(ends),
            parent_max_margin=float(margin.max()),actual_max_margin=float((margin+actual_delta).max()),
            tangent_max_margin=float((margin+tangent).max()),best_max_margin=float((margin+best).max()),
            actual_max_delta=float(actual_delta.max()),best_max_delta=float(best.max()),
            median_patch_dose=float(np.median(dose)),best_endpoint=int(ends[(margin+best).argmax()])))
    if state_digest(base)!=frozen:
        raise ValueError('frozen model changed')
    summary = {}
    for d,label in ((True,'disruptive'),(False,'safe')):
        rows = [r for r in records if r['disruptive']==d]
        summary[label] = dict(shots=len(rows),points=sum(r['points'] for r in rows),
            actual_cross=[r['shot'] for r in rows if r['actual_max_margin']>=0.],
            best_cross=[r['shot'] for r in rows if r['best_max_margin']>=0.],
            median_actual_max_delta=float(np.median([r['actual_max_delta'] for r in rows])),
            median_best_max_delta=float(np.median([r['best_max_delta'] for r in rows])))
    write(output,dict(scope='POSTSEAL_RECEIVER_WRITE_WITNESS_NOT_ONLINE_NOT_UPPER_BOUND',
        receiver='after first encoder block; valid patch tokens only; actual CLS change fixed',
        dose='same endpoint-wise L2 in patch tokens as the signed-path input intervention',
        steps=8,source_seal_sha256=sha(signed/'seal.json'),case_selection_sha256=sha(cases),
        summary=summary,records=records))
    print(summary,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('signed','cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    opt = parser.parse_args()
    run(opt.signed.resolve(),opt.cases.resolve(),opt.output.resolve())
