"""Project the frozen directions into an observable next-window comparison."""

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_direction_evidence_20261111 as policy
from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as training
from . import commissioning_split_learning_20261111 as direct
from . import simple_data
from .commissioning_source_teaching_east_20261111 import match_magnitude


FIRST_EVIDENCE = persistent.FIRST_EFFECT+persistent.STRIDE
STEPS = 256


@torch.no_grad()
def views(net, signal, ends, params, old_map, quiet, maps):
    """Issue two diagnostic counterfactuals; compare only after their targets arrive."""
    n = len(signal['values'])
    q = np.arange(direct.FIRST,n,persistent.STRIDE)
    points = np.union1d(ends,q)
    tr, pre, ready = training.observed_track(net,signal,points,params,old_map)
    v,_ = persistent.replay(*ready,quiet)
    k = torch.from_numpy(tr['key'])
    state = 4.*torch.from_numpy(v['sustained'][points])
    a,b = (torch.einsum('...k,...f,kfw->...w',k,state,maps[tag]) for tag in ('legacy','native'))
    b = match_magnitude(b,a)
    decoder, noise = params[2:4]
    base = direct.base_forecast(tr,decoder)
    bare = torch.from_numpy(tr['pre'])
    origin = np.searchsorted(points,q)
    h = bare[origin]
    projection = torch.from_numpy(decoder[0][:,:direct.WIDTH])
    forecast = []
    for d in (a,b):
        effect = ((h+d[origin]).tanh()-h.tanh()) @ projection.T
        forecast.append(base+effect.numpy().reshape(base.shape))
    e,stats = policy.decision_evidence(q,*forecast,direct.normalized_target(tr,noise),
                                       tr['observed'],n)
    take = np.searchsorted(points,ends)
    return dict(pre=pre[take], old=a[take], native=b[take], evidence=e[ends], bare=bare[take],
                current=training.view(pre[take],k[take],
                    4.*torch.from_numpy(v['current_equal_norm'][ends]),maps['legacy']),
                stats=stats)


def pack(net, manifest, ids, params, old_map, quiet, maps, role):
    # Union retains both teaching geometries. Outcome labels are used only in this
    # completed-source task, never by the forecasts or the direction selector.
    pair = sorted({(r['positive_shot'],r['positive'],r['donor_shot'],r['donor'])
                   for mode in ('coverage','native') for r in manifest[mode]})
    needs = {}
    for ds,dp,ss,sp in pair:
        needs.setdefault(ds,set()).add(dp)
        needs.setdefault(ss,set()).add(sp)
    if not set(needs)<=set(ids) or not pair:
        raise ValueError('source pair manifest must be within its locked role')
    table = {}
    keys = ('pre','old','native','evidence')
    for j,(sid,indices) in enumerate(sorted(needs.items()),1):
        raw = simple_data.read_shot('J-TEXT',sid)
        ends = np.asarray(sorted(indices))
        rows = views(net,raw,ends,params,old_map,quiet,maps)
        for i,end in enumerate(ends):
            table[sid,end] = {k:torch.as_tensor(rows[k][i]) for k in keys}
        if j%150==0 or j==len(needs):
            print(dict(stage=role,shots=j,total=len(needs)),flush=True)
    data = {k:torch.stack([torch.stack((table[ds,dp][k],table[ss,sp][k]))
                           for ds,dp,ss,sp in pair]) for k in keys}
    pos = np.asarray([a[0] for a in pair])
    _,inv,count = np.unique(pos,return_inverse=True,return_counts=True)
    bal = 1./count[inv]
    return data,torch.as_tensor(bal/bal.sum(),dtype=torch.float32),dict(
        pairs=len(pair),disruptive_shots=len(count),total_shots=len(needs),
        pairs_with_evidence=int(data['evidence'].abs().any((1,2)).sum()))


def fit(data, sample_weight, beta):
    """Fit a 5-variable source gate and the exact same objective without evidence."""
    x,pre,a,b = (data[k] for k in ('evidence','pre','old','native'))
    norm = x.reshape(-1,x.shape[-1]).std(dim=0).clamp_min(.05)
    slope = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    intercept = torch.nn.Parameter(torch.zeros(()))
    fixed = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam((slope,intercept,fixed),lr=.025)
    def loss(w):
        z = (pre+policy.mix_direction(a,b,w)).tanh() @ beta
        return F.softplus((z[:,1]-z[:,0])/.35) @ sample_weight
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        risk = loss((intercept+(x/norm)@slope).sigmoid())+loss(fixed.sigmoid())
        (risk + .005*slope.square().sum() + .005*(intercept.square()+fixed.square())).backward()
        opt.step()
    w = intercept.detach(), (slope.detach()/norm), fixed.detach()
    if not all(torch.isfinite(a).all() for a in w):
        raise FloatingPointError('nonfinite direction gate')
    return w


@torch.no_grad()
def diagnostics(data, beta, gate):
    x,pre,a,b = (data[k] for k in ('evidence','pre','old','native'))
    prior,coef,const = gate
    weights = dict(legacy=torch.zeros_like(x[...,0]),native=torch.ones_like(x[...,0]),
                   midpoint=torch.full_like(x[...,0],.5),constant=const.sigmoid(),
                   agreement=.5+.5*x[...,0],learned=(prior+x@coef).sigmoid(),
                   neutral=prior.sigmoid())
    result={}
    for k,w in weights.items():
        z = (pre+policy.mix_direction(a,b,w)).tanh() @ beta
        result[k] = dict(wins=int((z[:,0]>z[:,1]).sum()),
                         margin=float((z[:,0]-z[:,1]).mean()))
    # For a known disruption (safe window), the better candidate raises
    # (lowers) the risk component. This is a retrospective mechanism check.
    effect = ((pre+b).tanh()-(pre+a).tanh()) @ beta
    utility = effect*effect.new_tensor([1.,-1.])
    valid = (utility!=0.) & (x[...,0]!=0.)
    result['forecast_agreement_on_measured_windows'] = dict(
        windows=int(valid.sum()),correct_sign=int(((utility*x[...,0]>0.)&valid).sum()),
        native_preferred=int(((utility>0.)&valid).sum()))
    return result
