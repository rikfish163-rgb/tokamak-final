"""Teach a relation predictor and its two-step risk-aligned update on J-TEXT."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_lag_data_20261110 as lag
from . import commissioning_lag_state_20261110 as neural
from . import commissioning_transition_data_20261110 as univariate
from . import commissioning_transition_net_20261110 as update_rule
from . import io23, simple_data
from .commissioning_event_pairs_20261110 import match
from .commissioning_transfer_20261108 import shot_weights
from .commissioning_transition_net_20261110 import HORIZON, STRIDE
from .commissioning_transition_source_20261110 import (
    BATCH, SEED, STEPS, decision_points, episode, shot_coordinates,
)
from .history_gradient_metric_20261102.run import sha, write
from .mask_repair_20261104.east_evaluation import configure
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


def relation_inputs(z, mask, context, observed, ends):
    u = univariate.inputs(z, mask, context, observed, ends)
    s, valid = lag.signatures(z, mask, ends)
    return np.concatenate((u, s, valid.astype(np.float32)), axis=1)


def collect(ids, advanced, center, scale, phase):
    cols, used = [], []
    for j, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        shot = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        q = decision_points(shot['y'].astype(bool), shot['valid'])
        if len(q):
            z, m, c, cm = shot_coordinates(raw, center, scale)
            ends = q[:, None] + np.array([-2*STRIDE, -STRIDE, 0])
            x = relation_inputs(z, m, c, cm, ends.ravel()).reshape(len(q), 3, -1)
            r, s, yes = lag.feedback(z, m, ends[:, :2].ravel()+HORIZON)
            lead = (np.round((shot['down']-shot['times'][q])*1000.,9) if shot['is_disrupt']
                    else np.zeros(len(q)))
            cols.append((x, r.reshape(len(q),2,-1), s.reshape(len(q),2,-1), yes.reshape(len(q),2),
                         shot['y'][q], np.full(len(q),sid), q, np.full(len(q),shot['is_disrupt']), lead))
            used.append(sid)
        if j % 450 == 0 or j == len(ids):
            print(dict(stage=f'collect_{phase}',shots=j,total=len(ids)),flush=True)
    if not cols:
        raise ValueError(f'no episodes in {phase}')
    return dict(zip(('x','aligned','shifted','valid','risk','shot','end','disruptive','lead'),
                    (np.concatenate(c) for c in zip(*cols,strict=True)),strict=True)),used


def train(rows):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        net=neural.LagState(rows['x'].shape[-1],rows['aligned'].shape[-1]).cuda()
    x,y,m,r=(torch.from_numpy(rows[k]).cuda() for k in ('x','aligned','valid','risk'))
    w=torch.from_numpy(shot_weights(rows['shot'],rows['risk'].astype(bool))).cuda()
    rnd=torch.Generator(device='cuda').manual_seed(SEED)
    indices=torch.multinomial(w,STEPS*BATCH,replacement=True,generator=rnd).reshape(STEPS,BATCH)
    opt=torch.optim.Adam(net.parameters(),lr=.002)
    for i,index in enumerate(indices,1):
        opt.zero_grad(set_to_none=True)
        fixed,adapted,l0,l1=episode(net,x[index],y[index],m[index])
        bs,ba=(F.binary_cross_entropy_with_logits(s,r[index]) for s in (fixed,adapted))
        contrast=(l0+l1).mean()/2.
        (.5*(bs+ba)+.35*contrast).backward()
        opt.step()
        if i%240==0:
            print(dict(stage='fit',step=i,risk_fixed=float(bs.detach()),risk_adapted=float(ba.detach()),
                       order=float(contrast.detach())),flush=True)
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise FloatingPointError('nonfinite lag network')
    return net.eval().requires_grad_(False)


@torch.no_grad()
def held(net, rows, training):
    x,y,s,m=(torch.from_numpy(rows[k]).cuda() for k in ('x','aligned','shifted','valid'))
    f,r,*_=episode(net,x,y,m)
    _,c,*_=episode(net,x,s,m)
    scores=torch.stack((f,r,c),1).cpu().numpy()
    p,k=net.design(x)
    zero=net.initial(len(x))
    a,_=net.update(p[:,0],k[:,0],zero,y[:,0],m[:,0])
    b,_=net.update(p[:,0],k[:,0],zero,s[:,0],m[:,0])
    contexts=(p[:,1].tanh(),net.project(p[:,1],k[:,1],a),net.project(p[:,1],k[:,1],b))
    margin=torch.stack([(net.decoder(h)*y[:,1]).sum(-1) for h in contexts],1)[m[:,1]]
    if not len(margin):
        raise ValueError('no comparable held alignment outcomes')
    station=training['aligned'][training['valid']].mean(axis=0)
    station=station/max(np.linalg.norm(station),1e-12)
    station_margin=rows['aligned'][:,1][rows['valid'][:,1]] @ station
    pos,safe=[],[]

    def entry(i):
        return dict(shot=int(rows['shot'][i]),family=0,endpoint=int(rows['end'][i]),
                    context=np.r_[rows['x'][i,2,:2],0.],feature=scores[i])

    for sid in np.unique(rows['shot']):
        subset=rows['shot']==sid
        if rows['disruptive'][subset].any():
            ids=np.flatnonzero(subset & (rows['risk']>0) & (rows['lead']>10))
            if len(ids):
                lead=rows['lead'][ids]
                pos.append(entry(ids[np.abs(lead-np.median(lead)).argmin()]))
        else:
            safe.extend(entry(i) for i in np.flatnonzero(subset))
    pairs=match(pos,safe)
    if not pairs:
        raise ValueError('no held risk pairs')
    d=np.stack([a['feature']-b['feature'] for a,b in pairs])
    w=shot_weights(rows['shot'],rows['risk'].astype(bool))
    losses=(np.logaddexp(0.,scores)-rows['risk'][:,None]*scores).T@w
    return dict(scope='VAL160_EPISODE_DIAGNOSTIC_NOT_TEST220', pairs=len(pairs),
                risk_pair_wins=dict(zip(neural.MODES,(d>0).sum(0).tolist(),strict=True)),
                risk_bce=dict(zip(neural.MODES,losses.tolist(),strict=True)),
                pair_ids=[[a['shot'],a['endpoint'],b['shot'],b['endpoint']] for a,b in pairs],
                alignment_blocks=len(margin),
                alignment_wins=dict(zip(neural.MODES,(margin>0).sum(0).tolist(),strict=True)),
                alignment_nll=dict(zip(neural.MODES,F.softplus(-margin).mean(0).cpu().tolist(),strict=True)),
                stationary_alignment_wins=int((station_margin>0).sum()))


def run(out):
    started=time.monotonic()
    configure()
    base,_=load_parent(DEFAULT_PARENT)
    center,scale=(p.detach().numpy() for p in (base.center,base.scale))
    roles=io23.source_roles()
    adv=io23.advanced_times()
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(
        scope='SOURCE_TRAIN1353_AUXILIARY_CROSS_DIAGNOSTIC_LAG',parent_sha256=PARENT_SHA256,
        training_steps=STEPS,seed=SEED,source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        graph_data_sha256=sha(Path(lag.__file__)),model_code_sha256=sha(Path(neural.__file__)),
        univariate_code_sha256=sha(Path(univariate.__file__)),update_code_sha256=sha(Path(update_rule.__file__)),
        episodes_code_sha256=sha(Path(episode.__code__.co_filename)),producer_sha256=sha(Path(__file__))))
    rows,used=collect(roles['train'],adv,center,scale,'train')
    net=train(rows)
    test,test_used=collect(roles['val'],adv,center,scale,'val')
    val=held(net,test,rows)
    with (out/'model.npz').open('xb') as f:
        np.savez(f,**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    write(out/'result.json',dict(
        train_shots=used,val_shots=test_used,train_rows=len(rows['x']),eligible_train_feedback=int(rows['valid'].sum()),
        inputs=rows['x'].shape[-1],edges=rows['aligned'].shape[-1],held=val,
        model_sha256=sha(out/'model.npz'),seconds=time.monotonic()-started))
    print({k:v for k,v in val.items() if k!='pair_ids'},flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True,type=Path)
    run(cli.parse_args().output.resolve())
