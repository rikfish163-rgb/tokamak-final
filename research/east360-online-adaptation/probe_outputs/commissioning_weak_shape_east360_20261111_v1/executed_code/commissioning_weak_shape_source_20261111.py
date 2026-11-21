"""Train a shape adapter against similar safe transients on J-TEXT TRAIN."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import commissioning_lag_state_20261110 as neural
from . import commissioning_weak_shape_data_20261111 as shape
from . import io23, simple_data
from .commissioning_transfer_20261108 import features, shot_weights
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE
from .commissioning_transition_source_20261110 import (
    BATCH, SEED, STEPS, decision_points, episode, shot_coordinates,
)
from .history_gradient_metric_20261102.run import sha, write
from .mask_repair_20261104.east_evaluation import configure
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


TAIL_S = .125
PAIR_MARGIN = .5
PAIR_WEIGHT = .5
ORDER_WEIGHT = .35
FIRST_EFFECT = FIRST + HORIZON + 1


def code_bindings():
    paths = (Path(__file__), Path(shape.__file__), Path(neural.__file__),
             *(Path(f.__code__.co_filename) for f in
               (shape._window_mean, neural.move, episode, shot_coordinates, simple_data.read_shot,
                io23.read_signals)))
    return {p.name: sha(p) for p in paths}


def timely_mask(rows):
    return rows['disruptive'] & (rows['risk'] > 0) & (rows['lead'] > 10.)


def make_pairs(rows, scale, *, random=None):
    """Give each eligible timely endpoint a safe donor from each stratum."""
    ctx = rows['pair_context'].astype(np.float64) / scale
    anchors = np.flatnonzero(timely_mask(rows) & rows['matchable'])
    safe = ~rows['disruptive'] & rows['matchable']
    if not len(anchors):
        raise ValueError('no matchable timely source risk endpoints')
    donors = []
    for pool in (safe & rows['terminal'], safe & ~rows['terminal']):
        candidates = np.flatnonzero(pool)
        if not len(candidates):
            raise ValueError('each safe source donor stratum needs eligible windows')
        if random is None:
            nearest = []
            for block in np.array_split(anchors, (len(anchors)+127)//128):
                dist = ((ctx[block, None, :] - ctx[candidates][None, :, :])**2).mean(-1)
                nearest.append(candidates[dist.argmin(axis=1)])
            donors.append(np.concatenate(nearest))
        else:
            donors.append(random.choice(candidates, size=len(anchors)))
    pos = np.repeat(anchors, 2)
    neg = np.stack(donors, axis=1).ravel()
    return pos, neg, rows['terminal'][neg], np.sqrt(((ctx[pos]-ctx[neg])**2).mean(-1))


def collect(ids, advanced, center, scale, role):
    chunks, used = [], []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        shot = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        t, d = shot['times'], shot['is_disrupt']
        q = decision_points(shot['y'].astype(bool), shot['valid'])
        if not d:
            tail = np.flatnonzero(shot['valid'] & (t >= t[-1]-TAIL_S) &
                                  (np.arange(len(t)) >= FIRST + 2*STRIDE))
            q = np.union1d(q, tail[::8])
        if len(q):
            z, m, ctx, cm = shot_coordinates(raw, center, scale)
            ends = q[:, None] + np.array([-2*STRIDE, -STRIDE, 0])
            x, matched_on = shape.inputs(z, m, ctx, cm, ends.ravel())
            real, shifted, eligible = shape.feedback(z, m, ends[:, :2].ravel())
            bins = q[:, None] + np.arange(-47, 1)
            # Missing diagnostics are not low-amplitude negatives for matching.
            have = (m[bins[:, :32]].sum(1) >= 16) & (m[bins[:, 32:]].sum(1) >= 8)
            observed = (cm[bins[:, 16:]].reshape(len(q), 2, 16, 2).sum(2) >= 8).all((1, 2))
            for g in range(4):
                observed &= have[:, np.asarray(shape.GROUPS)==g].any(1)
            matched_on = matched_on.reshape(len(q), 3, 7)[:, -1]
            terminal = (~np.full(len(q), d) & (t[q]>=t[-1]-TAIL_S) & (matched_on[:, 2]<0))
            lead = np.round((shot['down']-t[q])*1000.,9) if d else np.zeros(len(q))
            chunks.append((x.reshape(len(q),3,-1), real.reshape(len(q),2,-1),
                           shifted.reshape(len(q),2,-1), eligible.reshape(len(q),2), shot['y'][q],
                           np.full(len(q),sid), q, np.full(len(q),d), lead, matched_on, observed, terminal))
            used.append(int(sid))
        if i%450 == 0 or i == len(ids):
            print(dict(stage=f'collect_{role}',shots=i,total=len(ids)),flush=True)
    if not chunks:
        raise ValueError(f'no usable source episodes in {role}')
    keys=('x','real','shifted','eligible','risk','shot','end','disruptive','lead',
          'pair_context','matchable','terminal')
    return dict(zip(keys,(np.concatenate(c) for c in zip(*chunks,strict=True)),strict=True)),used


def amplitude(rows, scale):
    return np.linalg.norm(rows['pair_context'][:, 3:] / scale[3:], axis=1)


def train(rows, pairs, weak, kind):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        net = neural.LagState(rows['x'].shape[-1], rows['real'].shape[-1]).cuda()
    x, real, eligible, y = (torch.from_numpy(rows[k]).cuda()
                           for k in ('x','real','eligible','risk'))
    bce_w = torch.from_numpy(shot_weights(rows['shot'], rows['risk'].astype(bool))).cuda()
    rng = torch.Generator(device='cuda').manual_seed(SEED)
    draws = torch.multinomial(bce_w, STEPS*BATCH, True, generator=rng).reshape(STEPS,BATCH)
    p, n, _, _ = pairs
    _, inv, count = np.unique(rows['shot'][p],return_inverse=True,return_counts=True)
    w = torch.from_numpy((1. + weak[p]) / count[inv]).cuda()
    draws_pair = torch.multinomial(w, STEPS*(BATCH//2), True, generator=rng).reshape(STEPS,-1)
    anchors, negatives = (torch.from_numpy(i).cuda() for i in (p, n))
    opt = torch.optim.Adam(net.parameters(), lr=.002)
    for step, (ix, ip) in enumerate(zip(draws, draws_pair, strict=True),1):
        opt.zero_grad(set_to_none=True)
        combined = torch.cat((ix, anchors[ip], negatives[ip]))
        fixed, adapted, a, b = episode(net,x[combined],real[combined],eligible[combined])
        point, rank = [], []
        for s in (fixed, adapted):
            current, pos, safe = s.split((len(ix),len(ip),len(ip)))
            point.append(F.binary_cross_entropy_with_logits(current,y[ix]))
            rank.append(F.softplus(PAIR_MARGIN-pos+safe).mean())
        order = (a[:len(ix)] + b[:len(ix)]).mean()/2.
        objective = .5*sum(point) + ORDER_WEIGHT*order + .5*PAIR_WEIGHT*sum(rank)
        objective.backward()
        opt.step()
        if step%240 == 0:
            print(dict(stage=kind, step=step, loss=float(objective.detach()),
                       paired=float((.5*sum(rank)).detach())),flush=True)
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise FloatingPointError('nonfinite shape network')
    return net.eval().requires_grad_(False)


@torch.no_grad()
def held(net, rows, pairs, weak):
    x, y, s, m = (torch.from_numpy(rows[k]).cuda() for k in ('x','real','shifted','eligible'))
    f, r, *_ = episode(net,x,y,m)
    _, c, *_ = episode(net,x,s,m)
    logits = torch.stack((f,r,c),1).cpu().numpy()
    p, n, tail, _ = pairs
    dif = logits[p] - logits[n]
    summary = {}
    for name, subset in (('all',np.ones(len(p),bool)),('weak',weak[p]),
                         ('falling_tail',tail),('other_safe',~tail),
                         ('weak_vs_falling_tail',weak[p] & tail)):
        delta = dif[subset]
        summary[name] = dict(n=len(delta),wins=dict(zip(neural.MODES,(delta>0).sum(0).tolist(),strict=True)),
                             ties=dict(zip(neural.MODES,(delta==0).sum(0).tolist(),strict=True)))
    pre, key = net.design(x)
    zero = net.initial(len(x))
    ar, _ = net.update(pre[:,0],key[:,0],zero,y[:,0],m[:,0])
    ac, _ = net.update(pre[:,0],key[:,0],zero,s[:,0],m[:,0])
    states = (pre[:,1].tanh(), net.project(pre[:,1],key[:,1],ar),net.project(pre[:,1],key[:,1],ac))
    q = torch.stack([(net.decoder(h)*y[:,1]).sum(-1) for h in states],1)[m[:,1]]
    if not len(q):
        raise ValueError('no released held feedback for order comparison')
    return dict(rank=summary, held_feedback=len(q),
                held_order_wins=dict(zip(neural.MODES,(q>0).sum(0).tolist(),strict=True)),
                held_order_nll=dict(zip(neural.MODES,F.softplus(-q).mean(0).cpu().tolist(),strict=True)))


def encode(base, net, signal, center, scale):
    raw, _ = features(base,signal,'cuda')
    z,m,c,cm = shot_coordinates(signal,center,scale)
    q = np.arange(FIRST,len(z)-HORIZON,STRIDE)
    x,_ = shape.inputs(z,m,c,cm,np.arange(len(z)))
    r,s,ok = shape.feedback(z,m,q)
    views,audit = neural.replay(net,x,q,r,s,ok)
    for view in views.values():
        view[:FIRST] = 0.
    return raw,views,audit


def pair_report(rows, pairs):
    p,n,tail,d = pairs
    return dict(n=len(p),positive_shots=int(len(np.unique(rows['shot'][p]))),
                safe_donors=int(len(np.unique(rows['shot'][n]))),
                falling_tail_donors=int(len(np.unique(rows['shot'][n[tail]]))),
                median_rms_distance=float(np.median(d)))


def run(out):
    started = time.monotonic()
    configure()
    base,_ = load_parent(DEFAULT_PARENT)
    center,scale = (p.detach().numpy() for p in (base.center,base.scale))
    roles,adv = io23.source_roles(),io23.advanced_times()
    out.mkdir(parents=True)
    write(out/'protocol.json',dict(
        scope='SOURCE_TRAIN1353_CONTEXT_MATCHED_WEAK_SHAPE',parent_sha256=PARENT_SHA256,
        source_label_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),code_sha256=code_bindings(),
        steps=STEPS,seed=SEED,batch=BATCH,margin=PAIR_MARGIN,pair_weight=PAIR_WEIGHT,
        order_weight=ORDER_WEIGHT,weak_weight=2,first_effect=FIRST_EFFECT,tail_s=TAIL_S,
        source_fit='TRAIN1353 only',holdout='VAL160 diagnostics only; not used to select EAST checkpoint',
        ordered_target='five observed five-point means, centered and normalized; same-window roll-by-two null',
        primary='context_matched',negative_control='random_safe_same_stratum'))
    rows,used = collect(roles['train'],adv,center,scale,'train')
    valid_context = rows['pair_context'][rows['matchable'] & (~rows['disruptive'] | timely_mask(rows))]
    quartiles = np.quantile(valid_context,[.25,.75],axis=0)
    match_scale = np.maximum(quartiles[1]-quartiles[0],.05)
    weak_cut = float(np.median(amplitude(rows,match_scale)[timely_mask(rows) & rows['matchable']]))
    weak = amplitude(rows,match_scale) <= weak_cut
    match = make_pairs(rows,match_scale)
    random = make_pairs(rows,match_scale,random=np.random.default_rng(SEED))
    print(dict(stage='matched',training=pair_report(rows,match),random=pair_report(rows,random)),flush=True)
    val,val_used = collect(roles['val'],adv,center,scale,'val')
    vp,vn,vt,vd = make_pairs(val,match_scale)
    # Report one timely anchor per held shot, with both types of negative.
    use = []
    for sid in np.unique(val['shot'][vp]):
        current = vp[::2][val['shot'][vp[::2]]==sid]
        lead = val['lead'][current]
        use.append(current[np.argmin(abs(lead-np.median(lead)))])
    m = np.isin(vp,use)
    test = (vp[m],vn[m],vt[m],vd[m])
    weak_val = amplitude(val,match_scale)<=weak_cut
    arms = {}
    for kind,pairs in (('context_matched',match),('random_safe_same_stratum',random)):
        net = train(rows,pairs,weak,kind)
        path = out/('model.npz' if kind=='context_matched' else 'random_pairs_model.npz')
        with path.open('xb') as f:
            np.savez(f,**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
        arms[kind] = dict(training_pairs=pair_report(rows,pairs),held=held(net,val,test,weak_val),
                          model_sha256=sha(path))
        print(dict(stage='source_held',arm=kind,**arms[kind]['held']),flush=True)
    write(out/'result.json',dict(
        train_shots=used,val_shots=val_used,train_rows=len(rows['x']),val_rows=len(val['x']),
        eligible_train_feedback=int(rows['eligible'].sum()),inputs=rows['x'].shape[-1],
        edges=rows['real'].shape[-1],model_sha256=arms['context_matched']['model_sha256'],
        match_scale=match_scale.tolist(),weak_amplitude_cutoff=weak_cut,
        held_pairs=[dict(positive=int(val['shot'][p]),positive_endpoint=int(val['end'][p]),
                         safe=int(val['shot'][n]),safe_endpoint=int(val['end'][n]),
                         falling_tail=bool(t),distance=float(d),weak=bool(weak_val[p]))
                    for p,n,t,d in zip(*test,strict=True)],
        held_pair_report=pair_report(val,test),arms=arms,seconds=time.monotonic()-started))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True,type=Path)
    run(cli.parse_args().output.resolve())
