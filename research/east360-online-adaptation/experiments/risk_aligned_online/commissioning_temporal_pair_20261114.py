"""Test whether same-shot cross-diagnostic matching supplies a risk update direction."""

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import logit
from torch import nn
from torch.nn import functional as F

from . import commissioning_diagnostic_feedback_20261114 as feedback
from . import io23, simple_data, simple_train, warning_lead_metrics as lead
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .operating_point_audit import best_at_count, peak_for_shot
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent


embed,history,group,lag = feedback.embed,feedback.history,feedback.group,feedback.lag
PATCH,HANDOFF,OFFSET = embed.PATCH,embed.PATCH,embed.OFFSET
PATCHES,PARTS = 4,3
SIDE_A = ('Mir_VV','n=1 amplitude','dx_a','dy_a')
SIDE_B = ('sxr_var','sxr_skew','xuv_var','xuv_skew')
VALUES = feedback.channels(*SIDE_A,*SIDE_B)
SELF,TRUE,OLDER,WRONG,JOINT,OPPOSITE = MODES = ('pair_only_current','risk_taught_current',
    'risk_taught_earlier','risk_taught_relative_shift','risk_taught_joint_reverse','risk_taught_opposite')
ARMS = (*embed.OLD,'legacy_common',*MODES)
PAIR_TEMP,RISK_TEMP,SEED = .25,.25,131
SOURCE_RADIUS = .89072  # Existing archived source teaching dose, not an EAST calibration.
SELF_STEPS,META_STEPS,FIT_BATCH,FIT_LR,PAIR_WEIGHT = 192,320,48,.002,.1
CASES = (81406,64581,90874,74928,70320,73405,74970,85422)


class Pairing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.left = nn.Linear(dim*2,16,bias=False)
        self.right = nn.Linear(dim*2,16,bias=False)


def observed_packets(base, signal, q):
    """Four eight-point blocks, ending with the last observed H25 sample."""
    clock = q[:,None]+np.arange(1+lag.H-PATCHES*PATCH,1+lag.H)
    x = signal['values'][clock][...,VALUES]
    m = signal['available'][clock][...,VALUES] & np.isfinite(x)
    value,valid = (torch.as_tensor(v,device=base.center.device) for v in (x,m))
    ctr,scale = (v[list(VALUES)] for v in (base.center,base.scale))
    z = ((torch.where(valid,value,ctr)-ctr)/scale).asinh()
    return z.reshape(-1,PATCHES,PATCH,len(VALUES)),m.all((1,2))


def value_weights(base):
    return base.embedding.weight.detach().unflatten(1,(PATCH,46))[:,:,list(VALUES)].flatten(1)


def similarity(head, packet, fixed, delta, *, view='current'):
    if view=='joint_reverse':
        packet = packet.flip(1)
    elif view not in ('current','relative'):
        raise ValueError(f'unknown pairing view {view}')
    # One vector for each new patch. Its two coordinates are change from the
    # immediately prior patch and departure from the first patch.
    w = (fixed[None]+delta).unflatten(-1,(PATCH,len(VALUES)))
    inc,diff = packet[:,1:]-packet[:,:-1],packet[:,1:]-packet[:,:1]
    enc = []
    for proj,cols in ((head.left,slice(0,len(SIDE_A))),(head.right,slice(len(SIDE_A),None))):
        u = [torch.einsum('btpc,bdpc->btd',z[...,cols],w[...,cols]) for z in (inc,diff)]
        z = proj(torch.cat(u,dim=-1)).tanh()
        # Near-invariant packets cannot acquire certainty by normalizing
        # numerical noise to unit length.
        enc.append(z/(z.square().sum(-1,keepdim=True)+.04).sqrt())
    sim = (enc[0]@enc[1].mT)/PAIR_TEMP
    # Keep each modality's already encoded trajectory unchanged; change only
    # its correspondence in the pairing loss.
    return sim.roll(1,dims=2) if view=='relative' else sim


def paired_loss(sim):
    positive = sim.diagonal(dim1=-2,dim2=-1)
    per = (.5*(sim.logsumexp(1)+sim.logsumexp(2))-positive).mean(1)
    marks = torch.arange(PARTS,device=sim.device)
    accuracy = .5*((sim.argmax(1)==marks).float().mean(1)+(sim.argmax(2)==marks).float().mean(1))
    return per,accuracy


def negative_gradient(head,packet,weight,*,view='current',create_graph=False):
    d = packet.new_zeros((len(packet),*weight.shape),requires_grad=True)
    loss,hit = paired_loss(similarity(head,packet,weight,d,view=view))
    g, = torch.autograd.grad(-loss.sum(),d,create_graph=create_graph)
    return g,torch.stack((loss.detach(),hit.detach()),dim=1)


def issued_grid(n):
    return np.arange(lag.FIRST,n-lag.H-1,lag.STRIDE,dtype=np.int64)


def training_examples(base,shots):
    sample,safe,disrupt = [],[],[]
    for s in shots:
        n = int(s['valid'].sum())
        q = issued_grid(n)
        packets,observed = observed_packets(base,s,q)
        for i in np.flatnonzero(observed):
            sample.append(packets[i].cpu())
        _,ready = embed.feedback_support(q,observed)
        ids = np.flatnonzero(ready)
        end = q[ids,None]+OFFSET+HANDOFF+np.array([0,8,16])
        keep = end<n
        if s['is_disrupt']:
            t = end.clip(max=n-1)
            ahead = np.round((s['down']-s['times'][t])*1000.,9)
            keep &= (s['y'][t]>0.) & (ahead>10.)
        row,col = np.nonzero(keep)
        if len(row):
            entry = (s,packets[ids[row]].cpu(),end[row,col])
            (disrupt if s['is_disrupt'] else safe).append(entry)
    if not sample or not safe or not disrupt:
        raise ValueError('both outcome classes and valid source matching examples are required')
    return torch.stack(sample),safe,disrupt


def fitting_batch(rng,groups,base):
    windows,packets = [],[]
    for cls in groups:
        for j in rng.integers(len(cls),size=FIT_BATCH//2):
            shot,frames,ends = cls[j]
            i = rng.integers(len(frames))
            windows.append(simple_data.window(shot,int(ends[i]),base.cfg.window))
            packets.append(frames[i])
    x,m = (torch.as_tensor(np.stack(z),device=base.center.device) for z in zip(*windows,strict=True))
    return x,m,torch.stack(packets).to(x.device)


def identity(x):
    return x


def directions(heads,weight,packet,prev,ready):
    shape = (len(MODES),len(packet)+1,*weight.shape)
    out = weight.new_zeros(shape)
    available = np.flatnonzero(ready)
    needed = np.unique(np.r_[available,prev[available]])
    stats = np.full((len(packet),4,2),np.nan,dtype=np.float32)
    for j,n in feedback.block_indices(needed):
        x = packet[j]
        estimates = [negative_gradient(h,x,weight,view=view) for h,view in (
            (heads[0],'current'),(heads[1],'current'),(heads[1],'relative'),(heads[1],'joint_reverse'))]
        values = torch.stack([embed.stable_direction(g,g)[:n] for g,_ in estimates])
        place = j[:n]+1
        out[0,place],out[1,place],out[3,place],out[4,place] = values
        out[5,place] = -values[1]
        # Record the diagnostic before adaptation, not hindsight efficacy.
        saved = [a[:n].detach().cpu().numpy() for _,a in estimates]
        stats[j[:n]] = np.stack(saved,axis=1)
    out[2,available+1] = out[1,prev[available]+1]
    return out,stats


def source_curves(base,heads,s,threshold):
    ends = np.flatnonzero(s['valid'])
    q = issued_grid(len(ends))
    data,ok = observed_packets(base,s,q)
    prev,keep = embed.feedback_support(q,ok)
    weights,stats = directions(heads,value_weights(base),data,prev,keep)
    slot,fraction = embed.action_clock(q,keep,ends)
    ans,spent = [],[]
    with torch.no_grad():
        for j in range(0,len(ends),group.BATCH):
            idx = slice(j,j+group.BATCH)
            x,m = lag._windows(base,s,ends[idx])
            if not fraction[idx].any():
                z = base(x,m)['logit']
                u = z.new_zeros(len(z))
                z = z[None].expand(len(MODES)+1,-1)
            else:
                wish = x.new_tensor(SOURCE_RADIUS*fraction[idx])
                h,(u,_) = embed.forward_embed(base,x,m,identity,weights[:,slot[idx]],
                                              requested=wish,values=VALUES)
                z = base.risk_head(h).squeeze(-1)
            ans.append(z.sigmoid().cpu().numpy())
            spent.append(u.cpu().numpy())
    score = np.concatenate(ans,axis=1)
    dose = np.concatenate(spent)
    if not np.isfinite(score).all() or np.max(abs(score[1:,dose==0.]-score[:1,dose==0.]),initial=0.)>3e-6:
        raise ValueError('inactive source witnesses must equal the parent')
    if s['is_disrupt']:
        valid = ((s['y'][ends]>0.) & (np.round((s['down']-s['times'][ends])*1000.,9)>10.))
    else:
        valid = np.ones(len(ends),bool)
    peaks = score[:,valid].max(1).tolist() if valid.any() else None
    common = dict(shot=s['shot'], is_disrupt=s['is_disrupt'],down=s['down'],times=s['times'][ends])
    relevant = np.flatnonzero(valid & (dose>0.))
    value = dict(shot=s['shot'],disruptive=s['is_disrupt'],n_written=int((dose>0.).sum()),
                 n_relevant=len(relevant),peaks=peaks,threshold=threshold,
                 alignment=stats[keep].mean(0).tolist() if keep.any() else None)
    return [dict(**common,scores=z) for z in score],value


def fit(out):
    start = monotonic()
    lag.configure()
    base,receipt = load_parent(DEFAULT_PARENT)
    base = base.cuda()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    roles,adv = simple_data.source_roles(),io23.advanced_times()
    out.mkdir()
    copyfile(__file__,out/'executed.py')
    learn = simple_train.load_shots(roles['train'],adv)
    normals,neg,pos = training_examples(base,learn)
    fixed = value_weights(base)
    matcher = Pairing(base.cfg.dim).cuda()
    opt = torch.optim.AdamW(matcher.parameters(),lr=FIT_LR)
    trace = []
    for step in range(SELF_STEPS):
        take = normals[rng.integers(len(normals),size=FIT_BATCH)].cuda()
        no_change = take.new_zeros((len(take),*fixed.shape))
        sim = similarity(matcher,take,fixed,no_change)
        aux,acc = paired_loss(sim)
        opt.zero_grad(set_to_none=True)
        aux.mean().backward()
        opt.step()
        if (step+1)%32==0:
            trace.append(dict(phase='matching',step=step+1,loss=float(aux.detach().mean()),
                              acc=float(acc.mean())))
    taught = deepcopy(matcher)
    opt = torch.optim.AdamW(taught.parameters(),lr=FIT_LR)
    tau = float(receipt['threshold'])
    mark = float(logit(tau))
    for step in range(META_STEPS):
        x,m,packets = fitting_batch(rng,(neg,pos),base)
        g,_ = negative_gradient(taught,packets,fixed,create_graph=True)
        delta = embed.stable_direction(g,g)
        h,(applied,_) = embed.forward_embed(base,x,m,identity,delta[None],
                                           requested=x.new_full((len(x),),SOURCE_RADIUS),values=VALUES)
        risk = base.risk_head(h[1]).flatten()
        label = risk.new_tensor([0.]*(FIT_BATCH//2)+[1.]*(FIT_BATCH//2))
        primary = F.binary_cross_entropy_with_logits((risk-mark)/RISK_TEMP,label)
        aux = paired_loss(similarity(taught,packets,fixed,g.new_zeros(g.shape)))[0].mean()
        loss = primary+PAIR_WEIGHT*aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(taught.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (step+1)%32==0:
            v = dict(phase='direction',step=step+1,risk=float(primary.detach()),
                     aux=float(aux.detach()),written=int((applied>0.).sum()))
            trace.append(v)
            print(v,flush=True)
    heads = [h.eval().requires_grad_(False) for h in (matcher,taught)]
    val = {k:[] for k in ('source_parent',*MODES)}
    witness = []
    for s in simple_train.load_shots(roles['val'],adv):
        rs,w = source_curves(base,heads,s,tau)
        witness.append(w)
        for k,r in zip(val,rs,strict=True):
            val[k].append(r)
    result = {k:simple_train.metrics(rs,tau) for k,rs in val.items()}
    torch.save([h.state_dict() for h in heads],out/'pairing.pt')
    write(out/'source_result.json',dict(source=PARENT_SHA256,train=roles['train'],val=roles['val'],
        label='max(25ms, AdvancedTime)',threshold=tau,loss=trace,channels=VALUES,
        groups=[SIDE_A,SIDE_B],fit_seconds=monotonic()-start,match_packets=len(normals),
        training_shots=[len(neg),len(pos)],training_endpoints=[sum(len(v[1]) for v in g) for g in (neg,pos)],
        seed=SEED,meta_steps=META_STEPS,match_steps=SELF_STEPS,lr=FIT_LR,pair_weight=PAIR_WEIGHT,
        source_radius=SOURCE_RADIUS,validation=result,witness=witness))
    print(dict(stage='source_done',fit_shots=[len(neg),len(pos)],packets=len(normals),
               val={k:(v['tp'],v['fp']) for k,v in result.items()}),flush=True)


def trial(signal,args,dec,path,heads,arc):
    base = args[0]
    tr = group.parent_track(signal,args,dec,path)
    q = tr[0]
    np.testing.assert_array_equal(q+lag.H+1,arc['release'])
    packet,good = observed_packets(base,signal,q)
    prev,common = embed.feedback_support(q,good)
    common &= arc['eligible']
    weights,stats = directions(heads,value_weights(base),packet,prev,common)
    curve,spent,err = embed.render(base,signal,tr,weights,common,arc,modes=MODES,values=VALUES)
    j = np.flatnonzero(common)+1
    angles = (weights[1:2,j].flatten(2)*weights[:,j].flatten(2)).sum(-1)
    return curve,dict(q=q,qualified=common,dose=spent,angles=angles.cpu().numpy(),
                      matching=stats[common]),err


def seal(out,source,only):
    start = monotonic()
    lock,p,old,args,decoder,path = history.inputs()
    frozen = {k:v.detach().cpu().clone() for k,v in args[0].state_dict().items()}
    receipt = read(source/'source_result.json')
    roles = simple_data.source_roles()
    if (p['source_sha256']!=receipt['source'] or receipt['train']!=roles['train'] or
            receipt['val']!=roles['val'] or receipt['channels']!=list(VALUES)):
        raise ValueError('matching source TRAIN-only pairing model required')
    heads = []
    for h in torch.load(source/'pairing.pt',map_location='cpu',weights_only=True):
        module = Pairing(args[0].cfg.dim)
        module.load_state_dict(h)
        heads.append(module.cuda().eval().requires_grad_(False))
    if len(heads)!=2:
        raise ValueError('both source-only pretraining and risk direction controls are required')
    out.mkdir()
    (out/'shots').mkdir()
    copyfile(__file__,out/'executed.py')
    write(out/'protocol.json',dict(**{k:p[k] for k in history.LOCK},source=str(source),only=only,arms=ARMS,
        scope='unlabeled within-shot numeric embedding, no persistent update or extra teacher labels',
        groups=[SIDE_A,SIDE_B],offset=OFFSET,handoff=HANDOFF,paired_budget='same realized token L2',
        feedback='in-shot pairing of successive eight-point changes in completed H25 packets'))
    names,prefix,max_error = [],{},np.zeros(2)
    with (patch.object(io23,'read_outcome',side_effect=RuntimeError('outcomes forbidden before seal')),
          patch.object(group.hdf,'_raw_truth',side_effect=RuntimeError('HDF5 truth forbidden before seal'))):
        for rank,sid in enumerate(p['order'],1):
            if only is not None and rank not in only:
                continue
            arc = history.archived_shot(rank,sid,old)
            if rank<=p['startup']:
                change = dict.fromkeys(ARMS[len(embed.OLD):],arc['bias_only'])
                detail = dict(q=np.empty(0,np.int64),qualified=np.empty(0,bool),dose=np.zeros_like(arc['dose']),
                              angles=np.empty((len(MODES),0)),matching=np.empty((0,4,2)))
            else:
                if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
                    raise ValueError('locked EAST signals changed')
                raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
                signal = lag.prepare_actual(raw,rank,lock['mapping'])
                change,detail,err = trial(signal,args,decoder,path,heads,arc)
                max_error = np.maximum(max_error,err)
                if rank in (21,120,360) or sid in CASES or only is not None:
                    n = min(len(raw['times']),650)
                    short,a = feedback.trimmed(signal,arc,n)
                    previous,d,_ = trial(short,args,decoder,path,heads,a)
                    np.testing.assert_array_equal(d['q'],detail['q'][:len(d['q'])])
                    np.testing.assert_array_equal(d['qualified'],detail['qualified'][:len(d['q'])])
                    np.testing.assert_allclose(d['dose'],detail['dose'][:n],atol=2e-6,rtol=0.)
                    for k,v in previous.items():
                        np.testing.assert_allclose(v,change[k][:n],atol=3e-6,rtol=0.)
                        for tau in p['calibration'].values():
                            np.testing.assert_array_equal(v>=tau,change[k][:n]>=tau)
                    prefix[str(sid)] = dict(points=n,probability=max(float(abs(v-change[k][:n]).max())
                                                                                for k,v in previous.items()))
            name = f'{rank:04d}_{sid}.npz'
            with (out/'shots'/name).open('xb') as f:
                np.savez_compressed(f,**{k:arc[k] for k in embed.OLD},**change,**detail)
            names.append(name)
            if only is not None or rank in (20,60,120,180,240,300,360):
                print(dict(rank=rank,seconds=round(monotonic()-start,2)),flush=True)
    for k,v in args[0].state_dict().items():
        if not torch.equal(v.detach().cpu(),frozen[k]):
            raise ValueError(f'unintended persistent change in {k}')
    write(out/'seal.json',dict(shots=names,prefix=prefix,maximum=max_error.tolist(),seconds=monotonic()-start))


def score(out):
    p,s = (read(out/f'{k}.json') for k in ('protocol','seal'))
    order = p['order']
    if p['only'] is not None or len(order)!=360 or s['shots']!=[f'{i:04d}_{j}.npz' for i,j in enumerate(order,1)]:
        raise ValueError('complete fixed stream required')
    old = read(history.ARCHIVE/'per_shot.json')
    rows = {b:{k:[] for k in ARMS} for b in p['calibration']}
    ranks = {k:[] for k in ARMS}
    cutoff = p['calibration']['0.2']
    late,eligibility,diag,angles = [],[],[],[]
    for i,(sid,name) in enumerate(zip(order,s['shots'],strict=True)):
        y,down,t = group.hdf._raw_truth(sid)
        with np.load(out/'shots'/name,allow_pickle=False) as f:
            z = dict(f)
        active = z['dose']>0.
        for k in ARMS:
            if k not in embed.OLD:
                np.testing.assert_array_equal(z[k][~active],z['bias_only'][~active])
            for c,tau in p['calibration'].items():
                item = dict(shot=sid,disruptive=y,down_s=down,**score_curve(t,z[k],y,down,
                            p['original'] if i<p['startup'] else tau))
                if k in embed.OLD and item!=old[c][k][i]:
                    raise ValueError('reference or scoring clock changed')
                rows[c][k].append(item)
            ranks[k].append(dict(disruptive=y,peak=peak_for_shot(t,z[k],y,down)))
        eligibility.append(dict(shot=sid,feedback=int(z['qualified'].sum()),written=int(active.sum())))
        if z['qualified'].any():
            diag.append(z['matching'].mean(0))
            angles.append(z['angles'].mean(1))
        item = rows['0.2']['current_5'][-1]
        if i>=p['startup'] and y and not item['timely']:
            time = np.round((down-t)*1000.,9)
            q = (time>30.) & (time<=125.) & active
            h = {k:bool((z[k][q]>=cutoff).any()) for k in ('bias_only',*MODES)}
            late.append(dict(shot=sid,kind='late' if item['alarm'] else 'miss',support=int(q.sum()),near=h,
                gap=float(logit(cutoff)-logit(z['bias_only'][q].max())) if q.any() else None,
                max_lift={k:float(max(logit(z[k][q].clip(1e-7,1.-1e-7))-
                        logit(z['bias_only'][q].clip(1e-7,1.-1e-7)))) for k in MODES} if q.any() else {}))
    metrics = {c:{k:lead.summarize(v) for k,v in a.items()} for c,a in rows.items()}
    pairs = [('bias_only',TRUE),('legacy_common',TRUE),('current_5',TRUE),*( (k,TRUE) for k in MODES if k!=TRUE)]
    comparison = {c:{f'{a}__to__{b}':flips(r[a],r[b]) for a,b in pairs} for c,r in rows.items()}
    cold = metrics['0.2']['bias_only']
    cold_fp = sum(not r['disruptive'] and r['alarm'] for r in rows['0.2']['bias_only'][:p['startup']])
    cold_tp = sum(r['timely'] for r in rows['0.2']['bias_only'][:p['startup']])
    budgets = {str(n):{k:dict(tp=b['tp']+cold_tp,fp=b['fp']+cold_fp) for k,v in ranks.items()
                       for b in (best_at_count(v[p['startup']:],n-cold_fp),)}
               for n in (int(.1*cold['n_safe']),int(.15*cold['n_safe']),int(.2*cold['n_safe']))}
    write(out/'per_shot.json',rows)
    write(out/'result.json',dict(metrics=metrics,paired=comparison,eligible=eligibility,hard=late,
        raw_verified=len(order)*len(ARMS)*len(metrics),suffix_rank_only=budgets,
        mean_pairing_per_shot=dict(zip(('pair_only','taught','relative','joint'),
            np.mean(diag,axis=0).tolist(),strict=True)),
        mean_direction_cosine=dict(zip(MODES,np.mean(angles,axis=0).tolist(),strict=True))))
    print({c:{k:(r['by_deadline']['30']['tp'],r['fp']) for k,r in a.items()} for c,a in metrics.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    cli.add_argument('--source',type=Path)
    action = cli.add_mutually_exclusive_group()
    action.add_argument('--fit',action='store_true')
    action.add_argument('--score',action='store_true')
    action.add_argument('--smoke-rank',action='append',type=int)
    a = cli.parse_args()
    if a.fit:
        fit(a.out.resolve())
    elif a.score:
        score(a.out.resolve())
    elif a.source is None:
        cli.error('--source is required to replay')
    else:
        seal(a.out.resolve(),a.source.resolve(),a.smoke_rank)
