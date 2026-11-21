"""Same-state danger/safe constrained writes on ended TRAIN shots."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from joint_paired_ball_20261106 import solve
import run_patch_utility_site_20261106 as placement
import run_prealarm_teaching_support_20261106 as teaching
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import first_alarm


b = teaching.b
PLAN = b.r.ROOT / "JOINT_PAIRED_DIRECTION_PROTOCOL_20261106.md"
SITES, SCALES = placement.SITES, placement.SCALES
ARMS = ("danger_only","paired")
BOUND, RADIUS = teaching.BOUND, b.pair.RADIUS
SMALL = 2e-5


def future_indices(t, clock, down, horizon):
    idx = np.arange(clock+9,min(clock+33,len(t)))
    lead = np.round((down-t[idx])*1000,9)
    timely = lead>10.
    return idx,timely & (lead<=horizon+1e-6),lead>horizon+1e-6,timely


def history(clock):
    return np.array([clock-16,clock-8,clock,clock+8],dtype=np.int64)


def inequalities(gd,gs,zd,zs,early,timely,tau,*,paired):
    # All these halfspaces contain zero because the frozen danger did not
    # timely-alarm and the selected safe shot never alarmed.
    pieces = [gd[:4],-gd[:4],gd[4:][early],gd[3:4]]
    caps = [np.full(8,BOUND),np.maximum(tau-zd[4:][early]-SMALL,0.),
            np.array([max(tau-float(zd[4:][timely].max())-SMALL,0.)])]
    if paired:
        pieces.extend((gs[:4],-gs[:4],gs[4:]))
        caps.extend((np.full(8,BOUND),np.maximum(tau-zs[4:]-SMALL,0.)))
    return np.concatenate(pieces),np.concatenate(caps)


def matched_grid(u):
    n = np.linalg.norm(u,axis=-1,keepdims=True)
    same = n.min(axis=1,keepdims=True)
    v = np.divide(u*same,n,out=np.zeros_like(u),where=n>1e-12)
    return v[:,:,None,:]*SCALES[None,None,:,None]


def pairing(tape, rows, prior):
    by = {q.sid:q for q in rows}
    valid = {}
    seen_late = 0
    for q in rows:
        if not q.disruptive or q.teacher["baseline"]["timely"]:
            continue
        seen_late += 1
        c = tape.data["clock"][q.blocks]
        families = teaching.families(tape,q)
        for m,fam in zip(c,families,strict=True):
            if fam not in (3,5,6,7) or m+9>=len(q.native):
                continue
            if (q.native[:m+9].astype(float)>=b.TAU).any():
                continue
            idx,canon,early,timely = future_indices(q.times,m,q.down,q.horizon)
            if canon.any():
                valid[q.sid,int(m)] = (q,int(fam),idx,canon,early,timely)
    pool = teaching.safe_pool(tape,rows)
    used = set()
    fixed = {}
    for r in prior["rows"]:
        if "safe" not in r:
            continue
        k, sk = (r["shot"],r["clock"]),(r["safe"],r["safe_clock"])
        same = (pool[0]==sk[0]) & (pool[1]==sk[1])
        if (k not in valid or sk[0] in used or same.sum()!=1
                or pool[2][same][0]!=valid[k][1]):
            raise ValueError("known reference pairing changed")
        used.add(sk[0])
        fixed[k] = (sk[0],sk[1],r["context_diff"])
    n_missing, matches = 0,[]
    for key,(q,fam,idx,canonical,early,timely) in valid.items():
        mt = key[1]
        feat = teaching.context(q,np.array([mt]))[0]
        got = fixed.get(key)
        if got is None:
            got = teaching.partner(pool,feat,fam,used)
        if got is None:
            n_missing += 1
            continue
        sid,st,dist = got
        matches.append((q,mt,by[sid],st,idx,canonical,early,timely,dict(
            shot=q.sid,clock=mt,safe=sid,safe_clock=st,fixed=key in fixed,
            context_diff=dist,canonical_frames=int(canonical.sum()))))
    return matches,dict(late=seen_late,eligible_releases=len(valid),unmatched=n_missing,
                        distinct_matched=len({k[0].sid for k in matches}))


def jacobian(base,pack,indices,site):
    p = pack.take(indices)
    u = p.tokens.new_zeros((len(indices),b.op.WIDTH),requires_grad=True)
    z = placement.forward(base,p,u,site)
    (g,) = torch.autograd.grad(z.sum(),u)
    return z.detach().cpu().numpy(),g.detach().cpu().numpy().astype(np.float64)


def solve_grid(gd,gs,zd,zs,canonical,early,timely):
    targets = np.flatnonzero(canonical)+4
    result,bounds,gaps = [],[],[]
    for paired in (False,True):
        a,c = inequalities(gd,gs,zd,zs,early,timely,b.TAU,paired=paired)
        data = [solve(gd[k],a,c,RADIUS) for k in targets]
        states,upper = zip(*data,strict=True)
        u = np.stack(states)
        ub = np.asarray(upper)
        gap = ub - np.einsum("kd,kd->k",gd[targets],u)
        if np.min(gap)<-1e-7 or (np.linalg.norm(u,axis=1)>RADIUS+1e-7).any():
            raise ValueError("affine solution exceeds its bound or radius")
        result.append(u)
        bounds.append(zd[targets]+ub-b.TAU)
        gaps.append(gap)
    return matched_grid(np.stack(result,axis=1)),np.stack(bounds,axis=1),np.stack(gaps,axis=1)


def batch_logits(base,p,indices,states,site):
    ix = np.broadcast_to(indices,(*states.shape[:-1],len(indices))).ravel()
    u = np.repeat(states.reshape(-1,b.op.WIDTH),len(indices),axis=0)
    return placement.scan(base,p,ix,p.tokens.new_tensor(u),site).reshape(*states.shape[:-1],len(indices))


def exact_forward(base,p,idx,grid,site):
    z = batch_logits(base,p,idx,grid,site)
    minus = batch_logits(base,p,idx[:4],-grid,site)
    return z,minus


def first_local(times, before, future, changed, down, tau):
    z = before.copy()
    z[future] = changed
    return first_alarm(times,z,down is not None,down,tau)


def score_grid(q,s,di,si,grid,zd,zs,minus_d,minus_s):
    gd = np.maximum(np.abs(zd[...,:4]-q.native[di[:4]]),
                    np.abs(minus_d-q.native[di[:4]])).max(-1)
    gs = np.maximum(np.abs(zs[...,:4]-s.native[si[:4]]),
                    np.abs(minus_s-s.native[si[:4]])).max(-1)
    offset = zd[...,3]-q.native[di[3]]
    dtime,stime = di[4:],si[4:]
    records = {}
    for k,name in enumerate(ARMS):
        counts = dict(danger=0,paired=0,neural_only=0,scalar_any=0,safe_reject=0)
        witness = None
        peak = None
        for a,scale in np.ndindex(grid.shape[0],grid.shape[2]):
            ix = (a,k,scale)
            if gd[ix]>BOUND:
                continue
            orig = q.native[dtime]
            alarm = first_local(q.times,q.native,dtime,zd[ix][4:],q.down,b.TAU)
            scalar = first_local(q.times,q.native,dtime,orig+offset[ix],q.down,b.TAU)
            safe = first_local(s.times,s.native,stime,zs[ix][4:],None,b.TAU)
            ok = gs[ix]<=BOUND and not safe["false_alarm"]
            if ok and scalar["timely"]:
                counts["scalar_any"]+=1
            if not alarm["timely"] or alarm["lead_ms"]>q.horizon+1e-6:
                continue
            counts["danger"]+=1
            if not ok:
                counts["safe_reject"]+=1
                continue
            counts["paired"]+=1
            if scalar["timely"]:
                continue
            counts["neural_only"]+=1
            if peak is None or alarm["lead_ms"]>peak:
                peak = alarm["lead_ms"]
                witness = dict(idx=[a,k,scale],lead_ms=peak,danger_guard=float(gd[ix]),
                               safe_guard=float(gs[ix]),anchor_delta=float(offset[ix]),
                               norm=float(np.linalg.norm(grid[ix])))
        records[name] = counts | {"witness":witness}
    return records


def probe(base,basis,view,gain,match,path):
    q,dc,s,sc,idx,canon,early,timely,note = match
    pd,ps = (b.op.encode(base,*b.bank.prior.views(base,teaching.source_raw(view,x.sid,gain)),basis)
             for x in (q,s))
    di,si = np.r_[history(dc),idx],np.r_[history(sc),np.arange(sc+9,sc+33)]
    if not (bool(pd.support[dc,0]) and bool(ps.support[sc,0])):
        return dict(note,supported=False)
    exact,arrays = {},{}
    for site in SITES:
        zd,gd = jacobian(base,pd,di,site)
        zs,gs = jacobian(base,ps,si,site)
        for a,c in ((zd,q.native[di]),(zs,s.native[si])):
            if (np.max(np.abs(a-c))>5e-5 or
                    not np.array_equal(a.astype(float)>=b.TAU,c.astype(float)>=b.TAU)):
                raise ValueError("wrong frozen zero-state risk")
        # Use the archived unedited baseline, including its float precision.
        grid,ub,gap = solve_grid(gd,gs,q.native[di],s.native[si],canon,early,timely)
        z,md = exact_forward(base,pd,di,grid,site)
        v,ms = exact_forward(base,ps,si,grid,site)
        record = score_grid(q,s,di,si,grid,z,v,md,ms)
        for j,arm in enumerate(ARMS):
            record[arm].update(linear_best_upper_margin=float(ub[:,j].max()),
                               linear_gap_max=float(gap[:,j].max()))
        arrays.update({f"{site}_{k}":x for k,x in dict(grid=grid,danger=z,safe=v,danger_minus=md,
                                                      safe_minus=ms,upper=ub,gap=gap).items()})
        exact[site] = record
    b.r.save_arrays(path,di=di,si=si,danger_time=q.times,safe_time=s.times,
        danger_native=q.native,safe_native=s.native,canon=canon,**arrays)
    return dict(note,supported=True,scalar_cap_cannot_rescue=bool(q.native[idx[timely]].max()+BOUND<b.TAU),
                exact=exact)


def summarize(records):
    keys = ("danger","paired","neural_only","scalar_any","safe_reject")
    return {site:{arm:{key:sorted({r["shot"] for r in records
                    if r["supported"] and r["exact"][site][arm][key]})
                    for key in keys}
            for arm in ARMS} for site in SITES}


def outcome(v):
    def ids(site,arm,key,views):
        return {p for name in views for p in v[name]["counts"][site][arm][key]}
    all_views = tuple(v)
    original = tuple(name for name in all_views if name in teaching.m.VIEWS)
    def site(s):
        # A selected action's scalar match is the primary test. This stronger
        # screen rejects a shot if *any* tested paired arm scalar rescues it.
        def subset(views):
            return ids(s,"paired","neural_only",views)-ids(s,"paired","scalar_any",views)-ids(s,"danger_only","scalar_any",views)
        strict = subset(original)
        new = strict - ids(s,"danger_only","neural_only",original)
        return dict(original_strict=sorted(strict),original_paired_new=sorted(new),
                    all_strict=sorted(subset(all_views)),source_learnability=len(strict)>=3)
    return {s:site(s) for s in SITES}


def run(native,masked,baseline,prior,output):
    if output.exists():
        raise FileExistsError(output)
    h,start = Hashes(),time.monotonic()
    verify = b.r.read(prior/"verification.json")
    evidence = b.r.read(prior/"result.json")
    if verify["status"]!="INDEPENDENT_FIRST_ALARM_RECOUNT_MATCH" or verify["late_menus"]!=53:
        raise ValueError("independently verified source-teacher census required")
    tapes,ids,_ = teaching.m.load(native,masked,baseline,h)
    no_mnm = tapes["no_MNM"]
    frozen = {name:(t,t.shots,None) for name,t in tapes.items()}
    proof,archive = prior/"stress_static_seal.json",prior/"stress_static.npz"
    if b.r.read(proof)["sha256"]!=h(archive):
        raise ValueError("prior stressed frozen trace seal changed")
    gains = b.r.read(prior/"donor_waveform_receipt.json")["profiles"]
    with np.load(archive,allow_pickle=False) as f:
        if not np.array_equal(f["shots"],ids):
            raise ValueError("unexpected stress shots")
        for name,profile in gains.items():
            scores = np.split(f[name],np.cumsum(f["lengths"])[:-1])
            rows = [replace(q,native=z,teacher={"baseline":first_alarm(q.times,z,q.disruptive,q.down,b.TAU)})
                    for q,z in zip(no_mnm.shots,scores,strict=True)]
            frozen[name] = (no_mnm,rows,profile["gain"])
    past = evidence["original"]|evidence["stress"]
    pair_map = {v:pairing(t,rows,past[v]) for v,(t,rows,_) in frozen.items()}
    b.r.configure()
    base,parent = b.r.load_parent(b.r.DEFAULT_PARENT)
    state = b.r.state_digest(base)
    if state!=b.r.read(native/"protocol.json")["source_state_sha256"] or parent["threshold"]!=b.r.THRESHOLD:
        raise ValueError("qualified frozen checkpoint changed")
    base.cuda()
    basis = b.op.basis(base)
    output.mkdir(parents=True,exist_ok=False)
    b.r.write_new(output/"protocol.json",dict(status="TRAIN_HINDSIGHT_DIRECTION",plan=h(PLAN),
        original=h(prior/"result.json"),source_state=state,tau=b.TAU,arms=ARMS,sites=SITES,
        bound=BOUND,norm=RADIUS,scale=SCALES.tolist(),
        matched={view:[p[-1] for p in matches] for view,(matches,_) in pair_map.items()},
        new_val_test_east_reads=0))
    results = {}
    for view,(_t,_rows,gain) in frozen.items():
        matches,note = pair_map[view]
        (output/view).mkdir()
        records = [probe(base,basis,view,gain,p,output/view/f"{p[0].sid}_{p[1]}.npz") for p in matches]
        report = summarize(records)
        results[view] = dict(**note,supported=sum(r["supported"] for r in records),counts=report,rows=records)
        print(view,note, {s:{a:{k:len(x) for k,x in y.items()} for a,y in r.items()} for s,r in report.items()},flush=True)
    if b.r.state_digest(base)!=state or any(p.grad is not None for p in base.parameters()):
        raise ValueError("source model changed")
    b.r.write_new(output/"result.json",dict(status="JOINT_DIRECTION_CAPACITY_NOT_ONLINE",views=results,
        decision=outcome(results),elapsed=time.monotonic()-start))
    print(outcome(results),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("native","masked","baseline","prior","output"):
        parser.add_argument(key,type=Path)
    opt = parser.parse_args()
    run(opt.native,opt.masked,opt.baseline,opt.prior,opt.output)
