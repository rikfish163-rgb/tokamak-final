"""Census source-only internal first-alarm teaching, then stress measured gains."""

import argparse
from dataclasses import replace
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_patch_sustained_margin_20261106 as latest
from run_layer0_query_memory_20261105 import raw_for
from verify_confirmation_raw_replay_20261105 import Hashes
from verify_patch_utility_selection_20261106 import deterministic_folds, first_alarm


b, m = latest.b, latest.m
DELAY, WIDTH = 8, 32
BOUND = latest.BOUND
PLAN = b.r.ROOT / "PREALARM_TEACHING_SUPPORT_PROTOCOL_20261106.md"
MNM, SXR_MEAN, SXR_VAR, XUV_VAR = (
    b.r.io23.RAW23.index(k) for k in ("MNM","sxr_mean","sxr_var","xuv_var")
)
FIELDS = ("clock", "states", "allowed", "curves")


def perturb(raw, gain):
    gs, gx = gain
    if min(gs, gx) <= 0 or not np.isfinite(gain).all():
        raise ValueError("positive finite detector gain required")
    x, mask = raw["values"].copy(), raw["available"].copy()
    for j, g in ((SXR_MEAN, gs), (SXR_VAR, gs**2), (XUV_VAR, gx**2)):
        x[mask[:,j],j] *= g
    mask[:,MNM], x[:,MNM] = False, 0
    return dict(raw, values=x, available=mask)


def donors(values):
    if len(values) < 10:
        raise ValueError("fewer than ten observed radiation-scale donors")
    ids = np.array(sorted(values))
    measures = np.array([values[sid] for sid in ids], dtype=np.float64)
    if measures.shape != (len(ids),2) or not np.isfinite(measures).all() or (measures <= 0).any():
        raise ValueError("paired positive observed variance medians required")
    gains = np.sqrt(measures / np.median(measures, axis=0))
    rank = np.argsort(np.log(gains).mean(1), kind="stable")
    rows = {name: dict(shot=int(ids[k]), gain=gains[k].tolist())
            for name, q in (("low",.25),("high",.75))
            for k in (rank[int(q*(len(ids)-1))],)}
    if np.array_equal(rows["low"]["gain"], rows["high"]["gain"]):
        raise ValueError("cold20 did not identify two distinct gain profiles")
    return rows


def calibrate(h):
    order, source, _ = b.r.reference()
    p = b.r.COLD / "input_map.json"
    source_ids = b.r.read(p)["calibration_ids"]
    if source_ids != order[:20] or h(p) != source["input_sha256"][str(p)]:
        raise ValueError("only the original completed EAST cold20 is authorized")
    values, seen = {}, []
    with patch.object(b.r.io23, "read_outcome", side_effect=RuntimeError("no EAST outcomes")):
        for sid in source_ids:
            raw = raw_for(sid, source["raw_sha256"])
            x, a = (raw[k][:256] for k in ("values","available"))
            bands = [x[a[:,k],k] for k in (SXR_VAR,XUV_VAR)]
            n = [len(u) for u in bands]
            med = [float(np.median(u)) if len(u) >= 64 else None for u in bands]
            seen.append(dict(shot=sid, n=n, medians=med))
            if all(x is not None and x > 0 for x in med):
                values[sid] = med
    return dict(count=len(source_ids), eligible=len(values), rows=seen, profiles=donors(values))


def first_in_suffix(c, times, curves, *, down, tau):
    ix = c[:,None] + np.arange(DELAY+1, WIDTH+1)
    present = ix < len(times)
    t = times[np.minimum(ix,len(times)-1)]
    if down is not None:
        present &= (down-t)*1000 >= -1e-6
    cross = (curves[:,:,DELAY:].astype(np.float64) >= tau) & present[:,None,:]
    return np.where(cross.any(-1), c[:,None]+DELAY+1+cross.argmax(-1), -1)


def teacher(c,t,actual,scalar,allowed,*,down,horizon,tau):
    def scored(curve):
        f = first_in_suffix(c,t,curve,down=down,tau=tau)
        valid = (f>=0) & allowed
        valid[:,0] = False
        lead = np.round((down-t[np.maximum(f,0)])*1000,9)
        timely = valid & (lead>10)
        return f,timely,timely & (lead <= horizon+1e-6)

    f,r,canonical = scored(actual)
    _,s,_ = scored(scalar)
    strict = None
    if canonical.any() and not s.any():
        j,a = np.argwhere(canonical)[0]
        strict = int(j),int(a),int(f[j,a])
    return dict(timely=int(r.sum()), canonical=int(canonical.sum()), scalar=int(s.sum()),
                strict=strict)


def offline_menu(q, menu):
    c,z,a = (menu[k] for k in ("clock","curves","allowed"))
    room = c+DELAY < len(q.native)-1
    avail = latest.old.commit_mask(a,z[:,:,:DELAY],room,np.maximum.accumulate(q.native)[c],b.TAU)
    offset = z[:,:,DELAY-1] - z[:,:1,DELAY-1]
    shifted = z[:,:1] + offset[...,None]
    shifted[:,:,:DELAY] = z[:,:1,:DELAY]
    return z,shifted,avail


def local_trace(native, curve, clock):
    out = native.copy()
    i,end = clock+DELAY+1, min(clock+WIDTH+1,len(native))
    out[i:end] = curve[DELAY:DELAY+end-i]
    return out


def context(q, clock):
    z = q.native[clock+DELAY].astype(np.float64)
    return np.column_stack((z,q.times[clock+DELAY]-q.times[0],z-q.native[clock-16]))


def families(tape, q):
    present = tape.data["ordered"][q.blocks,0,-9:] > 0
    return present.reshape(-1,3,3).any(2) @ np.array([1,2,4])


def safe_pool(tape, shots):
    data = []
    for q in shots:
        if q.disruptive or q.teacher["baseline"]["false_alarm"]:
            continue
        c = tape.data["clock"][q.blocks]
        f = families(tape,q)
        allowed = (c+WIDTH < len(q.native)) & np.isin(f,(3,5,6,7))
        cs = c[allowed]
        data.extend(zip(np.full(len(cs),q.sid),cs,f[allowed],context(q,cs),strict=True))
    ids,clock,fam,ctx = zip(*data, strict=True)
    return np.asarray(ids),np.asarray(clock),np.asarray(fam),np.asarray(ctx)


def partner(pool, feat, fam, used):
    ids,c,f,ctx = pool
    d = np.abs(ctx-feat)
    possible = np.flatnonzero((d[:,0]<=1.) & (d[:,1]<=.5) & (f==fam))
    cost = d[possible] @ np.array([1.,2.,.25])
    for j in possible[np.argsort(cost,kind="stable")]:
        if ids[j] not in used:
            used.add(int(ids[j]))
            return int(ids[j]),int(c[j]),d[j].tolist()
    return None


def source_raw(view, sid, gain):
    raw = b.r.simple_data.read_shot("J-TEXT",sid)
    if gain is not None:
        return perturb(raw,gain)
    return raw if view=="native" else latest.mask(raw,view)


def safe_forward(base, projection, raw, q, clock, state):
    x, a = b.bank.prior.views(base,raw)
    p = b.op.encode(base,x,a,projection)
    protect = np.array([clock-16,clock-8,clock,clock+8])
    endpoints = np.arange(clock+DELAY+1,clock+WIDTH+1)
    idx = np.r_[protect,endpoints]
    zero = b.risk_in_batches(base,p,idx,p.tokens.new_zeros((len(idx),b.op.WIDTH)))
    if np.max(np.abs(zero-q.native[idx])) > 8e-5:
        raise ValueError("paired safe state used a different frozen predictor")
    guard = b.risk_in_batches(base,p,np.tile(protect,2),
        p.tokens.new_tensor(np.repeat([state,-state],len(protect),axis=0))).reshape(2,4)-q.native[protect]
    if np.max(np.abs(guard)) > BOUND:
        return "guard_reject",guard,None
    z = b.risk_in_batches(base,p,endpoints,p.tokens.new_tensor(np.repeat(state[None],len(endpoints),axis=0)))
    trace = q.native.copy()
    trace[endpoints] = z
    y = first_alarm(q.times,trace,False,None,b.TAU)
    return ("false_alarm" if y["false_alarm"] else "protected"),guard,trace


def count_view(base, projection, view, tape, shots, gain, output):
    by_id = {q.sid:q for q in shots}
    population = safe_pool(tape,shots)
    used, notes = set(),[]
    output.mkdir()
    for q in shots:
        if not q.disruptive or q.teacher["baseline"]["timely"]:
            continue
        if gain is None:
            data = {k:tape.data[k][q.blocks] for k in FIELDS}
        else:
            all_data = b.generate(base,projection,source_raw(view,q.sid,gain))
            if not np.array_equal(all_data["clock"],tape.data["clock"][q.blocks]):
                raise ValueError("stress changed release schedule")
            if not np.array_equal(all_data["native"],q.native):
                raise ValueError("stress baseline was not reproduced by internal action generation")
            data = {k:all_data[k] for k in FIELDS}
        c = data["clock"]
        z,scalar,avail = offline_menu(q,data)
        n = teacher(c,q.times,z,scalar,avail,down=q.down,horizon=q.horizon,tau=b.TAU)
        row = dict(shot=q.sid,baseline=q.teacher["baseline"],horizon=q.horizon, **n)
        notes.append(row)
        if n["strict"] is None:
            continue
        j,act,first = n["strict"]
        dt = int(c[j])
        d_trace, ds_trace = (local_trace(q.native,x[j,act],dt) for x in (z,scalar))
        real = first_alarm(q.times,d_trace,True,q.down,b.TAU)
        null = first_alarm(q.times,ds_trace,True,q.down,b.TAU)
        state = data["states"][j,act]
        if not real["timely"] or null["timely"] or real["first"] != first:
            raise ValueError("complete trace disagrees with the source teaching witness")
        row.update(clock=dt, lead_ms=real["lead_ms"], action=b.ORDER[act],
                   state_norm=float(np.linalg.norm(state)))
        match = partner(population,context(q,c[j:j+1])[0],families(tape,q)[j],used)
        if match is None:
            row["pair"] = "no_match"
            continue
        sid,st,diff = match
        s = by_id[sid]
        verdict,guard,s_trace = safe_forward(base,projection,source_raw(view,sid,gain),s,st,state)
        row.update(pair=verdict,safe=sid,safe_clock=st,context_diff=diff,
                   safe_guard=float(np.max(np.abs(guard))))
        fields = dict(danger_time=q.times,danger_native=q.native,danger_updated=d_trace,
                      danger_scalar=ds_trace,safe_time=s.times,safe_native=s.native,state=state,
                      safe_guard=guard)
        if s_trace is not None:
            fields["safe_updated"] = s_trace
        b.r.save_arrays(output/f"{q.sid}.npz",**fields)
    counts = {k:[p["shot"] for p in notes if p[k]>0] for k in ("timely","canonical","scalar")}
    counts["strict"] = [q["shot"] for q in notes if q["strict"] is not None]
    for k in ("no_match","guard_reject","false_alarm","protected"):
        counts[k] = [p["shot"] for p in notes if p.get("pair")==k]
    print(view,"late",len(notes),"teacher counts",{k:len(v) for k,v in counts.items()},flush=True)
    return dict(late=len(notes),counts=counts,rows=notes)


@torch.no_grad()
def stress_baselines(base, source, profiles, out):
    curves = {name:[] for name in profiles}
    for i,q in enumerate(source,1):
        raw = b.r.simple_data.read_shot("J-TEXT",q.sid)
        if not np.array_equal(raw["times"],q.times):
            raise ValueError("stress may not change source clock")
        for name,row in profiles.items():
            x,a = b.bank.prior.views(base,perturb(raw,row["gain"]))
            score = b.bank.prior.scan(base,x,a).cpu().numpy()
            curves[name].append(score)
        if i%300 == 0 or i==len(source):
            print(f"sealed static stress {i}/{len(source)} source shots",flush=True)
    p = out/"stress_static.npz"
    b.r.save_arrays(p,shots=np.array([q.sid for q in source]),
                    lengths=np.array([len(q.native) for q in source]),
                    **{name:np.concatenate(vals) for name,vals in curves.items()})
    return curves,p


def gate(original, changed, source, fold):
    raw = {s for v in original.values() for s in v["counts"]["protected"]}
    extra = {s for v in changed.values() for s in v["counts"]["protected"]} - raw
    eligible = extra & {q.sid for q in source if q.teacher["baseline"]["timely"]}
    folds = [sum(int(fold[s])==k for s in eligible) for k in range(5)]
    return dict(raw_independent=sorted(raw), stress_added_independent=sorted(eligible),
                stress_added_per_fold=folds,
                enough_for_stress_teacher=len(eligible)>=20 and sum(n>=3 for n in folds)>=4)


def run(native, masked, prior, output):
    if output.exists():
        raise FileExistsError(output)
    h,started = Hashes(),time.monotonic()
    tapes,ids,_ = m.load(native,masked,prior,h)
    source = tapes["no_MNM"].shots
    fold = dict(zip(ids,deterministic_folds(ids),strict=True))
    b.r.configure()
    base,info = b.r.load_parent(b.r.DEFAULT_PARENT)
    digest = b.r.state_digest(base)
    if digest!=b.r.read(native/"protocol.json")["source_state_sha256"] or info["threshold"]!=b.r.THRESHOLD:
        raise ValueError("frozen source digest/threshold mismatch")
    base.cuda()
    projection = b.op.basis(base)
    output.mkdir(parents=True,exist_ok=False)
    b.r.write_new(output/"protocol.json",dict(plan=h(PLAN),state=digest,tau=b.TAU,views=m.VIEWS,
        raw_train=len(ids), source_bank_seal=h(native/"seal.json"), changed_device_outcome_reads=0))
    original = {v:count_view(base,projection,v,t,t.shots,None,output/v) for v,t in tapes.items()}
    original_count = len({q for v in original.values() for q in v["counts"]["protected"]})
    profiles,changed,static = None,{},{}
    if original_count<20:
        profiles = calibrate(h)
        b.r.write_new(output/"donor_waveform_receipt.json",profiles)
        print("radiation stress profiles",profiles["profiles"],flush=True)
        curves, static_path = stress_baselines(base,source,profiles["profiles"],output)
        b.r.write_new(output/"stress_static_seal.json",dict(status="SOURCE_STRESS_STATIC_SEALED",
            sha256=h(static_path),source_shots=len(source),east_signal_reads=20,east_outcome_reads=0))
        for name,row in profiles["profiles"].items():
            new = []
            for q,z in zip(source,curves[name],strict=True):
                score = first_alarm(q.times,z,q.disruptive,q.down,b.TAU)
                new.append(replace(q,native=z,teacher={"baseline":score}))
            late = [n.sid for q,n in zip(source,new,strict=True)
                    if q.teacher["baseline"]["timely"] and not n.teacher["baseline"]["timely"]]
            false = [n.sid for q,n in zip(source,new,strict=True)
                     if not q.teacher["baseline"]["false_alarm"] and n.teacher["baseline"]["false_alarm"]]
            static[name] = dict(timely=sum(q.teacher["baseline"]["timely"] for q in new),
                false_alarm=sum(q.teacher["baseline"]["false_alarm"] for q in new),
                induced_late=late,induced_fp=false)
            changed[name] = count_view(base,projection,name,tapes["no_MNM"],new,row["gain"],output/name)
    if digest!=b.r.state_digest(base) or any(p.grad is not None for p in base.parameters()):
        raise ValueError("frozen source model changed")
    result = gate(original,changed,source,fold)
    b.r.write_new(output/"result.json",dict(status="TEACHING_SUPPORT_NOT_ONLINE",
        original=original,stress=changed,stress_baseline=static,decision=result,
        cold_waveform_reads=profiles["count"] if profiles else 0, east_outcome_reads=0,
        duration=time.monotonic()-started))
    print(result,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("native","masked","prior","output"):
        cli.add_argument(arg,type=Path)
    p = cli.parse_args()
    run(p.native,p.masked,p.prior,p.output)
