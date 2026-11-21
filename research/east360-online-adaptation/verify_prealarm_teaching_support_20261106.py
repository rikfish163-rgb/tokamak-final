"""Recount the actual whole-shot and scalar action menus, plus raw safe writes."""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

import run_prealarm_teaching_support_20261106 as study
from verify_confirmation_raw_replay_20261105 import Hashes


b = study.b


def first(t,z,down,tau):
    hits = np.flatnonzero((z.astype(float)>=tau) & ((t<=down+1e-9) if down is not None else True))
    k = int(hits[0]) if len(hits) else None
    lead = round(float((down-t[k])*1000),9) if down is not None and k is not None else None
    return dict(first=k,lead_ms=lead,timely=lead is not None and lead>10,
                false_alarm=down is None and k is not None)


def enumerate_trace(q, data, tau):
    z,valid,clocks = (data[k] for k in ("curves","allowed","clock"))
    count = dict(timely=0,canonical=0,scalar=0)
    selected = None
    for j,m in enumerate(clocks):
        a,end = m+9,min(m+33,len(q.times))
        if a>=end or np.any(q.native[:a].astype(float)>=tau):
            continue
        for k in np.flatnonzero(valid[j,1:])+1:
            side = 1 + ((int(k)-1)^1)
            if not valid[j,side]:
                raise ValueError("stored symmetric action support mismatch")
            if np.max(np.abs(z[j,[k,side],7].astype(float)-float(q.native[m+8])))>study.BOUND:
                continue
            real,shifted = q.native.copy(),q.native.copy()
            real[a:end] = z[j,k,8:8+end-a]
            shifted[a:end] += z[j,k,7]-z[j,0,7]
            y,s = (first(q.times,v,q.down,tau) for v in (real,shifted))
            count["timely"] += int(y["timely"])
            count["scalar"] += int(s["timely"])
            if y["timely"] and y["lead_ms"]<=q.horizon+1e-6:
                count["canonical"] += 1
                if selected is None:
                    selected = [j,int(k),y["first"]]
    return count | {"strict": None if count["scalar"] else selected}


@torch.no_grad()
def check_pair(base, projection, path, view, gain, q, row, shots, tau):
    s = shots[row["safe"]]
    with np.load(path,allow_pickle=False) as f:
        for label, shot in (("danger",q),("safe",s)):
            if not np.array_equal(f[label+"_time"],shot.times) or not np.array_equal(f[label+"_native"],shot.native):
                raise ValueError("paired witness native bytes changed")
        u = f["state"]
        dt, st = row["clock"], row["safe_clock"]
        for label,trace,m in (("danger",f["danger_updated"],dt),("scalar",f["danger_scalar"],dt)):
            present = np.arange(len(q.native))
            off = (present<=m+8)|(present>m+32)
            if not np.array_equal(trace[off],q.native[off]):
                raise ValueError("teaching candidate rewrote an already observed sample")
            got = first(q.times,trace,q.down,tau)
            if (label=="danger" and got["lead_ms"]!=row["lead_ms"]) or (label=="scalar" and got["timely"]):
                raise ValueError("bad recorded danger witness")
        if np.linalg.norm(u)>b.pair.RADIUS+5e-7:
            raise ValueError("transferred state norm too large")
        raw = study.source_raw(view,s.sid,gain)
        pack = b.op.encode(base,*b.bank.prior.views(base,raw),projection)
        reached = np.array([st-16,st-8,st,st+8])
        def trace(indices,param):
            p = pack.take(indices)
            return b.op.forward(base,p,p.tokens.new_tensor(param).expand(len(indices),-1),0)["logit"].cpu().numpy()
        guard = np.stack([trace(reached,v)-s.native[reached] for v in (u,-u)])
        if np.max(np.abs(f["safe_guard"]-guard))>5e-5:
            raise ValueError("paired safe guard did not replay")
        if row["pair"]=="guard_reject":
            if np.max(np.abs(guard))<=study.BOUND:
                raise ValueError("pair was not rejected by the actual guard")
        else:
            future = np.arange(st+9,st+33)
            new = trace(future,u)
            signal = f["safe_updated"]
            if np.max(np.abs(signal[future]-new))>5e-5:
                raise ValueError("paired safe update did not replay")
            actual = "false_alarm" if first(s.times,signal,None,tau)["false_alarm"] else "protected"
            if actual!=row["pair"] or np.max(np.abs(guard))>study.BOUND or first(s.times,s.native,None,tau)["false_alarm"]:
                raise ValueError("paired safe decision changed")


def verify(root):
    started,h = time.monotonic(),Hashes()
    data,proto = (b.r.read(root/f"{name}.json") for name in ("result","protocol"))
    p = root.parent
    tapes,ids,_ = study.m.load(p/"patch_utility_bank_20261106_v1",p/"patch_utility_target_views_20261106_v1",
        p/"patch_utility_selection_20261106_v1"/"result.json",h)
    src = tapes["no_MNM"]
    b.r.configure()
    base,_ = b.r.load_parent(b.r.DEFAULT_PARENT)
    if b.r.state_digest(base)!=proto["state"]:
        raise ValueError("incorrect frozen parent")
    base.cuda()
    basis = b.op.basis(base)
    stage = {name:(t.shots,t,None) for name,t in tapes.items()}
    if data["stress"]:
        seal = b.r.read(root/"stress_static_seal.json")
        fixed = root/"stress_static.npz"
        receipt = b.r.read(root/"donor_waveform_receipt.json")
        if h(fixed)!=seal["sha256"] or receipt["count"]!=20 or seal["east_outcome_reads"]:
            raise ValueError("invalid stress frozen source record")
        with np.load(fixed,allow_pickle=False) as f:
            if not np.array_equal(f["shots"],ids) or not np.array_equal(f["lengths"],[len(q.native) for q in src.shots]):
                raise ValueError("stressed source identity or time length changed")
            for view,donor in receipt["profiles"].items():
                values = np.split(f[view],np.cumsum(f["lengths"])[:-1])
                new = [replace(q,native=z,teacher={"baseline":first(q.times,z,q.down,proto["tau"])})
                       for q,z in zip(src.shots,values,strict=True)]
                stat = dict(timely=sum(q.teacher["baseline"]["timely"] for q in new),
                    false_alarm=sum(q.teacher["baseline"]["false_alarm"] for q in new),
                    induced_late=[n.sid for q,n in zip(src.shots,new,strict=True)
                        if q.teacher["baseline"]["timely"] and not n.teacher["baseline"]["timely"]],
                    induced_fp=[n.sid for q,n in zip(src.shots,new,strict=True)
                        if not q.teacher["baseline"]["false_alarm"] and n.teacher["baseline"]["false_alarm"]])
                if stat!=data["stress_baseline"][view]:
                    raise ValueError("frozen stress score recount disagrees")
                stage[view] = (new,src,donor["gain"])
    count,pairs = 0,0
    for view,(rows,tape,gain) in stage.items():
        shot = {q.sid:q for q in rows}
        answer = data["original" if gain is None else "stress"][view]
        if [q.sid for q in rows if q.disruptive and not q.teacher["baseline"]["timely"]] != [r["shot"] for r in answer["rows"]]:
            raise ValueError("late original census was truncated")
        used = set()
        for row in answer["rows"]:
            q = shot[row["shot"]]
            if gain is None:
                d = {k:tape.data[k][q.blocks] for k in study.FIELDS}
            else:
                d = b.generate(base,basis,study.source_raw(view,q.sid,gain))
            observed = enumerate_trace(q,d,proto["tau"])
            if observed!={k:row[k] for k in observed}:
                raise ValueError(f"first alarm menu recount disagrees for {view}/{q.sid}")
            count+=1
            if "safe" in row:
                if row["safe"] in used:
                    raise ValueError("safe shot counted twice in a view")
                used.add(row["safe"])
                check_pair(base,basis,root/view/f"{q.sid}.npz",view,gain,q,row,shot,proto["tau"])
                pairs+=1
        print(view,"whole-shot menus",len(answer["rows"]),flush=True)
    b.r.write_new(root/"verification.json",dict(status="INDEPENDENT_FIRST_ALARM_RECOUNT_MATCH",
        late_menus=count, safe_raw_witnesses=pairs,
        static_stress_recount=len(ids)*len(data["stress"]),elapsed=time.monotonic()-started))


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    verify(cli.parse_args().output)
