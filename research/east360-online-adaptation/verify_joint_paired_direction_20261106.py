"""Independent full-first-alarm recount and actual module-hook witnesses."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import run_joint_paired_direction_20261106 as trial
from verify_adjacent_patch_intra_20261106 import direct, outcome_and_clock
from verify_prealarm_teaching_support_20261106 import first


b = trial.b


def recount(f,row,site,truth,tau):
    dn,sn,dt,st,di,si = (f[k] for k in
        ("danger_native","safe_native","danger_time","safe_time","di","si"))
    states,danger,safe,dminus,sminus = (f[f"{site}_{k}"] for k in
        ("grid","danger","safe","danger_minus","safe_minus"))
    for seq, m in ((di,row["clock"]),(si,row["safe_clock"])):
        np.testing.assert_array_equal(seq[:4],[m-16,m-8,m,m+8])
    d0,s0 = first(dt,dn,truth,tau),first(st,sn,None,tau)
    if (d0["timely"] or s0["false_alarm"] or np.any(di[4:]<=row["clock"]+8)
            or np.any(di[4:]>row["clock"]+32) or np.any(si[4:]<=row["safe_clock"]+8)
            or np.any(si[4:]>row["safe_clock"]+32)):
        raise ValueError("recount uses the wrong static strata or causal suffix")
    horizon = row["horizon"]
    norms = np.linalg.norm(states,axis=-1)
    if np.abs(norms[:,0]-norms[:,1]).max()>1e-8 or norms.max()>trial.RADIUS+5e-7:
        raise ValueError("actual search radii did not match")
    tally = {k:dict(danger=0,paired=0,neural_only=0,scalar_any=0,safe_reject=0)
             for k in trial.ARMS}
    best = None
    for ix in np.ndindex(states.shape[:-1]):
        dg = float(np.max(np.abs(np.stack((danger[ix][:4],dminus[ix]))-dn[di[:4]])))
        sg = float(np.max(np.abs(np.stack((safe[ix][:4],sminus[ix]))-sn[si[:4]])))
        if dg>trial.BOUND:
            continue
        d,scalar,s = dn.copy(),dn.copy(),sn.copy()
        d[di[4:]],s[si[4:]] = danger[ix][4:],safe[ix][4:]
        scalar[di[4:]] += danger[ix][3]-dn[di[3]]
        do,so,sa = first(dt,d,truth,tau),first(st,s,None,tau),first(dt,scalar,truth,tau)
        ok = not so["false_alarm"] and sg<=trial.BOUND
        c = tally[trial.ARMS[ix[1]]]
        c["scalar_any"] += int(ok and sa["timely"])
        if do["timely"] and do["lead_ms"]<=horizon+1e-6:
            c["danger"]+=1
            c["safe_reject"]+=int(not ok)
            c["paired"]+=int(ok)
            c["neural_only"]+=int(ok and not sa["timely"])
            if ok and not sa["timely"]:
                best = ix
    for arm,counts in tally.items():
        if counts!={k:row["exact"][site][arm][k] for k in counts}:
            raise ValueError(f"independent first alarm counts disagree on {site}/{arm}")
    return best,int(np.prod(states.shape[:-1]))


@torch.no_grad()
def replay_raw(base,basis,view,gain,sid,idx,state,site):
    x,a = b.bank.prior.views(base,trial.teaching.source_raw(view,sid,gain))
    x,a = x[idx].contiguous(),a[idx].contiguous()
    p = b.op.encode(base,x,a,basis)
    u = x.new_tensor(state).expand(len(idx),-1)
    if site=="input":
        return direct(base,x,a,p,u,0)
    dz = b.op.delta(p,u,0)
    def add(_, _args, value):
        return torch.cat((value[:,:-1]+dz,value[:,-1:]),dim=1)
    handle = base.encoder.layers[0].register_forward_hook(add)
    try:
        return base(x,a)["logit"].detach().cpu().numpy()
    finally:
        handle.remove()


def run(root,prealarm):
    started = time.monotonic()
    result,proto = (b.r.read(root/f"{p}.json") for p in ("result","protocol"))
    g = b.r.read(prealarm/"donor_waveform_receipt.json")["profiles"]
    advance = b.r.simple_data.advanced_times()
    b.r.configure()
    base,_ = b.r.load_parent(b.r.DEFAULT_PARENT)
    if b.r.state_digest(base)!=proto["source_state"]:
        raise ValueError("wrong raw witness base")
    base.cuda()
    basis = b.op.basis(base)
    checked,witness,records,maxerror = 0,0,0,0.
    for view,item in result["views"].items():
        gain = g[view]["gain"] if view in g else None
        used = set()
        for row in item["rows"]:
            if row["safe"] in used:
                raise ValueError("safe shot reused")
            used.add(row["safe"])
            if not row["supported"]:
                continue
            path = root/view/f"{row['shot']}_{row['clock']}.npz"
            with np.load(path,allow_pickle=False) as f:
                d,s = (outcome_and_clock(b.r.io23.path_for("J-TEXT",row[key]),len(f[role+"_time"]))
                       for key,role in (("shot","danger"),("safe","safe")))
                if not d[0] or s[0] or any(not np.array_equal(x[2],f[k+"_time"])
                                           for x,k in ((d,"danger"),(s,"safe"))):
                    raise ValueError("raw pair labels or clocks differ")
                row["horizon"] = b.r.io23.effective_horizon_ms(row["shot"],advance)[0]
                for site in trial.SITES:
                    ix,n = recount(f,row,site,d[1],proto["tau"])
                    checked += n
                    if row["fixed"] or ix is not None:
                        ix = ix if ix is not None else (0,1,0)
                        u = f[site+"_grid"][ix]
                        for role,key,indices in (("danger","shot",f["di"]),("safe","safe",f["si"])):
                            z = replay_raw(base,basis,view,gain,row[key],indices,u,site)
                            m = replay_raw(base,basis,view,gain,row[key],indices[:4],-u,site)
                            old,oldm = f[f"{site}_{role}"][ix],f[f"{site}_{role}_minus"][ix]
                            err = max(float(np.abs(z-old).max()),float(np.abs(m-oldm).max()))
                            if err>5e-5 or not np.array_equal(z.astype(float)>=proto["tau"],old.astype(float)>=proto["tau"]):
                                raise ValueError("actual hook differs from archived state write")
                            witness += 1
                            maxerror = max(maxerror,err)
            records += 1
    b.r.write_new(root/"verification.json",dict(status="INDEPENDENT_FULL_TRACE_AND_HOOK_MATCH",
        records=records,candidates=checked,raw_role_witnesses=witness,maxerror=maxerror,
        duration=time.monotonic()-started))
    print(records,checked,witness,maxerror,flush=True)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    cli.add_argument("prealarm",type=Path)
    a = cli.parse_args()
    run(a.output,a.prealarm)
