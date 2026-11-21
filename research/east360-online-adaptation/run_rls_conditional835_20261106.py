"""Replay archived direct diagnostic RLS on the existing conditional EAST cohort."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from experiments.risk_aligned_online import simple23
from experiments.risk_aligned_online.direct_fast_memory_20261104 import memory, model, rls, timeline
from rolling_safety_policy_20261106 import scalar_control
from run_layer0_query_memory_20261105 import raw_for
from run_rolling_safety_20261106 import cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


ARCHIVE = r.ROOT / "probe_outputs/direct_fast_memory_ridge360_20261104_v1"
FIRST, STRIDE, LAG, BATCH = 135, 32, 128, 256
WITNESS360 = (21,81,161,241,320,359)
WITNESS835 = (1,418,835)
ACTUAL = ("static","rls","lagged","constant")
ARMS = (*ACTUAL,"scalar")


def delayed_direction(delta, *, lag=LAG):
    """Scale the earlier direction to match the applied tanh-coefficient norm."""
    if lag <= 0 or delta.ndim != 2 or not len(delta):
        raise ValueError("positive delay and nonempty action matrix required")
    target = delta.tanh().square().sum(dim=1)
    i = torch.arange(len(delta),device=delta.device)
    latest = torch.cummax(torch.where(target>0.,i,-1),dim=0).values
    prior = torch.where(i>=lag,latest[(i-lag).clamp_min(0)],-1)
    past = delta[prior.clamp_min(0)]
    possible = (past!=0.).sum(dim=1)
    matched = (target>0.) & (prior>=0) & (target<=possible)
    source = torch.where(matched,prior,-1)
    out = delta.clone()
    d,goal = past[matched],target[matched]
    if len(d):
        a=d.abs()
        # At the low endpoint all coordinates are nearly zero; at the
        # high endpoint every nonzero coordinate has saturated.
        lo=np.log(1e-8)-a.amax(dim=1).log()
        hi=np.log(20.)-a.masked_fill(a==0.,torch.inf).amin(dim=1).log()
        for _ in range(25):
            mid=(lo+hi)*.5
            actual=(d*mid.exp()[:,None]).tanh().square().sum(dim=1)
            below=actual<goal
            lo,hi=torch.where(below,mid,lo),torch.where(below,hi,mid)
        out[matched]=d*((lo+hi)*.5).exp()[:,None]
    return out,source


def load_frozen():
    cfg,done=(r.read(ARCHIVE/f"{k}.json") for k in ("protocol","completed"))
    path=ARCHIVE/"fitted.pt"
    if (cfg["update_rule"]!="ridge" or cfg["threshold"]!=r.THRESHOLD
            or done["status"]!="COMPLETED_DIRECT_MEMORY_INTRA_DIAGNOSTIC"
            or r.sha(path)!=done["artifact_sha256"]["fitted.pt"]):
        raise ValueError("expected intact RLS fit from the original 360")
    for mod in (simple23,memory,model,rls,timeline):
        p=Path(mod.__file__).resolve()
        if r.sha(p)!=cfg["code_sha256"][str(p)]:
            raise ValueError(f"frozen RLS inference module changed: {p}")
    fixed=[q["shot"] for q in cfg["order"]]
    frozen=torch.load(path,map_location="cuda",weights_only=True)
    if frozen["update_rule"]!="ridge" or frozen["label_shots"]!=fixed[:20] or frozen["source_sha256"]!=cfg["source_sha256"]:
        raise ValueError("archived weights must be fit only from the original cold20")
    base,par=r.load_parent(r.DEFAULT_PARENT)
    base.cuda()
    if r.state_digest(base)!=cfg["source_state_sha256"] or par["threshold"]!=cfg["threshold"]:
        raise ValueError("wrong frozen J-TEXT checkpoint")
    injection=model.Injection().cuda().eval().requires_grad_(False)
    injection.load_state_dict(frozen["injection"],strict=True)
    map_path=r.COLD/"input_map.json"
    if r.sha(map_path)!=cfg["input_sha256"][str(map_path)]:
        raise ValueError("original EAST cold20 mapping changed")
    return base,injection,frozen["initial"],frozen["constant"],r.read(map_path),fixed,done


@torch.no_grad()
def predict_shot(base, injection, initial, constant, signal):
    device=base.center.device
    innovations,updates=timeline.trace(base,signal,initial,"real",device=device,rule="ridge")
    if any(z["arrival"]-z["origin"]!=8 or z["arrival"]!=FIRST+j*STRIDE for j,z in enumerate(updates)):
        raise ValueError("archived RLS arrival clock changed")
    x,a=r.latent.make_views(signal,device)
    n=len(x)
    cuts=list(range(0,n,BATCH))
    real=torch.cat([injection(innovations[k:k+BATCH]) for k in cuts])
    delayed,origins=delayed_direction(real)
    fixed=injection(constant[None]).expand(n,-1).clone()
    fixed[:FIRST]=0.
    if bool(real[:FIRST].abs().any()) or bool(delayed[:FIRST].abs().any()):
        raise ValueError("attempted internal write before the original activation time")
    active=real.norm(dim=1)>0.
    origin=base.initial_fast(n)
    eff=[((origin+z-origin).tanh()).norm(dim=1) for z in (real,delayed)]
    error=float((eff[0]-eff[1]).abs().max())
    if error>2e-5:
        raise ValueError(f"effective token adapter amplitude mismatch: {error:g}")
    eff_rel=(eff[0]-eff[1]).abs()/eff[0].clamp_min(1e-7)
    z={k:np.empty(n,dtype=np.float32) for k in ACTUAL}
    for k in cuts:
        s=slice(k,min(k+BATCH,n))
        xx,aa=x[s].contiguous(),a[s].contiguous()
        z["static"][s]=base(xx,aa)["logit"].sigmoid().cpu().numpy()
        for name,d in (("rls",real),("lagged",delayed),("constant",fixed)):
            z[name][s]=base(xx,aa,fast=origin[s]+d[s])["logit"].sigmoid().cpu().numpy()
    if any(not np.isfinite(p).all() for p in z.values()):
        raise FloatingPointError("nonfinite RLS risk trace")
    # Report diagnostics per shot; nothing below changes the state or risk.
    forecast=[np.mean([u[key] for u in updates]).item() for key in
              ("pre_update_mse","static_mse","hold_mse")] if updates else None
    matched=origins>=0
    return z,dict(releases=len(updates),active=int(active.sum()),delayed=int(matched.sum()),
                  effective_norm_diff=error,effective_rel_mean=float(eff_rel[matched].mean()) if bool(matched.any()) else None,
                  effective_rel_max=float(eff_rel.max()),forecast_mse=forecast)


def prefix_witness(base,inj,w0,const,signal,z,sid):
    stop=min(len(z["static"]),max(FIRST+1,len(z["static"])//2))
    prefix={k:signal[k][:stop] for k in ("values","available")}
    got,_=predict_shot(base,inj,w0,const,prefix)
    return dict(shot=sid,points=stop,maxerror=max(require_parity(p,z[k][:stop]) for k,p in got.items()))


def predict(out):
    start=time.monotonic()
    r.configure()
    base,inj,w0,const,mapping,fixed,old=load_frozen()
    frozen=(r.state_digest(base),r.state_digest(inj),w0.clone(),const.clone())
    reference,cfg,seal, (old_real,old_lag,warm)=cohort("conditional835")
    map_path=r.COLD/"input_map.json"
    if (warm!=0 or set(fixed)&set(cfg["order"]) or cfg["source_state_sha256"]!=frozen[0]
            or r.sha(map_path)!=cfg["input_sha256"][str(map_path)]):
        raise ValueError("original conditional deployment signals and mapping required")
    _,r360,_,_=cohort("fixed360")
    if r360["order"]!=fixed:
        raise ValueError("fixed source witness roster differs from training manifest")
    original_witness,prefix=[],[]
    out.mkdir(parents=True,exist_ok=False)
    (out/"shots").mkdir()
    protocol=out/"protocol.json"
    r.write_new(protocol,dict(status="FROZEN_RLS835_CONDITIONAL",order=cfg["order"],arms=ARMS,
        threshold=r.THRESHOLD,fit_sha256=r.sha(ARCHIVE/"fitted.pt"),source_state=frozen[0],
        cold20=fixed[:20],first=FIRST,stride=STRIDE,lag=LAG,
        match="l2_of_tanh_applied_gain_bias",added_fit_labels=0,inter_updates=0))
    records=[]
    with patch.object(r.io23,"read_outcome",side_effect=RuntimeError("outcomes sealed off during prediction")):
        for rank in WITNESS360:
            sid=fixed[rank-1]
            raw=raw_for(sid,r360["raw_sha256"])
            signal=r.prepare_actual(raw,rank,mapping)
            z,_=predict_shot(base,inj,w0,const,signal)
            p=ARCHIVE/"shots"/f"{rank:04d}_{sid}.npz"
            if r.sha(p)!=old["seals"][rank-1]["curve_sha256"]:
                raise ValueError("archived 360 curve changed")
            with np.load(p,allow_pickle=False) as a:
                np.testing.assert_array_equal(raw["times"],a["times"])
                agreement={k:require_parity(z[k],a[j])
                           for k,j in (("static","native"),("rls","real"),("constant","constant"))}
            original_witness.append(dict(shot=sid,rank=rank,errors=agreement))
            if rank in (21,161,359):
                prefix.append(prefix_witness(base,inj,w0,const,signal,z,sid))
        print("fixed360 archived and truncated parity established",original_witness,flush=True)
        for rank,row in enumerate(seal["sealed"],1):
            sid=row["shot"]
            t,static,_,_=recorded(reference,row,rank,old_real,old_lag)
            raw=raw_for(sid,cfg["raw_sha256"])
            np.testing.assert_array_equal(t,raw["times"])
            signal=r.prepare_actual(raw,21,mapping)
            z,info=predict_shot(base,inj,w0,const,signal)
            err=require_parity(z["static"],static)
            if rank in WITNESS835:
                prefix.append(prefix_witness(base,inj,w0,const,signal,z,sid))
            path=out/"shots"/f"{rank:04d}_{sid}.npz"
            r.save_arrays(path,**{k:p for k,p in z.items() if k!="static"})
            records.append(dict(rank=rank,shot=sid,curve_sha256=r.sha(path),static_error=err,**info))
            if rank%100==0:
                print("predicted",rank,"of",len(cfg["order"]),"seconds",round(time.monotonic()-start,1),flush=True)
    if (r.state_digest(base)!=frozen[0] or r.state_digest(inj)!=frozen[1]
            or not torch.equal(w0,frozen[2]) or not torch.equal(const,frozen[3])):
        raise ValueError("archived parameters were modified")
    r.write_new(out/"completed.json",dict(status="SEALED_FROZEN_RLS835",seconds=time.monotonic()-start,
        protocol_sha256=r.sha(protocol),archive360=original_witness,prefix=prefix,
        baseline_completion_sha256=r.sha(reference/"completed.json"),records=records))
    print("sealed",len(records),flush=True)


def recount(out):
    start=time.monotonic()
    info,done=(r.read(out/f"{k}.json") for k in ("protocol","completed"))
    if (done["status"]!="SEALED_FROZEN_RLS835" or info["threshold"]!=r.THRESHOLD
            or done["protocol_sha256"]!=r.sha(out/"protocol.json")):
        raise ValueError("HDF5 outcomes can be used only after predictions are sealed")
    reference,cfg,seal,(old_real,old_lag,warm)=cohort("conditional835")
    if (cfg["order"]!=info["order"] or warm or r.sha(reference/"completed.json")!=done["baseline_completion_sha256"]
            or [p["shot"] for p in done["records"]]!=cfg["order"]):
        raise ValueError("conditional 835 ordering or sealed baseline changed")
    rows={k:[] for k in ARMS}
    for rank,(q,b) in enumerate(zip(done["records"],seal["sealed"],strict=True),1):
        sid=q["shot"]
        t,static,_,_=recorded(reference,b,rank,old_real,old_lag)
        p=out/"shots"/f"{rank:04d}_{sid}.npz"
        if r.sha(p)!=q["curve_sha256"]:
            raise ValueError("sealed adaptation trace changed")
        with np.load(p,allow_pickle=False) as f:
            curves={k:f[k] for k in ACTUAL if k!="static"}
        for v in curves.values():
            require_parity(v[:FIRST],static[:FIRST])
        curves["static"]=static
        curves["scalar"]=scalar_control(static,curves["rls"],first=FIRST,stride=STRIDE)
        truth,clock=raw_truth(r.io23.path_for("EAST",sid),sid)
        np.testing.assert_array_equal(t,clock)
        for k in ARMS:
            rows[k].append(score(t,curves[k].astype(np.float64),truth,r.THRESHOLD))
    metrics={k:counts(v) for k,v in rows.items()}
    if (metrics["static"]["tp"],metrics["static"]["fp"])!=(257,68):
        raise ValueError("wrong archived baseline or HDF5 clock")
    for k,v in rows.items():
        item=metrics[k]
        item["tp_30_to_125"]=sum(row["status"]=="tp" and row["lead_ms"]<=125 for row in v)
        item["tp_125_to_3000"]=item["tp_30_to_3000"]-item["tp_30_to_125"]
        item["j_30_to_3000"]=item["tp_30_to_3000"]/item["n_disruptive"]-item["fpr"]
    paired={k:dict(strict=pair(rows["rls"],rows[k]), capped=pair(rows["rls"],rows[k],capped=True))
            for k in ARMS if k!="rls"}
    active=metrics["rls"]
    passes=(active["fpr"]<=.20 and active["tp"]>metrics["static"]["tp"]
            and active["j_30_to_3000"]>max(metrics[k]["j_30_to_3000"] for k in ("static","lagged","scalar")))
    r.write_new(out/"result.json",dict(status="HDF5835_RLS_RECOUNT",seconds=time.monotonic()-start,
        metrics=metrics,paired=paired,rows=rows,promote=passes))
    print({k:(v["tp"],v["tp_30_to_3000"],v["fp"]) for k,v in metrics.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("predict","score"))
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    (predict if a.phase=="predict" else recount)(a.out)
