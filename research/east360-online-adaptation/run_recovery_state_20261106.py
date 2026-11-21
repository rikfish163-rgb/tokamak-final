"""Frozen-parent EAST replay of a label-free, shot-local recovery state."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import recovery_state_20261106 as op
from rolling_safety_policy_20261106 import scalar_control
from run_layer0_query_memory_20261105 import parents, raw_for
from run_rolling_safety_20261106 import COHORTS, WITNESSES, cohort, recorded, require_parity
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


CHANNELS = ("ip","v_loop","dx_a","dy_a","Mir_VV","Mir_avg_amp","Mir_avg_fre",
            "n=1 amplitude","sxr_mean","sxr_var","xuv_var","CIII")
COLUMNS = np.array([r.io23.RAW23.index(k) for k in CHANNELS])
WRITE = np.array([k not in ("ip","v_loop") for k in CHANNELS])
WRITES = ("recovery","mismatched","combined","combined_mismatched")
MODES = ("static","original",*WRITES,"scalar","combined_scalar")


@torch.no_grad()
def shifted_probability(base,x,a,bias,fast=None):
    def append(_, __, out):
        return torch.cat((out[:,:-4], out[:,-4:]+bias),dim=1)

    h = base.embedding.register_forward_hook(append)
    try:
        if fast is None:
            z = base(x,a)["logit"]
        else:
            z = r.policy.routed_pair(base,x,a,r.latent.routes_all(a),fast)
        return z.sigmoid().cpu().numpy()
    finally:
        h.remove()


def latent_state(rule,ctx):
    true = r.actions(rule,ctx)[:, :128].contiguous()
    with torch.enable_grad():
        return r.policy.write_fast(torch.cat((true,torch.zeros_like(true)),dim=1))


def case(base, rule, raw, rank, mapping, static, original, *, verify_old=False):
    v = r.prepare_actual(raw,rank,mapping)
    col = COLUMNS
    a = v["available"][:,col]
    center,scale = (getattr(base,k).cpu().numpy()[col] for k in ("center","scale"))
    coords = np.arcsinh((np.where(a,v["values"][:,col],center)-center)/scale)
    q,u,note = op.recovery_field(coords,a,WRITE)
    n = len(static)
    result = {k:(original.copy() if k.startswith("combined") else static.copy()) for k in WRITES}
    energy = np.zeros(n,dtype=np.float32)
    eligible = np.flatnonzero(np.any(q,axis=1) & np.any(u,axis=1))
    if not len(eligible):
        return result,energy,note | {"native_error":0.,"original_error":0.,"norm_mismatch":0.}
    ctx = r.latent.native_context(base,v)
    x,mask,p0 = ctx[:3]
    native_error = require_parity(p0,static)
    old = latent_state(rule,ctx)
    weight = base.embedding.weight.reshape(64,8,46)[:,:,col]
    norm_error,old_error = 0.,0.
    for offset in range(0,len(eligible),r.BATCH):
        ix = eligible[offset:offset+r.BATCH]
        xx,aa = x[ix].contiguous(),mask[ix].contiguous()
        obs = aa[:,-32:,col] & torch.isfinite(xx[:,-32:,col])
        raw_bias = (op.value_bias(weight,obs,xx.new_tensor(field[ix])) for field in (q,u))
        true,wrong = op.matched_bias(*raw_bias)
        ne,nw = (z.norm(dim=(1,2)) for z in (true,wrong))
        mismatch = float((ne-nw).abs().max())
        if mismatch > 1e-6 or float(ne.max()) > op.CAP+1e-6:
            raise ValueError("paired token action energies disagree")
        norm_error = max(norm_error,mismatch)
        energy[ix] = ne.cpu().numpy()
        use = ne > 0.
        if not bool(use.any()):
            continue
        used = ix[use.cpu().numpy()]
        xb,ab = xx[use],aa[use]
        fast = old[(used-op.FIRST)//op.STRIDE]
        if verify_old and old_error == 0. and offset == 0:
            zero = torch.zeros_like(true[use])
            old_error = require_parity(shifted_probability(base,xb,ab,zero,fast),original[used])
        for name,bias,active in (("recovery",true,None),("mismatched",wrong,None),
                                 ("combined",true,fast),("combined_mismatched",wrong,fast)):
            result[name][used] = shifted_probability(base,xb,ab,bias[use],active)
    return result,energy,note | {"native_error":native_error,"original_error":old_error,"norm_mismatch":norm_error}


def prefix_witness(base,rule,raw,rank,mapping,static,original,expected,energy):
    n = min(len(static),max(op.FIRST+2, len(static)//2))
    clipped = {**raw, **{k:raw[k][:n] for k in ("times","values","available")}}
    got,e,_ = case(base,rule,clipped,rank,mapping,static[:n],original[:n],verify_old=True)
    np.testing.assert_array_equal(e>0.,energy[:n]>0.)
    np.testing.assert_allclose(e,energy[:n],atol=1e-6,rtol=0.)
    error = max(require_parity(got[k],expected[k][:n]) for k in WRITES)
    return dict(shot=raw["shot"], points=n, active=int((e>0.).sum()), max_error=error)


def predict(out):
    start = time.monotonic()
    r.configure()
    order, _, _, base, source, mapping, _, fit_path = parents()
    if ((base.cfg.window,base.cfg.patch,base.cfg.input_view)!=(128,8,"source")
            or base.embedding.in_features!=368):
        raise ValueError("expected frozen native 128/8/23 source embedding")
    rule = r.load_rule(fit_path,base.center.device)
    out.mkdir(parents=True,exist_ok=False)
    cfg_path = out/"protocol.json"
    r.write_new(cfg_path,dict(status="NO_LABEL_RECOVERY_STATE",modes=MODES,source_state=source,
        threshold=r.THRESHOLD,first_release=op.FIRST,stride=op.STRIDE,history=op.HISTORY,
        ridge=op.RIDGE,fit_floor=op.FLOOR,energy_cap=op.CAP,channels=CHANNELS,
        written=[k for k,yes in zip(CHANNELS,WRITE,strict=True) if yes],new_label_budget=0,
        inter_updates=0,evaluation_outcomes_during_prediction=0))
    all_rows = {}
    with patch.object(r.io23,"read_outcome",side_effect=RuntimeError("current outcome is unavailable")):
        for name in COHORTS:
            folder,cfg,prior,(real,stale,warm) = cohort(name)
            if (cfg["source_state_sha256"]!=source or cfg["fit_sha256"]!=r.sha(fit_path)
                    or (warm and cfg["order"]!=order) or (not warm and set(cfg["order"])&set(order))):
                raise ValueError("wrong original shot roster or source model")
            dest = out/name
            dest.mkdir()
            notes,witnesses = [],[]
            for rank,row in enumerate(prior["sealed"],1):
                sid = row["shot"]
                t,static,original,_ = recorded(folder,row,rank,real,stale)
                if rank<=warm:
                    continue
                raw = raw_for(sid,cfg["raw_sha256"])
                np.testing.assert_array_equal(t,raw["times"])
                prep_rank = rank if warm else 21
                probes = rank in WITNESSES[name]
                z,energy,diag = case(base,rule,raw,prep_rank,mapping,static,original,verify_old=probes)
                if probes:
                    witnesses.append(prefix_witness(base,rule,raw,prep_rank,mapping,static,original,z,energy))
                path = dest/f"{rank:04d}_{sid}.npz"
                r.save_arrays(path,energy=energy,**z)
                notes.append(dict(shot=sid,rank=rank,curve_sha256=r.sha(path),
                    active=int((energy>0.).sum()),max_energy=float(energy.max()),**diag))
                if rank%120==0:
                    print(name,rank,flush=True)
            all_rows[name] = dict(predictions=notes,witnesses=witnesses,
                                  parent_sha256=r.sha(folder/"completed.json"))
    if r.state_digest(base)!=source or any(p.grad is not None for p in base.parameters()):
        raise ValueError("frozen source changed")
    r.write_new(out/"completed.json",dict(status="SEALED_RECOVERY_PREDICTIONS",
        seconds=time.monotonic()-start,protocol_sha256=r.sha(cfg_path),cohorts=all_rows))
    print("sealed", {k:(len(v["predictions"]),sum(p["active"] for p in v["predictions"]))
                      for k,v in all_rows.items()},flush=True)


def recount(out):
    start=time.monotonic()
    done,protocol=(r.read(out/f"{x}.json") for x in ("completed","protocol"))
    if (done["status"]!="SEALED_RECOVERY_PREDICTIONS" or protocol["threshold"]!=r.THRESHOLD
            or done["protocol_sha256"]!=r.sha(out/"protocol.json")):
        raise ValueError("predictions must be sealed before opening HDF5 outcomes")
    evaluated={}
    for name in COHORTS:
        folder,cfg,prior,(real,stale,warm)=cohort(name)
        sealed=done["cohorts"][name]
        pred={row["rank"]:row for row in sealed["predictions"]}
        if (r.sha(folder/"completed.json")!=sealed["parent_sha256"]
                or set(pred)!=set(range(warm+1,len(cfg["order"])+1))):
            raise ValueError("missing predictions or original archived parent")
        rows={k:[] for k in MODES}
        categories=[]
        for rank,row in enumerate(prior["sealed"],1):
            sid=row["shot"]
            t,s,a,_=recorded(folder,row,rank,real,stale)
            curves={"static":s,"original":a}
            energy=np.zeros(len(t))
            if rank>warm:
                file=out/name/f"{rank:04d}_{sid}.npz"
                if sid!=pred[rank]["shot"] or r.sha(file)!=pred[rank]["curve_sha256"]:
                    raise ValueError("shot prediction seal changed")
                with np.load(file,allow_pickle=False) as f:
                    energy=f["energy"]
                    curves.update({k:f[k] for k in WRITES})
                if (np.any(energy[:op.FIRST+1]) or np.max(energy)>op.CAP+1e-6
                        or int((energy>0.).sum())!=pred[rank]["active"]):
                    raise ValueError("write timeline or energy bound violated")
                for k in WRITES:
                    old=a if k.startswith("combined") else s
                    np.testing.assert_array_equal(curves[k][energy==0.],old[energy==0.])
            else:
                curves.update({k:a if k.startswith("combined") else s for k in WRITES})
            curves["scalar"]=scalar_control(s,curves["recovery"],first=op.FIRST+1,stride=op.STRIDE)
            curves["combined_scalar"]=scalar_control(a,curves["combined"],first=op.FIRST+1,stride=op.STRIDE)
            truth,clock=raw_truth(r.io23.path_for("EAST",sid),sid)
            np.testing.assert_array_equal(t,clock)
            for mode in MODES:
                rows[mode].append(score(t,curves[mode].astype(np.float64),truth,r.THRESHOLD))
            if not truth["disruptive"]:
                above0,above1=(p.astype(float)>=r.THRESHOLD for p in (s,a))
                stops=np.arange(op.FIRST,len(t),op.STRIDE)
                before=bool(above0[:op.FIRST].any() or (rank<=warm and above0.any()))
                forced=not before and bool(np.any(np.logical_or.reduceat(above0,stops)
                                                & np.logical_or.reduceat(above1,stops)))
                if before or forced:
                    categories.append(dict(shot=sid,immutable_prefix=before,old_both_cross=forced,
                        recovery_clears=rows["recovery"][-1]["status"]=="tn",
                        combined_clears=rows["combined"][-1]["status"]=="tn",
                        matched_frames=int((energy>0.).sum())))
        metrics={k:counts(x) for k,x in rows.items()}
        s,a=(metrics[k] for k in ("static","original"))
        if (s["tp"],s["fp"],a["tp"],a["fp"])!=((116,29,120,33) if warm else (257,68,299,85)):
            raise ValueError("archived reference does not match original HDF5 scores")
        paired={k:{j:dict(strict=pair(rows[k],rows[j]),capped=pair(rows[k],rows[j],capped=True))
                 for j in ("static","original",*controls)} for k,controls in
                (("recovery",("mismatched","scalar")),("combined",("combined_mismatched","combined_scalar")))}
        evaluated[name]=dict(metrics=metrics,paired=paired,old_menu_false_alarms=categories,rows=rows)
        print(name,{k:(p["tp"],p["tp_30_to_3000"],p["fp"]) for k,p in metrics.items()},flush=True)
    primary,transfer=(evaluated[k]["metrics"] for k in COHORTS)
    decision={}
    for k,controls in (("recovery",("mismatched","scalar")),("combined",("combined_mismatched","combined_scalar"))):
        decision[k]=(primary[k]["tp"]>primary["static"]["tp"] and primary[k]["fpr"]<=.20
            and all(primary[k]["j"]>primary[j]["j"]+1e-12 for j in controls)
            and transfer[k]["fpr"]<=.20 and transfer[k]["tp"]>=transfer["static"]["tp"])
    r.write_new(out/"result.json",dict(status="EAST_HDF5_RECOVERY_RECOUNT",decision=decision,
        seconds=time.monotonic()-start,cohorts=evaluated))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("predict","score"))
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    (predict if a.phase=="predict" else recount)(a.out)
